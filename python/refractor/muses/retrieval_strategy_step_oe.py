from __future__ import annotations
from loguru import logger
import refractor.framework as rf  # type: ignore
from .creator_dict import CreatorDict
from .qa_data_handle import QaFlag
from .muses_levmar_solver import (
    MusesLevmarSolver,
    VerboseSolverLogging,
    SolverLogFileWriter,
)
from .cost_function import CostFunction
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_result import RetrievalResult
from .identifier import ProcessLocation
from .retrieval_array import RetrievalGridArray
from .retrieval_strategy_step import (
    RetrievalStrategyStep,
    RetrievalStrategyStepHandle,
    RetrievalStrategyStepSet,
)
import numpy as np
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .forward_model_combine import ForwardModelCombine
    from .retrieval_result import RetrievalResult
    from .muses_levmar_solver import SolverResult
    from .current_state import CurrentState
    from .process_location_observable import ProcessLocationObservable


class RetrievalStrategyStepOEBase(RetrievalStrategyStep):
    """This is a RetrievalStrategyStep with the additional stuff needed
    for Optimal Estimation (e.g. creating a forward model"""

    # Just to make the interface clear, we pull out the things we need
    # from RetrievalStrategy here. We may add to this list, or get these
    # somewhere else in the future
    def create_forward_model_combine(
        self,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
    ) -> ForwardModelCombine:
        return self.creator_dict[CostFunction].forward_model(
            self.creator_dict,
            self.current_strategy_step.instrument_name,
            self.current_state,
            self.current_strategy_step.spectral_window_dict,
            use_systematic=use_systematic,
            include_bad_sample=include_bad_sample,
            **self.kwargs,
        )

    def create_forward_model(self) -> rf.ForwardModel:
        if len(self.current_strategy_step.instrument_name) != 1:
            raise RuntimeError(
                "create_forward_model can only work with one instrument, we don't have handling for multiple."
            )
        iname = self.current_strategy_step.instrument_name[0]
        obs = self.creator_dict[rf.Observation].observation(
            iname, None, self.current_strategy_step.spectral_window_dict[iname], None
        )
        fm_sv = self.current_state.setup_fm_state_vector()
        fm = self.creator_dict[rf.ForwardModel].forward_model(
            iname,
            self.current_state,
            obs,
            fm_sv,
        )
        return fm

    def create_cost_function(self) -> CostFunction:
        return self.creator_dict[CostFunction].cost_function(
            self.creator_dict,
            self.current_strategy_step.instrument_name,
            self.current_state,
            self.current_strategy_step.spectral_window_dict,
            **self.kwargs,
        )


class RetrievalStrategyStepRetrieve(RetrievalStrategyStepOEBase):
    """Strategy step that does a retrieval (e.g., the default strategy
    step)."""

    def __init__(
        self,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            creator_dict, current_state, process_location_observable, **kwargs
        )
        self.slv: None | MusesLevmarSolver = None
        self.cfunc: None | CostFunction = None
        self.jacobian_sys: None | np.ndarray = None

    def get_state(self) -> dict[str, Any]:
        """Return a dictionary of values that can be used by
        set_state.  This allows us to skip pieces of the retrieval
        step. This is similar to a pickle serialization (which we also
        support), but only saves the things that change when we update
        the parameters.

        This is useful for unit tests of side effects of doing the
        retrieval step (e.g., generating output files) without needing
        to actually run the retrieval.

        """
        res: dict[str, Any] = {"slv": None, "jacobian_sys": None}
        if self.slv is not None:
            res["slv"] = self.slv.get_state()
        if self.jacobian_sys is not None:
            res["jacobian_sys"] = self.jacobian_sys.tolist()
        return res

    def retrieval_step_body(self) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        self.notify_process_location(ProcessLocation("retrieval input"))
        logger.info("Running run_retrieval ...")

        self.ret_res = self.run_retrieval()
        if self.cfunc is None:
            raise RuntimeError("self.cfunc should not be None")
        # self.cfunc.parameters set to the best iteration solution in MusesLevmarSolver
        self.current_strategy_step.notify_step_solution(
            self.current_state, self.cfunc.parameters.view(RetrievalGridArray)
        )
        logger.info("\n---")
        logger.info(str(self.strategy_step))
        logger.info(
            f"Best iteration {self.ret_res.bestIteration} out of {self.ret_res.num_iterations}"
        )
        logger.info("---\n")
        self.notify_process_location(ProcessLocation("run_retrieval_step"))

        # TODO jacobian_sys is only used in error_analysis_wrapper and
        # error_analysis.  I think we can leave bad sample out,
        # although I'm not positive. Would be nice not to have special
        # handling to add bad samples if we turn around and weed them
        # out.
        #
        # For right now, these are required, we would need to update
        # the error analysis to work without bad samples
        self.jacobian_sys = None
        if len(self.current_state.systematic_state_element_id) > 0:
            if self._saved_state is not None:
                # Skip forward model if we have a saved state.
                self.jacobian_sys = np.array(self._saved_state["jacobian_sys"])
            else:
                logger.info("Running run_forward_model for systematic jacobians ...")
                fm_sys = self.create_forward_model_combine(
                    use_systematic=True,
                    include_bad_sample=True,
                )
                self.jacobian_sys = fm_sys.model_measure_diff_jacobian().transpose()[
                    np.newaxis, :, :
                ]
                # Sanity check, if we need to look at this
                if False:
                    assert (
                        self.jacobian_sys[:, :, self.cfunc.good_point()].shape[-1]
                        == self.cfunc.max_a_posteriori.model_measure_diff_jacobian_fm.transpose().shape[
                            -1
                        ]
                    )
        self.notify_process_location(ProcessLocation("systematic_jacobian"))

        # Note a side effect of this is calling current_state.update_previous_aposteriori_cov_fm
        # and current_state.propagated_qa.update. See discussion in RetrievalResult.
        olist = [
            self.creator_dict[rf.Observation].observation(iname, None, None, None)
            for iname in self.strategy.instrument_name
        ]
        rfull = mpy_radiance_from_observation_list(olist, full_band=True)
        self.results = RetrievalResult(
            self.ret_res,
            self.current_state,
            self.current_strategy_step,
            self.cfunc.obs_list,
            rfull,
            self.current_state.propagated_qa,
            self.creator_dict[QaFlag],
            jacobian_sys=self.jacobian_sys,
        )
        self.notify_process_location(ProcessLocation("retrieval step"))

    def run_retrieval(self) -> SolverResult:
        """run_retrieval"""
        self.cfunc = self.create_cost_function()
        cost_function_params = self.kwargs["cost_function_params"]
        self.notify_process_location(ProcessLocation("create_cost_function"))
        chi2_tolerance = cost_function_params["chi2_tolerance"]
        if chi2_tolerance is None:
            r = mpy_radiance_from_observation_list(
                self.cfunc.obs_list, include_bad_sample=True
            )["NESR"]
            chi2_tolerance = 2.0 / len(r)  # theoretical value for tolerance
        logger.info(f"Initial State vector:\n{self.cfunc.fm_sv}")
        self.slv = MusesLevmarSolver(
            self.cfunc,
            self.process_location_observable,
            cost_function_params["max_iter"],
            cost_function_params["delta_value"],
            cost_function_params["conv_tolerance"],
            chi2_tolerance,
        )
        # For now, assume we want verbose logging
        # Move to retrieval_strategy
        if False:
            self.slv.add_observer(VerboseSolverLogging())
        if False and self.rs.write_output:
            levmar_log_file = f"{self.retrieval_config['output_directory']}/Step{self.strategy_step.step_number:02d}_{self.strategy_step.step_name}/LevmarSolver-{self.strategy_step.step_name}.log"
            self.slv.add_observer(SolverLogFileWriter(levmar_log_file))
        if self._saved_state is not None:
            # Skip solve if we have a saved state.
            self.slv.set_state(self._saved_state["slv"])
        else:
            if cost_function_params["max_iter"] > 0:
                self.slv.solve()
        logger.info(f"Solved State vector:\n{self.cfunc.fm_sv}")
        return self.slv.retrieval_results()


# Anything that isn't one of the special types is a generic retrieval, so
# set to this as the lowest priority fall back
RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(RetrievalStrategyStepRetrieve, None), priority_order=-1
)

__all__ = [
    "RetrievalStrategyStepOEBase",
    "RetrievalStrategyStepRetrieve",
]
