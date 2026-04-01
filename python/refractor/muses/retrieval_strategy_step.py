from __future__ import annotations
import abc
from loguru import logger
import refractor.framework as rf  # type: ignore
from .creator_dict import CreatorDict
from .creator_handle import CreatorHandle, CreatorHandleWithContextSet
from .qa_data_handle import QaFlag
from .muses_levmar_solver import (
    MusesLevmarSolver,
    VerboseSolverLogging,
    SolverLogFileWriter,
)
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_result import RetrievalResult
from .identifier import RetrievalType, ProcessLocation
from .retrieval_array import RetrievalGridArray
import numpy as np
import json
import gzip
import os
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import CurrentState
    from .muses_strategy_executor import CurrentStrategyStep
    from .retrieval_strategy import RetrievalStrategy
    from .muses_strategy import MusesStrategy
    from .forward_model_combine import ForwardModelCombine
    from .retrieval_result import RetrievalResult
    from .cost_function import CostFunction
    from .muses_levmar_solver import SolverResult
    from .muses_strategy_context import MusesStrategyContext
    from .identifier import StrategyStepIdentifier, InstrumentIdentifier
    from pathlib import Path


class RetrievalStrategyStepSet(CreatorHandleWithContextSet):
    """This takes the retrieval_type and determines a
    RetrievalStrategyStep to handle this, returning it so it can
    be called.

    Note RetrievalStrategyStep can assume that they are called for the
    same MusesStrategyContext, until notify_update_strategy_context is
    called. So if it makes sense, these objects can do internal
    caching for things that don't change when the context being
    retrieved is the same from one call to the next.
    """

    def __init__(self,
                 strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("retrieval_step", strategy_context)

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> RetrievalStrategyStep:
        return self.handle(retrieval_type, rs, creator_dict, **kwargs)


class RetrievalStrategyStepHandle(CreatorHandle):
    """Right now our strategy steps just key off of the retrieval_type
    being in a set (or None to match any). We have this handle for
    this case. We can certainly create other CreatorHandle if we have
    RetrievalStrategyStep with more complicated selection logic."""

    def __init__(
        self,
        cls: type[RetrievalStrategyStep],
        retrieval_type_set: set[RetrievalType] | None = None,
    ) -> None:
        super().__init__()
        self._create_cls = cls
        self._retrieval_type_set = retrieval_type_set

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> RetrievalStrategyStep | None:
        if (
            self._retrieval_type_set is None
            or retrieval_type in self._retrieval_type_set
        ):
            return self._create_cls(retrieval_type, rs, creator_dict, **kwargs)
        return None


class RetrievalStrategyStep(object, metaclass=abc.ABCMeta):
    """Do the retrieval step indicated by retrieval_type

    We *only* maintain state between steps in CurrentState (other than
    possibly internal caching for performance). If the same
    RetrievalStrategyStep is called with the same CurrentState, it
    will produce the same output.

    A RetrievalStrategyStep *only* modifies CurrentState and produces
    output through side effects. Note that the RetrievalStrategyStep
    doesn't directly produce output, instead we use the
    Observer/Observable pattern to decouple generation of output. A
    RetrievalStrategyStep calls RetrievalStrategy.notify_update_target
    at points during the processing, where the Observer can do
    whatever with the processing (e.g, produce an output file).

    Note this current class has a number of things related to optimal
    estimation (e.g., forward model) that don't really apply to a
    different kind of retrieval (e.g., machine learning). We may want
    to separate this out. Right now, it seems harmless - we just have
    a bunch of functions we don't need for machine learning. But it
    might be better to separate this out, once we have a clearer
    picture of the differences between the two.

    """

    def __init__(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> None:
        self._saved_state: None | dict[str, Any] = None
        self.retrieval_type = retrieval_type
        # TODO I think we will want to remove the direct use of RetrievalStrategy,
        # although I'm not sure. For now we use this.
        self.rs = rs
        self.creator_dict = creator_dict
        self.kwargs = kwargs

    def do_retrieval(
        self,
    ) -> None:
        self.set_state(self.kwargs.get("ret_state", None))
        self.retrieval_step_body()
        self.notify_update(ProcessLocation("end_retrieval_step"))

    @abc.abstractmethod
    def retrieval_step_body(self) -> None:
        """Actual do the retrieval step"""

    def get_state(self) -> dict[str, Any]:
        """Return a dictionary of values that can be used by
        set_state.  This allows us to skip pieces of the retrieval
        step. This is similar to a pickle serialization (which we also
        support), but only saves the things that change when we update
        the parameters.

        This is useful for unit tests of side effects of doing the retrieval
        step (e.g., generating output files) without needing to actually
        run the retrieval.

        """
        # Default, no state
        return {}

    def set_state(self, d: dict[str, Any] | None) -> None:
        """Set the state previously saved by get_state"""
        # Default, just put this into the _saved_state for use
        # in the rest of this object
        self._saved_state = d

    # Just to make the interface clear, we pull out the things we need
    # from RetrievalStrategy here. We may add to this list, or get these
    # somewhere else in the future
    def create_forward_model_combine(
        self,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
    ) -> ForwardModelCombine:
        return self.rs.create_forward_model_combine(
            use_systematic=use_systematic, include_bad_sample=include_bad_sample
        )

    def create_forward_model(self) -> rf.ForwardModel:
        return self.rs.strategy_executor.create_forward_model()

    def notify_update(self, ploc: ProcessLocation) -> None:
        self.rs.notify_update(ploc, retrieval_strategy_step=self)

    def create_cost_function(self) -> CostFunction:
        return self.rs.create_cost_function()

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        return self.rs.retrieval_config

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        return self.rs.current_strategy_step

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        return self.rs.strategy_step

    @property
    def strategy(self) -> MusesStrategy:
        return self.rs.strategy

    @property
    def current_state(self) -> CurrentState:
        return self.rs.current_state

    @property
    def instrument_name_all_step(self) -> list[InstrumentIdentifier]:
        return self.rs.instrument_name_all_step

    @property
    def write_output(self) -> bool:
        return self.rs.write_output

    @property
    def run_dir(self) -> Path:
        return self.rs.run_dir


class RetrievalStrategyStepNotImplemented(RetrievalStrategyStep):
    """There seems to be a few retrieval types that aren't implemented
    in py-retrieve. It might also be that we just don't have this
    implemented right in ReFRACtor, but in any case we don't have a
    test case for this.

    Throw an exception to indicate this, we can look at implementing
    this in the future - particularly if we have a test case to
    validate the code.

    """

    def retrieval_step_body(self) -> None:
        raise RuntimeError(
            f"We don't currently support retrieval_type {self.retrieval_type}"
        )


class RetrievalStrategyStepRetrieve(RetrievalStrategyStep):
    """Strategy step that does a retrieval (e.g., the default strategy
    step)."""

    def __init__(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> None:
        super().__init__(retrieval_type, rs, creator_dict, **kwargs)
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
        self.notify_update(ProcessLocation("retrieval input"))
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
        self.notify_update(ProcessLocation("run_retrieval_step"))

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
        self.notify_update(ProcessLocation("systematic_jacobian"))

        # Note a side effect of this is calling current_state.update_previous_aposteriori_cov_fm
        # and current_state.propagated_qa.update. See discussion in RetrievalResult.
        olist = [
            self.rs.observation_handle_set.observation(iname, None, None, None)
            for iname in self.instrument_name_all_step
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
        self.notify_update(ProcessLocation("retrieval step"))

    def run_retrieval(self) -> SolverResult:
        """run_retrieval"""
        self.cfunc = self.create_cost_function()
        cost_function_params = self.kwargs["cost_function_params"]
        self.notify_update(ProcessLocation("create_cost_function"))
        chi2_tolerance = cost_function_params["chi2_tolerance"]
        if chi2_tolerance is None:
            r = mpy_radiance_from_observation_list(
                self.cfunc.obs_list, include_bad_sample=True
            )["NESR"]
            chi2_tolerance = 2.0 / len(r)  # theoretical value for tolerance
        logger.info(f"Initial State vector:\n{self.cfunc.fm_sv}")
        self.slv = MusesLevmarSolver(
            self.cfunc,
            cost_function_params["max_iter"],
            cost_function_params["delta_value"],
            cost_function_params["conv_tolerance"],
            chi2_tolerance,
        )
        # For now, assume we want verbose logging
        if True:
            self.slv.add_observer(VerboseSolverLogging())
        if self.write_output:
            levmar_log_file = f"{self.run_dir}/Step{self.strategy_step.step_number:02d}_{self.strategy_step.step_name}/LevmarSolver-{self.strategy_step.step_name}.log"
            self.slv.add_observer(SolverLogFileWriter(levmar_log_file))
        if self._saved_state is not None:
            # Skip solve if we have a saved state.
            self.slv.set_state(self._saved_state["slv"])
        else:
            if cost_function_params["max_iter"] > 0:
                self.slv.solve()
        logger.info(f"Solved State vector:\n{self.cfunc.fm_sv}")
        return self.slv.retrieval_results()


RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(
        RetrievalStrategyStepNotImplemented,
        {RetrievalType("forwardmodel"), RetrievalType("omi_radiance_calibration")},
    )
)
# Anything that isn't one of the special types is a generic retrieval, so
# set to this as the lowest priority fall back
RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(RetrievalStrategyStepRetrieve, None), priority_order=-1
)

CreatorDict.register(RetrievalStrategyStep, RetrievalStrategyStepSet)


class RetrievalStepCaptureObserver:
    """Helper class, saves results of retrieval step so we can rerun skipping
    much of the calculation. Data saved when notify_update is called.
    Intended for unit tests and other kinds of debugging.

    Note this only saves the RetrievalStepCapture.get_state(),
    not the StateInfo. You will probably want to use this with
    something like StateInfoCaptureObserver.

    Note we can also save full pickles of the RetrievalStep, however
    since we are modifying the code this tends to be fragile. This might
    change in the future when things are more stable, but particularly for
    unit tests the json state is more fundamental and should be more stable.
    """

    def __init__(
        self, basefname: str, location_to_capture: str = "end_retrieval_step"
    ) -> None:
        self.basefname = basefname
        self.location_to_capture = ProcessLocation(location_to_capture)

    @classmethod
    def load_retrieval_state(cls, fname: str | os.PathLike[str]) -> dict[str, Any]:
        return json.loads(gzip.open(fname, mode="rb").read().decode("utf-8"))

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        if location != self.location_to_capture:
            return
        if retrieval_strategy_step is None:
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        fname = (
            f"{self.basefname}_{retrieval_strategy.strategy_step.step_number}.json.gz"
        )
        with gzip.open(fname, mode="wb") as fh:
            fh.write(
                json.dumps(
                    retrieval_strategy_step.get_state(), sort_keys=True, indent=4
                ).encode("utf-8")
            )


__all__ = [
    "RetrievalStrategyStepSet",
    "RetrievalStrategyStep",
    "RetrievalStrategyStepNotImplemented",
    "RetrievalStrategyStepRetrieve",
    "RetrievalStepCaptureObserver",
]
