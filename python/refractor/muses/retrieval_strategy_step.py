from __future__ import annotations
import abc
from loguru import logger
from pathlib import Path
from .priority_handle_set import PriorityHandleSet
from .muses_levmar_solver import MusesLevmarSolver
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
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_result import RetrievalResult
    from .cost_function import CostFunction
    from .muses_levmar_solver import SolverResult

# TODO clean up the usage for various internal objects of
# RetrievalStrategy, we want to rework this anyways as we introduce
# the MusesStrategyExecutor.


class RetrievalStrategyStepSet(PriorityHandleSet):
    """This takes the retrieval_type and determines a
    RetrievalStrategyStep to handle this. It then does the retrieval
    step.

    Note RetrievalStrategyStep can assume that they are called for the
    same target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next.

    """

    def retrieval_step(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: Any
    ) -> None:
        self.handle(retrieval_type, rs, **kwargs)

    def notify_update_target(self, rs: RetrievalStrategy) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(rs)

    def handle_h(
        self,
        h: RetrievalStrategyStep,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        **kwargs: Any,
    ) -> tuple[bool, None]:
        return h.retrieval_step(retrieval_type, rs, **kwargs)


class RetrievalStrategyStep(object, metaclass=abc.ABCMeta):
    """Do the retrieval step indicated by retrieval_type

    Note RetrievalStrategyStep can assume that they are called for the
    same target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next.

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

    """

    def __init__(self) -> None:
        self._uip = None
        self._saved_state: None | dict[str, Any] = None
        self.results: None | RetrievalResult = None

    def notify_update_target(self, rs: RetrievalStrategy) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        ret_state: None | dict[str, Any] = None,
        **kwargs: Any,
    ) -> tuple[bool, None]:
        """Returns (True, None) if we handle the retrieval step,
        (False, None) otherwise

        """
        self.set_state(ret_state)
        was_handled = self.retrieval_step_body(retrieval_type, rs, **kwargs)
        if was_handled:
            rs.notify_update(
                ProcessLocation("end_retrieval_step"), retrieval_strategy_step=self
            )
        return (was_handled, None)

    @abc.abstractmethod
    def retrieval_step_body(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: Any
    ) -> bool:
        """Returns True if we handle the retrieval step, False otherwise"""
        raise NotImplementedError()

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

    def radiance_step(self) -> dict[str, Any]:
        """We have a few places that need the old py-retrieve dict
        version of our observation data. This function calculates
        that- it is just a reformatting of our observation data.
        """
        # I don't think this ever gets called by something that doesn't have
        # a cfunc, but go ahead and have handling for this just in case
        if hasattr(self, "cfunc"):
            return mpy_radiance_from_observation_list(
                self.cfunc.obs_list, include_bad_sample=True
            )
        else:
            return {}

    def radiance_full(self, rs: RetrievalStrategy) -> dict[str, Any]:
        """The full set of radiance, for all instruments and full band."""
        olist = [
            rs.observation_handle_set.observation(iname, None, None, None)
            for iname in rs.instrument_name_all_step
        ]
        return mpy_radiance_from_observation_list(olist, full_band=True)


class RetrievalStrategyStepNotImplemented(RetrievalStrategyStep):
    """There seems to be a few retrieval types that aren't implemented
    in py-retrieve. It might also be that we just don't have this
    implemented right in ReFRACtor, but in any case we don't have a
    test case for this.

    Throw an exception to indicate this, we can look at implementing
    this in the future - particularly if we have a test case to
    validate the code.

    """

    def retrieval_step_body(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: Any
    ) -> bool:
        if retrieval_type not in (
            RetrievalType("forwardmodel"),
            RetrievalType("omi_radiance_calibration"),
        ):
            return False
        raise RuntimeError(
            f"We don't currently support retrieval_type {retrieval_type}"
        )


class RetrievalStrategyStepRetrieve(RetrievalStrategyStep):
    """Strategy step that does a retrieval (e.g., the default strategy
    step)."""

    def __init__(self) -> None:
        super().__init__()
        self.notify_update_target(None)
        self.slv: None | MusesLevmarSolver = None
        self.cfunc: None | CostFunction = None
        self.cfunc_sys: None | CostFunction = None

    def notify_update_target(self, rs: RetrievalStrategy | None) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        # Nothing currently needed

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
        res: dict[str, Any] = {"slv": None, "cfunc_sys": None}
        if self.slv is not None:
            res["slv"] = self.slv.get_state()
        if self.cfunc_sys is not None:
            res["cfunc_sys"] = self.cfunc_sys.get_state()
        return res

    def retrieval_step_body(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: Any
    ) -> bool:
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        rs.notify_update(
            ProcessLocation("retrieval input"), retrieval_strategy_step=self
        )
        logger.info("Running run_retrieval ...")

        # SSK 2023.  I find I get failures from glitches like reading
        # L1B files, if there are many processes running.  I re-run
        # several times to give every obs a chance to complete chance
        # to run, because I am running the same set with different
        # strategies and want to compare.  write out a token if it
        # gets here, indicating this obs already got a chance.  I then
        # copy all completed runs to either 00good/ or 00bad/
        # depending on the success flag, and re-run anything remaining
        # in the main directory.

        Path(f"{rs.run_dir}/-run_token.asc").touch()

        self.ret_res = self.run_retrieval(rs, **kwargs)
        if self.cfunc is None:
            raise RuntimeError("self.cfunc should not be None")
        # self.cfunc.parameters set to the best iteration solution in MusesLevmarSolver
        rs.current_strategy_step.notify_step_solution(
            rs.current_state, self.cfunc.parameters.view(RetrievalGridArray)
        )
        logger.info("\n---")
        logger.info(str(rs.strategy_step))
        logger.info(
            f"Best iteration {self.ret_res.bestIteration} out of {self.ret_res.num_iterations}"
        )
        logger.info("---\n")
        rs.notify_update(
            ProcessLocation("run_retrieval_step"), retrieval_strategy_step=self
        )

        # TODO jacobian_sys is only used in error_analysis_wrapper and
        # error_analysis.  I think we can leave bad sample out,
        # although I'm not positive. Would be nice not to have special
        # handling to add bad samples if we turn around and weed them
        # out.
        #
        # For right now, these are required, we would need to update
        # the error analysis to work without bad samples
        jacobian_sys = None
        if len(rs.current_state.systematic_state_element_id) > 0:
            self.cfunc_sys = rs.create_cost_function(
                do_systematic=True,
                include_bad_sample=True,
            )
            if self._saved_state is not None:
                # Skip forward model if we have a saved state.
                self.cfunc_sys.set_state(self._saved_state["cfunc_sys"])
            logger.info("Running run_forward_model for systematic jacobians ...")
            jacobian_sys = (
                self.cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[
                    np.newaxis, :, :
                ]
            )
        rs.notify_update(
            ProcessLocation("systematic_jacobian"), retrieval_strategy_step=self
        )

        # Note a side effect of this is calling current_state.update_previous_aposteriori_cov_fm
        # and current_state.propagated_qa.update. See discussion in RetrievalResult.
        self.results = RetrievalResult(
            self.ret_res,
            rs.current_state,
            rs.current_strategy_step,
            self.cfunc.obs_list,
            self.radiance_full(rs),
            rs.current_state.propagated_qa,
            rs.qa_data_handle_set,
            jacobian_sys=jacobian_sys,
        )
        rs.notify_update(
            ProcessLocation("retrieval step"), retrieval_strategy_step=self
        )
        return True

    def run_retrieval(
        self, rs: RetrievalStrategy, cost_function_params: dict = {}, **kwargs: Any
    ) -> SolverResult:
        """run_retrieval"""
        self.cfunc = rs.create_cost_function()
        rs.notify_update(
            ProcessLocation("create_cost_function"), retrieval_strategy_step=self
        )
        chi2_tolerance = cost_function_params["chi2_tolerance"]
        if chi2_tolerance is None:
            r = self.radiance_step()["NESR"]
            chi2_tolerance = 2.0 / len(r)  # theoretical value for tolerance
        if rs.write_output:
            levmar_log_file = f"{rs.run_dir}/Step{rs.strategy_step.step_number:02d}_{rs.strategy_step.step_name}/LevmarSolver-{rs.strategy_step.step_name}.log"
        else:
            levmar_log_file = None
        logger.info(f"Initial State vector:\n{self.cfunc.fm_sv}")
        self.slv = MusesLevmarSolver(
            self.cfunc,
            cost_function_params["max_iter"],
            cost_function_params["delta_value"],
            cost_function_params["conv_tolerance"],
            chi2_tolerance,
            verbose=True,
            log_file=levmar_log_file,
        )
        if self._saved_state is not None:
            # Skip solve if we have a saved state.
            self.slv.set_state(self._saved_state["slv"])
        else:
            if cost_function_params["max_iter"] > 0:
                self.slv.solve()
        logger.info(f"Solved State vector:\n{self.cfunc.fm_sv}")
        return self.slv.retrieval_results()


RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepNotImplemented())
# Anything that isn't one of the special types is a generic retrieval, so
# set to this as the lowest priority fall back
RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepRetrieve(), priority_order=-1
)


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
