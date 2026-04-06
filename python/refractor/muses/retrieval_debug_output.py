from __future__ import annotations
from .retrieval_output import RetrievalOutput
from .identifier import ProcessLocation
from loguru import logger
import os
import pickle
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .current_state import CurrentState

# We don't have all this in place yet, but put a few samples in place for output
# triggered by having "writeOutput" which is controlled by the --debug flag set


# TODO Clean this up, it has too tight a dependency on retrieval_strategy
class RetrievalInputOutput(RetrievalOutput):
    """Write out the retrieval inputs"""

    @property
    def error_current(self):  # type: ignore
        return self.retrieval_strategy.error_analysis.error_current

    @property
    def windows(self):  # type: ignore
        return self.retrieval_strategy.microwindows

    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [ProcessLocation("retrieval step")]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        super().notify_process_location(
            location,
            current_state,
            retrieval_strategy_step=retrieval_strategy_step,
        )
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        os.makedirs(self.input_directory, exist_ok=True)
        # May need to extend this logic here
        # detectorsUse = [1]
        # fstate_info = FakeStateInfo(self.current_state)
        # fretrieval_info = FakeRetrievalInfo(self.current_state)
        # mpy.write_retrieval_inputs(
        #    self.retrieval_strategy.rstrategy_table.strategy_table_dict,
        #    fstate_info,
        #    self.windows,
        #    fretrieval_info,
        #    self.step_number,
        #    self.error_current.__dict__,
        #    detectorsUse,
        # )
        # mpy.cdf_write_dict(
        #    fretrieval_info.__dict__,
        #    str(self.input_directory / "retrieval.nc"),
        # )


class RetrievalPickleResult(RetrievalOutput):
    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [ProcessLocation("retrieval step")]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        super().notify_process_location(
            location,
            current_state,
            retrieval_strategy_step=retrieval_strategy_step,
        )
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        os.makedirs(self.elanor_directory, exist_ok=True)
        with open(self.elanor_directory / "results.pkl", "wb") as fh:
            pickle.dump(self.results.__dict__, fh)


class RetrievalPlotResult(RetrievalOutput):
    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [ProcessLocation("retrieval step")]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        super().notify_process_location(
            location,
            current_state,
            retrieval_strategy_step=retrieval_strategy_step,
        )
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        os.makedirs(self.step_directory, exist_ok=True)
        # Just skip if we don't have muses_py. This is a pretty involved function, and
        # I'm not even sure these plots are used anymore. In any case, this is debug output
        # and skipping if we don't have muses-py is pretty reasonable
        from refractor.old_py_retrieve_wrapper import muses_py_plot_results

        muses_py_plot_results(
            self.current_state,  # type:ignore[arg-type]
            self.step_directory,
            self.results,  # type:ignore[arg-type]
        )


class RetrievalPlotRadiance(RetrievalOutput):
    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [ProcessLocation("retrieval step")]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        super().notify_process_location(
            location,
            current_state,
            retrieval_strategy_step=retrieval_strategy_step,
        )
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        os.makedirs(self.analysis_directory, exist_ok=True)
        # Just skip if we don't have muses_py. This is a pretty involved function, and
        # I'm not even sure these plots are used anymore. In any case, this is debug output
        # and skipping if we don't have muses-py is pretty reasonable
        from refractor.old_py_retrieve_wrapper import muses_py_plot_radiance

        muses_py_plot_radiance(
            str(self.analysis_directory) + "/",
            self.results,
            self.radiance_step.__dict__,
            None,
        )


__all__ = [
    "RetrievalInputOutput",
    "RetrievalPickleResult",
    "RetrievalPlotResult",
    "RetrievalPlotRadiance",
]
