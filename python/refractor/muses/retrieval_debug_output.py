from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .retrieval_output import RetrievalOutput
from .identifier import ProcessLocation
from .fake_state_info import FakeStateInfo
from loguru import logger
import os
import pickle
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .retrieval_info import RetrievalInfo

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
    def retrieval_info(self) -> RetrievalInfo:
        return self.retrieval_strategy.current_state.retrieval_info

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        os.makedirs(self.input_directory, exist_ok=True)
        # May need to extend this logic here
        detectorsUse = [1]
        fstate_info = FakeStateInfo(self.current_state)
        mpy.write_retrieval_inputs(
            self.retrieval_strategy.rstrategy_table.strategy_table_dict,
            fstate_info,
            self.windows,
            self.retrieval_info.retrieval_info_obj,
            self.step_number,
            self.error_current.__dict__,
            detectorsUse,
        )
        mpy.cdf_write_dict(
            self.retrieval_info.retrieval_info_obj.__dict__,
            str(self.input_directory / "retrieval.nc"),
        )


class RetrievalPickleResult(RetrievalOutput):
    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        os.makedirs(self.elanor_directory, exist_ok=True)
        with open(self.elanor_directory / "results.pkl", "wb") as fh:
            pickle.dump(self.results.__dict__, fh)


class RetrievalPlotResult(RetrievalOutput):
    @property
    def retrieval_info(self) -> RetrievalInfo:
        return self.retrieval_strategy.current_state.retrieval_info

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        os.makedirs(self.step_directory, exist_ok=True)
        fstate_info = FakeStateInfo(self.current_state)
        mpy.plot_results(
            str(self.step_directory) + "/",
            self.results,
            self.retrieval_info.retrieval_info_obj,
            fstate_info,
        )


class RetrievalPlotRadiance(RetrievalOutput):
    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        os.makedirs(self.analysis_directory, exist_ok=True)
        mpy.plot_radiance(
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
