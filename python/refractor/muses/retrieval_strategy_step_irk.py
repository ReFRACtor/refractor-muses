from __future__ import annotations
from .retrieval_strategy_step import (
    RetrievalStrategyStepSet,
    RetrievalStrategyStepHandle,
)
from .retrieval_strategy_step_oe import RetrievalStrategyStepOEBase
from .misc import ResultIrk
from .identifier import RetrievalType, ProcessLocation
from loguru import logger
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .creator_dict import CreatorDict
    from .current_state import CurrentState
    from .process_location_observable import ProcessLocationObservable


class RetrievalStrategyStepIRK(RetrievalStrategyStepOEBase):
    """IRK strategy step."""

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
        self.results_irk: ResultIrk | None = None

    def retrieval_step_body(self) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        logger.info("Running run_irk ...")
        fm = self.create_forward_model()
        if not hasattr(fm, "irk"):
            raise RuntimeError(
                f"The forward model {fm.__class__.__name__} does not support calculating the irk"
            )
        if self._saved_state is None:
            self.results_irk = fm.irk(self.current_state)
        else:
            # Use saved results instead of calculating
            # unit testing where we use a precomputed result
            self.results_irk = ResultIrk()
            self.results_irk.set_state(self._saved_state["results_irk"])
        self.notify_process_location(ProcessLocation("IRK step"))

    def get_state(self) -> dict[str, Any]:
        res: dict[str, Any] = {"results_irk": None}
        if self.results_irk is not None:
            res["results_irk"] = self.results_irk.get_state()
        return res


RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(
        RetrievalStrategyStepIRK,
        {
            RetrievalType("irk"),
        },
    )
)


__all__ = [
    "RetrievalStrategyStepIRK",
]
