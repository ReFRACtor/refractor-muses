from __future__ import annotations
from .retrieval_strategy_step import (
    RetrievalStrategyStep,
    RetrievalStrategyStepSet,
    RetrievalStrategyStepHandle,
)
from .misc import ResultIrk
from .identifier import RetrievalType, ProcessLocation
from loguru import logger
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .creator_dict import CreatorDict


class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    """IRK strategy step."""

    def __init__(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> None:
        super().__init__(retrieval_type, rs, creator_dict, **kwargs)
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
        self.notify_update(ProcessLocation("IRK step"))

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
