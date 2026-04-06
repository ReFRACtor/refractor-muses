from __future__ import annotations
from refractor.muses import (
    RetrievalStrategy,
    ProcessLocation,
    RetrievalStrategyStep,
    CurrentState,
)
from loguru import logger
from typing import Any


class RetrievalMlOutput:
    def notify_add(self, retrieval_strategy: RetrievalStrategy) -> None:
        self.retrieval_strategy = retrieval_strategy

    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [
            ProcessLocation("ML step"),
        ]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState | None = None,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        self.retrieval_strategy_step = retrieval_strategy_step
        logger.info("Fake output")
