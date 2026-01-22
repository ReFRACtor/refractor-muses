from __future__ import annotations
from refractor.muses import RetrievalStrategy, ProcessLocation, RetrievalStrategyStep
from loguru import logger
from typing import Any


class RetrievalMlOutput:
    def notify_add(self, retrieval_strategy: RetrievalStrategy) -> None:
        self.retrieval_strategy = retrieval_strategy

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.retrieval_strategy_step = retrieval_strategy_step
        logger.info("Fake output")
