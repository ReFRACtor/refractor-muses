from __future__ import annotations
from refractor.muses import (
    RetrievalStrategyStep,
    RetrievalType,
    RetrievalStrategy,
    ProcessLocation,
)
from typing import Any
from loguru import logger


class RetrievalStrategyStepMl(RetrievalStrategyStep):
    def retrieval_step_body(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: Any
    ) -> bool:
        """Returns True if we handle the retrieval step, False otherwise"""
        if retrieval_type != RetrievalType("ML"):
            return False
        logger.info("Fake ML retrieval")
        rs.notify_update(
            ProcessLocation("retrieval step"), retrieval_strategy_step=self
        )
        return True


__all__ = [
    "RetrievalStrategyStepMl",
]
