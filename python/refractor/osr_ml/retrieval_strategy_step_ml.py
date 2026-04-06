from __future__ import annotations
from refractor.muses import (
    RetrievalStrategyStep,
    ProcessLocation,
)
from loguru import logger


class RetrievalStrategyStepMl(RetrievalStrategyStep):
    def retrieval_step_body(self) -> None:
        """Returns True if we handle the retrieval step, False otherwise"""
        logger.info("Fake ML retrieval")
        self.notify_process_location(ProcessLocation("ML step"))


__all__ = [
    "RetrievalStrategyStepMl",
]
