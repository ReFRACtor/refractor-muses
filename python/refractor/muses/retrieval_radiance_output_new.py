from __future__ import annotations
import numpy as np
from .declarative_output import register_dataset, DeclarativeOutput
from .identifier import ProcessLocation
from .retrieval_output_file import RetrievalOutputFile
from loguru import logger
from pathlib import Path
import os
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy_step import RetrievalStrategyStep
    from .current_state import CurrentState


class RetrievalRadianceOutputNew(DeclarativeOutput):
    """New version of RetrievalRadianceOutput, that uses the DeclarativeOutput interface.
    We will likely rename the old RetrievalRadianceOutput to RetrievalRadianceOutputOld,
    and rename this to RetrievalRadianceOutput when we have this all in place. But for
    now leave the old one in place and have this as the "new" version."""

    def __init__(self, output_filename: str | os.PathLike[str]) -> None:
        self.output_filename = Path(output_filename)
        self.output = RetrievalOutputFile(output_filename)
        self.output.register_instances((self,))

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState,
        retrieval_strategy_step: RetrievalStrategyStep,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.current_state = current_state
        self.retrieval_strategy_step = retrieval_strategy_step
        self.write()

    @register_dataset("/RADIANCEFULLBAND")
    def radiance_full_band(self) -> np.ndarray:
        # Placeholder
        return np.zeros((2223,))

    def write(self) -> None:
        self.output.write()


__all__ = [
    "RetrievalRadianceOutputNew",
]
