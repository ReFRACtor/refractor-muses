from .templated_output import TemplatedOutput
import os
import numpy as np
from .declaritve_output import register_dataset, DeclarativeOutput


class ColumnCoFile(DeclarativeOutput):
    """We might try to get a general file rather than hardcoding CO, but start
    simply for now."""

    def __init__(
        self, output_filename: str | os.PathLike[str], pspec: str | os.PathLike[str]
    ) -> None:
        self.output = TemplatedOutput(pspec, output_filename)
        self.output.register_instances((self,))

    @register_dataset("/col")
    def co_column(self) -> np.ndarray:
        return np.zeros((500, 5))

    def write(self):
        self.output.write()


__all__ = [
    "ColumnCoFile",
]
