# This contains various support routines for reading OSP data.
from __future__ import annotations
from .tes_file import TesFile
import numpy as np
from pathlib import Path
from typing import Any, Self
from functools import cache
import re
import typing

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier


class RangeFind:
    """This is a simple dict like object that allows us to find an item
    by seeing if it is in a range. This isn't super efficient for a large
    number of entries, but our use has just a handful of entries.

    So with something like d[(0,10)] = 5, d[(10,20)] = 3 we have
    d[1] return 5, d[9] return 5, and d[11] return 3.

    We don't try to do anything intelligent here, we just search in the order that
    the data was put into this object. So if you have overlapping ranges, we just
    return the data for the first one found."""

    def __init__(self) -> None:
        self.data: list[tuple[tuple[float, float], Any]] = []

    def __setitem__(self, r: tuple[float, float], v: Any) -> None:
        self.data.append((r, v))

    def __getitem__(self, x: float) -> Any:
        for r, v in self.data:
            if x >= r[0] and x < r[1]:
                return v
        raise KeyError()


class OspCovarianceMatrix:
    """This reads file found in OSP/Covariance/Covariance"""

    def __init__(self, covariance_directory: Path) -> None:
        """This looks through the given directory (e.g., OSP/Covariance/Covariance) and
        maps all the files found into a simple structure that we can use to look up
        the file to read for a particular StateElement"""
        self.filename_data: dict[str, dict[str, RangeFind]] = {}
        for fname in covariance_directory.glob("Covariance_Matrix_*.asc"):
            m = re.match(
                r"Covariance_Matrix_(\w+)_(\w+)_(\d+)([SN])_(\d+)([SN]).asc", fname.name
            )
            if m:
                sid = m[1]
                maptype = m[2].lower()
                r1 = float(m[3]) * (-1 if m[4] == "S" else 1)
                r2 = float(m[5]) * (-1 if m[6] == "S" else 1)
                if sid not in self.filename_data:
                    self.filename_data[sid] = {}
                if maptype not in self.filename_data[sid]:
                    self.filename_data[sid][maptype] = RangeFind()
                self.filename_data[sid][maptype][(r1, r2)] = fname

    def read_cov(
        self, sid: StateElementIdentifier, map_type: str, latitude: float
    ) -> np.ndarray:
        # Right now, we only read scalar data. We just need to put in handling for
        # data on levels once we get to this.
        # The table has a column with the species name, a column with pressure, a
        # column with map type, and then the data. So we just subset for that
        d = np.array(
            TesFile(self.filename_data[str(sid)][map_type.lower()][latitude]).table
        )[:, 3:].astype(float)
        if d.shape != (1, 1):
            raise RuntimeError("Only handle scalar data right now")
        return d

    @classmethod
    @cache
    def read_dir(cls, covariance_directory: Path) -> Self:
        return cls(covariance_directory)


__all__ = ["RangeFind", "OspCovarianceMatrix"]
