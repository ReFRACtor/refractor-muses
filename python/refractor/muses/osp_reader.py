# This contains various support routines for reading OSP data.
from __future__ import annotations
from .tes_file import TesFile
import numpy as np
from pathlib import Path
from typing import Any, Self
from functools import cache
import re
from .identifier import StateElementIdentifier, RetrievalType


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


class OspCovarianceMatrixReader:
    """This reads file found in OSP/Covariance/Covariance"""

    def __init__(self, covariance_directory: Path) -> None:
        """This looks through the given directory (e.g., OSP/Covariance/Covariance) and
        maps all the files found into a simple structure that we can use to look up
        the file to read for a particular StateElement"""
        self.filename_data: dict[StateElementIdentifier, dict[str, RangeFind]] = {}
        for fname in covariance_directory.glob("Covariance_Matrix_*.asc"):
            m = re.match(
                r"Covariance_Matrix_(\w+)_(\w+)_(\d+)([SN])_(\d+)([SN]).asc", fname.name
            )
            if m:
                sid = StateElementIdentifier(m[1])
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
            TesFile(self.filename_data[sid][map_type.lower()][latitude]).table
        )[:, 3:].astype(float)
        if d.shape != (1, 1):
            raise RuntimeError("Only handle scalar data right now")
        return d

    @classmethod
    @cache
    def read_dir(cls, covariance_directory: Path) -> Self:
        return cls(covariance_directory)


class OspSpeciesReader:
    """This reads file found in directories like
    OSP/Strategy_Tables/ops/OSP-OMI-AIRS-v10/Species-66"""

    def __init__(self, species_directory: Path) -> None:
        """This looks through the given directory (e.g.,
        OSP/Strategy_Tables/ops/OSP-OMI-AIRS-v10/Species-66) and maps
        all the files found into a simple structure that we can use to
        look up the file to read for a particular StateElement

        """
        self.filename_data: dict[StateElementIdentifier, dict[RetrievalType, Path]] = {}
        self._default_cache: dict[StateElementIdentifier, np.ndarray] = {}
        for fname in species_directory.glob("*.asc"):
            m = re.match(r"([A-Z0-9_]+)(_([a-z0-9_]+))?.asc", fname.name)
            if m:
                sid = StateElementIdentifier(m[1])
                rtype = (
                    RetrievalType(m[3])
                    if m[3] is not None
                    else RetrievalType("default")
                )
                if sid not in self.filename_data:
                    self.filename_data[sid] = {}
                self.filename_data[sid][rtype] = fname

    def read_constraint_matrix(
        self, sid: StateElementIdentifier, retrieval_type: RetrievalType
    ) -> np.ndarray:
        if retrieval_type in self.filename_data[sid] or sid not in self._default_cache:
            t = self.read_file(sid, retrieval_type)
            ctype = t["constraintType"].lower()
            if ctype == "diagonal":
                s = float(t["sSubaDiagonalValues"])
                cov = np.array([[1.0 / (s * s)]])
            else:
                raise RuntimeError(f"Don't know how to handle {ctype}")
            if retrieval_type not in self.filename_data[
                sid
            ] or retrieval_type == RetrievalType("default"):
                self._default_cache[sid] = cov
            return cov
        return self._default_cache[sid]

    def read_file(
        self, sid: StateElementIdentifier, retrieval_type: RetrievalType
    ) -> TesFile:
        if retrieval_type in self.filename_data[sid]:
            fname = self.filename_data[sid][retrieval_type]
        else:
            fname = self.filename_data[sid][RetrievalType("default")]
        return TesFile(fname)

    @classmethod
    @cache
    def read_dir(cls, species_directory: Path) -> Self:
        return cls(species_directory)


__all__ = ["RangeFind", "OspCovarianceMatrixReader", "OspSpeciesReader"]
