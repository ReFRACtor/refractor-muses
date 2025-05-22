# This contains various support routines for reading OSP data.
from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .tes_file import TesFile
import collections.abc
import numpy as np
import os
from pathlib import Path
from typing import Any, Self, Iterator
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


class CovarianceMatrix:
    """Small class to handle the apriori_cov_fm read by OspCovarianceMatrixReader.

    For data on pressure levels, the data is read with one set of pressure levels, but
    we need to convert this to the forward model pressure levels. This class handles that.

    Just so we have one interface, we also use the same class for covariances that aren't
    on pressure levels - external class can just grab the covariance matrix without doing
    interpolation."""

    def __init__(self, pressure_input: np.ndarray, original_cov: np.ndarray) -> None:
        """Pressure should be in hPa (what is used by muses-py)."""
        self.pressure_input = pressure_input
        self.original_cov = original_cov

    def interpolated_covariance(self, pressure_level: np.ndarray) -> np.ndarray:
        """Interpolate the original_cov to the given pressure levels.
        Pressure should be in hPa (what is used by muses-py)."""
        p = self.pressure_input
        # If out of range at top, just move top level. This is what muses-py does,
        # I think the code for making the interpolation matrix requires this.
        if p.min() > pressure_level.min():
            p = p.copy()
            p[-1] = pressure_level.min()
        # TODO - short term convert to float32, just so we can directly compare
        # with muses-py. We'll change this to float64, but good to do that as a single
        # step so we don't mix in other changes
        m = mpy.make_interpolation_matrix_susan(
            np.log(p.astype(np.float32)), np.log(pressure_level)
        )
        return np.matmul(
            np.matmul(m, self.original_cov.astype(np.float32)), m.transpose()
        )


class OspFileHandle:
    """Small mixin, since I tend to put this code in a lot of places. Adds a
    _abs_path that handles the relative "../OSP" that tends to appear in the OSP
    files."""

    def __init__(
        self,
        osp_dir: str | os.PathLike[str] | None = None,
        nlevel: int = 4,
        ref_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if osp_dir is None:
            if ref_path is None:
                raise RuntimeError("Need to supply either osp_dir or ref_path")
            # Assume a directory structure,
            p = Path(ref_path)
            for i in range(nlevel):
                p = p.parent
            self.osp_dir = p
        else:
            self.osp_dir = Path(osp_dir).absolute()

    def _abs_path(self, v: str) -> Path:
        m = re.match(r"^\.\./OSP/(.*)", v)
        if m:
            return self.osp_dir / m[1]
        return self.osp_dir / v


class OspCovarianceMatrixReader:
    """This reads file found in OSP/Covariance/Covariance"""

    def __init__(self, covariance_directory: Path) -> None:
        """This looks through the given directory (e.g., OSP/Covariance/Covariance) and
        maps all the files found into a simple structure that we can use to look up
        the file to read for a particular StateElement"""
        self.filename_data: dict[
            StateElementIdentifier, dict[str, dict[str, RangeFind]]
        ] = {}
        for fname in covariance_directory.glob("Covariance_Matrix_*.asc"):
            m = re.match(
                r"Covariance_Matrix_(\w+)_(\w+)_(\d+)([SN])_(\d+)([SN])(_([A-Z]+))?.asc",
                fname.name,
            )
            if m:
                sid = StateElementIdentifier(m[1])
                maptype = m[2].lower()
                r1 = float(m[3]) * (-1 if m[4] == "S" else 1)
                r2 = float(m[5]) * (-1 if m[6] == "S" else 1)
                if sid not in self.filename_data:
                    self.filename_data[sid] = {}
                spectype = m[8] if m[8] is not None else ""
                if spectype not in self.filename_data[sid]:
                    self.filename_data[sid][spectype] = {}
                if maptype not in self.filename_data[sid][spectype]:
                    self.filename_data[sid][spectype][maptype] = RangeFind()
                self.filename_data[sid][spectype][maptype][(r1, r2)] = fname

    def read_cov(
        self,
        sid: StateElementIdentifier,
        map_type: str,
        latitude: float,
        spectype: str | None = None,
    ) -> CovarianceMatrix:
        """Read the covariance file."""
        # The table has a column with the species name, a column with pressure, a
        # column with map type, and then the data. So we just subset for that
        if spectype is None:
            spectype = ""
        tf = TesFile(self.filename_data[sid][spectype][map_type.lower()][latitude])
        d = np.array(tf.table)[:, 3:].astype(float)
        pressure_sa = np.array(tf.table)[:, 1].astype(float)
        return CovarianceMatrix(pressure_sa, d)
        if d.shape == (1, 1):
            return d

    @classmethod
    @cache
    def read_dir(cls, covariance_directory: Path) -> Self:
        return cls(covariance_directory)


class OspSpeciesReader(OspFileHandle):
    """This reads file found in directories like
    OSP/Strategy_Tables/ops/OSP-OMI-AIRS-v10/Species-66"""

    def __init__(
        self,
        species_directory: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """This looks through the given directory (e.g.,
        OSP/Strategy_Tables/ops/OSP-OMI-AIRS-v10/Species-66) and maps
        all the files found into a simple structure that we can use to
        look up the file to read for a particular StateElement

        """
        super().__init__(osp_dir=osp_dir, ref_path=species_directory, nlevel=4)
        self.filename_data: dict[
            StateElementIdentifier,
            dict[StateElementIdentifier | None, dict[RetrievalType, Path]],
        ] = {}
        self._default_cache: dict[
            StateElementIdentifier, dict[StateElementIdentifier | None, np.ndarray]
        ] = {}
        for fname in Path(species_directory).glob("*.asc"):
            m = re.match(r"([A-Z0-9]+)(_([A-Z0-9]+))?(_([a-z0-9_]+))?.asc", fname.name)
            if m:
                sid = StateElementIdentifier(m[1])
                sid2 = None if m[3] is None else StateElementIdentifier(m[3])
                rtype = (
                    RetrievalType(m[5])
                    if m[5] is not None
                    else RetrievalType("default")
                )
                if sid not in self.filename_data:
                    self.filename_data[sid] = {}
                if sid2 not in self.filename_data[sid]:
                    self.filename_data[sid][sid2] = {}
                self.filename_data[sid][sid2][rtype] = fname

    def read_constraint_matrix(
        self,
        sid: StateElementIdentifier,
        retrieval_type: RetrievalType,
        num_retrieval: int,
        sid2: StateElementIdentifier | None = None,
        spectype: str | None = None,
    ) -> np.ndarray:
        if (
            retrieval_type in self.filename_data[sid][sid2]
            or sid not in self._default_cache
            or sid2 not in self._default_cache[sid]
        ):
            t = self.read_file(sid, retrieval_type, sid2=sid2)
            ctype = t["constraintType"].lower()
            if ctype == "diagonal":
                s = float(t["sSubaDiagonalValues"])
                cov = np.array([[1.0 / (s * s)]])
            elif ctype == "premade":
                # In a fairly obscure way, we replace the default "87" found in the
                # constraint file name with the number of retrieval levels to find
                # the constraint file - e.g., rather than reading
                # Constraint_Matrix_TATM_NADIR_LINEAR_90S_90N_87.asc we read
                # Constraint_Matrix_TATM_NADIR_LINEAR_90S_90N_30.asc if we have 30 levels
                fname = self._abs_path(t["constraintFilename"])
                fname = Path(
                    re.sub(
                        r"_87_87.asc$",
                        f"_{num_retrieval}_{num_retrieval}.asc",
                        str(self._abs_path(t["constraintFilename"])),
                    )
                )
                if spectype is None:
                    fname = Path(
                        re.sub(r"_87.asc$", f"_{num_retrieval}.asc", str(fname))
                    )
                else:
                    fname = Path(
                        re.sub(
                            r"_87.asc$", f"_{num_retrieval}_{spectype}.asc", str(fname)
                        )
                    )
                d = TesFile(fname)
                # First column is name of species, second is pressure, third is map type.
                # We chop off just to get the data
                cov = np.array(d.table)[:, 3:].astype(float)
            else:
                raise RuntimeError(f"Don't know how to handle {ctype}")
            if sid not in self._default_cache:
                self._default_cache[sid] = {}
            if retrieval_type not in self.filename_data[sid][
                sid2
            ] or retrieval_type == RetrievalType("default"):
                self._default_cache[sid][sid2] = cov
            return cov
        return self._default_cache[sid][sid2]

    def read_file(
        self,
        sid: StateElementIdentifier,
        retrieval_type: RetrievalType,
        sid2: StateElementIdentifier | None = None,
    ) -> TesFile:
        if retrieval_type in self.filename_data[sid][sid2]:
            fname = self.filename_data[sid][sid2][retrieval_type]
        else:
            fname = self.filename_data[sid][sid2][RetrievalType("default")]
        return self._tes_file(fname)

    @cache
    def _tes_file(self, fname: Path) -> TesFile:
        return TesFile(fname)

    @classmethod
    @cache
    def read_dir(cls, species_directory: Path) -> Self:
        return cls(species_directory)


class OspL2SetupControlInitial(collections.abc.Mapping, OspFileHandle):
    """muses-py has a L2_Setup_Control_Initial.asc that lists state elements and
    what the source of the initial guess is.

    Note that this doesn't really seem to be a "control" file, so much as a description.
    So for example, we can't move a state element from Species_List_From_Single to
    Species_List_From_GMAO and suddenly be able to read that from the GMAO files. Instead,
    the function get_state_initial.py has lots of specific code for different species.

    However, it can still be useful to read this file to if nothing else check that our
    StateElement implementation matches (e.g., if we get H2O from GMAO but the file
    says it should be "Single", then that should at least warrant a warning of if not an
    error."""

    def __init__(
        self,
        fname: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
    ):
        OspFileHandle.__init__(self, osp_dir=osp_dir, ref_path=fname, nlevel=4)
        self._file = TesFile(fname)
        self._sid_to_type: dict[StateElementIdentifier, str] = {}
        typ = [
            "Zero",
            "Single",
            "Climatology",
            "GMAO",
            "AIRS_Initial",
            "TES",
            "TES_Initial",
            "TES_Constraint",
            "OCO2_Initial",
            "OCO2",
        ]
        for t in typ:
            # For some reason, Zero uses a different naming convention
            if t == "Zero":
                slist = self._file["Species_List_Zero"].split(",")
            else:
                slist = self._file[f"Species_List_From_{t}"].split(",")
            for s in slist:
                if s != "-":
                    self._sid_to_type[StateElementIdentifier(s)] = t

    # Make other parts of the file available as a dict like object

    def __getitem__(self, ky: str) -> str | Path | None:
        res = self._file[ky]
        if res == "-":
            return None
        if re.match(r".*_Directory", ky):
            return self._abs_path(self._file[ky])
        return res

    def __len__(self) -> int:
        return len(self._file)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self._file.__iter__()

    @property
    def sid_to_type(self) -> dict[StateElementIdentifier, str]:
        return self._sid_to_type

    @classmethod
    @cache
    def read(
        cls,
        initial_guess_setup_directory: Path,
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> Self:
        # muses-py uses a hardcoded file name given the "initialGuessSetupDirectory"
        # found in the target file.
        return cls(
            initial_guess_setup_directory / "L2_Setup_Control_Initial.asc", osp_dir
        )


__all__ = [
    "RangeFind",
    "OspCovarianceMatrixReader",
    "OspSpeciesReader",
    "OspL2SetupControlInitial",
]
