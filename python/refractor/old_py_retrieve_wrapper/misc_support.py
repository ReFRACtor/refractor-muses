from __future__ import annotations
import refractor.muses_py as mpy  # type: ignore
from loguru import logger
import json
import os
import importlib
from typing import Any
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    from refractor.muses import (
        InputFilePath,
        RetrievalType,
        StateElementIdentifier,
        MusesSpectralWindow,
    )

# This has various things that we use to have in refractor.muses. We
# pulled all this into refractor.old_py_retrieve_wrapper just so we
# have a clean separation, and also because most of this is just for
# backwards testing, i.e., it really does belong in
# old_py_retrieve_wrapper, but needs to be in refractor.muses objects


def create_retrieval_output_json() -> None:
    """muses-py has a lot of hard coded things related to the species
    names and netcdf output.  It would be good a some point to just
    replace this all with a better thought out output format. But for
    now, we need to support the existing output format.
    """
    # TODO - Replace with better thought out output format
    if not importlib.resources.is_resource("refractor.muses", "retrieval_output.json"):
        if not mpy.have_muses_py:
            raise RuntimeError(
                "Require muses-py to create the file retrieval_output.json"
            )
        d = {
            "cdf_var_attributes": mpy.cdf_var_attributes,
            "groupvarnames": mpy.cdf_var_names(),
            "exact_cased_variable_names": mpy.cdf_var_map(),
        }
        with importlib.resources.path(
            "refractor.muses", "retrieval_output.json"
        ) as fspath:
            logger.info(f"Creating the file {fspath}")
            with open(fspath, "w") as fh:
                json.dump(d, fh, indent=4)


def muses_py_radiance_data(
    radiance_fm: np.ndarray, nesr_fm: np.ndarray, freq_fm: np.ndarray
) -> dict[str, np.ndarray]:
    # Some values that are just placeholder
    detectors = [-1]
    # Not sure what filters is, but fm_wrapper just supplies this
    # as a empty array
    filters: list[int] = []
    instrument = ""
    return mpy.radiance_data(
        radiance_fm, nesr_fm, detectors, freq_fm, filters, instrument
    )


def muses_microwindows_fname_from_muses_py(
    cls: MusesSpectralWindow,
    viewing_mode: str,
    spectral_window_directory: str | os.PathLike[str] | InputFilePath,
    retrieval_elements: list[StateElementIdentifier],
    step_name: str,
    retrieval_type: RetrievalType,
    spec_file: str | os.PathLike[str] | InputFilePath | None = None,
) -> str:
    """For testing purposes, this calls the old mpy.table_get_spectral_filename to
    determine the microwindow file name use. This can be used to verify that
    we are finding the right name. This shouldn't be used for real code,
    instead use the SpectralWindowHandleSet."""
    # creates a dummy strategy_table dict with the values it expects to find
    stable: dict[str, Any] = {}
    stable["preferences"] = {
        "viewingMode": viewing_mode,
        "spectralWindowDirectory": str(spectral_window_directory),
    }
    t1 = [
        ",".join([str(i) for i in retrieval_elements])
        if len(retrieval_elements) > 0
        else "-",
        step_name,
        str(retrieval_type),
    ]
    t2 = ["retrievalElements", "stepName", "retrievalType"]
    if spec_file is not None:
        t1.append(str(spec_file))
        t2.append("specFile")
    stable["data"] = [
        " ".join(t1),
    ]
    stable["labels1"] = " ".join(t2)
    stable["numRows"] = 1
    stable["numColumns"] = len(t2)
    return mpy.table_get_spectral_filename(stable, 0)


def muses_microwindows_from_muses_py(
    cls: MusesSpectralWindow,
    default_spectral_window_fname: str,
    viewing_mode: str,
    spectral_window_directory: str | os.PathLike[str] | InputFilePath,
    retrieval_elements: list[StateElementIdentifier],
    step_name: str,
    retrieval_type: RetrievalType,
    spec_file: str | os.PathLike[str] | InputFilePath | None = None,
) -> list[dict[str, Any]]:
    """For testing purposes, this calls the old mpy.table_new_mw_from_step. This can
    be used to verify that the microwindows we generate are correct. This shouldn't
    be used for real code, instead use the SpectralWindowHandleSet."""
    # Wrap arguments into format expected by table_new_mw_from_step. This
    # creates a dummy strategy_table dict with the values it expects to find
    stable: dict[str, Any] = {}
    stable["preferences"] = {
        "defaultSpectralWindowsDefinitionFilename": str(default_spectral_window_fname),
        "viewingMode": viewing_mode,
        "spectralWindowDirectory": str(spectral_window_directory),
    }
    t1 = [
        ",".join([str(i) for i in retrieval_elements]),
        step_name,
        str(retrieval_type),
    ]
    t2 = ["retrievalElements", "stepName", "retrievalType"]
    if spec_file is not None:
        t1.append(str(spec_file))
        t2.append("specFile")
    stable["data"] = [
        " ".join(t1),
    ]
    stable["labels1"] = " ".join(t2)
    stable["numRows"] = 1
    stable["numColumns"] = len(t2)
    return mpy.table_new_mw_from_step(stable, 0)


def muses_py_radiance_get_indices(d: dict[str, Any], mw: list[dict]) -> np.ndarray:
    return mpy.radiance_get_indices(d, mw)


__all__ = [
    "create_retrieval_output_json",
    "muses_microwindows_fname_from_muses_py",
    "muses_microwindows_from_muses_py",
    "muses_py_radiance_data",
    "muses_py_radiance_get_indices",
]
