from __future__ import annotations
from .misc import osp_setup
from .muses_spectral_window import MusesSpectralWindow
from .identifier import InstrumentIdentifier, FilterIdentifier
from .muses_observation import (
    MusesObservationImp,
    MeasurementId,
    MusesObservationHandle,
)
from .observation_handle import ObservationHandleSet
from .mpy import (
    mpy_read_noaa_cris_fsr,
    mpy_read_nasa_cris_fsr,
    mpy_radiance_data,
)
import os
import numpy as np
import refractor.framework as rf  # type: ignore
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .input_file_helper import InputFileHelper


class MusesCrisObservation(MusesObservationImp):
    def __init__(
        self,
        o_cris: dict[str, Any],
        sdesc: dict[str, Any],
        num_channels: int = 1,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping = None,
    ) -> None:
        """Note you don't normally create an object of this class with the
        __init__. Instead, call one of the create_xxx class methods."""
        super().__init__(o_cris, sdesc)
        # This is just hardcoded in py-retrieve, see about line 395 in
        # script_retrieval_setup_ms.py
        self._filter_data_name = [
            FilterIdentifier("CrIS-fsr-lw"),
            FilterIdentifier("CrIS-fsr-mw"),
            FilterIdentifier("CrIS-fsr-sw"),
        ]
        mw_range = np.zeros((3, 1, 2))
        mw_range[0, 0, :] = 0.0, 1200.00
        mw_range[1, 0, :] = 1200.01, 2145.00
        mw_range[2, 0, :] = 2145.01, 9999.00
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str],
        granule: int,
        xtrack: int,
        atrack: int,
        pixel_index: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        o_cris = cls.read_cris(filename, xtrack, atrack, pixel_index, ifile_hlp)
        # Add in RADIANCESTRUCT. Not sure if this is used, but easy enough to put in
        radiance = o_cris["RADIANCE"]
        frequency = o_cris["FREQUENCY"]
        nesr = o_cris["NESR"]
        filters = np.full((len(nesr),), "CrIS-fsr-lw")
        filters[frequency > 1200] = "CrIS-fsr-mw"
        filters[frequency > 2145] = "CrIS-fsr-sw"
        o_cris["RADIANCESTRUCT"] = mpy_radiance_data(
            radiance, nesr, [0], frequency, filters, "CRIS"
        )
        # We can perhaps clean this up, but for now there is some metadata  written
        # in the output file that depends on getting the l1b_type through o_cris,
        # so set that up
        o_cris["l1bType"] = cls.l1b_type_from_filename(filename)
        sdesc = {
            "CRIS_GRANULE": np.int16(granule),
            "CRIS_ATRACK_INDEX": np.int16(atrack),
            "CRIS_XTRACK_INDEX": np.int16(xtrack),
            "CRIS_PIXEL_INDEX": np.int16(pixel_index),
            "POINTINGANGLE_CRIS": abs(o_cris["SCANANG"]),
            "CRIS_L1B_TYPE": np.int16(cls.l1b_type_int_from_filename(filename)),
        }
        return (o_cris, sdesc)

    @classmethod
    def read_cris(
        cls,
        filename: str | os.PathLike[str],
        xtrack: int,
        atrack: int,
        pixel_index: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        i_fileid = {
            "CRIS_filename": os.path.abspath(str(filename)),
            "CRIS_XTrack_Index": xtrack,
            "CRIS_ATrack_Index": atrack,
            "CRIS_Pixel_Index": pixel_index,
        }
        filename = os.path.abspath(str(filename))
        with osp_setup(ifile_hlp):
            if cls.l1b_type_from_filename(filename) in ("snpp_fsr", "noaa_fsr"):
                o_cris = mpy_read_noaa_cris_fsr(i_fileid)
            else:
                o_cris = mpy_read_nasa_cris_fsr(i_fileid)
        return o_cris

    @classmethod
    def l1b_type_int_from_filename(cls, filename: str | os.PathLike[str]) -> int:
        """Enumeration used in output metadata for the l1b_type"""
        return [
            "suomi_nasa_nsr",
            "suomi_nasa_fsr",
            "suomi_nasa_nomw",
            "jpss1_nasa_fsr",
            "suomi_cspp_fsr",
            "jpss1_cspp_fsr",
            "jpss2_cspp_fsr",
        ].index(cls.l1b_type_from_filename(filename))

    @classmethod
    def l1b_type_from_filename(cls, filename: str | os.PathLike[str]) -> str:
        """There are a number of sources for the CRIS data, and two
        different file format types. This determines the l1b_type by
        looking at the path/filename. This isn't particularly robust,
        it depends on the specific directory structure. However it
        isn't clear what a better way to handle this would be - this
        is really needed metadata that isn't included in the
        Measurement_ID file but inferred by where the CRIS data comes
        from.

        """
        if "nasa_nsr" in str(filename):
            return "suomi_nasa_nsr"
        elif "nasa_fsr" in str(filename):
            return "suomi_nasa_fsr"
        elif "jpss_1_fsr" in str(filename):
            return "jpss1_nasa_fsr"
        elif "snpp_fsr" in str(filename):
            return "suomi_cspp_fsr"
        elif "noaa_fsr" in str(filename):
            return "suomi_noaa_fsr"
        else:
            raise RuntimeError(
                f"Don't recognize CRIS file type from path/filename {filename}"
            )

    @property
    def l1b_type_int(self) -> int:
        return self.l1b_type_int_from_filename(self.filename)

    @property
    def l1b_type(self) -> str:
        return self.l1b_type_from_filename(self.filename)

    def desc(self) -> str:
        return "MusesCrisObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("CRIS")

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str],
        granule: int,
        xtrack: int,
        atrack: int,
        pixel_index: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> Self:
        """Create from just the filenames. Note that spectral window
        doesn't get set here, but this can be useful if you just want
        access to the underlying data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).

        """
        o_cris, sdesc = cls._read_data(
            str(filename), granule, xtrack, atrack, pixel_index, ifile_hlp=ifile_hlp
        )
        return cls(o_cris, sdesc)

    @classmethod
    def create_from_id(
        cls,
        mid: MeasurementId,
        existing_obs: MusesCrisObservation | None,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        ifile_hlp: InputFileHelper,
        **kwargs: Any,
    ) -> Self:
        """Create from a MeasurementId. If this depends on any state
        information, you can pass in the CurrentState. This can be
        given as None if you just want to use default values, e.g. you
        aren't doing a retrieval. If the CurrentState is supplied, you
        can also pass a StateVector to add this class to as needed.

        """
        if existing_obs is not None:
            # Take data from existing observation
            obs = cls(
                existing_obs.muses_py_dict,
                existing_obs.sounding_desc,
                num_channels=existing_obs.num_channels,
            )
        else:
            filename = mid["CRIS_filename"]
            granule = mid["CRIS_Granule"]
            xtrack = int(mid["CRIS_XTrack_Index"])
            atrack = int(mid["CRIS_ATrack_Index"])
            pixel_index = int(mid["CRIS_Pixel_Index"])
            o_cris, sdesc = cls._read_data(
                filename, granule, xtrack, atrack, pixel_index, ifile_hlp=ifile_hlp
            )
            obs = cls(o_cris, sdesc)
        obs.spectral_window = (
            spec_win if spec_win is not None else MusesSpectralWindow(None, None)
        )
        obs.spectral_window.add_bad_sample_mask(obs)
        if fm_sv is not None:
            if current_state is None:
                raise RuntimeError(
                    "If fm_sv is not None, current_state needs to also be not None"
                )
            current_state.add_fm_state_vector_if_needed(
                fm_sv,
                obs.state_element_name_list(),
                [
                    obs,
                ],
            )
        return obs

    def radiance_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> np.ndarray:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict["RADIANCE"]

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict["FREQUENCY"]

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self.muses_py_dict["NESR"]

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        # TODO the old muses-py code replaces negative altitude with 0. Not
        # sure if this should be done or not, but that is the only instrument that
        # has this behavior. Assume this is right
        r = float(self.muses_py_dict["SURFACEALTITUDE"])
        return rf.DoubleWithUnit(0.0 if r < 0 else r, "m")


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("CRIS"), MusesCrisObservation)
)

__all__ = [
    "MusesCrisObservation",
]
