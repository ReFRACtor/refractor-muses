from __future__ import annotations
from .misc import osp_setup
from .observation_handle import ObservationHandleSet
from .muses_observation import (
    MusesObservationImp,
    MusesObservationHandle,
    MeasurementId,
)
from .muses_spectral_window import MusesSpectralWindow
from .mpy import (
    mpy_read_airs_l1b,
)
import os
import numpy as np
import refractor.framework as rf  # type: ignore
from typing import Any, Self
import typing
from .identifier import InstrumentIdentifier, FilterIdentifier

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .input_file_helper import InputFileHelper


class MusesAirsObservation(MusesObservationImp):
    def __init__(
        self,
        o_airs: dict[str, Any],
        sdesc: dict[str, Any],
        num_channels: int = 1,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping = None,
    ) -> None:
        """Note you don't normally create an object of this class with
        the __init__. Instead, call one of the create_xxx class
        methods.

        """
        super().__init__(o_airs, sdesc)
        # This is just hardcoded in py-retrieve, see about line 62 in
        # read_airs.py
        self._filter_data_name = [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("2A3"),
            FilterIdentifier("1A1"),
        ]
        mw_range = np.zeros((5, 1, 2))
        mw_range[0, 0, :] = 0.0, 950.00
        mw_range[1, 0, :] = 950.01, 1119.80
        mw_range[2, 0, :] = 1119.81, 1444.00
        mw_range[3, 0, :] = 1444.01, 1890.80
        mw_range[4, 0, :] = 1890.81, 9999.00
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str],
        granule: int,
        xtrack: int,
        atrack: int,
        filter_list: list[str],
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        i_window = []
        for cname in filter_list:
            i_window.append({"filter": cname})
        o_airs = cls.read_airs(filename, xtrack, atrack, ifile_hlp)
        sdesc = {
            "AIRS_GRANULE": np.int16(granule),
            "AIRS_ATRACK_INDEX": np.int16(atrack),
            "AIRS_XTRACK_INDEX": np.int16(xtrack),
            "POINTINGANGLE_AIRS": abs(o_airs["scanAng"]),
        }
        return (o_airs, sdesc)

    @classmethod
    def read_airs(
        cls,
        filename: str | os.PathLike[str],
        xtrack: int,
        atrack: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        """This is probably a bit over complicated, we don't really need the full
        o_airs structure. But for now, just duplicate the old muses-py code so we
        have a starting point for possibly cleaning up."""
        if ifile_hlp is not None:
            ifile_hlp.notify_file_input(filename)
            # Hardcoded path
            ifile_hlp.notify_file_input(
                ifile_hlp.osp_dir / "AIRS/Bad_Frequencies/airs_bad_frequencies.nc"
            )
        with osp_setup(ifile_hlp):
            o_airs = mpy_read_airs_l1b(os.path.abspath(str(filename)), xtrack, atrack)
        # Not sure why, but data isn't fully sorted by wavenumber. Go ahead and
        # fix this. Also convert radiance to float64 (it is float32). frequency
        # and NESR are already float64
        frequency = o_airs["frequency"]
        ss = np.argsort(frequency)
        o_airs["radiance"] = o_airs["radiance"][ss].astype(np.float64)
        o_airs["frequency"] = o_airs["frequency"][ss]
        o_airs["NESR"] = o_airs["NESR"][ss]
        return o_airs

    def desc(self) -> str:
        return "MusesAirsObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("AIRS")

    @property
    def spacecraft_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(float(self._muses_py_dict["satheight"]), "km")

    @property
    def scan_angle(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(self._muses_py_dict["scanAng"], "deg")

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str],
        granule: int,
        xtrack: int,
        atrack: int,
        filter_list: list[FilterIdentifier],
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create from just the filenames. Note that spectral window
        doesn't get set here, but this can be useful if you just want
        access to the underlying data.

        You might also want to use create_from_id, which sets up everything
        (spectral window, coefficients, attaching to a fm_sv).

        """
        o_airs, sdesc = cls._read_data(
            str(filename),
            granule,
            xtrack,
            atrack,
            [str(i) for i in filter_list],
            ifile_hlp=ifile_hlp,
        )
        return cls(o_airs, sdesc)

    @classmethod
    def create_from_id(
        cls,
        mid: MeasurementId,
        existing_obs: MusesAirsObservation | None,
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
                existing_obs._muses_py_dict,  # noqa: SLF001
                existing_obs.sounding_desc,
                num_channels=existing_obs.num_channels,
            )
        else:
            # Read the data from disk, because it doesn't already exist.
            filter_list = mid.filter_list_dict[InstrumentIdentifier("AIRS")]
            filename = mid["AIRS_filename"]
            granule = mid["AIRS_Granule"]
            xtrack = int(mid["AIRS_XTrack_Index"])
            atrack = int(mid["AIRS_ATrack_Index"])
            o_airs, sdesc = cls._read_data(
                filename,
                granule,
                xtrack,
                atrack,
                [str(i) for i in filter_list],
                ifile_hlp=ifile_hlp,
            )
            obs = cls(o_airs, sdesc)
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
        return self._muses_py_dict["radiance"]

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["frequency"]

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["NESR"]

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(float(self._muses_py_dict["surfaceAltitude"]), "m")


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("AIRS"), MusesAirsObservation)
)


__all__ = [
    "MusesAirsObservation",
]
