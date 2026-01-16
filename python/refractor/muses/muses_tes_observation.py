from __future__ import annotations
from .misc import osp_setup
from .observation_handle import ObservationHandleSet
from .muses_observation import (
    MusesObservationImp,
    MusesObservationHandle,
    MeasurementId,
)
from .muses_spectral_window import MusesSpectralWindow, TesSpectralWindow
from .mpy import (
    mpy_read_tes_l1b,
    mpy_radiance_apodize,
)
import os
import numpy as np
import refractor.framework as rf  # type: ignore
import copy
from typing import Any, Self
import typing
from .identifier import InstrumentIdentifier, FilterIdentifier

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .input_file_helper import InputFileHelper, InputFilePath


class MusesTesObservation(MusesObservationImp):
    def __init__(
        self,
        o_tes: dict[str, Any],
        sdesc: dict[str, Any],
        num_channels: int = 1,
        coeff: np.ndarray | None = None,
        mp: rf.StateMapping = None,
    ) -> None:
        """Note you don't normally create an object of this class with
        the __init__. Instead, call one of the create_xxx class
        methods.

        """
        super().__init__(o_tes, sdesc)
        # Set up stuff for the filter_data metadata
        # Hardcoded, see about line 29 of read_test_l1b.py
        self._filter_data_name = [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ]
        self._filter_data_name = [
            FilterIdentifier(str(i)) for i in o_tes["radianceStruct"]["filterNames"]
        ]
        # Note that the bands of tes actually overlap. This is confusing, but we
        # need to match the old way this ran. We have handling in filter_data_full
        # to work around this.
        mw_range = np.zeros((len(self._filter_data_name), 1, 2))
        sindex = 0
        for i in range(mw_range.shape[0]):
            eindex = o_tes["radianceStruct"]["filterSizes"][i] + sindex
            freq = o_tes["radianceStruct"]["frequency"][sindex:eindex]
            mw_range[i, 0, :] = min(freq), max(freq)
            sindex = eindex
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str],
        l1b_index: list[int],
        l1b_avgflag: int,
        run: int,
        sequence: int,
        scan: int,
        filter_list: list[str],
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        windows = []
        for cname in filter_list:
            windows.append({"filter": cname})
        o_tes = cls.read_tes(filename, l1b_index, l1b_avgflag, windows, ifile_hlp)
        bangle = rf.DoubleWithUnit(o_tes["boresightNadirRadians"], "rad")
        sdesc = {
            "TES_RUN": np.int16(run),
            "TES_SEQUENCE": np.int16(sequence),
            "TES_SCAN": np.int16(scan),
            "POINTINGANGLE_TES": abs(bangle.convert("deg").value),
        }
        return (o_tes, sdesc)

    @classmethod
    def read_tes(
        cls,
        filename: str | os.PathLike[str],
        l1b_index: list[int],
        l1b_avgflag: int,
        windows: list[dict[str, Any]],
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        i_fileid = {}
        i_fileid["preferences"] = {
            "TES_filename_L1B": os.path.abspath(str(filename)),
            "TES_filename_L1B_Index": l1b_index,
            "TES_L1B_Average_Flag": l1b_avgflag,
        }
        with osp_setup(ifile_hlp):
            o_tes = mpy_read_tes_l1b(i_fileid, windows)
        return o_tes

    @classmethod
    def _apodization(
        cls,
        o_tes: dict[str, Any],
        func: str,
        strength: str,
        flt: np.ndarray,
        maxopd: np.ndarray,
        spacing: np.ndarray,
    ) -> None:
        """Apply apodization to the radiance and NESR. o_tes is
        updated in place"""
        if func != "NORTON_BEER":
            raise RuntimeError(f"Don't know how to apply apodization function {func}")
        rstruct = mpy_radiance_apodize(
            o_tes["radianceStruct"], strength, flt, maxopd, spacing
        )
        o_tes["radianceStruct"] = rstruct

    @property
    def boresight_angle(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(self._muses_py_dict["boresightNadirRadians"], "rad")

    @property
    def spacecraft_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(self._muses_py_dict["instrumentAltitude"], "km")

    @property
    def filter_data(self) -> list[tuple[FilterIdentifier, int]]:
        # Need to override, because self._filter_data_swin overlaps (so the
        # total of the sizes is greater than the actual full spectral_domain)
        res: list[tuple[FilterIdentifier, int]] = []
        d = self._muses_py_dict["radianceStruct"]
        for i, fltname in enumerate(self._filter_data_name):
            res.append((fltname, int(d["filterSizes"][i])))
        return res

    @property
    def filter_data_full(self) -> list[tuple[FilterIdentifier, int]]:
        # Need to override, because self._filter_data_swin overlaps (so the
        # total of the sizes is greater than the actual full spectral_domain)
        res: list[tuple[FilterIdentifier, int]] = []
        d = self._muses_py_dict["radianceStruct"]
        for i, fltname in enumerate(self._filter_data_name):
            res.append((fltname, int(d["filterSizes"][i])))
        return res

    def desc(self) -> str:
        return "MusesTesObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("TES")

    @classmethod
    def create_fake_for_irk(
        cls,
        tes_frequency_fname: str | os.PathLike[str] | InputFilePath,
        swin: MusesSpectralWindow,
        ifile_hlp: InputFileHelper,
    ) -> Self:
        """For the RetrievalStrategyStepIrk (Instantaneous Radiative
        Kernels) AIRS frequencies gets replaced with TES. I think TES
        is a more full grid.  We don't really need a full
        MusesObservation for this, but the OSS MusesTesForwardModel
        gets the frequency grid from an observation. So this reads a
        frequency netcdf file and uses that to create a fake
        observation. This doesn't actually have the radiance or
        anything in it, just the frequency part. But this is all that
        is needed by the IRK calculation.

        """
        my_file: dict[str, Any] = {}
        with ifile_hlp.open_ncdf(tes_frequency_fname) as nci:
            nci.set_auto_maskandscale(False)
            # We can probably reduce this list, but this is what py-retrieve currently
            # reads
            for fldname, vname in [
                ("FILENAME", "filename"),
                ("INSTRUMENT", "instrument"),
                ("COMMENTS", "comments"),
                ("PREFERENCES", "preferences"),
                ("DETECTORS", "detectors"),
                ("MWS", "mws"),
                ("NUMDETECTORSORIG", "numDetectorsOrig"),
                ("NUMDETECTORS", "numDetectors"),
                ("NUM_FREQUENCIES", "num_frequencies"),
                ("RADIANCE", "radiance"),
                ("NESR", "NESR"),
                ("FREQUENCY", "frequency"),
                ("VALID", "valid"),
                ("FILTERSIZES", "filterSizes"),
                ("FILTERNAMES", "filterNames"),
                ("INSTRUMENTSIZES", "instrumentSizes"),
                ("INSTRUMENTNAMES", "instrumentNames"),
                ("PIXELSUSED", "pixelsUsed"),
                ("INTERPIXELVAR", "interpixelVar"),
                ("FREQSHIFT", "freqShift"),
                ("IMAGINARYMEAN", "imaginaryMean"),
                ("IMAGINARYRMS", "imaginaryRMS"),
                ("BT8", "bt8"),
                ("BT10", "bt10"),
                ("BT11", "bt11"),
                ("SCANDIRECTION", "scanDirection"),
            ]:
                my_file[vname] = nci[fldname][:]
                if (
                    isinstance(my_file[vname], np.ndarray)
                    and len(my_file[vname].shape) >= 2
                    and vname != "filterNames"
                    and my_file[vname].shape[-1] != 1
                ):
                    my_file[vname] = np.transpose(my_file[vname])

        # Convert filterNames from array of bytes (ASCII) to array of
        # strings since the NetCDF file store the strings as bytes.
        my_file["filterNames"] = [
            "".join(chr(i) for i in t) for t in my_file["filterNames"]
        ]
        # Convert instrumentNames from bytes (ASCII) to string.
        my_file["instrumentNames"] = [
            "".join(chr(i) for i in my_file["instrumentNames"])
        ]
        # Remove first element from 'instrumentSizes' if it is 0
        if my_file["instrumentSizes"][0] == 0:
            my_file["instrumentSizes"] = np.delete(my_file["instrumentSizes"], 0)
        # Change 'instrumentSizes' from ndarray to list since some
        # later function expects a list.
        if isinstance(my_file["instrumentSizes"], np.ndarray):
            my_file["instrumentSizes"] = my_file["instrumentSizes"].tolist()
        # Change 'instrumentNames' from ndarray to list since some
        # later function expects a list.
        if isinstance(my_file["instrumentNames"], np.ndarray):
            my_file["instrumentNames"] = my_file["instrumentNames"].tolist()
            # Change 'numDetectorsOrig' from ndarray to scalar.
        if isinstance(my_file["numDetectorsOrig"], np.ndarray) or isinstance(
            my_file["numDetectorsOrig"], list
        ):
            my_file["numDetectorsOrig"] = my_file["numDetectorsOrig"][0]
        # Remove first element from 'filterSizes' if it is 0 and
        # perform calculation of actual filter sizes.
        if my_file["filterSizes"][0] == 0:
            # Because the way the file specifies the
            # my_file['filterSizes'] as [ 0 4451 8402 12553 18554] We
            # have to subtract each number from the previous to get
            # the exact numbers of the frequencies for each filter to
            # get [4451 3951 4151 6001]
            actual_filter_sizes = []
            for ii in range(1, len(my_file["filterSizes"])):
                # Start with the 2nd index, subtract the 2nd index
                # from the first to get the actual filter size for
                # each filter.
                actual_filter_sizes.append(
                    my_file["filterSizes"][ii] - my_file["filterSizes"][ii - 1]
                )
            my_file["filterSizes"] = np.asarray(actual_filter_sizes)
        my_file["radiance"] = my_file["radiance"][:, 0]
        my_file["NESR"] = my_file["NESR"][:, 0]
        tes_struct = {
            "radianceStruct": my_file,
            "boresightNadirRadians": 0.0,
            "instrumentAltitude": 0.0,
        }
        sdesc = {
            "TES_RUN": 0,
            "TES_SEQUENCE": 0,
            "TES_SCAN": 0,
            "POINTINGANGLE_TES": -999,
        }
        res = cls(tes_struct, sdesc)
        swin2 = copy.deepcopy(swin)
        swin2.instrument_name = InstrumentIdentifier("TES")
        res.spectral_window = TesSpectralWindow(swin2, res)
        return res

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str],
        l1b_index: list[int],
        l1b_avgflag: int,
        run: int,
        sequence: int,
        scan: int,
        filter_list: list[str],
        ifile_hlp: InputFileHelper | None = None,
    ) -> Self:
        """Create from just the filenames. Note that spectral window
        doesn't get set here, but this can be useful if you just want
        access to the underlying data.

        You might also want to use create_from_id, which sets up
        everything (spectral window, coefficients, attaching to a
        fm_sv).

        """
        o_tes, sdesc = cls._read_data(
            str(filename),
            l1b_index,
            l1b_avgflag,
            run,
            sequence,
            scan,
            [str(i) for i in filter_list],
            ifile_hlp=ifile_hlp,
        )
        return cls(o_tes, sdesc)

    @classmethod
    def create_from_id(
        cls,
        mid: MeasurementId,
        existing_obs: MusesTesObservation | None,
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
            filter_list = mid.filter_list_dict[InstrumentIdentifier("TES")]
            filename = mid["TES_filename_L1B"]
            l1b_index = mid["TES_filename_L1B_Index"].split(",")
            l1b_avgflag = int(mid["TES_L1B_Average_Flag"])
            run = int(mid["TES_Run"])
            sequence = int(mid["TES_Sequence"])
            scan = int(mid["TES_Scan"])
            o_tes, sdesc = cls._read_data(
                filename,
                l1b_index,
                l1b_avgflag,
                run,
                sequence,
                scan,
                [str(i) for i in filter_list],
                ifile_hlp=ifile_hlp,
            )
            func = mid["apodizationFunction"]
            if func == "NORTON_BEER":
                strength = mid["NortonBeerApodizationStrength"]
                sdef = ifile_hlp.open_tes(
                    mid["defaultSpectralWindowsDefinitionFilename"]
                )
                maxopd = np.array(sdef.checked_table["MAXOPD"])
                flt = np.array(sdef.checked_table["FILTER"])
                spacing = np.array(sdef.checked_table["RET_FRQ_SPC"])
                cls._apodization(o_tes, func, strength, flt, maxopd, spacing)
            obs = cls(o_tes, sdesc)
        # Note that TES has a particularly complicated spectral window needed
        # to match what gets uses of OSS in MusesTesForwardModel. Use this
        # adapter in place of spec_win to match.
        obs.spectral_window = (
            TesSpectralWindow(spec_win, obs)
            if spec_win is not None
            else MusesSpectralWindow(None, None)
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
        return self._muses_py_dict["radianceStruct"]["radiance"]

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["radianceStruct"]["frequency"]

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["radianceStruct"]["NESR"]

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(float(self._muses_py_dict["surfaceElevation"]), "m")


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("TES"), MusesTesObservation)
)

__all__ = [
    "MusesTesObservation",
]
