from __future__ import annotations
from .observation_handle import ObservationHandleSet
from .muses_observation import (
    MusesObservationImp,
    MusesObservationHandle,
    MeasurementId,
)
from .muses_spectral_window import MusesSpectralWindow, TesSpectralWindow
import os
import numpy as np
import scipy
import refractor.framework as rf  # type: ignore
import copy
from typing import Any, Self
import typing
import itertools
from .identifier import InstrumentIdentifier, FilterIdentifier
from .input_file_helper import InputFileHelper, InputFilePath

if typing.TYPE_CHECKING:
    from .current_state import CurrentState


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
        l1b_avgflag: bool,
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
        filename: str | os.PathLike[str] | InputFilePath,
        l1b_index: list[int],
        l1b_avgflag: bool,
        windows: list[dict[str, Any]],
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        if ifile_hlp is None:
            ifile_hlp = InputFileHelper()
        filters = ["2B1", "1B2", "2A1", "1A1"]
        fpas = ["2B", "1B", "2A", "1A"]
        wavenumberOut = np.zeros((20000), dtype=np.float32)
        spectraOut = np.zeros((20000), dtype=np.float64)
        nesrOut = np.zeros((20000), dtype=np.float64)
        filterOut = np.empty((20000), dtype="<U3")

        count = 0
        filename = InputFilePath.create_input_file_path(filename)
        for ifilter, (flt, nmpas, indx) in enumerate(zip(filters, fpas, l1b_index)):
            with ifile_hlp.open_h5(filename.sub_fname("FP2B", f"FP{nmpas}")) as fh:
                delta = fh[f"Filter{flt}"].attrs["Delta_Frequency"]
                start = fh[f"Filter{flt}"].attrs["Start_Frequency"]
                n_freq = fh[f"Filter{flt}/NESR"].shape[2]
                # get all things same across all filters, e.g. surface elevation.
                if ifilter == 1:
                    # surface elevation is in index 20.  You can count below in Geolocation or stop the reader and look
                    # at the variable o_var_array in hdf5_reader.py
                    geo_info = fh[f"Filter{flt}/Geolocation"][indx]

                    surface_elevation = geo_info[20]

                    # boresightNadirRadians
                    boresight_nadir_radians = geo_info[30] * np.pi / 180.0
                    # orbitInclinationAngle
                    orbit_inclination_angle = geo_info[6]
                    # viewMode
                    view_mode = "Nadir"
                    # instrumentAzimuth
                    instrument_azimuth = geo_info[33]
                    # instrumentLatitude
                    instrument_latitude = geo_info[40]
                    # geoPointing: not used for nadir
                    geo_pointing = -999
                    # targetRadius: not used for nadir
                    target_radius = -999
                    # instrumentRadius: not used for nadir
                    instrument_radius = -999
                    # orbitAscending
                    orbit_ascending = geo_info[2]
                    instrument_altitude = geo_info[42]

                # calculate wavenumber
                wavenumberOut[count : count + n_freq] = (
                    np.array(range(n_freq)) * delta + start
                )
                filterOut[count : count + n_freq] = flt
                if not l1b_avgflag:
                    # 1 scene
                    nesr = fh[f"Filter{flt}/NESR"][:, indx, :]
                    spectra = fh[f"Filter{flt}/Spectra"][:, indx, :]
                    error = fh[
                        f"Filter{flt}/QA/L1B_Target_Spectra/L1B_General_Error_Flag"
                    ][:, indx]
                    indgood = np.where(error == 0)[0]

                    # do averaging over the good of the 16 pixels
                    for ii in range(n_freq):
                        ind = np.where(nesr[indgood, ii] > 0)[0]
                        if len(ind) > 0:
                            indgood2 = indgood[ind]
                            nesrOut[count + ii] = np.sqrt(
                                1 / np.sum(1 / nesr[indgood2, ii] / nesr[indgood2, ii])
                            )
                            spectraOut[count + ii] = (
                                np.sum(
                                    spectra[indgood2, ii]
                                    / nesr[indgood2, ii]
                                    / nesr[indgood2, ii]
                                )
                                * nesrOut[count + ii]
                                * nesrOut[count + ii]
                            )
                        else:
                            # all bad, set nesrOut to negative
                            nesrOut[count + ii] = np.sqrt(
                                1 / np.sum(1 / nesr[indgood, ii] / nesr[indgood, ii])
                            )
                            spectraOut[count + ii] = (
                                np.sum(
                                    spectra[indgood, ii]
                                    / nesr[indgood, ii]
                                    / nesr[indgood, ii]
                                )
                                * nesrOut[count + ii]
                                * nesrOut[count + ii]
                            )
                            nesrOut[count + ii] = -nesrOut[count + ii]
                else:
                    # 2 adjacent scenes
                    nesr = fh[f"Filter{flt}/NESR"][:, indx : indx + 2, :]
                    spectra = fh[f"Filter{flt}/Spectra"][:, indx : indx + 2, :]
                    error = fh[
                        f"Filter{flt}/QA/L1B_Target_Spectra/L1B_General_Error_Flag"
                    ][:, indx : indx + 2].flatten()
                    indgood = np.where(error == 0)[0]

                    # do averaging over the good of the 16 pixels and 2 observations
                    for ii in range(n_freq):
                        nx = nesr[
                            :, :, ii
                        ].flatten()  # flattened so we can index good detectors
                        sx = spectra[
                            :, :, ii
                        ].flatten()  # flattened so we can index good detectors
                        ind = np.where(nx[indgood] > 0)[0]
                        if len(ind) > 0:
                            indgood2 = indgood[ind]
                            nesrOut[count + ii] = np.sqrt(
                                1 / np.sum(1 / nx[indgood2] / nx[indgood2])
                            )
                            spectraOut[count + ii] = (
                                np.sum(sx[indgood] / nx[indgood2] / nx[indgood2])
                                * nesrOut[count + ii]
                                * nesrOut[count + ii]
                            )
                        else:
                            # all bad, set nesrOut to negative
                            nesrOut[count + ii] = np.sqrt(
                                1 / np.sum(1 / nx[indgood] / nx[indgood])
                            )
                            spectraOut[count + ii] = (
                                np.sum(sx[indgood] / nx[indgood] / nx[indgood])
                                * nesrOut[count + ii]
                                * nesrOut[count + ii]
                            )
                            nesrOut[count + ii] = -nesrOut[count + ii]

                count = count + n_freq

        # trim to actual size
        wavenumberOut = wavenumberOut[0:count]
        spectraOut = spectraOut[0:count]
        nesrOut = nesrOut[0:count]
        filterOut = filterOut[0:count]

        radianceStruct = cls.radiance_data(
            spectraOut, nesrOut, wavenumberOut, filterOut, "TES"
        )

        o_tes = {
            "radianceStruct": radianceStruct,
            "surfaceElevation": surface_elevation,
            "boresightNadirRadians": boresight_nadir_radians,
            "orbitInclinationAngle": orbit_inclination_angle,
            "viewMode": view_mode,
            "instrumentAzimuth": instrument_azimuth,
            "instrumentLatitude": instrument_latitude,
            "geoPointing": geo_pointing,
            "targetRadius": target_radius,
            "instrumentRadius": instrument_radius,
            "orbitAscending": orbit_ascending,
            "instrumentAltitude": instrument_altitude / 1000,  # km
        }

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
        # o_tes["radianceStruct"] updated in place.
        cls._radiance_apodize(o_tes["radianceStruct"], strength, flt, maxopd, spacing)

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
        l1b_avgflag: bool,
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
            l1b_index = [int(i) for i in mid["TES_filename_L1B_Index"].split(",")]
            l1b_avgflag = False if int(mid["TES_L1B_Average_Flag"]) == 0 else True
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

    @classmethod
    def _radiance_apodize(
        cls,
        i_radianceStruct: dict[str, Any],
        apodStrength: str,
        i_filter: np.ndarray,
        i_maxOPD: np.ndarray,
        i_spacing: np.ndarray,
    ) -> None:
        # Update i_radianceStruct in place

        apodStrength = apodStrength.lower()

        # this radiance may have different filters or windows.  Go through and
        # pull out ranges of each filter or window by seeing where frequency
        # jumps.
        instrument = (
            i_radianceStruct["instrumentNames"] * i_radianceStruct["instrumentSizes"][0]
        )
        filter = list(
            itertools.chain.from_iterable(
                [
                    nm,
                ]
                * sz
                for nm, sz in zip(
                    i_radianceStruct["filterNames"], i_radianceStruct["filterSizes"]
                )
            )
        )
        frequency = i_radianceStruct["frequency"]

        nf = len(filter)

        # get all places the frequency makes a big jump.
        diff = (frequency[1:] - frequency[0:-1]) / (frequency[1] - frequency[0])
        indf = np.where(abs(diff - 1) > 0.01)[0]
        ind = [0]
        if indf[0] > 0:
            for ii in indf:
                ind.append(ii + 1)
        ind.append(nf)

        radiance = i_radianceStruct["radiance"]
        nesr = i_radianceStruct["NESR"]

        # count new # points
        ntotal = 0
        for iwin in range(0, len(ind) - 1):
            i1 = ind[iwin]
            i2 = ind[iwin + 1] - 1

            indFilter = np.where(i_filter == filter[i1])[0]
            convSpacing = np.mean(
                float(i_spacing[indFilter])
            )  # take mean to ensure is a # not an array

            n = int((frequency[i2] - frequency[i1]) / convSpacing + 1 + 0.5)
            ntotal = ntotal + n

        radNew = np.zeros(ntotal, dtype=np.float64)
        freqNew = np.zeros(ntotal, dtype=np.float64)
        filtNew = np.empty(ntotal, dtype="<U3")
        nesrNew = np.empty(ntotal, dtype=np.float64)
        satNew = np.empty(ntotal, dtype="<U3")

        # record negative nesrs
        indNeg = np.where(nesr < 0)[0]
        if len(indNeg) > 0:
            freqNeg = frequency[indNeg]
        else:
            freqNeg = []

        ntotal = 0
        for iwin in range(0, len(ind) - 1):
            i1 = ind[iwin]
            i2 = ind[iwin + 1]

            indFilter = np.where(i_filter == filter[i1])[0]
            convSpacing = np.mean(np.float64(i_spacing[indFilter]))
            maxOPD = float(np.mean(np.float64(i_maxOPD[indFilter])))

            n = int((frequency[i2 - 1] - frequency[i1]) / convSpacing + 1 + 0.5)
            freqNew[ntotal : ntotal + n] = frequency[i1] + np.array(
                [i * convSpacing for i in range(n)]
            )

            f = frequency[i1:i2]
            r = radiance[i1:i2]
            nes = nesr[i1:i2]
            fn = freqNew[ntotal : ntotal + n]

            ###################### guts of the apodization #######################
            frq_conv, rad_conv = cls.apodize(apodStrength, f, r, maxOPD, o_frequency=fn)

            # calculate nesr... note could conv rad with this but padding
            # is set to NESR.  So just have 2 calls
            # twpr_conv_nesr, apodStrength, f, r, fn, rad_conv2, abs(nes), nesr_conv, maxOPD[0]
            # I was getting 2.34e-8 for original ave rad for 2147-24-2/3.
            # This function (for strong) was resulting in 3.74e-8 even
            # though max value was 3e-8.  Something messed up so go back
            # to strength (bleah)

            if apodStrength == "strong":
                strength = 0.6066
            if apodStrength == "moderate":
                strength = 0.6604
            if apodStrength == "weak":
                strength = 0.7347

            # limb:
            # ENDIF ELSE IF (abs(spacing - 0.02) LT 0.005) THEN BEGIN
            #    IF strengthString EQ 'strong' THEN strength = 0.6066
            #    IF strengthString EQ 'moderate' THEN strength = 0.6604
            #   IF strengthString EQ 'weak' THEN strength = 0.7347
            #    noise.NESR = noise.NESR * strength

            # have to interpolate
            # multiply by strength.  Apodization reduces the NESR because multiple frequencies
            # are averaged.
            nesr_conv = np.interp(fn, f, nes)
            nesr_conv = nesr_conv * strength

            radNew[ntotal : ntotal + n] = rad_conv
            filtNew[ntotal : ntotal + n] = i_filter[indFilter][0]
            satNew[ntotal : ntotal + n] = instrument[i1]
            nesrNew[ntotal : ntotal + n] = nesr_conv
            ntotal = ntotal + n
        # end of filter/window loop

        # locate negative NESRs by frquency and re-negativize them.
        for ii in range(0, len(freqNeg)):
            ind2 = np.where(
                np.abs(freqNew - freqNeg[ii]) < (frequency[1] - frequency[0]) / 2
            )[0]
            if len(ind2) > 0:
                nesrNew[ind2] = -np.abs(nesrNew[ind2])

        i_radianceStruct["frequency"] = freqNew
        filterNames = [str(t) for t in list(dict.fromkeys(filtNew))]
        filterSizes = [
            int(np.count_nonzero(np.array(filtNew) == v)) for v in filterNames
        ]
        i_radianceStruct["filterSizes"] = filterSizes
        i_radianceStruct["instrumentSizes"] = [
            freqNew.shape[0],
        ]
        i_radianceStruct["radiance"] = radNew
        i_radianceStruct["NESR"] = nesrNew

    @classmethod
    def apodize(
        cls,
        i_apodstr: str,
        i_frequency: np.ndarray,
        i_radiance: np.ndarray,
        i_maxopd: float,
        o_frequency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apodize spectra.  Allow apodization functions:  rectangle and
        Norton-Beer weak, moderate, or strong.

        :param i_apodstr: rectangle, weak, moderate, or strong
        :param i_frequency: input frequency array, wavenumber, in units cm-1
        :param i_radiance: rectangle, weak, moderate, or strong
        :param i_maxopd: maximum optical path difference in cm
        :param o_frequency: output frequency array
        """

        if o_frequency is None:
            spacing = 0.5 / i_maxopd
            nn = (max(i_frequency) - min(i_frequency)) / spacing + 0.5
            o_frequency = min(i_frequency) + np.array(range(int(nn))) * spacing

        spacing_in = i_frequency[1] - i_frequency[0]
        spacing_out = o_frequency[1] - o_frequency[0]

        nfactor = int((spacing_out / spacing_in) + 0.5)
        n_in = len(i_frequency)
        n_out = len(o_frequency)

        # find next power of 2 for mono points
        i2 = 1
        p2 = 2
        while p2 < n_in:
            i2 = i2 + 1
            p2 = p2 * 2

        # zero-pad to next power of 2
        padmono = np.zeros((p2), dtype=np.float64)
        padmono[0:n_in] = i_radiance

        # pad with last data point
        # (comment out next line if zero-padding is wanted)
        padmono[n_in : p2 - 1] = i_radiance[n_in - 1]

        # real-to-complex FFTW to interferogram space
        ifgm = scipy.fft.fft(padmono)

        # find ifgm spacing and perform apodization
        apodfn = np.zeros((p2 // 2), dtype=np.float64)
        dx = 1 / (p2 * spacing_in)
        A = 1 / i_maxopd
        Xi = np.array(range(p2 // 2)) * dx
        XiA = Xi * A
        ind = np.where(Xi < i_maxopd)[0]

        if i_apodstr.lower() == "rectangle":
            apodfn[ind] = 1.0
        elif i_apodstr.lower() == "weak":
            U = 1 - XiA[ind] * XiA[ind]
            apodfn[ind] = 0.384093 - 0.087577 * U + 0.703484 * (U * U)
        elif i_apodstr.lower() == "moderate":
            U = 1 - XiA[ind] * XiA[ind]
            apodfn[ind] = 0.152442 - 0.136176 * U + 0.983734 * (U * U)
        elif i_apodstr.lower() == "strong":
            U = 1 - XiA[ind] * XiA[ind]
            U2 = U * U
            apodfn[ind] = 0.045355 + 0.554883 * U2 + 0.399782 * (U2 * U2)

        # make hermitian apodization function to apply to double-sided ifgm
        hermapod = cls.hermitian_array(apodfn)
        ifgm_apod = ifgm * hermapod

        # complex-to-real FFTW to spectral space
        convolved = scipy.fft.ifft(ifgm_apod)

        # calculate first point for output grid
        ifirst = max(np.where(i_frequency < min(o_frequency) + spacing_in / 2))

        # sample points to convolved grid
        isample = np.array(range(n_out)) * nfactor + ifirst
        # o_radiance = np.zeros((n_out), dtype = np.float64)
        o_radiance = convolved[isample]

        return o_frequency, o_radiance

    @classmethod
    def hermitian_array(
        cls, i_array: np.ndarray, f_expand: int = 1, direction: int = 1
    ) -> np.ndarray:
        """Make input array hermitian.  Output array is returned that is twice
        the size of the input array.  If direction is 1, the returned array
        has the original array on the left side of the final array.
        If direction is 2, the returned array has the original array on the right
        side of the final array

        :param i_array: array on evenly spaced grid, presumably in interferogram space
        :param f_expand: expansion of final array.  Default = 1.
        :param direction:  f direction is 1, the returned array
        has the original array on the left side of the final array.  Right side is mirror
        image.  If direction is 2, the returned array has the original array on the right
        side of the final array, left side is mirror image.
        """
        asize = len(i_array)
        hsize = 2 * asize * f_expand
        herm_arr = np.zeros((hsize), dtype=np.float64)
        ind = asize - np.array(range(asize)) - 1

        if direction == 1:
            herm_arr[0:asize] = i_array
            herm_arr[asize:hsize] = i_array[ind]
        else:
            herm_arr[asize:] = i_array
            herm_arr[ind] = i_array

        return herm_arr

    @classmethod
    def radiance_data(
        cls,
        i_radiance: np.ndarray,
        i_nesr: np.ndarray,
        i_frequency: np.ndarray,
        filters: np.ndarray,
        i_instrument: str,
    ) -> dict[str, Any]:
        # Create a standard structure (dictionary in Python).
        o_radianceStruct = cls.radiance_new_struct(i_frequency, filters, i_instrument)

        # Put 3 more elements in the dictionary.
        o_radianceStruct["radiance"] = i_radiance.astype(np.float64)
        o_radianceStruct["NESR"] = i_nesr.astype(np.float64)
        o_radianceStruct["pixelsUsed"][:, :] = 1

        return o_radianceStruct

    @classmethod
    def radiance_new_struct(
        cls, i_frequency: np.ndarray, i_filterArray: np.ndarray, i_instrument: str
    ) -> dict[str, Any]:
        nfreq = len(i_frequency)

        filterArray = i_filterArray
        filterNames = [str(t) for t in list(dict.fromkeys(filterArray))]

        nfilters = len(filterNames)
        filterSizes = [
            int(np.count_nonzero(np.array(filterArray) == v)) for v in filterNames
        ]

        uniqueInstruments = [
            i_instrument,
        ]
        instrumentSizes = [
            nfreq,
        ]

        o_radianceStruct = {
            "filename": "",
            "instrument": "",
            "comments": "",
            "preferences": "",
            "detectors": [
                0,
            ],
            "mws": "",
            "numDetectorsOrig": 1,
            "numDetectors": 1,
            "num_frequencies": nfreq,
            "radiance": "dummy_radiance",
            "NESR": "dummy_NESR",
            "frequency": i_frequency,
            "valid": "yes",
            "filterSizes": filterSizes,
            "filterNames": filterNames,
            "instrumentSizes": instrumentSizes,
            "instrumentNames": uniqueInstruments,
            "pixelsUsed": np.zeros(shape=(1, nfilters), dtype=int),
            "interpixelVar": np.zeros(shape=(nfilters), dtype=np.float32),
            "freqShift": 0.0,
            "imaginaryMean": 0.0,
            "imaginaryRMS": 0.0,
            "bt8": 0.0,
            "bt10": 0.0,
            "bt11": 0.0,
            "scanDirection": 0,
        }

        return o_radianceStruct


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("TES"), MusesTesObservation)
)

__all__ = [
    "MusesTesObservation",
]
