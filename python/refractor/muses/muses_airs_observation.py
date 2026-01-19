from __future__ import annotations
from .misc import osp_setup
from .observation_handle import ObservationHandleSet
from .muses_observation import (
    MusesObservationImp,
    MusesObservationHandle,
    MeasurementId,
)
from .muses_spectral_window import MusesSpectralWindow
import os
import numpy as np
import refractor.framework as rf  # type: ignore
from typing import Any, Self
import typing
from .identifier import InstrumentIdentifier, FilterIdentifier
from .input_file_helper import InputFileHelper, InputFilePath

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    import pyhdf.SD  # type: ignore
    import pyhdf.HDF  # type: ignore
    import pyhdf.VS  # type: ignore


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
        filename: str | os.PathLike[str] | InputFilePath,
        xtrack: int,
        atrack: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        """This is probably a bit over complicated, we don't really need the full
        o_airs structure. But for now, just duplicate the old muses-py code so we
        have a starting point for possibly cleaning up."""
        if ifile_hlp is None:
            ifile_hlp = InputFileHelper()
        # Hardcoded path
        ifile_hlp.notify_file_input(
            ifile_hlp.osp_dir / "AIRS/Bad_Frequencies/airs_bad_frequencies.nc"
        )
        f_sd = None
        f_hdf = None
        f_vs = None
        with osp_setup(ifile_hlp):
            try:
                f_sd = ifile_hlp.open_hdf4_sd(filename)
                f_hdf = ifile_hlp.open_hdf4(filename)
                f_vs = f_hdf.vstart()
                o_airs = cls.read_airs_l1b(f_sd, f_hdf, f_vs, xtrack, atrack)
            finally:
                if f_sd is not None:
                    f_sd.end()
                if f_vs is not None:
                    f_vs.end()
                if f_hdf is not None:
                    f_hdf.close()
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

    @classmethod
    def read_airs_l1b(
        cls,
        f_sd: pyhdf.SD.SD,
        f_hdf: pyhdf.HDF.HDF,
        f_vs: pyhdf.VS.VS,
        iXtrack: int,
        iTrack: int,
    ) -> dict[str, Any]:
        from refractor.muses_py import cdf_read_bad_frequencies_cache

        # Get the 3 geolocation fields: 'Latitude', 'Longitude', and 'Time'

        lon_var = f_sd.select("Longitude")[iTrack, iXtrack]
        lat_var = f_sd.select("Latitude")[iTrack, iXtrack]
        time_var = f_sd.select("Time")[iTrack, iXtrack]
        freq_var = np.array(f_vs.attach("nominal_freq")[:])[:, 0]
        rad0 = f_sd.select("radiances")[iTrack, iXtrack, :]

        # Bit field, by channel, for the current scanline.
        # Zero means the channel was well calibrated, for this scanline.
        # Bit 7 (MSB): scene over/underflow
        # Bit 6: (value 64) anomaly in offset calculation
        # Bit 5: (value 32) anomaly in gain calculation
        # Bit 4: (value 16) pop detected
        # Bit 3: (value 8) DCR Occurred
        # Bit 2: (value 4) Moon in View
        # Bit 1: (value 2) telemetry out of limit condition
        # Bit 0: (LSB, value 1) cold scene noise

        CalFlag = f_sd.select("CalFlag")[iTrack, :]
        satheight = np.float32(f_vs.attach("satheight")[iTrack][0])

        # Bit field. Bitwise OR of CalFlag, by channel, over all scanlines.
        # Noise threshold and spectral quality added.
        # Zero means the channel was well calibrated for all scanlines
        # Bit 7 (MSB): scene over/underflow
        # Bit 6: (value 64) anomaly in offset calculation
        # Bit 5        : (value 32) anomaly in gain calculation
        # Bit 4: (value 16) pop detected
        # Bit 3: (value 8) noise out of bounds
        # Bit 2: (value 4) anomaly in spectral calibration
        # Bit 1: (value 2) Telemetry
        # Bit 0: (LSB, value 1) unused (reserved)

        CalChanSummary = np.array(f_vs.attach("CalChanSummary")[:])[:, 0]

        # An integer 0-6, indicating A/B detector weights.
        # Used in L1B processing. 0 - A weight = B weight. Probably better that channels with state > 2
        # 1 - A-side only. Probably better that channels with state >2
        # 2 - B-side only. Probably better that channels with state >2

        ExcludedChans = np.array(f_vs.attach("ExcludedChans")[:])[:, 0]

        # Noise-equivalent Radiance (radiance units) for an assumed 250K scene
        NESR = np.array(f_vs.attach("NeN")[:])[:, 0]

        # Scanning angle of AIRS instrument with respect to the AIRS
        # instrument for this footprint (- 180.0 ... 180.0, negative
        # at start of scan, 0 at nadir)

        scanang = f_sd.select("scanang")[iTrack, iXtrack]

        # Spacecraft zenith angle (0.0 ... 180.0) degrees from zenith
        # (measured relative to the geodetic vertical on the reference
        # (WGS84) spheroid and including corrections outlined in EOS
        # SDP toolkit for normal accuracy.)

        satzen = f_sd.select("satzen")[iTrack, iXtrack]

        # Spacecraft azimuth angle (-180.0 ... 180.0) degrees E of N GEO)

        satazi = f_sd.select("satazi")[iTrack, iXtrack]

        # Solar zenith angle (0.0 ... 180.0) degrees from zenith
        # (measured relative to the geodetic vertical on the reference
        # (WGS84) spheroid and including corrections outlined in EOS
        # SDP toolkit for normal accuracy.)

        solzen = f_sd.select("solzen")[iTrack, iXtrack]

        # Solar azimuth angle (-180.0 ... 180.0) degrees E of N GEO)

        solazi = f_sd.select("solazi")[iTrack, iXtrack]

        # Distance (km) from footprint center to location of the sun
        # glint (-9999 for unknown, 30000 for no glint visible because
        # spacecraft is in Earth's shadow)

        sun_glint_distance = float(f_sd.select("sun_glint_distance")[iTrack, iXtrack])

        # Mean topography in meters above reference ellipsoid

        topog = f_sd.select("topog")[iTrack, iXtrack]

        # Fraction of spot that is land (0.0 ... 1.0)

        landFrac = f_sd.select("landFrac")[iTrack, iXtrack]

        # Data state: 0:Process, 1:Special, 2:Erroneous, 3:Missing

        state = f_sd.select("state")[iTrack, iXtrack]

        # Assign DayNightFlag using solar zenth angle.
        DayNightFlag = 0
        if solzen > 0 and solzen < 180:
            DayNightFlag = 1

        # For every place where the value of the radiance is negative
        # (less than 0), we set them to some value.  Not recommend
        # them to be not a number.
        rad0[rad0 < 0] = np.nan
        NESR[NESR < 0] = np.nan

        # AIRS currently in m2/m2/sr/cm-1
        # convert AIRS to w/cm2/sr/cm-1 by multiplying by 10-7

        airs = {
            "latitude": lat_var,
            "longitude": lon_var,
            "time": time_var,
            "satheight": satheight,
            "radiance": rad0 * 1e-7,
            "DaytimeFlag": DayNightFlag,
            "CalChanSummary": CalChanSummary,
            "ExcludedChans": ExcludedChans,
            "NESR": NESR * 1e-7,
            "frequency": freq_var,
            "scanAng": scanang,
            "satZen": satzen,
            "satAzi": satazi,
            "sza": solzen,
            "solazi": solazi,
            "sunGlintDistance": sun_glint_distance,
            "surfaceAltitude": topog,
            "landFraction": landFrac,
            "state": state,
            "valid": "Yes",
        }

        # Set negative nesr for all channels with bad radiance.
        if state == 2 or state == 3:
            airs["nesr"] = -abs(airs["nesr"])
            airs["valid"] = "No"

        airs["NESR"][CalFlag != 0] = -999.00
        airs["radiance"][CalFlag != 0] = 0
        airs["NESR"][CalChanSummary != 0] = -999.00
        airs["radiance"][CalChanSummary != 0] = 0

        # non-finite = bad Look for bad values in both arrays 'NESR'
        # and 'radiance'.  If any of them have bad values, we set both
        # arrays to indices where they are bad.

        airs["NESR"][~np.isfinite(airs["radiance"])] = -999.00
        airs["radiance"][~np.isfinite(airs["radiance"])] = 0
        airs["NESR"][~np.isfinite(airs["NESR"])] = -999.00
        airs["radiance"][~np.isfinite(airs["NESR"])] = 0

        # DEVELOPER_NOTE: Because using the built-in tools of Numpy,
        # causes the program to crash, we are doing it manually of
        # building the uu, vv and indbad arrays.
        uu = []
        uu.append(abs(-999 * 20))

        for i in range(len(airs["NESR"])):
            uu.append(abs(airs["NESR"][i]) * 20)

        vv = []
        for i in range(1, len(airs["NESR"])):
            vv.append(abs(airs["NESR"][i]) * 20)

        # Look for bad values and use the indices to set NESR and
        # radiance arrays to their default bad values.
        num_bad_values = 0
        indbad = []
        for i in range(0, len(vv)):
            if (
                (airs["NESR"][i] > uu[i])
                or (airs["NESR"][i] > vv[i])
                and (airs["NESR"][i] > 0)
            ):
                indbad.append(i)  # Save the index where bad values happen.
                num_bad_values = num_bad_values + 1

        if len(indbad) > 0:
            airs["NESR"][np.asarray(indbad)] = -999.0
            airs["radiance"][np.asarray(indbad)] = 0

        # Take out the following points which have persistent biases on the
        # order of 4x the NESR variable.
        indbad = []
        for i in range(0, len(airs["frequency"])):
            if (
                (abs(airs["frequency"][i] - 679.635) < 0.1)
                or (abs(airs["frequency"][i] - 679.887) < 0.1)
                or (abs(airs["frequency"][i] - 688.140) < 0.1)
                or (abs(airs["frequency"][i] - 688.410) < 0.1)
            ):
                indbad.append(i)  # Save the index where bad values happen.

        if len(indbad) > 0:
            airs["NESR"][np.asarray(indbad)] = -999.0
            airs["radiance"][np.asarray(indbad)] = 0

        # Take out AIRS points that are intermittently bad.
        freqlist = [
            717.700,
            728.357,
            728.660,
            732.005,
            732.923,
            733.229,
            735.690,
            736.308,
            738.167,
            741.599,
            743.170,
            743.485,
            743.800,
            744.747,
            745.380,
            747.285,
            749.519,
            749.839,
            751.123,
            751.445,
            752.411,
            754.676,
            756.301,
            758.261,
            758.917,
            759.245,
            759.573,
            761.550,
            762.542,
            762.873,
            763.205,
            763.537,
            764.867,
            766.536,
            767.205,
            781.534,
            789.263,
            789.971,
            842.310,
            843.111,
            900.655,
            901.347,
            1225.650,
            1227.190,
            1263.250,
            1264.890,
            1265.430,
            1290.570,
            1300.300,
        ]

        # this point the spectra seems to smooth over a line at this
        # place; and many AIRS obs are marked bad at this freq.
        freqlist = [761.550]
        for ii in range(0, len(freqlist)):
            indbad = []
            for i in range(0, len(airs["frequency"])):
                if abs(airs["frequency"][i] - freqlist[ii]) < 0.1:
                    indbad.append(i)  # Save the index where bad values happen.

            if len(indbad) > 0:
                airs["NESR"][np.asarray(indbad)] = -999.0
                airs["radiance"][np.asarray(indbad)] = 0

        # take out AIRS points identified bad through comparisons of OSS and
        # ELANOR (places that are hard to model by RT or issues with ILS
        # my_file = cdf_read('../OSP/AIRS/Bad_Frequencies/airs_bad_frequencies.nc')
        cdf_file_name = "../OSP/AIRS/Bad_Frequencies/airs_bad_frequencies.nc"
        my_file = cdf_read_bad_frequencies_cache(cdf_file_name)

        ind = np.where(my_file["BAD"] == 1)[0]

        freqlist = my_file["FREQUENCY"]
        freqlist = freqlist[ind]

        num_bad_points = 0
        for ii in range(0, len(freqlist)):
            indbad2 = np.where(np.abs(airs["frequency"] - freqlist[ii]) < 0.1)[0]
            if len(indbad2) > 0:
                airs["NESR"][indbad2] = -999.0
                airs["radiance"][indbad2] = 0
                num_bad_points += 1

        return airs


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("AIRS"), MusesAirsObservation)
)


__all__ = [
    "MusesAirsObservation",
]
