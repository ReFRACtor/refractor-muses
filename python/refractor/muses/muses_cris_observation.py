from __future__ import annotations
from .muses_spectral_window import MusesSpectralWindow
from .identifier import InstrumentIdentifier, FilterIdentifier
from .input_file_helper import InputFileHelper, InputFilePath
from .muses_observation import (
    MusesObservationImp,
    MeasurementId,
    MusesObservationHandle,
)
from .observation_handle import ObservationHandleSet
import os
import re
import math
import numpy as np
import refractor.framework as rf  # type: ignore
import h5py  # type: ignore
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState


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
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("cm^-1"))
        self._filter_data_swin = rf.SpectralWindowRange(mw_range)

    @classmethod
    def _read_data(
        cls,
        filename: str | os.PathLike[str] | InputFilePath,
        granule: int,
        xtrack: int,
        atrack: int,
        pixel_index: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        o_cris = cls.read_cris(filename, xtrack, atrack, pixel_index, ifile_hlp)
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
        filename: str | os.PathLike[str] | InputFilePath,
        xtrack: int,
        atrack: int,
        pixel_index: int,
        ifile_hlp: InputFileHelper | None = None,
    ) -> dict[str, Any]:
        filename = InputFilePath.create_input_file_path(filename)
        if ifile_hlp is None:
            ifile_hlp = InputFileHelper()
        with ifile_hlp.open_h5(filename) as fh:
            # Check group in noaa but not nasa to determine which reader to use
            if "All_Data/CrIS-FS-SDR_All" in fh:
                # Find geo file by looking in same  directory
                # Have same base, with SCRIF replaced with GCRSO
                nm = re.sub(
                    r"SCRIF_([a-zA-Z]+_d\d+_t\d+_e\d+_).*",
                    r"GCRSO_\1*.h5",
                    filename.name,
                )
                flist = list(filename.parent.glob(nm))
                if len(flist) == 0:
                    raise RuntimeError(
                        f"Could not find GCRSO file to go with {filename}"
                    )
                with ifile_hlp.open_h5(flist[0]) as fh_geo:
                    o_cris = cls.read_noaa_cris_fsr(
                        fh, fh_geo, xtrack, atrack, pixel_index
                    )
            else:
                o_cris = cls.read_nasa_cris_fsr(fh, xtrack, atrack, pixel_index)
        return o_cris

    @classmethod
    def l1b_type_int_from_filename(
        cls, filename: str | os.PathLike[str] | InputFilePath
    ) -> int:
        """Enumeration used in output metadata for the l1b_type"""
        return [
            "suomi_nasa_nsr",
            "suomi_nasa_fsr",
            "suomi_nasa_nomw",
            "jpss1_nasa_fsr",
            "suomi_cspp_fsr",
            "suomi_noaa_fsr",
        ].index(cls.l1b_type_from_filename(filename))

    @classmethod
    def l1b_type_from_filename(
        cls, filename: str | os.PathLike[str] | InputFilePath
    ) -> str:
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
        return [
            "suomi_nasa_nsr",
            "suomi_nasa_fsr",
            "suomi_nasa_nomw",
            "jpss1_nasa_fsr",
            "suomi_cspp_fsr",
            "suomi_noaa_fsr",
        ].index(self.l1b_type)

    @property
    def l1b_type(self) -> str:
        return self._muses_py_dict["l1bType"]

    def desc(self) -> str:
        return "MusesCrisObservation"

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return InstrumentIdentifier("CRIS", self.l1b_type)

    @property
    def spacecraft_altitude(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(float(self._muses_py_dict["SATALT"]), "m")

    @property
    def scan_angle(self) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(float(self._muses_py_dict["SCANANG"]), "deg")

    @property
    def pointing_angle(self) -> rf.DoubleWithUnit:
        return self.scan_angle
    
    def window_fix_for_uip(self, win: dict[str, Any]) -> None:
        """This is bit of kludge for use in RefractorUip.create_uip. This adjusts
        windows needed for doing a joint retrieval with tropomi.

        I'm not sure if this still matters, or what exactly this does. But
        we at least pull to out so RefractorUip isn't mucking around with
        internal variables of MusesCrisObservation.

        win is updated in place."""
        tempind = (self._muses_py_dict["FREQUENCY"] >= win["start"]) & (
            self._muses_py_dict["FREQUENCY"] <= win["endd"]
        )
        MAXOPD = np.unique(self._muses_py_dict["MAXOPD"][tempind])
        SPACING = np.unique(self._muses_py_dict["SPACING"][tempind])
        if len(MAXOPD) > 1 or len(SPACING) > 1:
            raise RuntimeError(
                "Microwindows across CrIS filter bands leading to spacing and OPD does not uniform in this MW!"
            )
        win["maxopd"] = np.float32(MAXOPD[0])
        win["spacing"] = np.float32(SPACING[0])
        win["monoextend"] = np.float32(SPACING[0]) * 4.0

    @classmethod
    def create_from_filename(
        cls,
        filename: str | os.PathLike[str] | InputFilePath,
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
                existing_obs._muses_py_dict,  # noqa: SLF001
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

    def spectrum_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> rf.Spectrum:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        sd = self.spectral_domain_full(sensor_index)
        sr = rf.SpectralRange(
            self.radiance_full(sensor_index, skip_jacobian=skip_jacobian),
            rf.Unit("W / (cm^2 sr cm^-1)"),
            self.nesr_full(sensor_index),
        )
        return rf.Spectrum(sd, sr)
    
    def radiance_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> np.ndarray:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["RADIANCE"]

    def spectral_domain_full(self, sensor_index: int) -> rf.SpectralDomain:
        """Spectral domain before we have removed bad samples or
        applied the microwindows."""
        # By convention, sample index starts with 1. This was from OCO-2, I'm not
        # sure if that necessarily makes sense here or not. But I think we have code
        # that depends on the 1 base.
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        return rf.SpectralDomain(freq, sindex, rf.Unit("cm^-1"))
    
    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["FREQUENCY"]

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        return self._muses_py_dict["NESR"]

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        # TODO the old muses-py code replaces negative altitude with 0. Not
        # sure if this should be done or not, but that is the only instrument that
        # has this behavior. Assume this is right
        r = float(self._muses_py_dict["SURFACEALTITUDE"])
        return rf.DoubleWithUnit(0.0 if r < 0 else r, "m")

    @classmethod
    def read_nasa_cris_fsr(
        cls,
        fh: h5py.File,
        xtrack: int,
        atrack: int,
        pixel_index: int,
    ) -> dict[str, Any]:
        # evaluate the spectral resolution of CrIS L1B file
        max_opd_lw = fh["aux/max_opd_lw"][()]
        max_opd_mw = fh["aux/max_opd_mw"][()]
        max_opd_sw = fh["aux/max_opd_sw"][()]

        # read wavelength grid
        freqScaleLW = fh["wnum_lw"]
        freqScaleMW = fh["wnum_mw"]
        freqScaleSW = fh["wnum_sw"]

        np_lw = freqScaleLW.shape[0]
        np_mw = freqScaleMW.shape[0]
        np_sw = freqScaleSW.shape[0]

        MAXOPD = np.ndarray(shape=(np_lw + np_mw + np_sw), dtype=np.float32)
        MAXOPD[0:np_lw] = max_opd_lw
        MAXOPD[np_lw : np_lw + np_mw] = max_opd_mw
        MAXOPD[np_lw + np_mw : np_lw + np_mw + np_sw] = max_opd_sw

        SPACING = np.ndarray(shape=(np_lw + np_mw + np_sw), dtype=np.float32)
        SPACING[0:np_lw] = freqScaleLW[1] - freqScaleLW[0]
        SPACING[np_lw : np_lw + np_mw] = freqScaleMW[1] - freqScaleMW[0]
        SPACING[np_lw + np_mw : np_lw + np_mw + np_sw] = freqScaleSW[1] - freqScaleSW[0]

        # ===============
        # GEO data fields
        # ===============
        # 45 x 30
        FORTime = fh["obs_time_tai93"][atrack, xtrack]

        # 45 x 30 x 9
        Height = fh["surf_alt"][atrack, xtrack, pixel_index]

        # 45 x 30 x 9
        latitude = fh["lat"][atrack, xtrack, pixel_index]
        latitude_nadir = fh["lat"][atrack, 15, 4]

        # 45 x 30 x 9
        longitude = fh["lon"][atrack, xtrack, pixel_index]
        longitude_nadir = fh["lon"][atrack, 15, 4]

        # 45 x 30 x 9
        SatelliteAzimuthAngle = fh["sat_azi"][atrack, xtrack, pixel_index]

        # 45 x 30 x 9
        SatelliteZenithAngle = fh["sat_zen"][atrack, xtrack, pixel_index]

        # 45 x 30 x 9
        SCANANG = fh["view_ang"][atrack, xtrack, pixel_index]

        if SatelliteZenithAngle == 0.0:
            SatelliteZenithAngle = 0.000001

        if SatelliteZenithAngle < 0.0:
            SatelliteZenithAngle = abs(SatelliteZenithAngle)

        sat_alt = fh["sat_alt"][atrack]

        # 45 x 30 x 9
        SolarAzimuthAngle = fh["sol_azi"][atrack, xtrack, pixel_index]

        # 45 x 30 x 9
        SolarZenithAngle = fh["sol_zen"][atrack, xtrack, pixel_index]

        DAYTIMEFLAG = 0
        if SolarZenithAngle >= 0 and SolarZenithAngle <= 90.0:
            DAYTIMEFLAG = 1

        rad_lw0 = fh["rad_lw"][atrack, xtrack, pixel_index, :]
        rad_mw0 = fh["rad_mw"][atrack, xtrack, pixel_index, :]
        rad_sw0 = fh["rad_sw"][atrack, xtrack, pixel_index, :]

        nedn_lw0 = fh["nedn_lw"][pixel_index]
        nedn_mw0 = fh["nedn_mw"][pixel_index]
        nedn_sw0 = fh["nedn_sw"][pixel_index]

        # see: https://docserver.gesdisc.eosdis.nasa.gov/public/project/JPSS-1/NASA_CrIS_L1B_Product_Users_Guide_V2.11.pdf
        # Table	5.5- 1 CrIS	L1B	Science Variables.

        # NOTE: CrIS L1B noise is NEDR (noise equivalent delta radiance), not NESR (noise equivalent spectral radiance)

        # unit convert from mW/(m2 sr cm-1) to w/(cm^2*sr*cm-1)
        rad_unit_f = rf.conversion(rf.Unit("mW / (m^2 sr cm^-1)"), rf.Unit("W / (cm^2 sr cm^-1)"))

        rad_conv_lw = rad_lw0 * rad_unit_f
        nesr_conv_lw = nedn_lw0 * rad_unit_f
        rad_conv_mw = rad_mw0 * rad_unit_f
        nesr_conv_mw = nedn_mw0 * rad_unit_f
        rad_conv_sw = rad_sw0 * rad_unit_f
        nesr_conv_sw = nedn_sw0 * rad_unit_f

        rad_conv_arr = np.concatenate((rad_conv_lw, rad_conv_mw, rad_conv_sw), axis=0)
        nesr_conv_arr = np.concatenate(
            (nesr_conv_lw, nesr_conv_mw, nesr_conv_sw), axis=0
        )
        FREQUENCY_ARR = np.concatenate((freqScaleLW, freqScaleMW, freqScaleSW), axis=0)

        o_cris_rad = {
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "TIME": FORTime,
            "RADIANCE": rad_conv_arr,
            "DAYTIMEFLAG": DAYTIMEFLAG,
            "NESR": nesr_conv_arr,
            "FREQUENCY": FREQUENCY_ARR,
            "MAXOPD": MAXOPD,
            "SPACING": SPACING,
            "SCANANG": SCANANG,
            "SATLAT": latitude_nadir,
            "SATLONG": longitude_nadir,
            "SATZEN": SatelliteZenithAngle,
            "SATAZI": SatelliteAzimuthAngle,
            "SATALT": sat_alt,
            "SZA": SolarZenithAngle,
            "SOLAZI": SolarAzimuthAngle,
            "SURFACEALTITUDE": Height,
            "L1BQUALITYFLAG": 0,
            "GEOQUALITYFLAG": 0,
            "APODIZATION": "YES",
            "APODIZATIONFUNCTION": "NORTON_BEER_STRONG",
        }
        return o_cris_rad

    @classmethod
    def read_noaa_cris_fsr(
        cls,
        fh: h5py.File,
        fh_geo: h5py.File,
        xtrack: int,
        atrack: int,
        pixel_index: int,
    ) -> dict[str, Any]:
        # mimic what we do in read_nasa_cris_fsr.pro
        # to read-in CrIS CSPP data from the l1b and geo files.
        # return exactly the same data structure.

        # to get the data in sdr l1b data
        # and to calculate the wanted extra data

        es_reallw = fh["/All_Data/CrIS-FS-SDR_All/ES_RealLW"]
        es_realmw = fh["/All_Data/CrIS-FS-SDR_All/ES_RealMW"]
        es_realsw = fh["/All_Data/CrIS-FS-SDR_All/ES_RealSW"]
        es_nednlw = fh["/All_Data/CrIS-FS-SDR_All/ES_NEdNLW"]
        es_nednmw = fh["/All_Data/CrIS-FS-SDR_All/ES_NEdNMW"]
        es_nednsw = fh["/All_Data/CrIS-FS-SDR_All/ES_NEdNSW"]

        # now, Dejian code
        np_lw = es_reallw.shape[3]
        np_mw = es_realmw.shape[3]
        np_sw = es_realsw.shape[3]

        fsr_l1b_flag = 0
        if np_lw == 717 and np_mw == 869 and np_sw == 637:
            fsr_l1b_flag = 1

        if fsr_l1b_flag == 0:
            max_opd_lw = 0.80
            max_opd_mw = 0.40
            max_opd_sw = 0.20
        elif fsr_l1b_flag == 1:
            max_opd_lw = 0.8
            max_opd_mw = 0.8
            max_opd_sw = 0.8
        else:
            raise RuntimeError(
                f"Could not recognize fsr_l1b_flag neither 0 or 1 {fsr_l1b_flag}"
            )

        delta_lw = 0.5 / max_opd_lw
        delta_mw = 0.5 / max_opd_mw
        delta_sw = 0.5 / max_opd_sw

        # define wavelength grid, not provided in the files
        freqScaleLW = np.arange(0, np_lw)
        freqScaleLW = freqScaleLW * delta_lw + 650.0 - delta_lw * 2.0

        freqScaleMW = np.arange(0, np_mw)
        freqScaleMW = freqScaleMW * delta_mw + 1210.0 - delta_mw * 2.0

        freqScaleSW = np.arange(0, np_sw)
        freqScaleSW = freqScaleSW * delta_sw + 2155.0 - delta_sw * 2.0

        MAXOPD = np.ndarray(shape=(np_lw + np_mw + np_sw), dtype=np.float32)
        MAXOPD[0:np_lw] = max_opd_lw  # Note that the right-hand-side is just a float
        MAXOPD[np_lw : np_lw + np_mw] = (
            max_opd_mw  # Note that the right-hand-side is just a float
        )
        MAXOPD[np_lw + np_mw : np_lw + np_mw + np_sw] = (
            max_opd_sw  # Note that the right-hand-side is just a float
        )

        SPACING = np.ndarray(shape=(np_lw + np_mw + np_sw), dtype=np.float32)
        SPACING[0:np_lw] = freqScaleLW[1] - freqScaleLW[0]
        SPACING[np_lw : np_lw + np_mw] = freqScaleMW[1] - freqScaleMW[0]
        SPACING[np_lw + np_mw : np_lw + np_mw + np_sw] = freqScaleSW[1] - freqScaleSW[0]

        # Ming: I only have filename for the sdr data.
        #       All geo data should be in the input cris_file_id

        # to get data from the geo fn_geo
        # and to pick the atrack, xtrack, pix wanted

        FORTime = fh_geo["/All_Data/CrIS-SDR-GEO_All/FORTime"][atrack, xtrack]

        # Ming: since CSPP time starts
        # 1/1/1958 and in micro-sec and NASA tai time starts 1/1/1993,
        # I calc NASA time by subtracting the years (there were 9 leap years)

        delta_time = ((1993 - 1958) * 365 + 9) * np.float64(86400.0)
        FORTime = FORTime * np.float64(1.0) - 6 - delta_time

        Height = fh_geo["All_Data/CrIS-SDR-GEO_All/Height"][atrack, xtrack, pixel_index]

        latitude = fh_geo["All_Data/CrIS-SDR-GEO_All/Latitude"][
            atrack, xtrack, pixel_index
        ]
        # get the satellite lat at nadir xtrack, 15
        latitude_nadir = fh_geo["All_Data/CrIS-SDR-GEO_All/Latitude"][
            atrack, 15, pixel_index
        ]

        longitude = fh_geo["All_Data/CrIS-SDR-GEO_All/Longitude"][
            atrack, xtrack, pixel_index
        ]
        # get the satellite lat at nadir xtrack, 15
        longitude_nadir = fh_geo["All_Data/CrIS-SDR-GEO_All/Longitude"][
            atrack, 15, pixel_index
        ]

        SatelliteZenithAngle = fh_geo["All_Data/CrIS-SDR-GEO_All/SatelliteZenithAngle"][
            atrack, xtrack, pixel_index
        ]

        SatelliteAzimuthAngle = fh_geo[
            "All_Data/CrIS-SDR-GEO_All/SatelliteAzimuthAngle"
        ][atrack, xtrack, pixel_index]

        # Dejians

        degree2sr = math.pi / np.float64(180.0)
        sr2degree = np.float64(180.0) / math.pi

        if SatelliteZenithAngle == 0.0:
            SatelliteZenithAngle = 0.000001

        if SatelliteZenithAngle < 0.0:
            SatelliteZenithAngle = abs(SatelliteZenithAngle)

        temp_rad = cls.earth_radius(latitude)
        SCANANG = (
            math.asin(
                temp_rad
                * math.sin((180.0 - np.float64(SatelliteZenithAngle)) * degree2sr)
                / (824000.0 + temp_rad)
            )
            * sr2degree
        )

        SATALT = fh_geo["All_Data/CrIS-SDR-GEO_All/SCPosition"][atrack, :]
        SATALT = math.sqrt(
            SATALT[0] * SATALT[0] + SATALT[1] * SATALT[1] + SATALT[2] * SATALT[2]
        )
        SATALT = SATALT - temp_rad

        SolarZenithAngle = fh_geo["All_Data/CrIS-SDR-GEO_All/SolarZenithAngle"][
            atrack, xtrack, pixel_index
        ]

        SolarAzimuthAngle = fh_geo["All_Data/CrIS-SDR-GEO_All/SolarAzimuthAngle"][
            atrack, xtrack, pixel_index
        ]

        DAYTIMEFLAG = 0
        if SolarZenithAngle >= 0 and SolarZenithAngle <= 90.0:
            DAYTIMEFLAG = 1

        # to pick the atrack, xtrack, pix wanted in l1b data

        rad_lw0 = es_reallw[atrack, xtrack, pixel_index, :np_lw]
        rad_mw0 = es_realmw[atrack, xtrack, pixel_index, :np_mw]
        rad_sw0 = es_realsw[atrack, xtrack, pixel_index, :np_sw]
        nedn_lw0 = es_nednlw[atrack, xtrack, pixel_index, :np_lw]
        nedn_mw0 = es_nednmw[atrack, xtrack, pixel_index, :np_mw]
        nedn_sw0 = es_nednsw[atrack, xtrack, pixel_index, :np_sw]

        rad_unit_f = rf.conversion(rf.Unit("mW / (m^2 sr cm^-1)"), rf.Unit("W / (cm^2 sr cm^-1)"))
        rad_conv_lw = rad_lw0 * rad_unit_f
        nesr_conv_lw = nedn_lw0 * rad_unit_f
        rad_conv_mw = rad_mw0 * rad_unit_f
        nesr_conv_mw = nedn_mw0 * rad_unit_f
        rad_conv_sw = rad_sw0 * rad_unit_f
        nesr_conv_sw = nedn_sw0 * rad_unit_f

        rad_conv_list = np.concatenate((rad_conv_lw, rad_conv_mw, rad_conv_sw), axis=0)
        nesr_conv_list = np.concatenate(
            (nesr_conv_lw, nesr_conv_mw, nesr_conv_sw), axis=0
        )
        frequency_list = np.concatenate((freqScaleLW, freqScaleMW, freqScaleSW), axis=0)

        # put the data to a structure consisting
        # the same data as in NASA data reader

        o_cris = {
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "TIME": FORTime,
            "RADIANCE": rad_conv_list,
            "DAYTIMEFLAG": DAYTIMEFLAG,
            "NESR": nesr_conv_list,
            "FREQUENCY": frequency_list,
            "MAXOPD": MAXOPD,
            "SPACING": SPACING,
            "SCANANG": SCANANG,
            "SATLAT": latitude_nadir,
            "SATLONG": longitude_nadir,
            "SATZEN": SatelliteZenithAngle,
            "SATAZI": SatelliteAzimuthAngle,
            "SATALT": SATALT,
            "SZA": SolarZenithAngle,
            "SOLAZI": SolarAzimuthAngle,
            "SURFACEALTITUDE": Height,
            "L1BQUALITYFLAG": 0,
            "GEOQUALITYFLAG": 0,
            "APODIZATION": "YES",
            "APODIZATIONFUNCTION": "NORTON_BEER_STRONG",
        }

        return o_cris

    @classmethod
    def earth_radius(cls, latitude: float, tes_pge: bool = False) -> float:
        if tes_pge:
            # TES pge (note TES PGE in km).
            polar_radius = 6.356779e06
            equatorial_radius = 6.378160e06

            ratio = (
                polar_radius * polar_radius / (equatorial_radius * equatorial_radius)
            )
            sine = math.sin(math.radians(latitude))

            res = equatorial_radius * math.sqrt(
                (1.0 + (ratio * ratio - 1.0) * sine * sine)
                / (1.0 + (ratio - 1.0) * sine * sine)
            )
        else:
            #            res = 6.356779e06 + (6.37816e6 - 6.356779e06) * math.cos(
            #                math.radians(latitude)
            #            )
            # TODO We get slight round off differences from muses-py if we use
            # math.radians instead of this calculation. For now, match muses-py just
            # so we don't need to worry in our comparisons
            res = 6.356779e06 + (6.37816e6 - 6.356779e06) * math.cos(
                math.pi * latitude / 180.0
            )

        return res


ObservationHandleSet.add_default_handle(
    MusesObservationHandle(InstrumentIdentifier("CRIS"), MusesCrisObservation)
)

__all__ = [
    "MusesCrisObservation",
]
