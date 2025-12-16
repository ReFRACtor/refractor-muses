from __future__ import annotations
from functools import cached_property, lru_cache
from refractor.muses import (
    RefractorFmObjectCreator,
    ForwardModelHandle,
    MusesRaman,
    CurrentState,
    SurfaceAlbedo,
    InstrumentIdentifier,
    StateElementIdentifier,
    InputFileMonitor
)
from functools import cache
import refractor.framework as rf  # type: ignore
from loguru import logger
import numpy as np
import re
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import MeasurementId, MusesObservation, RetrievalConfiguration


class TropomiSurfaceAlbedo(SurfaceAlbedo):
    def __init__(self, ground: rf.GroundWithCloudHandling, spec_index: int) -> None:
        self.ground = ground
        self.spec_index = spec_index

    def surface_albedo(self) -> float:
        # We just directly use the coefficients for the constant term. Could
        # do something more clever, but this is what py-retrieve does
        if self.ground.do_cloud:
            return self.ground.ground_cloud.albedo_coefficients(self.spec_index)[
                0
            ].value
        else:
            return self.ground.ground_clear.albedo_coefficients(self.spec_index)[
                0
            ].value


class TropomiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        use_raman: bool = True,
        use_oss: bool = False,
        oss_training_data: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            current_state,
            measurement_id,
            retrieval_config,
            InstrumentIdentifier("TROPOMI"),
            observation,
            use_raman=use_raman,
            **kwargs,
        )
        self.use_oss = use_oss
        self.oss_training_data = oss_training_data

    @cached_property
    def instrument_correction(self) -> list[list[rf.InstrumentCorrection]]:
        res = []
        for i in range(self.num_channels):
            v: list[rf.InstrumentCorrection] = []
            v.append(self.radiance_scaling[i])
            res.append(v)
        return res

    def ils_params_preconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf = 4
        wn, sindex = self.observation.wn_and_sindex(sensor_index)
        return self.get_tropomi_ils(
            self.observation.frequency_full(sensor_index),
            sindex,
            self.observation.wavelength_filter,
            num_fwhm_srf,
        )

    def ils_params_postconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        # Place holder, this doesn't work yet. Copy of what is
        # done in make_uip_tropomi.
        num_fwhm_srf = 4

        # Kind of a convoluted way to just get the monochromatic list, but this is what
        # make_uip_tropomi does, so we duplicate it here. We have the observation frequencies,
        # and the monochromatic for all the windows added into one long list, and then give
        # the index numbers that are actually relevant for the ILS.
        mono_list, mono_filter_list, mono_list_length = (
            self.observation.spectral_window.muses_monochromatic()
        )
        wn, _ = self.observation.wn_and_sindex(sensor_index)
        wn_list = np.concatenate([wn, mono_list], axis=0)
        wn_filter = np.concatenate(
            [self.observation.wavelength_filter, mono_filter_list], axis=0
        )
        startmw_fm = len(wn) + sum(mono_list_length[:sensor_index])

        sindex = np.arange(0, mono_list_length[sensor_index]) + startmw_fm
        return self.get_tropomi_ils(
            wn_list,
            sindex,
            wn_filter,
            num_fwhm_srf,
        )

    def ils_params_fastconv(self, sensor_index: int) -> dict[str, Any]:
        # Note, I'm not sure of fastconv actually works. This fails in muses-py if
        # we try to use fastconv. From the code, I *think* this is what was intended,
        # but I don't know if this was actually tested anywhere since you can't actuall
        # run this. But put this in here for completeness.

        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf = 4
        wn, sindex = self.observation.wn_and_sindex(sensor_index)
        # Kind of a convoluted way to just get the monochromatic list, but this is what
        # make_uip_tropomi does, so we duplicate it here. We have the observation frequencies,
        # and the monochromatic for all the windows added into one long list, and then give
        # the index numbers that are actually relevant for the ILS.
        mono_list, mono_filter_list, mono_list_length = (
            self.observation.spectral_window.muses_monochromatic()
        )
        wn_list = np.concatenate([wn, mono_list], axis=0)
        wn_filter = np.concatenate(
            [self.observation.wavelength_filter, mono_filter_list], axis=0
        )
        startmw_fm = len(wn) + sum(mono_list_length[:sensor_index])

        sindex2 = np.arange(0, mono_list_length[sensor_index]) + startmw_fm

        return self.get_tropomi_ils_fastconv(
            wn_list,
            sindex,
            wn_filter,
            num_fwhm_srf,
            i_monochromfreq=wn_list[sindex2],
            i_interpmethod="INTERP_MONOCHROM",
        )

    def ils_params(self, sensor_index: int) -> dict[str, Any]:
        """ILS parameters"""
        if self.ils_method(sensor_index) == "APPLY":
            return self.ils_params_preconv(sensor_index)
        elif self.ils_method(sensor_index) == "POSTCONV":
            return self.ils_params_postconv(sensor_index)
        elif self.ils_method(sensor_index) == "FASTCONV":
            return self.ils_params_fastconv(sensor_index)
        else:
            raise RuntimeError(
                f"Unrecognized ils_method {self.ils_method(sensor_index)}"
            )

    def ils_method(self, sensor_index: int) -> str:
        """Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV"""
        # Note in principle we could have this be a function of the sensor band,
        # however the current implementation just has one value set here.
        #
        # This is currently the same for all sensor_index values. We can extend this
        # if needed, but we also need to change the absorber to handle this, since the
        # selection is based off the ils_method. We could probably wrap the different
        # absorber types and select which one is used by some kind of logic.
        #
        # We really need a test case to work through the logic here before changing this
        return self.retrieval_config["ils_tropomi_xsection"]

    def instrument_hwhm(self, sensor_index: int) -> rf.DoubleWithUnit:
        band_name = str(self.filter_list[sensor_index])
        if band_name == "BAND7":
            # JLL: testing different values of HWHM with the IlsGrating component,
            # this value (= a 0.2 nm difference at 2330 nm) gave output spectra that
            # compared well with TROPOMI radiances. This is wider than the HWHM in
            # the Landgraf CO paper (doi: 10.5194/amt-9-4955-2016), which gives 0.25 nm
            # as the FULL width half max in Appendix B, but a HWHM of 0.25/2 nm didn't
            # compare as well. (NB in their Appendix B, there is a 0.1 nm FWHM, but that
            # is only part of the ISRF).
            return rf.DoubleWithUnit(0.36, "cm^-1")
        else:
            raise NotImplementedError(f"HWHM for band {band_name} not defined")

    def reference_wavelength(self, sensor_index: int) -> float:
        """Calculate the reference wavelength used in couple of places. By convention, this
        is the middle of the band used for MusesRaman, including bad samples."""
        with self.observation.modify_spectral_window(
            do_raman_ext=True, include_bad_sample=True
        ):
            t = self.observation.spectral_domain(sensor_index).data
            if len(t) < 2:
                return 0
            return (t[0] + t[-1]) / 2

    @cached_property
    def radiance_scaling(self) -> list[rf.InstrumentCorrection]:
        # By convention, the reference band is the middle of the full
        # frequency (see rev_and_fm_map
        res = []
        for i in range(self.num_channels):
            filter_name = self.filter_list[i]

            selem = [
                StateElementIdentifier(f"TROPOMIRESSCALEO0{filter_name}"),
                StateElementIdentifier(f"TROPOMIRESSCALEO1{filter_name}"),
                StateElementIdentifier(f"TROPOMIRESSCALEO2{filter_name}"),
            ]
            coeff, mp = self.current_state.object_state(selem)
            rscale = rf.RadianceScalingSvMusesFit(
                coeff,
                rf.DoubleWithUnit(self.reference_wavelength(i), "nm"),
                str(filter_name),
            )
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv,
                selem,
                [
                    rscale,
                ],
            )
            res.append(rscale)
        return res

    @cached_property
    def temperature(self) -> rf.Temperature:
        tlev_fm, _ = self.current_state.object_state(
            [
                StateElementIdentifier("TATM"),
            ]
        )
        selem = [
            StateElementIdentifier("TROPOMITEMPSHIFTBAND3"),
        ]
        coeff, _ = self.current_state.object_state(selem)
        tlevel = rf.TemperatureLevel(tlev_fm, self.pressure_fm)
        t = rf.TemperatureLevelOffset(
            self.pressure_fm, tlevel.temperature_profile(), coeff[0]
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [
                t,
            ],
        )
        return t

    @cached_property
    def ground_clear(self) -> rf.Ground:
        albedo = np.zeros((self.num_channels, 3))
        band_reference = np.zeros(self.num_channels)
        selem = []
        for i in range(self.num_channels):
            filt_name = self.filter_list[i]
            if re.match(r"BAND\d$", str(filt_name)) is not None:
                band_reference[i] = self.reference_wavelength(i)
                selem.extend(
                    [
                        StateElementIdentifier(f"TROPOMISURFACEALBEDO{filt_name}"),
                        StateElementIdentifier(f"TROPOMISURFACEALBEDOSLOPE{filt_name}"),
                        StateElementIdentifier(
                            f"TROPOMISURFACEALBEDOSLOPEORDER2{filt_name}"
                        ),
                    ]
                )
            else:
                raise RuntimeError("Don't recognize filter name")

        coeff, mp = self.current_state.object_state(selem)
        albedo[:, :] = np.reshape(coeff, albedo.shape)
        res = rf.GroundLambertian(
            albedo,
            rf.ArrayWithUnit(band_reference, "nm"),
            rf.Unit("nm"),
            [str(i) for i in self.filter_list],
            mp,
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [
                res,
            ],
        )
        return res

    @cached_property
    def ground_cloud(self) -> rf.Ground:
        albedo = np.zeros((self.num_channels, 1))
        np.full((self.num_channels, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)
        band_reference[:] = 1000
        selem = [
            StateElementIdentifier("TROPOMICLOUDSURFACEALBEDO"),
        ]
        coeff, mp = self.current_state.object_state(selem)
        albedo[:, 0] = coeff[0]
        res = rf.GroundLambertian(
            albedo,
            rf.ArrayWithUnit(band_reference, "nm"),
            [
                "Cloud",
            ]
            * self.num_channels,
            mp,
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [
                res,
            ],
        )
        return res

    @cached_property
    def cloud_fraction(self) -> float:
        selem = [
            StateElementIdentifier("TROPOMICLOUDFRACTION"),
        ]
        coeff, mp = self.current_state.object_state(selem)
        cf = rf.CloudFractionFromState(float(coeff[0]))
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [
                cf,
            ],
        )
        return cf

    @lru_cache(maxsize=None)
    def raman_effect(self, i: int) -> rf.RamanSiorisEffect:
        if self.match_py_retrieve:
            return self.raman_effect_muses(i)
        else:
            return self.raman_effect_refractor(i)

    @lru_cache(maxsize=None)
    def raman_effect_muses(self, i: int) -> rf.RamanSiorisEffect:
        # Note we should probably look at this sample grid, and
        # make sure it goes RamanSioris.ramam_edge_wavenumber past
        # the edges of our spec_win. Also there isn't any particular
        # reason that the solar data/optical depth should be calculated
        # on the muses_fm_spectral_domain. But this is what muses-py
        # does, so we'll match that for now.
        selem = [
            StateElementIdentifier(f"TROPOMIRINGSF{self.filter_list[i]}"),
        ]
        if str(self.filter_list[i]) in ("BAND1", "BAND2", "BAND3"):
            coeff, mp = self.current_state.object_state(selem)
            scale_factor = float(coeff[0])
        elif str(self.filter_list[i]) in ("BAND7", "BAND8"):
            # JLL: The SWIR bands should not need to account for Raman scattering -
            # Vijay has never seen Raman scattering accounted for in the CO band.
            scale_factor = None
        else:
            raise RuntimeError("Unrecognized filter_list")
        if scale_factor is None:
            return None
        else:
            with self.observation.modify_spectral_window(do_raman_ext=True):
                wlen = self.observation.spectral_domain(i)
            # This is short if we aren't actually running this filter
            if wlen.data.shape[0] < 2:
                return None
            salbedo = TropomiSurfaceAlbedo(self.ground, i)
            ram = MusesRaman(
                salbedo,
                self.ray_info,
                wlen,
                float(scale_factor),
                i,
                rf.DoubleWithUnit(self.sza[i], "deg"),
                rf.DoubleWithUnit(self.oza[i], "deg"),
                rf.DoubleWithUnit(self.raz[i], "deg"),
                self.atmosphere,
                self.solar_model(i),
                rf.StateMappingLinear(),
            )
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv,
                selem,
                [
                    ram,
                ],
            )
            return ram

    @cached_property
    def underlying_forward_model(self) -> rf.ForwardModel:
        if self.use_oss:
            res = rf.director.OSSForwardModel(
                self.instrument,
                self.spec_win,
                self.radiative_transfer,
                self.spectrum_sampling,
                self.spectrum_effect,
                self.oss_training_data,
            )
            res.setup_grid()
        else:
            res = rf.StandardForwardModel(
                self.instrument,
                self.spec_win,
                self.radiative_transfer,
                self.spectrum_sampling,
                self.spectrum_effect,
            )
            res.setup_grid()
        return res

    @lru_cache(maxsize=None)
    def raman_effect_refractor(self, i: int) -> rf.RamanSiorisEffect:
        selem = [
            StateElementIdentifier(f"TROPOMIRINGSF{self.filter_list[i]}"),
        ]
        if str(self.filter_list[i]) in ("BAND1", "BAND2", "BAND3"):
            coeff, mp = self.current_state.object_state(selem)
            scale_factor = float(coeff[0])
        elif str(self.filter_list[i]) in ("BAND7", "BAND8"):
            # JLL: The SWIR bands should not need to account for Raman scattering -
            # Vijay has never seen Raman scattering accounted for in the CO band.
            scale_factor = None
        else:
            raise RuntimeError("Unrecognized filter_list")
        if scale_factor is None:
            return None
        else:
            with self.observation.modify_spectral_window(do_raman_ext=True):
                wlen = self.observation.spectral_domain(i)
            # This is short if we aren't actually running this filter
            if wlen.data.shape[0] < 2:
                return None
            ram = rf.RamanSiorisEffect(
                wlen,
                scale_factor,
                i,
                rf.DoubleWithUnit(self.sza[i], "deg"),
                rf.DoubleWithUnit(self.oza[i], "deg"),
                rf.DoubleWithUnit(self.raz[i], "deg"),
                self.atmosphere,
                self.solar_model(i),
                rf.StateMappingLinear(),
            )
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv,
                selem,
                [
                    ram,
                ],
            )
            return ram

    def get_tropomi_ils(
        self,
        i_Frequency: np.ndarray,
        i_freqIndex: np.ndarray,
        i_WAVELENGTH_FILTER: list[str],
        i_num_fwhm_srf: int,
    ) -> dict[str, Any]:
        # ILS files
        ils_dir_bands_1_6 = (
            self.osp_dir / "TROPOMI/isrf_release/isrf/binned_uvn_spectral_unsampled/"
        )
        fn_1_6 = (
            ils_dir_bands_1_6
            / "S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc"
        )

        ils_dir_bands_7_8 = (
            self.osp_dir / "TROPOMI/isrf_release/isrf/binned_uvn_swir_sampled/"
        )
        fn_7_8 = (
            ils_dir_bands_7_8
            / "S5P_OPER_AUX_ISRF___00000101T000000_99991231T235959_20180320T084215.nc"
        )

        # number of data points
        num_points = len(i_freqIndex)
        band_index_list = np.zeros((num_points,)).astype("int")

        num_bands = 8
        band_num_delta_wavelengths = np.zeros((num_bands,))

        # Determine all bands used:
        for i_temp in range(0, num_points):
            FilterBand = i_WAVELENGTH_FILTER[i_freqIndex[i_temp]]
            band_index_list[i_temp] = int(FilterBand[-1])

        use_bands_1_6_flag = np.min(band_index_list) < 7
        use_bands_7_8_flag = np.max(band_index_list) >= 7

        if use_bands_1_6_flag:
            with InputFileMonitor.open_h5(fn_1_6, self.ifile_mon) as f:
                for i_band in range(6):
                    FilterBand_group = f"band_{i_band + 1}"
                    isrf_var = f[FilterBand_group]["isrf"]
                    band_num_delta_wavelengths[i_band] = isrf_var.shape[2]

        if use_bands_7_8_flag:
            with InputFileMonitor.open_h5(fn_7_8, self.ifile_mon) as f:
                for i_band in range(6, num_bands):
                    FilterBand_group = f"band_{i_band + 1}"
                    delta_wavelength_var = f[FilterBand_group]["delta_wavelength"]
                    band_num_delta_wavelengths[i_band] = len(delta_wavelength_var)

        # MT: We are going to interpolate the isrf functions, as suggested in:
        # https://sentinel.esa.int/documents/247904/2476257/Sentinel-5P-TROPOMI-Level-1B-ATBD, Appendix B.2

        central_wavelength = np.zeros((num_points,)) + np.float64(-999.0)

        num_delta_wavelengths = np.zeros((num_points,)).astype("int")
        max_num_delta_wavelengths = int(np.max(band_num_delta_wavelengths))
        delta_wavelength = np.zeros(
            (
                num_points,
                max_num_delta_wavelengths,
            )
        ) + np.float64(-999.0)

        isrf = np.zeros(
            (
                num_points,
                max_num_delta_wavelengths,
            )
        ) + np.float64(-999.0)

        for tempi in range(0, num_points):
            FilterBand = i_WAVELENGTH_FILTER[i_freqIndex[tempi]]

            band_index = int(FilterBand[-1])
            FilterBand_group = f"band_{band_index}"

            filter_band_name = np.asarray(
                self.observation.observation_table["Filter_Band_Name"]
            )
            xtrack = np.asarray(self.observation.observation_table["XTRACK"])

            where_filter_occured = np.where(filter_band_name == FilterBand)[0]
            PixelIndex = xtrack[where_filter_occured]

            tempfreq = i_Frequency[i_freqIndex[tempi]]

            # MT: Some of the wavelengths in the instrument grid seem to be outside the range of the central wavelengths
            # So we will interpolate between bands if necessary.

            if band_index < 7:
                if self.ifile_mon is not None:
                    self.ifile_mon.notify_file_input(fn_1_6)
                central_wavelength_var, delta_wavelength_var, isrf_var = _tropomi_ils(
                    fn_1_6, band_index
                )
                temp_central_wavelength = central_wavelength_var[
                    PixelIndex, :
                ].flatten()
                temp_delta_wavelength = delta_wavelength_var
            elif band_index >= 7:
                if self.ifile_mon is not None:
                    self.ifile_mon.notify_file_input(fn_7_8)
                central_wavelength_var, delta_wavelength_var, isrf_var = _tropomi_ils(
                    fn_7_8, band_index
                )

                temp_central_wavelength = central_wavelength_var
                temp_delta_wavelength = delta_wavelength_var

            if (temp_central_wavelength[0] <= tempfreq + 0.0001) and (
                temp_central_wavelength[-1] >= tempfreq - 0.0001
            ):
                tempind_low = np.where(temp_central_wavelength <= tempfreq + 0.0001)[0][
                    -1
                ]
                if tempind_low < len(temp_central_wavelength) - 1:
                    tempind_high = tempind_low + 1
                    central_wavelength_low = temp_central_wavelength[tempind_low]
                    central_wavelength_high = temp_central_wavelength[tempind_high]
                    lambda_low = (tempfreq - central_wavelength_low) / (
                        central_wavelength_high - central_wavelength_low
                    )
                    lambda_high = 1.0 - lambda_low
                else:
                    tempind_high = tempind_low
                    central_wavelength_low = temp_central_wavelength[tempind_low]
                    central_wavelength_high = temp_central_wavelength[tempind_high]
                    lambda_low = 1.0
                    lambda_high = 0.0

                central_wavelength[tempi] = tempfreq

                delta_wavelength[tempi, : len(temp_delta_wavelength)] = (
                    temp_delta_wavelength
                )
                num_delta_wavelengths[tempi] = len(temp_delta_wavelength)

                if band_index < 7:
                    if tempind_high != tempind_low:
                        temp_isrf_low = isrf_var[PixelIndex, tempind_low, :].flatten()
                        temp_isrf_high = isrf_var[PixelIndex, tempind_high, :].flatten()
                        temp_isrf = (lambda_low * temp_isrf_low) + (
                            lambda_high * temp_isrf_high
                        )
                    else:
                        temp_isrf = isrf_var[PixelIndex, tempind_low, :].flatten()
                else:
                    if tempind_high != tempind_low:
                        temp_isrf_low = isrf_var[PixelIndex, tempind_low, :].flatten()
                        temp_isrf_high = isrf_var[PixelIndex, tempind_high, :].flatten()
                        temp_isrf = (lambda_low * temp_isrf_low) + (
                            lambda_high * temp_isrf_high
                        )
                    else:
                        temp_isrf = isrf_var[PixelIndex, tempind_low, :].flatten()

                isrf[tempi, : len(temp_isrf)] = temp_isrf / np.sum(temp_isrf)
            else:
                logger.error(
                    "Wavelength ",
                    tempfreq,
                    " outside of range [",
                    temp_central_wavelength[0],
                    ", ",
                    temp_central_wavelength[-1],
                    "] by more than 0.0001 ",
                )
                assert False

        o_ils = {
            "central_wavelength": central_wavelength,
            "central_wavelength_fm": i_Frequency[i_freqIndex],
            "freqIndex_fm": i_freqIndex,
            "delta_wavelength": delta_wavelength,
            "isrf": isrf,
            "num_fwhm_srf": i_num_fwhm_srf,
            # This tells RefractorFmObjectCreator and its children that it can use the
            # central_wavelength array from this dictionary as the monochromatic
            # wavelength grid for the RT. If that is not the case, set this to False.
            "central_wavelength_is_mono_grid": True,
        }

        return o_ils

    def get_tropomi_ils_fastconv(
        self,
        i_Frequency: np.ndarray,
        i_freqIndex: np.ndarray,
        i_WAVELENGTH_FILTER: list[str],
        i_num_fwhm_srf: int,
        i_monochromfreq: np.ndarray,
        i_interpmethod: str = "NO_INTERP",
        i_sv_threshold: float = 0.99,
    ) -> dict[str, Any]:
        ilsInfo = self.get_tropomi_ils(
            i_Frequency, i_freqIndex, i_WAVELENGTH_FILTER, i_num_fwhm_srf
        )
        central_wavelength = ilsInfo["central_wavelength"]
        central_wavelength = central_wavelength[np.where(central_wavelength > -999)]
        N_conv_wl = len(central_wavelength)
        # We will have several options: 1) Interpolate monochromatic radiance to the ILS wavelength grid (requires i_monochromfreq input),
        # 2) Interpolate ILS to the monochromatic grid, 3) Do not interpolate (either because they are already on the same grid, or because we
        # will interpolate the monochromatic radiance to the ILS frequency grid later).

        def integrate_trap_rule(x_vec: np.ndarray, y_vec: np.ndarray) -> float:
            y_integral = 0
            for i in range(len(x_vec) - 1):
                y_integral = y_integral + 0.5 * (x_vec[i + 1] - x_vec[i]) * (
                    y_vec[i] + y_vec[i + 1]
                )

            return y_integral

        ind_monochrom = np.where(i_monochromfreq > -999)[0]
        monochromgrid = i_monochromfreq[
            ind_monochrom
        ]  # MT: It looks like this will likely be wavelength instead of frequency, so we'll call it a generic "grid."
        N_monochrom = len(monochromgrid)

        if (
            i_interpmethod == "INTERP_MONOCHROM"
        ):  # Interpolate monochromatic grid to ILS wavelength grid
            # Normalize and rescale ILS curves:
            isrf_normalized = np.zeros(ilsInfo["isrf"].shape)  # + -999
            isrf_rescaled = np.zeros((N_conv_wl, N_monochrom))  # + -999
            # temp_isrf = np.zeros(ilsInfo['isrf'].shape[1],)

            for i_isrf in range(N_conv_wl):
                temp_delta_wavelength = ilsInfo["delta_wavelength"][i_isrf, :].flatten()
                temp_ind = np.where(temp_delta_wavelength > -999)
                temp_delta_wavelength = temp_delta_wavelength[temp_ind]
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                temp_isrf = ilsInfo["isrf"][i_isrf, :].flatten()

                isrf_normalized[i_isrf, temp_ind[0]] = temp_isrf[:]

                TF_1 = isrf_wavelength >= monochromgrid[0]

                for i_monochrom in range(N_monochrom - 1):
                    wl_0 = monochromgrid[i_monochrom]
                    wl_1 = monochromgrid[i_monochrom + 1]

                    TF_0 = TF_1
                    TF_1 = isrf_wavelength >= wl_1

                    temp_isrf[:] = 0
                    temp_isrf[TF_0] = (isrf_normalized[i_isrf, temp_ind[0][TF_0]]) / (
                        wl_1 - wl_0
                    )
                    temp_isrf[TF_1] = 0

                    response_component_1 = np.sum(temp_isrf)
                    response_component_2 = np.sum(
                        np.multiply(isrf_wavelength, temp_isrf)
                    )

                    isrf_rescaled[i_isrf, i_monochrom] = (
                        isrf_rescaled[i_isrf, i_monochrom]
                        + wl_1 * response_component_1
                        - response_component_2
                    )
                    isrf_rescaled[i_isrf, i_monochrom + 1] = (
                        isrf_rescaled[i_isrf, i_monochrom + 1]
                        - wl_0 * response_component_1
                        + response_component_2
                    )

                wl_end = monochromgrid[-1]
                temp_isrf[:] = 0
                temp_isrf[isrf_wavelength == wl_end] = 1
                isrf_rescaled[i_isrf, -1] = isrf_rescaled[i_isrf, -1] + np.sum(
                    np.multiply(temp_isrf, isrf_normalized[i_isrf, temp_ind[0]])
                )

                del temp_isrf

        elif i_interpmethod == "INTERP_ILS":  # Interpolate ILS to monochromatic grid
            # Normalize and rescale ILS curves:
            isrf_normalized = np.zeros((N_conv_wl, N_monochrom))  # + -999
            isrf_rescaled = np.zeros((N_conv_wl, N_monochrom))  # + -999
            # temp_isrf = np.zeros(ilsInfo['isrf'].shape[1],)

            for i_isrf in range(N_conv_wl):
                temp_delta_wavelength = ilsInfo["delta_wavelength"][i_isrf, :].flatten()
                temp_ind = np.where(temp_delta_wavelength > -999)
                temp_delta_wavelength = temp_delta_wavelength[temp_ind]
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                temp_isrf_1 = np.interp(
                    monochromgrid,
                    isrf_wavelength,
                    temp_delta_wavelength,
                    ilsInfo["isrf"][i_isrf, temp_ind[0]].flatten(),
                )

                temp_isrf_2 = temp_isrf_1

                # isrf_normalized[i_isrf, temp_ind[0]] = temp_isrf_2[:]
                isrf_normalized[i_isrf, :] = temp_isrf_2[:]

                for i_monochrom in range(1, N_monochrom - 1):
                    # isrf_rescaled[i_isrf, i_monochrom] = 0.5*(monochromgrid[i_monochrom + 1] - monochromgrid[i_monochrom - 1])*isrf_normalized[i_isrf, temp_ind[0][i_monochrom]]
                    isrf_rescaled[i_isrf, i_monochrom] = (
                        0.5
                        * (
                            monochromgrid[i_monochrom + 1]
                            - monochromgrid[i_monochrom - 1]
                        )
                        * isrf_normalized[i_isrf, i_monochrom]
                    )

                # isrf_rescaled[i_isrf, 0] = 0.5*(monochromgrid[1] - monochromgrid[0])*isrf_normalized[i_isrf, temp_ind[0][0]]
                # isrf_rescaled[i_isrf, -1] = 0.5*(monochromgrid[-1] - monochromgrid[-2])*isrf_normalized[i_isrf, temp_ind[0][-1]]
                isrf_rescaled[i_isrf, 0] = (
                    0.5
                    * (monochromgrid[1] - monochromgrid[0])
                    * isrf_normalized[i_isrf, 0]
                )
                isrf_rescaled[i_isrf, -1] = (
                    0.5
                    * (monochromgrid[-1] - monochromgrid[-2])
                    * isrf_normalized[i_isrf, -1]
                )

                del temp_isrf_1
                del temp_isrf_2

        else:  # Do not interpolate either grid
            # Normalize and rescale ILS curves:
            isrf_normalized = np.zeros((N_conv_wl, N_monochrom))  # + -999
            isrf_rescaled = np.zeros((N_conv_wl, N_monochrom))  # + -999

            for i_isrf in range(N_conv_wl):
                temp_delta_wavelength = ilsInfo["delta_wavelength"][i_isrf, :].flatten()
                temp_ind = np.where(temp_delta_wavelength > -999)
                temp_delta_wavelength = temp_delta_wavelength[temp_ind]
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                temp_isrf = ilsInfo["isrf"][i_isrf, temp_ind[0]].flatten()

                isrf_normalized[i_isrf, temp_ind[0]] = temp_isrf[:]
                isrf_rescaled[i_isrf, temp_ind[0]] = temp_isrf[:]

                del temp_isrf

        # Normalize the ils response functions by their L1-norm for a (hopefully) lower-rank SVD.
        isrf_norms = np.sum(isrf_rescaled, 1)
        for i_isrf in range(N_conv_wl):
            if np.abs(isrf_norms[i_isrf]) > 0:
                isrf_rescaled[i_isrf, :] = isrf_rescaled[i_isrf, :] / isrf_norms[i_isrf]

        isrf_mat = np.zeros(
            (N_conv_wl, N_monochrom), dtype=float
        )  # num pixel samples x num high res
        center_wl_indices = np.zeros(isrf_rescaled.shape[0], dtype=int)
        monochrom_center_index = N_monochrom // 2

        for i_isrf in range(N_conv_wl):
            isrf_resp = isrf_rescaled[i_isrf, :]
            max_resp_idx = int(
                np.round(np.mean(np.where(isrf_resp >= 0.90 * max(isrf_resp))))
            )
            center_wl_indices[i_isrf] = max_resp_idx

            # Roll center of ILS response to center of of array
            isrf_mat[i_isrf, :] = np.roll(
                isrf_resp, monochrom_center_index - max_resp_idx
            )

        # Determine where to cut off centered ILS curves to reduce SVD time
        # truncate = True
        truncate = False
        if truncate:
            beg_idx = isrf_mat.shape[1]
            end_idx = 0
            for i_isrf in range(isrf_mat.shape[0]):
                where_not_zero = np.where(isrf_mat[i_isrf, :] != 0)
                if where_not_zero[0].size > 0:
                    beg_idx = min(beg_idx, where_not_zero[0][0])
                    end_idx = max(beg_idx, where_not_zero[0][-1])

            isrf_mat = isrf_mat[:, beg_idx:end_idx]

        else:
            beg_idx = 0
            end_idx = isrf_mat.shape[1]

        isrf_length = isrf_mat.shape[1]
        isrf_center_ind = monochrom_center_index - beg_idx

        # COMPUTE SVD, AND THEN RE-INTRODUCE NORMS TO U_ISRF (OR DO THIS LATER BELOW... AFTER REDUCING NUMBER OF SVS).
        isrf_mat_rev = isrf_mat[:, ::-1]
        [u_isrf, s_isrf, vh_isrf] = np.linalg.svd(isrf_mat_rev, full_matrices=False)

        # Only keep singular vectors up to sv_threshold percentage
        if i_sv_threshold > 1:
            logger.warning(
                "sv_threshold is meant to be in the range [0, 1]. Rounding to 1."
            )
            i_sv_threshold = 1
        elif i_sv_threshold < 0:
            logger.warning(
                "sv_threshold is meant to be in the range [0, 1]. Rounding to 0."
            )
            i_sv_threshold = 0

        num_svs = 0
        sum_svs = 0
        max_sum_svs = np.sum(s_isrf)
        while (sum_svs < (i_sv_threshold * max_sum_svs)) and (num_svs < len(s_isrf)):
            sum_svs = sum_svs + s_isrf[num_svs]
            num_svs = num_svs + 1
        u_isrf = u_isrf[:, 0:num_svs]
        s_isrf = s_isrf[0:num_svs]
        vh_isrf = vh_isrf[0:num_svs, :]
        svh_isrf = np.matmul(np.diag(s_isrf), vh_isrf)

        # Compute necessary FFTs of the svh_isrf rows
        fft_size = int(np.power(2, np.floor(np.log2(N_monochrom) + 2)))
        svh_isrf_fft = np.fft.fft(svh_isrf, n=fft_size)

        # Scale the u_isrf by the norms (if not already done above)
        scaled_u_isrf = np.matmul(np.diag(isrf_norms), u_isrf)
        scaled_uh_isrf = np.transpose(scaled_u_isrf)

        # Account for any offset due to reversing of ils_mat previously
        where_extract = (isrf_length - 1) - isrf_center_ind + center_wl_indices

        # Need to store: fft_size, svh_isrf_fft, scaled_uh_isrf, where_extract, central_wavelength(?), monochromgrid(?), ind_monochrom(?), sv_threshold(?)
        # Note: We will store central_wavelength and monochromgrid in case we need to interpolate anything.
        # But right now, it looks like our setup will be such that the central_wavelength's are the instrument grid, and
        # we will require no interpolation.

        o_ils = {
            "fft_size": fft_size,
            "svh_isrf_fft_real": np.real(
                svh_isrf_fft
            ),  # MT: Trying to avoid passing around complex-valued arrays.
            "svh_isrf_fft_imag": np.imag(svh_isrf_fft),
            #'svh_isrf_fft': svh_isrf_fft,
            "scaled_uh_isrf": scaled_uh_isrf,
            "where_extract": where_extract,
            "central_wavelength": central_wavelength,
            "monochromgrid": monochromgrid,
            "ind_monochrom": ind_monochrom,
            "sv_threshold": i_sv_threshold,
            # This tells RefractorFmObjectCreator and its children that it can use the
            # central_wavelength array from this dictionary as the monochromatic
            # wavelength grid for the RT. If that is not the case, set this to False.
            "central_wavelength_is_mono_grid": True,
        }

        return o_ils


class TropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.measurement_id: None | MeasurementId = None
        self.retrieval_config: None | RetrievalConfiguration = None

    def notify_update_target(self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("TROPOMI"):
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating forward model using using TropomiFmObjectCreator")
        obj_creator = TropomiFmObjectCreator(
            current_state,
            self.measurement_id,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Tropomi Forward model\n{fm}")
        return fm


# This is separated out so we can cache this
@cache
def _tropomi_ils(i_fn: Path, i_band: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # We catch this file at a higher level
    with InputFileMonitor.open_h5(i_fn, None) as f:
        FilterBand_group = f"band_{i_band}"
        if i_band < 7:
            wav = f[FilterBand_group]["wavelength"][...]
            deltawav = f[FilterBand_group]["delta_wavelength"][...]
            isrf = f[FilterBand_group]["isrf"][...]
        else:
            wav = f[FilterBand_group]["central_wavelength"][...]
            deltawav = f[FilterBand_group]["delta_wavelength"][...]
            isrf = f[FilterBand_group]["isrf"][...]
        return wav, deltawav, isrf


__all__ = ["TropomiFmObjectCreator", "TropomiForwardModelHandle"]
