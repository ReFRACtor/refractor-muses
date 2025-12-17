from __future__ import annotations
from functools import cached_property, lru_cache
from refractor.muses import (
    RefractorFmObjectCreator,
    ForwardModelHandle,
    MusesRaman,
    SurfaceAlbedo,
    InstrumentIdentifier,
    FilterIdentifier,
    StateElementIdentifier,
    InputFileHelper,
)
import refractor.framework as rf  # type: ignore
import os
from functools import cache
from pathlib import Path
from loguru import logger
import numpy as np

from typing import Any
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        CurrentState,
        MeasurementId,
        RetrievalConfiguration,
        MusesObservation,
    )


class OmiSurfaceAlbedo(SurfaceAlbedo):
    def __init__(self, ground: rf.GroundWithCloudHandling, spec_index: int):
        self.ground = ground
        self.spec_index = spec_index

    def surface_albedo(self) -> float:
        # We just directly use the coefficients for the constant term. Could
        # do something more clever, but this is what py-retrieve does
        if self.ground.do_cloud:
            # TODO Reevaluate using a fixed value here
            # py-retrieve returns a hard coded value. Not sure why we don't
            # just use the cloud albedo, but for now match the old code
            # return self.ground_cloud.coefficient[0].value
            return 0.80
        else:
            return self.ground.ground_clear.albedo_coefficients(self.spec_index)[
                0
            ].value


class OmiFmObjectCreator(RefractorFmObjectCreator):
    def __init__(
        self,
        current_state: CurrentState,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        observation: MusesObservation,
        use_eof: bool = False,
        eof_dir: None | str | os.PathLike[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            current_state,
            measurement_id,
            retrieval_config,
            InstrumentIdentifier("OMI"),
            observation,
            **kwargs,
        )
        self.use_eof = use_eof
        self.eof_dir = Path(eof_dir) if eof_dir is not None else None

    def ils_params_preconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf = 4
        wn, sindex = self.observation.wn_and_sindex(sensor_index)
        return self.get_omi_ils(
            wn,
            sindex,
            self.observation.wavelength_filter,
            num_fwhm_srf,
        )

    def ils_params_postconv(self, sensor_index: int) -> dict[str, Any]:
        # make_uip_omi does the same thing for postconv as preconv
        return self.ils_params_preconv(sensor_index)

    def ils_params_fastconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_omi, so we duplicate that here
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

        return self.get_omi_ils_fastconv(
            wn_list,
            sindex,
            wn_filter,
            num_fwhm_srf,
            i_monochromfreq=wn_list[sindex2],
            i_interpmethod="INTERP_MONOCHROM",
        )

    def ils_params(self, sensor_index: int) -> dict[str, Any]:
        """ILS parameters"""
        # TODO Pull out of rf_uip. This is in make_uip_tropomi.py
        # Note that this seems to fold in determine the high resolution grid.
        # We have a separate class MusesSpectrumSampling for doing that, which
        # currently just returns what ils_params has. When we clean this up, we
        # may want to put part of the functionality there - e.g., read the whole
        # ILS table here and then have the calculation of the spectrum in
        # MusesSpectrumSampling
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
        return self.retrieval_config["ils_omi_xsection"]

    @cached_property
    def instrument_correction(self) -> list[list[rf.InstrumentCorrection]]:
        res = []
        for i in range(self.num_channels):
            v: list[rf.InstrumentCorrection] = []
            if self.use_eof:
                if self.observation.filter_list[i] in self.eof:
                    for e in self.eof[self.observation.filter_list[i]]:
                        v.append(e)
            res.append(v)
        return res

    @cached_property
    def eof(self) -> dict[FilterIdentifier, list[rf.EmpiricalOrthogonalFunction]]:
        res: dict[FilterIdentifier, list[rf.EmpiricalOrthogonalFunction]] = {}
        for i in range(self.num_channels):
            filter_name = self.observation.filter_list[i]
            if str(filter_name) not in ("UV1", "UV2"):
                continue
            selem = [
                StateElementIdentifier(f"OMIEOF{filter_name}"),
            ]
            coeff, mp = self.current_state.object_state(selem)
            r = []
            # This is the full instrument size of the given filter_name. We only run
            # the forward model on a subset of these, but the eof is in terms of
            # the full instrument size. Note the values outside of what we actually
            # run the forward model on don't really matter - we don't end up using
            # them. We can set these to zero if desired.
            npixel = len(self.observation.frequency_full(i))
            # Note: npixel is 577 for UV2, which doesn't make sense given indexes in EOF
            # Muses uses fullbandfrequency subset by microwindow info (shown below) instead

            if self.eof_dir is not None:
                uv1_index, uv2_index = self.observation.across_track
                uv_basename = "EOF_xtrack_{0}-{1:02d}_window_{0}.nc"
                uv1_fname = self.eof_dir / uv_basename.format("uv1", uv1_index)
                uv2_fname = self.eof_dir / uv_basename.format("uv2", uv2_index)

                if filter_name == FilterIdentifier("UV1"):
                    eof_fname = uv1_fname
                elif filter_name == FilterIdentifier("UV2"):
                    eof_fname = uv2_fname

                # Note: We hit this code when retrieving cloud fraction which uses 7 freq. and 1 microwin
                if len(self.observation.filter_data) <= 1:
                    res[FilterIdentifier("UV2")] = []
                    continue

                eof_path = "/eign_vector"
                eof_index_path = "/Index"
                with InputFileHelper.open_ncdf(eof_fname, self.ifile_hlp) as eof_ds:
                    eofs = eof_ds[eof_path][:]
                    pixel_indexes = eof_ds[eof_index_path][:]

                # Sample index is 1 based (by convention),
                # so subtract 1 to get index into array. Include bad pixels here
                with self.observation.modify_spectral_window(include_bad_sample=True):
                    nonzero_eof_index = (
                        self.observation.spectral_domain(i).sample_index - 1
                    )
                for basis_index in range(len(coeff)):
                    eof_full = np.zeros((npixel,))
                    eof_channel = np.zeros((len(nonzero_eof_index),))
                    eof_channel[pixel_indexes] = eofs[basis_index, :]
                    eof_full[nonzero_eof_index] = eof_channel

                    # Not sure about the units, but this is what we assigned to our
                    # forward model, and the EOF needs to match
                    wform = rf.ArrayWithUnit(eof_full, rf.Unit("sr^-1"))
                    r.append(
                        rf.EmpiricalOrthogonalFunction(
                            coeff[basis_index], wform, basis_index + 1, filter_name
                        )
                    )
                res[filter_name] = r
                self.current_state.add_fm_state_vector_if_needed(
                    self.fm_sv,
                    selem,
                    r,
                )
            else:
                # If we don't have EOF data, use all zeros. This should go
                # away, this is just so we can start using the EOF before
                # we have all the data sorted out
                wform = rf.ArrayWithUnit(np.zeros(npixel), rf.Unit("sr^-1"))
                for i in range(len(coeff)):
                    r.append(
                        rf.EmpiricalOrthogonalFunction(
                            coeff[i], wform, i + 1, str(filter_name)
                        )
                    )
                res[filter_name] = r
                self.current_state.add_fm_state_vector_if_needed(
                    self.fm_sv,
                    selem,
                    r,
                )

        return res

    @cached_property
    def omi_solar_model(self) -> rf.SolarModel:
        """We read a 3 year average solar file HDF file for omi. This
        duplicates what mpy.read_omi does, which is then stored in the pickle
        file that solar_model uses."""
        f = InputFileHelper.open_h5(
            self.retrieval_config["omiSolarReference"], self.ifile_hlp
        )
        res = []
        for i in range(self.num_channels):
            ind = self.observation.across_track[i]
            wav_vals = f[f"WAV_{self.filter_list[i]}"][:, ind]
            irad_vals = f[f"SOL_{self.filter_list[i]}"][:, ind]
            one_au = 149597870691
            irad_vals *= (one_au / self.observation.earth_sun_distance) ** 2
            # File does not have units contained within it
            # Same units as the OMI L1B files, but use irradiance units
            sol_domain = rf.SpectralDomain(wav_vals, rf.Unit("nm"))
            sol_range = rf.SpectralRange(irad_vals, rf.Unit("ph / nm / s"))
            sol_spec = rf.Spectrum(sol_domain, sol_range)
            ref_spec = rf.SolarReferenceSpectrum(sol_spec, None)
            res.append(ref_spec)
        return res

    def instrument_hwhm(self, sensor_index: int) -> rf.DoubleWithUnit:
        filter_name = self.observation.filter_list[sensor_index]
        raise NotImplementedError(f"HWHM for band {filter_name} not defined")

    @cached_property
    def ground_clear(self) -> rf.Ground:
        albedo = np.zeros((self.num_channels, 3))
        which_retrieved = np.full((self.num_channels, 3), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)
        selem_all = []
        for i in range(self.num_channels):
            if self.filter_list[i] == FilterIdentifier("UV1"):
                band_reference[i] = (315 + 262) / 2.0
                selem = [
                    StateElementIdentifier("OMISURFACEALBEDOUV1"),
                ]
                coeff, mp = self.current_state.object_state(selem)
                albedo[i, 0:1] = coeff
                which_retrieved[i, mp.retrieval_indexes] = True
                selem_all.extend(selem)
            elif self.filter_list[i] == FilterIdentifier("UV2"):
                # Note this value is hardcoded in print_omi_surface_albedo
                band_reference[i] = 320.0
                selem = [
                    StateElementIdentifier("OMISURFACEALBEDOUV2"),
                    StateElementIdentifier("OMISURFACEALBEDOSLOPEUV2"),
                ]
                coeff, mp = self.current_state.object_state(selem)
                albedo[i, 0:2] = coeff
                which_retrieved[i, mp.retrieval_indexes] = True
                selem_all.extend(selem)
            else:
                raise RuntimeError("Don't recognize filter name")
        res = rf.GroundLambertian(
            albedo,
            rf.ArrayWithUnit(band_reference, "nm"),
            rf.Unit("nm"),
            [str(i) for i in self.filter_list],
            rf.StateMappingAtIndexes(np.ravel(which_retrieved)),
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem_all,
            [
                res,
            ],
        )
        return res

    @cached_property
    def ground_cloud(self) -> rf.Ground:
        albedo = np.zeros((self.num_channels, 1))
        which_retrieved = np.full((self.num_channels, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channels)
        band_reference[:] = 1000

        # This is hardcoded in py-retrieve (see script_retrieval_setup_ms.py),
        # unlike for tropomi.
        albedo[:, 0] = 0.8

        return rf.GroundLambertian(
            albedo,
            rf.ArrayWithUnit(band_reference, "nm"),
            [
                "Cloud",
            ]
            * self.num_channels,
            rf.StateMappingAtIndexes(np.ravel(which_retrieved)),
        )

    @cached_property
    def cloud_fraction(self) -> float:
        selem = [
            StateElementIdentifier("OMICLOUDFRACTION"),
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
        # This is hardcoded in py-retrieve (see script_retrieval_setup_ms.py),
        # unlike for tropomi.
        scale_factor = 1.9
        with self.observation.modify_spectral_window(do_raman_ext=True):
            wlen = self.observation.spectral_domain(i)
        # This is short if we aren't actually running this filter
        if wlen.data.shape[0] < 2:
            return None
        salbedo = OmiSurfaceAlbedo(self.ground, i)
        return MusesRaman(
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

    @lru_cache(maxsize=None)
    def raman_effect_refractor(self, i: int) -> rf.RamanSiorisEffect:
        scale_factor = 1.9
        with self.observation.modify_spectral_window(do_raman_ext=True):
            wlen = self.observation.spectral_domain(i)
        # This is short if we aren't actually running this filter
        if wlen.data.shape[0] < 2:
            return None
        return rf.RamanSiorisEffect(
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

    def get_omi_ils(
        self,
        i_Frequency: np.ndarray,
        i_freqIndex: np.ndarray,
        i_WAVELENGTH_FILTER: list[str],
        i_num_fwhm_srf: int,
    ) -> dict[str, Any]:
        # ILS files
        ils_dir = self.osp_dir / "OMI/OMI_ILS/"

        # number of data points
        num_points = len(i_freqIndex)

        # get number of data points for each AIRS SRF profile
        NP_SINGLE_SIDE = np.array([-9999])
        SUM = [np.float64(-999.0)]
        X0 = [np.float64(-999.0)]
        X0_ind = np.array([-999])
        v1_mono = [np.float64(-999.0)]
        v2_mono = [np.float64(-999.0)]

        for tempi in range(0, num_points):
            # Get OMI ILS filename
            FilterBand = i_WAVELENGTH_FILTER[i_freqIndex[tempi]]
            where_filter_occured = np.where(
                np.asarray(self.observation.observation_table["Filter_Band_Name"])
                == FilterBand
            )[0]
            OMIMODE = np.asarray(self.observation.observation_table["MeasurementMode"])[
                where_filter_occured
            ]
            PixelIndex = np.asarray(self.observation.observation_table["XTRACK"])[
                where_filter_occured
            ]

            tempfreq = i_Frequency[i_freqIndex[tempi]]

            formatted_pixel_index = str("{:02d}".format(PixelIndex[0]))
            fn = (
                ils_dir
                / OMIMODE[0]
                / FilterBand
                / f"OMI_ILS_{OMIMODE[0]}_{FilterBand}_{formatted_pixel_index}.h5"
            )

            if not fn.exists():
                raise RuntimeError(f"File not found: {fn}")

            # get freq index
            self.ifile_hlp.notify_file_input(fn)
            temp_XCF0 = _omi_ils_read_variable(fn, "XCF0")
            tempind = np.where(
                abs(temp_XCF0 - tempfreq) == np.amin(abs(temp_XCF0 - tempfreq))
            )[0]
            temp_XCF0 = temp_XCF0[tempind]
            X0 = np.concatenate((X0, temp_XCF0), axis=0)

            temp_ILS_SUM = _omi_ils_read_variable(fn, "ILS_SUM")
            temp_ILS_SUM = temp_ILS_SUM[tempind]
            SUM = np.concatenate((SUM, temp_ILS_SUM), axis=0)
            X0_ind = np.concatenate((X0_ind, tempind), axis=0)

            temp_V1_MONO = _omi_ils_read_variable(fn, "V1_MONO")
            temp_V1_MONO = temp_V1_MONO[tempind]
            v1_mono = np.concatenate((v1_mono, temp_V1_MONO), axis=0)

            temp_V2_MONO = _omi_ils_read_variable(fn, "V2_MONO")
            temp_V2_MONO = temp_V2_MONO[tempind]
            v2_mono = np.concatenate((v2_mono, temp_V2_MONO), axis=0)

            temp_NP_SINGLE_SIDE = _omi_ils_read_variable(fn, "NP_SINGLE_SIDE")
            NP_SINGLE_SIDE = np.concatenate(
                (NP_SINGLE_SIDE, temp_NP_SINGLE_SIDE), axis=0
            )
            # end for tempi in range(0, num_points):

        X0 = X0[1 : num_points + 1]
        X0_ind = X0_ind[1 : num_points + 1]
        SUM = SUM[1 : num_points + 1]
        v1_mono = v1_mono[1 : num_points + 1]
        v2_mono = v2_mono[1 : num_points + 1]
        NP_SINGLE_SIDE = NP_SINGLE_SIDE[1 : num_points + 1]

        # get OMI ILS profile
        maximum_ils_np = np.amax(np.asarray([505, 337]))

        ilsval = np.ndarray(shape=(maximum_ils_np, num_points), dtype=np.float64)
        ilsval.fill(-999.0)

        ilsfreq = np.ndarray(shape=(maximum_ils_np, num_points), dtype=np.float64)
        ilsfreq.fill(0.0)

        ilsnp = np.ndarray(shape=(num_points), dtype=np.int64)

        mononfreq_spacing = np.ndarray(shape=(num_points), dtype=np.float64)
        mononfreq_spacing.fill(0)

        for tempi in range(0, num_points):
            FilterBand = i_WAVELENGTH_FILTER[i_freqIndex[tempi]]
            where_filter_occured = np.where(
                np.asarray(self.observation.observation_table["Filter_Band_Name"])
                == FilterBand
            )[0]
            OMIMODE = np.asarray(self.observation.observation_table["MeasurementMode"])[
                where_filter_occured
            ]
            # IDL: observationtable.XTRACK[where(observationtable.FILTER_BAND_NAME eq FilterBand)]
            PixelIndex = np.asarray(self.observation.observation_table["XTRACK"])[
                where_filter_occured
            ]

            # We have to loop len(PixelIndex) times to look for each file name and append the name to fn_all
            # Python works differently than IDL when it comes to building a string if some substrings contains more than one elements.
            fn_all = []
            for io in range(0, len(OMIMODE)):
                for ip in range(0, len(PixelIndex)):
                    # Note that we have to use [ii] to index into OMIMODE and PixelIndex since we are going through each element of OMIMODE and PixelIndex
                    formatted_pixel_index = str("{:02d}".format(PixelIndex[ip]))
                    temp_fn = (
                        ils_dir
                        / OMIMODE[io]
                        / FilterBand
                        / f"OMI_ILS_{OMIMODE[io]}_{FilterBand}_{formatted_pixel_index}.h5"
                    )

                    # If found at least one name, save it to fn_all array.
                    if temp_fn.exists():
                        fn_all.append(temp_fn)

            # eliminate duplicates
            fn_all = list(dict.fromkeys(fn_all))

            tempcnt = len(fn_all)
            if tempcnt == 0:
                logger.error("File not found:", temp_fn)
                assert False

            ind1 = X0_ind[tempi]

            # Process if only found 1 file.
            if tempcnt == 1:
                temp = _omi_ils_read_variable(
                    fn_all[0], "ILS_MONO"
                )  # Read just one file.
                temp = temp[:, ind1]
                ilsnp[tempi] = len(temp)
                ilsval[0 : ilsnp[tempi], tempi] = temp

                temp = _omi_ils_read_variable(fn_all[0], "FREQ_MONO")
                temp = temp[:, ind1]
                ilsfreq[0 : ilsnp[tempi], tempi] = temp
                # end if tempcnt == 1:

            # Process if found more than 1 file.
            if tempcnt > 1:
                for Ipix in range(0, tempcnt):
                    temp = _omi_ils_read_variable(fn_all[Ipix], "ILS_MONO")
                    temp = temp[:, ind1]

                    ilsnp[tempi] = len(temp)

                    if Ipix == 0:
                        ilsval[0 : ilsnp[tempi], tempi] = temp

                    if Ipix != 0:
                        ilsval[0 : ilsnp[tempi], tempi] = (
                            ilsval[0 : ilsnp[tempi], tempi] + temp
                        )
                        # end for Ipix in range(0,tempcnt):

                ilsval[0 : ilsnp[tempi], tempi] = (
                    ilsval[0 : ilsnp[tempi], tempi] / tempcnt
                )

                temp = _omi_ils_read_variable(fn_all[0], "FREQ_MONO")
                temp = temp[:, ind1]

                ilsfreq[0 : ilsnp[tempi], tempi] = temp
                # end if tempcnt > 1:

            mononfreq_spacing[tempi] = ilsfreq[1, tempi] - ilsfreq[0, tempi]
            # end for tempi in range(0, num_points):

        o_ils = {
            "X0": X0,
            "X0_fm": i_Frequency[i_freqIndex],
            "freqIndex_fm": i_freqIndex,
            "ilsnp": ilsnp,
            "ilsfreq": ilsfreq,
            "ilsval": ilsval,
            "sum": SUM,
            "NP_SINGLE_SIDE": NP_SINGLE_SIDE,
            "v1_mono": v1_mono,
            "v2_mono": v2_mono,
            "monofreq_spacing": mononfreq_spacing,
            "num_fwhm_srf": i_num_fwhm_srf,
        }
        return o_ils

    def get_omi_ils_fastconv(
        self,
        i_Frequency: np.ndarray,
        i_freqIndex: np.ndarray,
        i_WAVELENGTH_FILTER: list[str],
        i_num_fwhm_srf: int,
        i_monochromfreq: np.ndarray,
        i_interpmethod: str = "NO_INTERP",
        i_sv_threshold: float = 0.99,
    ) -> dict[str, Any]:
        ilsInfo = self.get_omi_ils(
            i_Frequency, i_freqIndex, i_WAVELENGTH_FILTER, i_num_fwhm_srf
        )

        central_wavelength = ilsInfo["X0"][:]
        central_wavelength = central_wavelength[np.where(central_wavelength > -999)]
        N_conv_wl = len(central_wavelength)

        # The current ILS convolution interpolates the monochromatic radiance to the ILS wavelength grid,
        # then performs the convolution. We will alter each ILS curve to avoid having to interpolate

        # MONOCHROMATIC FREQUENCY SHOULD BE GIVEN BY i_uip['fullbandfrequency'][i_uip['freqIndex_fm']] in apply_omi_isrf_fastconv.py

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
            # isrf_normalized = np.zeros(ilsInfo['ilsval'].shape) #+ -999  #DOUBLE CHECK: MIGHT NEED NP_SINGLE_SIDE FROM APPLY_OMI_ILS
            isrf_normalized = np.zeros(
                (
                    ilsInfo["ilsval"].shape[1],
                    ilsInfo["ilsval"].shape[0],
                )
            )  # + -999
            isrf_rescaled = np.zeros((N_conv_wl, N_monochrom))  # + -999

            for i_isrf in range(N_conv_wl):
                # x0 = central_wavelength[i_isrf] #i_omifreq[i_omifreqIndex[tempj]] #MT: Double check this
                NP_SINGLE_SIDE = ilsInfo["NP_SINGLE_SIDE"][i_isrf]

                # Replacing xcf with isrf_wavelength
                temp_delta_wavelength = np.arange(
                    0, NP_SINGLE_SIDE * 2 + 1
                ) * np.float64(0.01)
                temp_delta_wavelength = temp_delta_wavelength - np.mean(
                    temp_delta_wavelength
                )
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                NP_DOUBLE_SIDE = len(isrf_wavelength)  # = NP_SINGLE_SIDE * 2 + 1

                temp_isrf = ilsInfo["ilsval"][
                    :, i_isrf
                ]  # MT: MIGHT HAVE TO DIVIDE BY SUM
                temp_isrf = temp_isrf[0 : NP_SINGLE_SIDE * 2 + 1].flatten()

                # isrf_normalized[i_isrf, :NP_DOUBLE_SIDE] = temp_isrf[:]
                isrf_normalized[i_isrf, :NP_DOUBLE_SIDE] = temp_isrf[:] / np.sum(
                    temp_isrf[:]
                )  # MT: DIVIDING BY SUM TO BE SAFE

                TF_1 = isrf_wavelength >= monochromgrid[0]

                for i_monochrom in range(N_monochrom - 1):
                    wl_0 = monochromgrid[i_monochrom]
                    wl_1 = monochromgrid[i_monochrom + 1]

                    TF_0 = TF_1
                    TF_1 = isrf_wavelength >= wl_1

                    temp_isrf[:] = 0
                    temp_isrf[TF_0] = (
                        isrf_normalized[i_isrf, :NP_DOUBLE_SIDE][TF_0]
                    ) / (wl_1 - wl_0)
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
                    np.multiply(temp_isrf, isrf_normalized[i_isrf, :NP_DOUBLE_SIDE])
                )

                del temp_isrf

        elif i_interpmethod == "INTERP_ILS":  # Interpolate ILS to monochromatic grid
            # Normalize and rescale ILS curves:
            isrf_normalized = np.zeros((N_conv_wl, N_monochrom))  # + -999
            isrf_rescaled = np.zeros((N_conv_wl, N_monochrom))  # + -999
            # temp_isrf = np.zeros(ilsInfo['isrf'].shape[1],)

            for i_isrf in range(N_conv_wl):
                NP_SINGLE_SIDE = ilsInfo["NP_SINGLE_SIDE"][i_isrf]

                # Replacing xcf with isrf_wavelength
                temp_delta_wavelength = np.arange(
                    0, NP_SINGLE_SIDE * 2 + 1
                ) * np.float64(0.01)
                temp_delta_wavelength = temp_delta_wavelength - np.mean(
                    temp_delta_wavelength
                )
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                NP_DOUBLE_SIDE = len(isrf_wavelength)  # = NP_SINGLE_SIDE * 2 + 1

                # MT: WE WILL DIVIDE BY THE SUM OF THE ILS VALUES IN CASE NOT ALREADY DONE IN PRE-PROCESSING.
                temp_isrf_0 = ilsInfo["ilsval"][
                    : NP_SINGLE_SIDE * 2 + 1, i_isrf
                ].flatten()
                temp_isrf_0 = temp_isrf_0 / np.sum(temp_isrf_0)

                temp_isrf_1 = np.interp(
                    monochromgrid, isrf_wavelength, temp_delta_wavelength, temp_isrf_0
                )

                temp_isrf_2 = temp_isrf_1

                isrf_normalized[i_isrf, :] = temp_isrf_2[:]

                for i_monochrom in range(1, N_monochrom - 1):
                    isrf_rescaled[i_isrf, i_monochrom] = (
                        0.5
                        * (
                            monochromgrid[i_monochrom + 1]
                            - monochromgrid[i_monochrom - 1]
                        )
                        * isrf_normalized[i_isrf, i_monochrom]
                    )

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
                NP_SINGLE_SIDE = ilsInfo["NP_SINGLE_SIDE"][i_isrf]

                # Replacing xcf with isrf_wavelength
                temp_delta_wavelength = np.arange(
                    0, NP_SINGLE_SIDE * 2 + 1
                ) * np.float64(0.01)
                temp_delta_wavelength = temp_delta_wavelength - np.mean(
                    temp_delta_wavelength
                )
                isrf_wavelength = central_wavelength[i_isrf] + temp_delta_wavelength

                NP_DOUBLE_SIDE = len(isrf_wavelength)  # = NP_SINGLE_SIDE * 2 + 1

                # MT: WE WILL DIVIDE BY THE SUM OF THE ILS VALUES IN CASE NOT ALREADY DONE IN PRE-PROCESSING.
                temp_isrf = ilsInfo["ilsval"][
                    : NP_SINGLE_SIDE * 2 + 1, i_isrf
                ].flatten()
                temp_isrf = temp_isrf / np.sum(temp_isrf)

                isrf_normalized[i_isrf, :NP_DOUBLE_SIDE] = temp_isrf[:]
                isrf_rescaled[i_isrf, :NP_DOUBLE_SIDE] = temp_isrf[:]

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
            "v1_mono": ilsInfo["v1_mono"],
            "v2_mono": ilsInfo["v2_mono"],
            "NP_SINGLE_SIDE": ilsInfo["NP_SINGLE_SIDE"],
            "ilsval": ilsInfo["ilsval"],
        }

        return o_ils


class OmiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.measurement_id: None | MeasurementId = None
        self.retrieval_config: None | RetrievalConfiguration = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
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
        if instrument_name != InstrumentIdentifier("OMI"):
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating forward model using using OmiFmObjectCreator")
        obj_creator = OmiFmObjectCreator(
            current_state,
            self.measurement_id,
            self.retrieval_config,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"OMI Forward model\n{fm}")
        return fm


@cache
def _omi_ils_read_variable(i_filename: Path, i_variable_name: str) -> np.ndarray:
    # We catch this file at a higher level
    with InputFileHelper.open_h5(i_filename, None) as f:
        return f[i_variable_name][:]


__all__ = ["OmiFmObjectCreator", "OmiForwardModelHandle"]
