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
)
from refractor.muses import fake_muses_py as mpy
import refractor.framework as rf  # type: ignore
import os
from pathlib import Path
from loguru import logger
import numpy as np
import h5py  # type: ignore

# for netCDF3 support which is not supported in h5py
from netCDF4 import Dataset
from typing import Callable, Any
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import (
        CurrentState,
        MeasurementId,
        MusesObservation,
        RefractorUip,
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
        observation: MusesObservation,
        use_eof: bool = False,
        eof_dir: None | str | os.PathLike[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            current_state,
            measurement_id,
            InstrumentIdentifier("OMI"),
            observation,
            **kwargs,
        )
        self.use_eof = use_eof
        self.eof_dir = Path(eof_dir) if eof_dir is not None else None

    def ils_params_preconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_tropomi, so we duplicate that here
        num_fwhm_srf = 4.0
        wn, sindex = self.observation.wn_and_sindex(sensor_index)
        return mpy.get_omi_ils(
            str(self.osp_dir),
            wn,
            sindex,
            self.observation.wavelength_filter,
            self.observation.observation_table,
            num_fwhm_srf,
        )

    def ils_params_postconv(self, sensor_index: int) -> dict[str, Any]:
        # make_uip_omi does the same thing for postconv as preconv
        return self.ils_params_preconv(sensor_index)

    def ils_params_fastconv(self, sensor_index: int) -> dict[str, Any]:
        # This is hardcoded in make_uip_omi, so we duplicate that here
        num_fwhm_srf = 4.0
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

        return mpy.get_omi_ils_fastconv(
            str(self.osp_dir),
            wn_list,
            sindex,
            wn_filter,
            self.observation.observation_table,
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
        return self.measurement_id["ils_omi_xsection"]

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
                with Dataset(eof_fname) as eof_ds:
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
                self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, r)
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
                self.current_state.add_fm_state_vector_if_needed(self.fm_sv, selem, r)

        return res

    @cached_property
    def omi_solar_model(self) -> rf.SolarModel:
        """We read a 3 year average solar file HDF file for omi. This
        duplicates what mpy.read_omi does, which is then stored in the pickle
        file that solar_model uses."""
        f = h5py.File(self.measurement_id["omiSolarReference"], "r")
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


class OmiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs: Any) -> None:
        self.creator_kwargs = creator_kwargs
        self.measurement_id: None | MeasurementId = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.measurement_id = measurement_id

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        **kwargs: Any,
    ) -> rf.ForwardModel:
        if instrument_name != InstrumentIdentifier("OMI"):
            return None
        if self.measurement_id is None:
            raise RuntimeError("Call notify_update_target first")
        logger.debug("Creating forward model using using OmiFmObjectCreator")
        obj_creator = OmiFmObjectCreator(
            current_state,
            self.measurement_id,
            obs,
            rf_uip_func=rf_uip_func,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"OMI Forward model\n{fm}")
        return fm


__all__ = ["OmiFmObjectCreator", "OmiForwardModelHandle"]
