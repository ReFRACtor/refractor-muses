# Don't both typechecking the file. This is old code, only used for backwards testing.
# Silence mypy, just so we don't get a lot of noise in the output
# type: ignore

from __future__ import annotations
import refractor.framework as rf  # type: ignore
from refractor.muses import RefractorUip
import numpy as np

# Older interface for observations, which wraps around muses-py code


class MusesObservationBaseOld(rf.ObservationSvImpBase):
    # Note the handling of include_bad_sample is important here. muses-py
    # expects to get all the samples in the forward model run in the routine
    # run_forward_model/fm_wrapper. I'm not sure what it does with the bad
    # data, but we need to have the ability to include it.
    # run_retrieval/residual_fm_jacobian on the other hand does the normal
    # filtering of bad samples. We handle this by toggling the behavior of
    # bad_sample_mask, either masking bad samples or having a empty mask that
    # lets everything pass through.
    def __init__(
        self,
        rf_uip: RefractorUip,
        instrument_name,
        obs_rad,
        meas_err,
        include_bad_sample=False,
        **kwargs,
    ):
        super().__init__([])
        self.rf_uip = rf_uip
        self.instrument_name = instrument_name
        self.obs_rad = obs_rad
        self.meas_err = meas_err
        self.include_bad_sample = include_bad_sample

    def _v_num_channels(self):
        return 1

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample:
            gmask[:] = True
        return rf.SpectralDomain(
            self.rf_uip.frequency_list(self.instrument_name)[gmask], rf.Unit("nm")
        )

    def bad_sample_mask(self, sensor_index):
        subset = [str(t) == self.instrument_name for t in self.rf_uip.instrument_list]
        uncer = self.meas_err[subset]
        bmask = np.array(uncer < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def radiance_all_with_bad_sample(self):
        return self.radiance(0, skip_jacobian=True, inc_bad_sample=True)

    def radiance(self, sensor_index, skip_jacobian=False, inc_bad_sample=False):
        if sensor_index != 0:
            raise ValueError("sensor_index must be 0")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample:
            gmask[:] = True
        sd = self.spectral_domain(sensor_index, inc_bad_sample)
        subset = [str(t) == self.instrument_name for t in self.rf_uip.instrument_list]
        r = self.obs_rad[subset][gmask]
        uncer = self.meas_err[subset][gmask]
        sr = rf.SpectralRange(r, rf.Unit("sr^-1"), uncer)
        if sr.data.shape != sd.data.shape:
            raise RuntimeError("sd and sr are different lengths")
        return rf.Spectrum(sd, sr)


class MusesCrisObservationOld(MusesObservationBaseOld):
    """Wrapper that just returns the passed in measured radiance
    and uncertainty for CRIS"""

    def __init__(self, rf_uip: RefractorUip, obs_rad, meas_err, **kwargs):
        super().__init__(rf_uip, "CRIS", obs_rad, meas_err, **kwargs)

    def desc(self):
        return "MusesCrisObservationOld"


class MusesAirsObservationOld(MusesObservationBaseOld):
    """Wrapper that just returns the passed in measured radiance
    and uncertainty for AIRS"""

    def __init__(self, rf_uip: RefractorUip, obs_rad, meas_err, **kwargs):
        super().__init__(rf_uip, "AIRS", obs_rad, meas_err, **kwargs)

    def desc(self):
        return "MusesAirsObservationOld"


__all__ = ["MusesCrisObservationOld", "MusesAirsObservationOld"]
