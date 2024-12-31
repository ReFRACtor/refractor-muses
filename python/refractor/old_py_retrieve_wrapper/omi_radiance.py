import refractor.framework as rf
import numpy as np
import pickle


class OmiRadiance(rf.ObservationSvImpBase):
    """This is a wrapper for the py-retrieve function get_omi_radiance
    to make it look like a Observation.

    We have handling for the portion of the jacobian that gets handled by
    get_omi_radiance.
    """

    def __init__(self, coeff, which_retrieved, include_bad_sample=False):
        super().__init__(coeff, rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
        self.include_bad_sample = include_bad_sample
        # Easy way for radiance_all_with_bad_sample to get radiance_all to
        # include bad samples
        self._force_include_bad_sample = False

    def state_vector_name_i(self, i):
        if i == 0:
            return "Radiance Shift UV1"
        if i == 1:
            return "Radiance Shift UV2"
        if i == 2:
            return "Solar Shift UV1"
        if i == 3:
            return "Solar Shift UV2"
        if i == 4:
            return "Solar Slope UV1"
        if i == 5:
            return "Solar Slope UV2"
        return f"State vector element {i}"

    def radiance_all_with_bad_sample(self):
        try:
            self._force_include_bad_sample = True
            return self.radiance_all(True)
        finally:
            self._force_include_bad_sample = False

    def desc(self):
        return "OmiRadiance"


class OmiRadiancePyRetrieve(OmiRadiance):
    """Version of OmiRadiance that just turns around and calls
    get_omi_radiance.
    """

    def __init__(self, rf_uip, include_bad_sample=False, skip_jac=False):
        """This is a OmiRadiance that calls get_omi_radiance.
        Note that if we are using a PyRetrieve forward model the jacobian
        is already included. To prevent double counting of this, we can
        skip the jacobian.
        """
        self.rf_uip = rf_uip
        self.skip_jac = skip_jac
        coeff = [
            self.rf_uip.omi_params["nradwav_uv1"],
            self.rf_uip.omi_params["nradwav_uv2"],
            self.rf_uip.omi_params["odwav_uv1"],
            self.rf_uip.omi_params["odwav_uv2"],
            self.rf_uip.omi_params["odwav_slope_uv1"],
            self.rf_uip.omi_params["odwav_slope_uv2"],
        ]
        which_retrieved = [
            "OMINRADWAVUV1" in self.rf_uip.state_vector_params("OMI"),
            "OMINRADWAVUV2" in self.rf_uip.state_vector_params("OMI"),
            "OMIODWAVUV1" in self.rf_uip.state_vector_params("OMI"),
            "OMIODWAVUV1" in self.rf_uip.state_vector_params("OMI"),
            "OMIODWAVSLOPEUV1" in self.rf_uip.state_vector_params("OMI"),
            "OMIODWAVSLOPEUV2" in self.rf_uip.state_vector_params("OMI"),
        ]
        self.mw = [
            slice(0, self.rf_uip.nfreq_mw(0, "OMI")),
            slice(self.rf_uip.nfreq_mw(0, "OMI"), None),
        ]
        self.include_bad_sample = include_bad_sample
        # Easy way for radiance_all_with_bad_sample to get radiance_all to
        # include bad samples
        self._force_include_bad_sample = False
        super().__init__(coeff, which_retrieved, include_bad_sample=include_bad_sample)

    def _v_num_channels(self):
        return len(self.mw)

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        if sensor_index < 0 or sensor_index >= len(self.mw):
            raise RuntimeError("sensor_index out of range")
        o_measured_radiance_omi = self.rf_uip.measured_radiance("OMI")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        sd = rf.SpectralDomain(
            o_measured_radiance_omi["wavelength"][self.mw[sensor_index]][gmask],
            rf.Unit("nm"),
        )
        return sd

    def bad_sample_mask_full(self, sensor_index):
        """bad_sample_mask is subsetted by the microwindow. This instead returns
        the full bad pixel mask for all the pixels, including pixels outside of
        the microwindow range."""
        o_measured_radiance_omi = self.rf_uip.measured_radiance(
            "OMI", sensor_index, full_freq=True
        )
        bmask = np.array(o_measured_radiance_omi["measured_nesr"] < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def bad_sample_mask(self, sensor_index):
        """Indicate data that is bad. Note we filter this out of the
        returned radiance"""
        if sensor_index < 0 or sensor_index >= len(self.mw):
            raise RuntimeError("sensor_index out of range")
        lmw = self.mw[sensor_index]
        o_measured_radiance_omi = self.rf_uip.measured_radiance("OMI")
        bmask = np.array(o_measured_radiance_omi["measured_nesr"][lmw] < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def radiance(self, sensor_index, skip_jacobian=False, inc_bad_sample=False):
        if sensor_index < 0 or sensor_index >= len(self.mw):
            raise RuntimeError("sensor_index out of range")
        lmw = self.mw[sensor_index]
        o_measured_radiance_omi = self.rf_uip.measured_radiance("OMI")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        sd = rf.SpectralDomain(
            o_measured_radiance_omi["wavelength"][lmw][gmask], rf.Unit("nm")
        )
        if self.skip_jac or self.mapped_state.is_constant:
            jac = np.empty(
                (len(o_measured_radiance_omi["measured_radiance_field"][lmw][gmask]), 0)
            )
        else:
            jac = (
                np.outer(
                    -o_measured_radiance_omi["normwav_jac"][lmw][gmask],
                    self.mapped_state.jacobian[0 + sensor_index, :],
                )
                + np.outer(
                    -o_measured_radiance_omi["odwav_jac"][lmw][gmask],
                    self.mapped_state.jacobian[2 + sensor_index, :],
                )
                + np.outer(
                    -o_measured_radiance_omi["odwav_slope_jac"][lmw][gmask],
                    self.mapped_state.jacobian[4 + sensor_index, :],
                )
            )
        sr = rf.SpectralRange(
            rf.ArrayAd_double_1(
                o_measured_radiance_omi["measured_radiance_field"][lmw][gmask], jac
            ),
            rf.Unit("sr^-1"),
            o_measured_radiance_omi["measured_nesr"][lmw][gmask],
        )
        return rf.Spectrum(sd, sr)


class OmiRadianceToUip(rf.CacheInvalidatedObserver):
    def __init__(self, rf_uip, omi_radiance):
        super().__init__()
        self.rf_uip = rf_uip
        self.omi_radiance = omi_radiance
        self.omi_radiance.add_cache_invalidated_observer(self)
        # Make sure data is synchronized initially
        self.invalidate_cache()

    def invalidate_cache(self):
        """Called with self.omi_radiance changes"""
        x = self.omi_radiance.mapped_state.value
        self.rf_uip.omi_params["nradwav_uv1"] = x[0]
        self.rf_uip.omi_params["nradwav_uv2"] = x[1]
        self.rf_uip.omi_params["odwav_uv1"] = x[2]
        self.rf_uip.omi_params["odwav_uv2"] = x[3]
        self.rf_uip.omi_params["odwav_slope_uv1"] = x[4]
        self.rf_uip.omi_params["odwav_slope_uv2"] = x[5]
        self.cache_valid_flag = True


__all__ = ["OmiRadiance", "OmiRadiancePyRetrieve", "OmiRadianceToUip"]
