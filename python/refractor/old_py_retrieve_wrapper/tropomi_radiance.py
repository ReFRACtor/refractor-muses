import refractor.framework as rf
import numpy as np
import pickle


# The "!= True" syntax is actually correct, although it looks wrong to
# ruff. Turn this rule off so we pass
# ruff: noqa: E712
class TropomiRadiance(rf.ObservationSvImpBase):
    """This is the base class for TROPOMI measured radiance.
    Right now we only support BAND3, we can extend this but for now
    we'll keep things simple.

    The radiance takes the parameters solarshift_BAND3, radianceshift_BAND3 and
    radsqueeze_BAND3."""

    def __init__(self, coeff, which_retrieved, include_bad_sample=False, band=3):
        super().__init__(coeff, rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
        self.include_bad_sample = include_bad_sample
        self._band = band
        # Easy way for radiance_all_with_bad_sample to get radiance_all to
        # include bad samples
        self._force_include_bad_sample = False

    def state_vector_name_i(self, i):
        if i == 0:
            return f"Solar shift Band {self._band}"
        if i == 1:
            return f"Radiance shift Band {self._band}"
        if i == 2:
            return f"Radiance squeeze Band {self._band}"
        return f"State vector element {i}"

    def radiance_all_with_bad_sample(self):
        try:
            self._force_include_bad_sample = True
            return self.radiance_all(True)
        finally:
            self._force_include_bad_sample = False

    def desc(self):
        return "TropomiRadiance"


class TropomiRadiancePyRetrieve(TropomiRadiance):
    """Version of TropomiRadiance that just turns around and calls
    get_tropomi_radiance.

    Note that the solar shift jacobian is actually *wrong*, but we duplicate
    what py-retrieve does here to get the mechanics in place for  testing."""

    def __init__(self, rf_uip, skip_jac=False, include_bad_sample=False):
        """This is a TropomiRadiance that calls get_tropomi_radiance.
        Note that if we are using a PyRetrieve forward model the jacobian
        is already included. To prevent double counting of this, we can
        skip the jacobian.
        """
        self.rf_uip = rf_uip
        self.skip_jac = skip_jac
        coeff = [
            self.rf_uip.tropomi_params["solarshift_BAND3"],
            self.rf_uip.tropomi_params["radianceshift_BAND3"],
            self.rf_uip.tropomi_params["radsqueeze_BAND3"],
        ]
        which_retrieved = [
            "TROPOMISOLARSHIFTBAND3" in self.rf_uip.state_vector_params("TROPOMI"),
            "TROPOMIRADIANCESHIFTBAND3" in self.rf_uip.state_vector_params("TROPOMI"),
            "TROPOMIRADSQUEEZEBAND3" in self.rf_uip.state_vector_params("TROPOMI"),
        ]
        super().__init__(coeff, which_retrieved, include_bad_sample=include_bad_sample)

    def _v_num_channels(self):
        return 1

    def bad_sample_mask(self, sensor_index):
        """Indicate data that is bad. Note we filter this out of the
        returned radiance"""
        o_measured_radiance_tropomi = self.rf_uip.measured_radiance("TROPOMI")
        bmask = np.array(o_measured_radiance_tropomi["measured_nesr"] < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        o_measured_radiance_tropomi = self.rf_uip.measured_radiance("TROPOMI")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        sd = rf.SpectralDomain(
            o_measured_radiance_tropomi["wavelength"][gmask], rf.Unit("nm")
        )
        return sd

    def radiance(self, sensor_index, skip_jacobian=False, inc_bad_sample=False):
        if sensor_index != 0:
            raise RuntimeError(f"Out of range sensor_index {sensor_index}")
        o_measured_radiance_tropomi = self.rf_uip.measured_radiance("TROPOMI")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        sd = rf.SpectralDomain(
            o_measured_radiance_tropomi["wavelength"][gmask], rf.Unit("nm")
        )
        if self.skip_jac or self.mapped_state.is_constant:
            jac = np.empty(
                (len(o_measured_radiance_tropomi["measured_radiance_field"][gmask]), 0)
            )
        else:
            mw = [
                slice(0, self.rf_uip.nfreq_mw(0, "TROPOMI")),
                slice(self.rf_uip.nfreq_mw(0, "TROPOMI"), None),
            ]
            jac = (
                np.outer(
                    o_measured_radiance_tropomi["normwav_jac"][mw[0]][gmask],
                    self.mapped_state.jacobian[0, :],
                )
                + np.outer(
                    o_measured_radiance_tropomi["odwav_jac"][mw[0]][gmask],
                    self.mapped_state.jacobian[1, :],
                )
                + np.outer(
                    o_measured_radiance_tropomi["odwav_slope_jac"][mw[0]][gmask],
                    self.mapped_state.jacobian[2, :],
                )
            )

        sr = rf.SpectralRange(
            rf.ArrayAd_double_1(
                o_measured_radiance_tropomi["measured_radiance_field"][gmask], jac
            ),
            rf.Unit("sr^-1"),
            o_measured_radiance_tropomi["measured_nesr"][gmask],
        )
        return rf.Spectrum(sd, sr)


class TropomiRadianceRefractor(TropomiRadiance):
    """Version of TropomiRadiance that we calculate. Jacobians should
    be correct.
    """

    def __init__(
        self,
        rf_uip,
        bands,
        fname="./Input/Radiance_TROPOMI_.pkl",
        include_bad_sample=False,
    ):
        self.rf_uip = rf_uip

        if len(bands) != 1:
            raise NotImplementedError("Cannot handle >1 band yet")
        band = bands[0]  # JLL: this should be something like "BAND3", "BAND7", etc.

        coeff = [
            self.rf_uip.tropomi_params[f"solarshift_{band}"],
            self.rf_uip.tropomi_params[f"radianceshift_{band}"],
            self.rf_uip.tropomi_params[f"radsqueeze_{band}"],
        ]
        which_retrieved = [
            f"TROPOMISOLARSHIFT{band}" in self.rf_uip.state_vector_params("TROPOMI"),
            f"TROPOMIRADIANCESHIFT{band}" in self.rf_uip.state_vector_params("TROPOMI"),
            f"TROPOMIRADSQUEEZE{band}" in self.rf_uip.state_vector_params("TROPOMI"),
        ]
        super().__init__(
            coeff, which_retrieved, include_bad_sample=include_bad_sample, band=band
        )
        # Can perhaps get this more directly, but current set up is
        # to read a pickle file.
        tropomi = pickle.load(open(fname, "rb"))
        original_wave = tropomi["Earth_Radiance"]["Wavelength"]
        solar_rad = tropomi["Solar_Radiance"]["AdjustedSolarRadiance"]
        # Right now, only work with one band. We can extend this later.
        # Index where we match the band, and have good data
        good_ind = (
            (tropomi["Earth_Radiance"]["EarthWavelength_Filter"] == band)
            & (tropomi["Earth_Radiance"]["EarthRadianceNESR"] > 0.0)
            & (solar_rad > 0.0)
        )
        # Bit klunky, but LinearInterpolateAutoDerivative is fairly low
        # level and has a clumsy python interface. We create a vector
        # of wavelength and solar radiance values, and use to create the
        # interpolator
        self.orgwav = rf.vector_auto_derivative()
        orgwav_good = rf.vector_auto_derivative()
        self.orgwav_mean = np.mean(original_wave[good_ind])
        y = rf.vector_auto_derivative()
        for v in original_wave:
            self.orgwav.append(rf.AutoDerivativeDouble(float(v)))
        for v in original_wave[good_ind]:
            orgwav_good.append(rf.AutoDerivativeDouble(float(v)))
        for v in solar_rad[good_ind]:
            y.append(rf.AutoDerivativeDouble(float(v)))
        self.sol_interp = rf.LinearInterpolateAutoDerivative(orgwav_good, y)
        self.earth_rad = tropomi["Earth_Radiance"]["CalibratedEarthRadiance"][
            tropomi["Earth_Radiance"]["EarthWavelength_Filter"] == band
        ]
        self.sd_data = original_wave[self.rf_uip.freq_index("TROPOMI")]
        self.nesr = tropomi["Earth_Radiance"]["EarthRadianceNESR"][
            tropomi["Earth_Radiance"]["EarthWavelength_Filter"] == band
        ]

    def _v_num_channels(self):
        return 1

    def bad_sample_mask_full(self, sensor_index):
        """bad_sample_mask is subsetted by the microwindow. This instead returns
        the full bad pixel mask for all the pixels, including pixels outside of
        the microwindow range."""
        bmask = np.array(self.nesr < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def bad_sample_mask(self, sensor_index):
        """Indicate data that is bad. Note we filter this out of the
        returned radiance"""
        bmask = np.array(self.nesr[self.rf_uip.freq_index("TROPOMI")] < 0)
        if self.include_bad_sample:
            bmask[:] = False
        return bmask

    def spectral_domain(self, sensor_index, inc_bad_sample=False):
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        return rf.SpectralDomain(self.sd_data[gmask], rf.Unit("nm"))

    def norm_radiance(self):
        """Calculate the normalized radiance. This is for good data only,
        self.radiance handles adding in bad_samples of -999 if requested."""
        x = rf.vector_auto_derivative()
        y = rf.vector_auto_derivative()
        sol_rad = self.solar_radiance()
        for i, j in enumerate(self.good_freq_index()):
            x.append(self.orgwav[j])
            y.append(rf.AutoDerivativeDouble(float(self.earth_rad[j])) / sol_rad[i])

        norm_rad_interp = rf.LinearInterpolateAutoDerivative(x, y)
        delta_wav = self.mapped_state[1]
        delta_wav_slope = self.mapped_state[2]
        res = rf.ArrayAd_double_1(
            len(self.good_freq_index()), self.mapped_state.number_variable
        )
        for i, j in enumerate(self.good_freq_index()):
            w = self.orgwav[j]
            wnew = w - (delta_wav + (w - self.orgwav_mean) * delta_wav_slope)
            res[i] = norm_rad_interp(wnew)
        return res

    def norm_rad_nesr(self):
        sol_rad = self.solar_radiance()
        return np.array(
            [
                self.nesr[j] / sol_rad[i].value
                for i, j in enumerate(self.good_freq_index())
            ]
        )

    def good_freq_index(self):
        return [
            int(i) for i in self.rf_uip.freq_index("TROPOMI") if (self.nesr[i] >= 0)
        ]

    def solar_radiance(self):
        """Use our interpolator to get the solar model at the
        shifted spectrum."""
        res = rf.vector_auto_derivative()
        delta_wav = self.mapped_state[0]
        for i in self.good_freq_index():
            wnew = self.orgwav[i] - delta_wav
            res.append(self.sol_interp(wnew))
        return res

    def radiance(self, sensor_index, skip_jacobian=False, inc_bad_sample=False):
        if sensor_index != 0:
            raise RuntimeError(f"Out of range sensor_index {sensor_index}")
        gmask = self.bad_sample_mask(sensor_index) != True
        if inc_bad_sample or self._force_include_bad_sample:
            gmask[:] = True
        # This duplicates the limit to snr set in tropomi
        nrad = self.norm_radiance()
        t = self.norm_rad_nesr()
        snr = nrad.value / t
        uplimit = 500.0
        tind = np.asarray(snr > uplimit)
        t[tind] = nrad.value[tind] / uplimit
        if inc_bad_sample or self._force_include_bad_sample or self.include_bad_sample:
            gmask2 = np.array(self.nesr[self.rf_uip.freq_index("TROPOMI")] >= 0)
            nrad_with_bad_data = rf.ArrayAd_double_1(
                gmask2.shape[0], nrad.number_variable
            )
            j = 0
            for i, g in enumerate(gmask2):
                if g:
                    nrad_with_bad_data[i] = nrad[j]
                    j += 1
                else:
                    nrad_with_bad_data[i] = rf.AutoDerivativeDouble(-999.0)
            t_with_bad_data = np.full(gmask2.shape, -999.0)
            t_with_bad_data[gmask2] = t
            sr = rf.SpectralRange(nrad_with_bad_data, rf.Unit("sr^-1"), t_with_bad_data)
        else:
            sr = rf.SpectralRange(nrad, rf.Unit("sr^-1"), t)
        sd = rf.SpectralDomain(self.sd_data[gmask], rf.Unit("nm"))
        return rf.Spectrum(sd, sr)


__all__ = ["TropomiRadiance", "TropomiRadiancePyRetrieve", "TropomiRadianceRefractor"]
