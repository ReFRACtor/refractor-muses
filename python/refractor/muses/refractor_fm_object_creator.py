from functools import cached_property, lru_cache
from .muses_optical_depth_file import MusesOpticalDepthFile
from .muses_altitude import MusesAltitude
from .muses_spectrum_sampling import MusesSpectrumSampling
from .muses_raman import MusesRaman
from .refractor_uip import RefractorUip
from .muses_forward_model import RefractorForwardModel
from .muses_ray_info import MusesRayInfo
import refractor.framework as rf
import os
from pathlib import Path
import logging
import numpy as np
import glob
import abc
import copy

from typing import Sequence

logger = logging.getLogger("py-retrieve")

class RefractorFmObjectCreator(object, metaclass=abc.ABCMeta):
    '''There are a lot of interrelated object needed to be created to
    make a ForwardModel.

    This class provides a framework for a lot of the common pieces. It
    is intended that derived classes override certain functionality (e.g.,
    have a different Pressure object). The changes then ripple through
    the creation - so anything that needs a Pressure object to create itself
    will use whatever the derived class supplies.

    Note that this class is a convenience - for things like CostFunctionCreator
    we just need *a* ForwardModel. There is no requirement that you use this
    particular class to create the ForwardModel. But because most of the
    time the steps are pretty similar this can be a useful class to start with.

    Take a look at TropomiFmObjectCreator and OmiObjectCreator for examples
    of modifying things.

    A note for developers, the various @cached_property decorators are 
    important, this isn't just for performance (which is pretty minor
    for most of the objects). It is important that if two different
    piece of code access for example the pressure object, it gets the
    *same* pressure object, not just two objects that have the same 
    pressure levels. Because these objects get updated by for example the 
    rf.StateVector, we need to have only one instance of them.
    '''

    def __init__(self, current_state : 'CurrentState',
                 measurement_id : 'MeasurementId',
                 instrument_name: str, observation : 'MusesObservation',
                 # Temp, we are moving away from this
                 rf_uip : "Optional(RefractorUip)" = None,
                 fm_sv : "Optional(rf.StateVector)" = None,
                 # Values, so we can flip between using pca and not
                 use_pca=True, use_lrad=False, lrad_second_order=False,
                 use_raman=True,
                 skip_observation_add=False
                 ):
        '''Constructor. The StateVector to add things to can be passed in, or if this
        isn't then we create a new StateVector.
        '''
        self.use_pca = use_pca
        self.use_lrad = use_lrad
        self.use_raman = use_raman
        self.instrument_name = instrument_name
        self.lrad_second_order = lrad_second_order
        self.observation = observation
        # TODO This is needed by MusesOpticalDepthFile. It would be nice to
        # move away from needing this. This is uses in make_uip_tropomi to
        # write out the file, which we then read
        self.step_directory = current_state.step_directory
        if(fm_sv):
            self.fm_sv = fm_sv
        else:
            self.fm_sv = rf.StateVector()
            self.fm_sv.observer_claimed_size = current_state.fm_state_vector_size
        self.current_state = current_state
        # Note MeasurementId also has access to all the stuff in RetrievalConfiguration
        self.measurement_id = measurement_id

        # Depending on when the StateVector is created, the observation may or may
        # not have been added. Note it is safe to add this multiple times, so we
        # don't need to worry if it is already there. I don't think this will ever
        # cause a problem, but we have a option to skip this if needed.
        if(not skip_observation_add):
            coeff,mp = self.current_state.object_state(self.observation.state_element_name_list())
            self.observation.update_coeff_and_mapping(coeff,mp)
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv, self.observation.state_element_name_list(), [self.observation,])
        
        # We are moving away from this, but leave now as a reference
        self.rf_uip = rf_uip

        self.filter_list = self.observation.filter_list
        self.num_channels = self.observation.num_channels

        self.sza = self.observation.solar_zenith
        self.oza = self.observation.observation_zenith
        self.raz = self.observation.relative_azimuth

        # This is what OMI currently uses, and TROPOMI band 3. We may put all this
        # together, but right now tropomi_fm_object_creator may replace this.
        self._inner_absorber = O3Absorber(self)

        
    def solar_model(self, sensor_index):
        with self.observation.modify_spectral_window(do_raman_ext=True):
            sol_rad = self.observation.solar_spectrum(sensor_index)
        return rf.SolarReferenceSpectrum(sol_rad, None)

    @cached_property
    def ray_info(self):
        '''Return MusesRayInfo.'''
        return MusesRayInfo(self.rf_uip, self.instrument_name, self.pressure)

    @property
    def spec_win(self):
        return self.observation.spectral_window

    @cached_property
    def spectrum_sampling(self):
        # TODO
        # Note MusesSpectrumSampling doesn't have the logic in place to skip
        # highres wavelengths that we don't need - e.g., because of bad pixels.
        # For now, just follow the logic that py-retrieve uses, but we may
        # want to change this.
        # See comment in ils_params
        hres_spec = []
        for i in range(self.num_channels):
            if(self.ils_method(i) in ("FASTCONV", "POSTCONV")):
                cwave = self.ils_params(i)["central_wavelength"]
                hres_spec.append(rf.SpectralDomain(cwave, rf.Unit("nm")))
            else:
                hres_spec.append(None)
        return MusesSpectrumSampling(hres_spec)

    @cached_property
    def dispersion(self):
        # return creator.instrument.SampleGridSpectralDomain.create_direct(
        #    spectral_domains = self.rf_uip.sample_grid,
        #    desc_band_name = self.channel_names,
        #    num_channels = self.num_channels,
        #    spec_win = self.spec_win)
        # Not currently used
        return None

    def ils_params(self, sensor_index : int):
        '''ILS parameters'''
        raise NotImplementedError
    
    def ils_method(self, sensor_index : int) -> str:
        '''Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV'''
        raise NotImplementedError
    
    @cached_property
    def ils_function(self):
        # This is the "real" ILS, but py-retrieve applies this to
        # the input data is some way. We don't actually want to apply
        # this is the forward model
        # return OmiIlsTable.create_direct(
        #    ils_path = self.input_dir + "ils/normal",
        #    hdf_band_name = self.channel_names,
        #    desc_band_name = self.channel_names,
        #    dispersion = self.dispersion,
        #    num_channels = self.num_channels,
        #    across_track_indexes = self.across_track_indexes)
        return None

    @cached_property
    def instrument_correction(self):
        res = []
        for i in range(self.num_channels):
            v = []
            res.append(v)
        return res

    @cached_property
    def ils_half_width(self):
        # For the "real" ILS
        # return rf.ArrayWithUnit(np.array([0.63/2.0, 0.42/2.0]), "nm")
        return None

    @cached_property
    def instrument(self):
        ils_vec = []
        for i in range(self.num_channels):
            sg = rf.SampleGridSpectralDomain(self.observation.spectral_domain_full(i),
                                             self.observation.filter_list[i])
            if self.ils_method(i) == "FASTCONV":
                iparms = self.ils_params(i)

                # High res extensions unused by IlsFastApply
                high_res_ext = rf.DoubleWithUnit(0, rf.Unit("nm"))

                ils_obj = rf.IlsFastApply(iparms["scaled_uh_isrf"].transpose(),
                                          iparms["svh_isrf_fft_real"].transpose(),
                                          iparms["svh_isrf_fft_imag"].transpose(),
                                          iparms["where_extract"],
                                          sg,
                                          high_res_ext,
                                          self.filter_list[i], self.filter_list[i])
            elif self.ils_method(i) == "POSTCONV":
                iparms = self.ils_params(i)
                # Calculate the wavelength grid first - deltas in
                # wavelength don't translate to wavenumber deltas JLL:
                # I *think* that "central_wavelength" from the UIP ILS
                # parameters will be the wavelengths that the ISRF is
                # defined on. Not sure what "central_wavelength_fm"
                # is; in testing, it was identical to
                # central_wavelength.
                response_wavelength = iparms['central_wavelength'].reshape(-1,1) + iparms['delta_wavelength']

                # Convert to frequency-ordered wavenumber arrays. The
                # 2D arrays need flipped on both axes since the order
                # reverses in both
                center_wn = np.flip(rf.ArrayWithUnit(iparms['central_wavelength'], 'nm').convert_wave('cm^-1').value)
                response_wn = np.flip(np.flip(rf.ArrayWithUnit(response_wavelength, 'nm').convert_wave('cm^-1').value, axis=1), axis=0)
                response = np.flip(np.flip(iparms['isrf'], axis=1), axis=0)

                # Calculate the deltas in wavenumber space
                delta_wn = response_wn - center_wn.reshape(-1,1)

                # Build a table of ILSs at the sampled wavelengths/frequencies
                interp_wavenumber = True
                band_name = self.filter_list[i]
                ils_func = rf.IlsTableLinear(center_wn, delta_wn, response, band_name,
                                             band_name, interp_wavenumber)
                
                # That defines the ILS function, but now we need to
                # get the actual grating object.  Technically we've
                # conflating things here; POSTCONV doesn't necessarily
                # need to mean a grating spectrometer - we could be
                # working with an FTIR. But we'll deal with that when
                # ReFRACtor has an FTIR ILS object.
                #
                # It also seems to be important that the hwhm be in
                # the same units as the ILS table. In my CO
                # development, when I tried using a HWHM in nm,
                # increasing it from ~0.2 nm to ~0.25 nm make the line
                # widths simulated by ReFRACtor compare worse to
                # measured TROPOMI radiances, but discussion with Matt
                # Thill suggested that the HWHM input to the
                # IlsGrating should just set how wide a window around
                # the central wavelength that component does its
                # calculations over, so a wider window should always
                # produce a more accurate result. Converting HWHM to
                # wavenumber seems to fix that issue; once I did that
                # the 0.2 and 0.25 nm HWHM gave similar results.
                hwhm = self.instrument_hwhm(i)
                if hwhm.units.name != 'cm^-1':
                    # Don't try to convert non-wavenumber values -
                    # remember, this is a delta, and delta wavelengths
                    # can't be simply converted to delta wavenumbers
                    # without knowing what wavelength we're working
                    # at.
                    raise ValueError('Half width at half max values for POSTCONV ILSes must be given in wavenumbers')

                # Needs to be in wavenumbers.
                sg2 = rf.SampleGridSpectralDomain(
                    rf.SpectralDomain(self.observation.spectral_domain_full(i).convert_wave("cm^-1"),
                                   rf.Unit("cm^-1")),
                    self.observation.filter_list[i])
                ils_obj = rf.IlsGrating(sg2, ils_func, hwhm)
            else:
                ils_obj = rf.IdentityIls(sg)

            ils_vec.append(ils_obj)
        return rf.IlsInstrument(ils_vec, self.instrument_correction)
    
    @abc.abstractmethod
    def instrument_hwhm(self, sensor_index: int) -> rf.DoubleWithUnit:
        '''Grating spectrometers like OMI and TROPOMI require a fixed
        half width at half max for the IlsGrating object. This can
        vary from band to band. This function must return the HWHM in
        wavenumbers for the band indicated by `sensor_index`<`, which
        will be the index from `self.channel_list()` for the current
        band.
        '''
        raise NotImplementedError

    @cached_property
    def pressure_fm(self):
        '''Pressure grid. Note this is always on the full forward model
        grid. Various objects (e.g. AbsorberVmrLevel) may take a state
        vector on a subset of the pressure grid (the "retrieval grid"), but
        this is handled by having StateMapping for those objects. In all
        cases, this pressure object is what is needed by the ForwardModel,
        which is the pressure on the forward model grid.'''
        # 100 is to convert hPa used by py-retrieve to Pa we use here.
        plev = self.rf_uip.atmosphere_column("pressure") * 100

        surface_pressure = plev[0]
        return rf.PressureSigma(plev, surface_pressure,
                                rf.Pressure.PREFER_DECREASING_PRESSURE)

    @property
    def cloud_pressure(self):
        '''Pressure to use for cloud top'''
        return self.observation.cloud_pressure

    @cached_property
    def pressure(self):
        # We sometimes get negative cloud pressure (e.g. -32767), which later shows as bad_alloc errors
        if self.cloud_pressure < 0:
            raise RuntimeError(f"Invalid cloud pressure: {self.cloud_pressure}.")

        # Note, there is a bit of a difference between the use of
        # cloud_pressure in py_retrieve vs. ReFRACtor. py_retrieve compares
        # the cloud pressure against the pressure of *layers*, while ReFRACtor
        # does this against the pressure at *levels*. In practice the
        # cloud_pressure is only used to determine what levels/layers are
        # included in cloudy forward model. So we determine an equivalent
        # "cloud pressure level" that gives the same number of layers. We
        # could change ReFRACtor to use layers, but there doesn't seem to be
        # much point.
        rinfo = MusesRayInfo(self.rf_uip, self.instrument_name, self.pressure_fm)
        ncloud_lay = rinfo.number_cloud_layer(self.cloud_pressure)
        pgrid = self.pressure_fm.pressure_grid().value.value
        if(ncloud_lay+1 < pgrid.shape[0]):
            cloud_pressure_level = (pgrid[ncloud_lay] + pgrid[ncloud_lay+1]) / 2
        else:
            cloud_pressure_level = pgrid[ncloud_lay]
            
        p = rf.PressureWithCloudHandling(self.pressure_fm, cloud_pressure_level)
        return p

    @cached_property
    def temperature(self):
        tlev_fm, _ = self.current_state.object_state(["TATM",])
        tlevel = rf.TemperatureLevel(tlev_fm, self.pressure_fm)
        return tlevel

    @cached_property
    def constants(self):
        return rf.DefaultConstant()

    @cached_property
    def rayleigh(self):
        return rf.RayleighBodhaine(self.pressure, self.altitude,
                                   self.constants)

    @cached_property
    def absorber_vmr(self):
        return self._inner_absorber.absorber_vmr

    @cached_property
    def absorber(self):
        '''Absorber to use. This is a pass through method to the inner absorber component.'''
        return self._inner_absorber.absorber

    @abc.abstractproperty
    @cached_property
    def ground_clear(self):
        raise NotImplementedError

    @abc.abstractproperty
    @cached_property
    def ground_cloud(self):
        raise NotImplementedError

    @cached_property
    def ground(self):
        return rf.GroundWithCloudHandling(self.ground_clear,self.ground_cloud)

    @cached_property
    def relative_humidity(self):
        return rf.RelativeHumidity(self.absorber, self.temperature,
                                   self.pressure)

    @cached_property
    def altitude(self):
        res = []
        for i in range(self.num_channels):
            # chan_alt = rf.AltitudeHydrostatic(self.pressure,
            #     self.temperature, self.rf_uip.latitude_with_unit(i),
            #     self.rf_uip.surface_height_with_unit(i))

            chan_alt = MusesAltitude(self.ray_info, self.pressure,
                                     self.observation.latitude[i])
            res.append(chan_alt)
        return res

    @cached_property
    def atmosphere(self):
        atm = rf.AtmosphereStandard(self.absorber, self.pressure,
            self.temperature, self.rayleigh, self.relative_humidity,
            self.ground, self.altitude, self.constants)

        patm = atm.pressure.pressure_grid().value.value / 100
        tatm = atm.temperature.temperature_grid(atm.pressure).value.value
        alt = atm.altitude(0).value

        return atm

    @cached_property
    def radiative_transfer(self):
        '''RT to use. This just gives us a simple place to switch
        between Lidort and PCA.'''
        # Not sure that PCA is working, right now we'll only run this if
        # use_pca is set to true
        if(self.use_pca):
            return self.radiative_transfer_pca
        else:
            return self.radiative_transfer_lidort

    @cached_property
    def radiative_transfer_pca(self):
        # Can compare between original l_rad and optimized by changing
        # number of stokes from 4 to 1. Should actually be the same
        # value calculated
        # a = np.zeros((self.num_channel, 4))
        a = np.zeros((self.num_channels, 1))
        a[:, 0] = 1
        stokes = rf.StokesCoefficientConstant(a)
        primary_absorber = self._inner_absorber.primary_absorber_name
        bin_method = rf.PCABinning.UVVSWIR_V4
        num_bins = 11
        num_eofs = 4
        num_streams = 4
        num_mom = 3
        use_solar_sources = True
        use_thermal_emission = False
        do_3m_correction = False

        # Use lrad for first order SS computation and correction
        first_order_rt = None
        if self.use_lrad:
            num_stokes = 1  # Use optimized i_only second order correction
            pure_nadir = False  # Do we want any logic here to set this?

            # Use the first order results since we are using l_rad for SS as well as
            # the corrections
            use_first_order_results = True

            first_order_rt = rf.LRadRt(stokes,
                                       self.atmosphere,
                                       self.spec_win.spectral_bound,
                                       self.sza, self.oza, self.raz,
                                       pure_nadir, num_stokes,
                                       self.lrad_second_order,
                                       num_streams)

        rt = rf.PCARt(self.atmosphere, primary_absorber,
                      bin_method, num_bins, num_eofs,
                      stokes, self.sza, self.oza, self.raz,
                      num_streams, num_mom, use_solar_sources,
                      use_thermal_emission, do_3m_correction, first_order_rt)

        # Change RT flags to match py-retrieve
        lid_interface = rt.lidort.rt_driver.lidort_interface
        lid_interface.lidort_modin.mbool().ts_do_focorr(False)
        lid_interface.lidort_modin.mbool().ts_do_focorr_nadir(False)
        lid_interface.lidort_modin.mbool().ts_do_focorr_outgoing(False)
        lid_interface.lidort_modin.mbool().ts_do_rayleigh_only(False)
        lid_interface.lidort_modin.mbool().ts_do_double_convtest(False)
        lid_interface.lidort_modin.mbool().ts_do_deltam_scaling(False)
        lid_interface.lidort_modin.mchapman().ts_earth_radius(6371.0)

        return rt

    @cached_property
    def radiative_transfer_lidort(self):
        num_streams = 4
        num_mom = 2
        use_thermal_emission = False
        use_solar_sources = True
        pure_nadir = False

        # Use lrad for the single scattering to avoid double computation
        if self.use_lrad:
            multiple_scattering_only = True
        else:
            multiple_scattering_only = False

        use_thermal_scattering = True
        # Can compare between original l_rad and optimized by changing
        # number of stokes from 4 to 1. Should actually be the same
        # value calculated
        # a = np.zeros((self.num_channel, 4))
        a = np.zeros((self.num_channels, 1))
        a[:, 0] = 1
        stokes = rf.StokesCoefficientConstant(a)

        rt = rf.LidortRt(self.atmosphere, stokes,
                         self.sza, self.oza, self.raz,
                         pure_nadir, num_streams, num_mom,
                         multiple_scattering_only, use_solar_sources,
                         use_thermal_emission, use_thermal_scattering)

        # Change RT flags to match py-retrieve
        lid_interface = rt.rt_driver.lidort_interface
        lid_interface.lidort_modin.mbool().ts_do_focorr(True)
        lid_interface.lidort_modin.mbool().ts_do_focorr_nadir(True)
        lid_interface.lidort_modin.mbool().ts_do_focorr_outgoing(False)
        lid_interface.lidort_modin.mbool().ts_do_rayleigh_only(True)
        lid_interface.lidort_modin.mbool().ts_do_double_convtest(False)
        lid_interface.lidort_modin.mbool().ts_do_deltam_scaling(False)
        lid_interface.lidort_modin.mchapman().ts_earth_radius(6371.0)

        # Add in LRadRt
        if self.use_lrad:
            pure_nadir = False  # Do we want any logic here to set this?

            # Use the first order results since we have turned LIDORT into MS only mode above
            # Since l_rad has to compute the SS anyways, we take advantage to not to compute
            # it more than once.
            use_first_order_results = True

            rt = rf.LRadRt(rt, self.spec_win.spectral_bound,
                           self.sza, self.oza, self.raz,
                           pure_nadir, use_first_order_results, self.lrad_second_order)

        return rt

    @cached_property
    def spectrum_effect(self):
        res = []
        for i in range(self.num_channels):
            per_channel_eff = []
            if(self.use_raman):
                reffect = self.raman_effect(i)
                if(reffect is not None):
                    per_channel_eff.append(reffect)
            res.append(per_channel_eff)
        return res

    @cached_property
    def underlying_forward_model(self):
        res = rf.StandardForwardModel(self.instrument, self.spec_win,
                  self.radiative_transfer, self.spectrum_sampling,
                  self.spectrum_effect)
        res.setup_grid()
        return res

    @abc.abstractproperty
    @cached_property
    def cloud_fraction(self):
        raise NotImplementedError

    @cached_property
    def forward_model(self):
        res = rf.ForwardModelWithCloudHandling(self.underlying_forward_model,
                                               self.cloud_fraction)
        res.add_cloud_handling_object(self.pressure)
        res.add_cloud_handling_object(self.ground)
        if(False):
            # Add a wrapper in python, so we can get timings include ReFRACtor
            res = RefractorForwardModel(res)
        #logger.debug("Forward Model: %s", res)
        return res

    @lru_cache(maxsize=None)
    def raman_effect(self, i):
        raise NotImplementedError


class AbstractAbsorber(object, metaclass=abc.ABCMeta):
    @abc.abstractproperty
    @cached_property
    def absorber(self):
        """Return the `rf.Absorber` subclass instance that the FM object creator should use to calculate 
        trace gase absorbance.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def available_species(self) -> Sequence[str]:
        """Return a list of species names which this class can calculate absorbances for.
        The intended use is for the class to check this list against the list of species
        required by the UIP, and skip trying to simulate species it does not have the 
        nesessary spectroscopy for.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def primary_absorber_name(self) -> str:
        """Return the name of the absorber specie that the PCA radiative transfer method
        should use as the "primary" absorber when determining its bins.
        """
        raise NotImplementedError

    @abc.abstractproperty
    @cached_property
    def absorber_vmr(self):
        """Return the vector of gas VMR absorber instances to use in the main absorber class.
        """
        raise NotImplementedError

    def find_absco_fname(self, pattern, join_to_absco_base_path=True):
        if join_to_absco_base_path:
            fname_pat = os.path.join(self.absco_base_path, pattern)
        else:
            fname_pat = pattern
            
        flist = glob.glob(fname_pat)
        if(len(flist) > 1):
            raise RuntimeError(f"Found more than one ABSCO file at {fname_pat}")
        if(len(flist) == 0):
            raise RuntimeError(f"No ABSCO files found at {fname_pat}")
        return flist[0]

    @property
    def absco_base_path(self):
        # JLL: if the MUSES_OSP_PATH environmental variable isn't set, assume that it's
        # the standard MUSES OSP path and we are currently in a sounding directory
        # which has the OSPs linked in its parent directory.
        osp_path = os.environ.get('MUSES_OSP_PATH', '../OSP')
        return os.path.join(osp_path, 'ABSCO')



class O3Absorber(AbstractAbsorber):
    """A class representing absorbance hard-coded for O3.

    This is the original implementation of O3 absorbance for TROPOMI/OMI. It has aspects
    that are specifically written assuming O3 is the only absorber, so it may not be the
    best choice or need modification if implementing new UV/visible absorbers (e.g. NO2).
    """
    def __init__(self, parent_obj_creator: RefractorFmObjectCreator):
        # JLL: I'm not thrilled about introducing a circular reference here, but since the
        # absorber methods need to be cached properties, this was the easiest way to keep
        # that structure.
        self._parent = parent_obj_creator


    @property
    def available_species(self) -> Sequence[str]:
        return ['O3']
        
    @property
    def primary_absorber_name(self) -> str:
        return 'O3'
    
    @cached_property
    def absorber(self):
        '''Absorber to use. This just gives us a simple place to switch
        between absco and cross section.'''

        # Use higher resolution xsec when using FASTCONV
        if self._parent.ils_method(0) == "FASTCONV":
            return self.absorber_xsec
        else:
            return self.absorber_muses

    @cached_property
    def absorber_muses(self):
        '''Uses MUSES O3 optical files, which are precomputed ahead
        of the forward model. They may include a convolution with the ILS.
        '''
        # TODO Note that MusesOpticalDepthFile reads files that get created
        # in make_uip_tropomi.py. This get read in by functions like
        # get_tropomi_o3xsec_without_ils and then written to file.
        # We should move this into MusesOpticalDepthFile without having
        # a file.
        return MusesOpticalDepthFile(self._parent.ray_info,
                                     self._parent.pressure,
                                     self._parent.temperature, self._parent.altitude,
                                     self.absorber_vmr, self._parent.num_channels,
                                     f"{self._parent.step_directory}/vlidort/input")

    @cached_property
    def absorber_xsec(self):
        '''Use the O3 cross section files for calculation absorption.
        This does not include the ILS at the absorption calculation level,
        so to get good results we should include an ILS with our forward
        model.'''
        xsectable = []
        for gas in ["O3", ]:
            xsec_data = np.loadtxt(rf.cross_section_filenames[gas])
            cfac = rf.cross_section_file_conversion_factors.get(gas, 1.0)
            spec_grid = rf.ArrayWithUnit(xsec_data[:, 0], "nm")
            xsec_values = rf.ArrayWithUnit(xsec_data[:, 1:], "cm^2")
            if xsec_data.shape[1] >= 4:
                xsectable.append(rf.XSecTableTempDep(spec_grid, xsec_values,
                                                        cfac))
            else:
                xsectable.append(rf.XSecTableSimple(spec_grid, xsec_values,
                                                       cfac))
        return rf.AbsorberXSec(self.absorber_vmr, self._parent.pressure,
                               self._parent.temperature, self._parent.altitude,
                               xsectable)

    @cached_property
    def absorber_absco(self):
        '''Use ABSCO tables to calculation absorption.'''
        absorptions = []
        absco_filename = self.find_absco_fname("O3_*_v0.0_init.nc")
        absorptions.append(rf.AbscoAer(absco_filename, 1.0, 5000,
                              rf.AbscoAer.NEAREST_NEIGHBOR_WN))
        return rf.AbsorberAbsco(self.absorber_vmr, self._parent.pressure,
                                self._parent.temperature,
                                self._parent.altitude, absorptions, self._parent.constants)

    @cached_property
    def absorber_vmr(self):
        vmrs = []

        # Log mapping must come first to convert state vector elements from log first
        # before mapping to a different number of levels
        # TODO from JLL: if the approach used in the SwirAbsorber (using `available_species`
        # and the UIP to determine which species are absorbers) is correct and more general,
        # then this should be modified to be consistent with that method. 
        mappings = []
        mappings.append(rf.StateMappingLog())

        smap = rf.StateMappingComposite(mappings)

        vmrs.append(rf.AbsorberVmrLevel(self._parent.pressure_fm,
                         self._parent.rf_uip.atmosphere_column("O3"),
                         "O3", smap))
        return vmrs


class SwirAbsorber(AbstractAbsorber):
    def __init__(self, parent_obj_creator: RefractorFmObjectCreator):
        # JLL: I'm not thrilled about introducing a circular reference here, but since the
        # absorber methods need to be cached properties, this was the easiest way to keep
        # that structure.
        self._parent = parent_obj_creator

    @property
    def available_species(self) -> Sequence[str]:
        return ['CO', 'CH4', 'H2O', 'HDO']
        
    @property
    def primary_absorber_name(self) -> str:
        # JLL: Depending on how the PCA RT uses the primary absorber, this may need
        # to be updated to figure this out from the UIP in case we want to target one
        # of the other available species. I don't think there's anything in strategy tables
        # at the moment that allows us to manually specify a "primary absorber", but
        # depending on exactly what that means, we might be able to infer a reasonable
        # answer.
        return 'CO'
    
    @cached_property
    def absorber(self):
        '''Use ABSCO tables to calculation absorption.'''
        absorptions = []
        species = self._parent.rf_uip.atm_params(self._parent.instrument_name)['species']
        skipped_species = []
        for spec in species:
            if spec in self.available_species:
                absco_filename = self.find_swir_absco_filename(spec)
                # JLL: during development, I used an AbscoStub class that inherited from rf.Absco.
                # Not sure if there are key differences with the rf.AbscoAer class.
                absorptions.append(rf.AbscoAer(absco_filename, 1.0, 5000,
                                               rf.AbscoAer.NEAREST_NEIGHBOR_WN))
            else:
                skipped_species.append(spec)

        if skipped_species:
            skipped_species = ', '.join(skipped_species)
            logger.info(f'One or species from the strategy table will not be simulated by ReFRACtor because SWIR absorbances are not implemented for them: {skipped_species}')

        return rf.AbsorberAbsco(self.absorber_vmr, self._parent.pressure,
                                self._parent.temperature,
                                self._parent.altitude, absorptions, self._parent.constants)
        


    def find_swir_absco_filename(self, specie, version='latest'):
        # allow one to pass in "latest" or a version number like either "1.0" or "v1.0"
        if version == 'latest':
            vpat = 'v*'
        elif version.startswith('v'):
            vpat = version
        else:
            vpat = f'v{version}'

        # Assumes that in the top level of the ABSCO directory there are
        # subdirectories such as "v1.0_SWIR_CO" which contain our ABSCO files.
        absco_subdir_pattern = f'{vpat}_SWIR_{specie.upper()}'
        absco_subdirs = sorted(Path(self.absco_base_path).glob(absco_subdir_pattern))
        if version == 'latest' and len(absco_subdirs) == 0:
            full_pattern = Path(self.absco_base_path) / absco_subdir_pattern
            raise RuntimeError(f'Found no ABSCO directories for specie "{specie}" matching {full_pattern}')
        elif version == 'latest':
            # Assumes that the latest version will be the last after sorting (e.g. v1.1
            # > v1.0). Should technically use a semantic version parser to ensure e.g.
            # v1.0.1 would be selected over v1.0.
            specie_subdir = absco_subdirs[-1]
            logger.info(f'Using ABSCO files from {specie_subdir} for {specie}')
        elif len(absco_subdirs) == 1:
            specie_subdir = absco_subdirs[0]
        else:
            raise RuntimeError(f'{len(absco_subdirs)} were found for {specie} {version} in {self.absco_base_path}')

        specie_pattern = (specie_subdir / 'nc_ABSCO' / f'{specie.upper()}_*_v0.0_init.nc').as_posix()
        return self.find_absco_fname(specie_pattern, join_to_absco_base_path=False)

    @cached_property
    def absorber_vmr(self):        
        vmrs = []

        # Log mapping must come first to convert state vector elements from log first
        # before mapping to a different number of levels
        # (JLL: I take it the mappings are applied from the end of the vector to the front?)
        for specie in self._parent.rf_uip.atm_params(self._parent.instrument_name)['species']:
            mappings = []
            if specie in self._parent.rf_uip.uip['speciesListFM']:
                map_type = self._parent.rf_uip.species_lin_log_mapping(specie).lower()
            else:
                # JLL: When the specie isn't listed in speciesListFM we can't know the mapping, however
                # Mike indicated that it doesn't matter because it won't be used.
                map_type = 'log'

            if map_type == 'log':
                mappings.append(rf.StateMappingLog())
            elif map_type != 'linear':
                raise NotImplementedError(f'Unknown map type "{map_type}"')

            smap = rf.StateMappingComposite(mappings)

            vmrs.append(rf.AbsorberVmrLevel(self._parent.pressure_fm,
                             self._parent.rf_uip.atmosphere_column(specie),
                             specie, smap))
        return vmrs
    

__all__ = ["RefractorFmObjectCreator", "O3Absorber", "SwirAbsorber"]
