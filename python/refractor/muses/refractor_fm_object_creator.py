from functools import cached_property, lru_cache
from .muses_optical_depth_file import MusesOpticalDepthFile
from .muses_optical_depth import MusesOpticalDepth
from .muses_altitude import MusesAltitude
from .muses_spectrum_sampling import MusesSpectrumSampling
from .muses_raman import MusesRaman
from .refractor_uip import RefractorUip
from .muses_forward_model import RefractorForwardModel
from .muses_ray_info import MusesRayInfo
import refractor.framework as rf
import os
from pathlib import Path
from loguru import logger
import numpy as np
import glob
import abc
import copy

from typing import Sequence

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
                 rf_uip_func : "Optional(Callable[{instrument:None}, RefractorUip])" = None,
                 fm_sv : "Optional(rf.StateVector)" = None,
                 # Values, so we can flip between using pca and not
                 use_pca=True, use_lrad=False, lrad_second_order=False,
                 use_raman=True,
                 skip_observation_add=False,
                 match_py_retrieve=False,
                 osp_dir=None,
                 absorption_gases = ["O3",],
                 primary_absorber = "O3"
                 ):
        '''Constructor. The StateVector to add things to can be passed in, or if this
        isn't then we create a new StateVector.

        There are a number of number of options for exactly how we construct the
        ForwardModel.

        match_py_retrieve - We have some classes that purposely mimic the way py-retrieves
             forward model works. These may have only minor differences with standard
             ReFRACtor classes, but for doing an initial comparison against py-retrieve
             it can be useful to remove minor differences to uncover anything that might
             be a real, unexpected difference. Normally you want to the default False
             value, but for testing purposes you might want to turn this on.
        '''
        self.use_pca = use_pca
        self.use_lrad = use_lrad
        self.use_raman = use_raman
        self.instrument_name = instrument_name
        self.lrad_second_order = lrad_second_order
        self.match_py_retrieve = match_py_retrieve
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
        if(not skip_observation_add and len(self.observation.state_element_name_list()) > 0):
            coeff,mp = self.current_state.object_state(self.observation.state_element_name_list())
            self.observation.update_coeff_and_mapping(coeff,mp)
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv, self.observation.state_element_name_list(), [self.observation,])
        
        # We are moving away from this, but leave now as a reference
        self.rf_uip_func = rf_uip_func

        self.filter_list = self.observation.filter_list
        self.num_channels = self.observation.num_channels

        self.sza = self.observation.solar_zenith
        self.oza = self.observation.observation_zenith
        self.raz = self.observation.relative_azimuth

        self.osp_dir = osp_dir if osp_dir is not None else os.environ.get('MUSES_OSP_PATH', '../OSP')
        # We may put in logic to determine this, but for right now just
        # take the list of absorption species as a input list and the
        # primary absorber needed by PCA
        self.absorption_gases = copy.copy(absorption_gases)
        self.primary_absorber = primary_absorber
        
    def solar_model(self, sensor_index):
        with self.observation.modify_spectral_window(do_raman_ext=True):
            sol_rad = self.observation.solar_spectrum(sensor_index)
        return rf.SolarReferenceSpectrum(sol_rad, None)

    @cached_property
    def ray_info(self):
        '''Return MusesRayInfo.'''
        return MusesRayInfo(self.rf_uip_func(), self.instrument_name, self.pressure)

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

    @property
    def ils_method(self, sensor_index : int) -> str:
        '''Return the ILS method to use. This is APPLY, POSTCONV, or FASTCONV.
        '''
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
        plev_fm, _ = self.current_state.object_state(["pressure",])
        # 100 is to convert hPa used by py-retrieve to Pa we use here.
        plev_fm *= 100.0

        surface_pressure = plev_fm[0]
        return rf.PressureSigma(plev_fm, surface_pressure,
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
        rinfo = MusesRayInfo(self.rf_uip_func(), self.instrument_name, self.pressure_fm)
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
        return rf.RayleighBodhaine(self.pressure, self.alt_vec(),
                                   self.constants)

    @cached_property
    def absorber_vmr(self):
        vmrs = []
        for gas in self.absorption_gases:
            selem = [gas,]
            coeff, mp = self.current_state.object_state(selem)
            # Need to get mp to be the log mapping in current_state, but for
            # now just work around this
            mp = rf.StateMappingLog()
            vmr = rf.AbsorberVmrLevel(self.pressure_fm, coeff, gas, mp)
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv, selem, [vmr,])
            vmrs.append(vmr)
        return vmrs

    @cached_property
    def absorber_muses_file(self):
        '''Uses MUSES O3 optical files, which are precomputed ahead
        of the forward model. They may include a convolution with the ILS.
        '''
        # TODO Note that MusesOpticalDepthFile reads files that get created
        # in make_uip_tropomi.py. This get read in by functions like
        # get_tropomi_o3xsec_without_ils and then written to file.
        # We should move this into MusesOpticalDepthFile without having
        # a file.
        # MusesOpticalDepthFile only support O3
        vmr_list = [vmr for vmr in self.absorber_vmr if vmr.gas_name == "O3"]
        return MusesOpticalDepthFile(self.ray_info,
                                     self.pressure,
                                     self.temperature, self.altitude,
                                     self.absorber_vmr, self.num_channels,
                                     f"{self.step_directory}/vlidort/input")

    @cached_property
    def absorber_muses(self):
        '''Uses MUSES code for O3 absorption, which are precomputed ahead
        of the forward model. They may include a convolution with the ILS.
        '''
        # Temp, force UIP to generate O3 file so we can compare
        _ = self.ray_info
        vmr_list = [vmr for vmr in self.absorber_vmr if vmr.gas_name == "O3"]
        ils_params_list = []
        for i in range(self.num_channels):
            ils_params_list.append(self.ils_params(i))
        return MusesOpticalDepth(self.pressure,
                                 self.temperature, self.altitude,
                                 self.absorber_vmr,
                                 self.observation,
                                 ils_params_list,
                                 self.osp_dir)
    
    @cached_property
    def absorber_xsec(self):
        '''Use the O3 cross section files for calculation absorption.
        This does not include the ILS at the absorption calculation level,
        so to get good results we should include an ILS with our forward
        model.'''
        xsectable = []
        for gas in self.absorption_gases:
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
        return rf.AbsorberXSec(self.absorber_vmr, self.pressure,
                               self.temperature, self.alt_vec(),
                               xsectable)
    

    def find_absco_pattern(self, pattern, join_to_absco_base_path=True):
        if join_to_absco_base_path:
            fname_pat = f"{self.absco_base_path}/{pattern}"
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
        return f"{self.osp_dir}/ABSCO/"
    

    def absco_filename(self, gas):
        if(gas != "O3"):
            return None
        return self.find_absco_pattern("O3_*_v0.0_init.nc")

    @cached_property
    def absorber_absco(self):
        '''Use ABSCO tables to calculation absorption.'''
        absorptions = []
        skipped_gases = []
        for gas in self.absorption_gases:
            fname = self.absco_filename(gas)
            if(fname is not None):
                absorptions.append(rf.AbscoAer(fname, 1.0, 5000,
                                               rf.AbscoAer.NEAREST_NEIGHBOR_WN))
            else:
                skipped_gases.append(gas)
        if len(skipped_gases) > 0:
            logger.info(f"One or absorption_gases does not have a ABSCO file, so won't be include. Skipped gases: {', '.join(skipped_gases)}")
        
        return rf.AbsorberAbsco(self.absorber_vmr, self.pressure,
                                self.temperature,
                                self.alt_vec(), absorptions, self.constants)
            
            
    @cached_property
    def absorber(self):
        '''Absorber to use. This just gives us a simple place to switch
        between absco and cross section.'''

        # Use higher resolution xsec when not using APPLY (which means
        # pre convolve)
        #
        # Note see commend in ils_method, this assumes all spectral bands are
        # the same. We can probably relax that, but really need a test case to
        # work through the logic
        if self.ils_method(0) != "APPLY":
            return self.absorber_xsec
        elif(self.match_py_retrieve):
            return self.absorber_muses_file
        else:
            return self.absorber_muses

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
    def altitude_muses(self):
        res = []
        for i in range(self.num_channels):
            chan_alt = MusesAltitude(self.ray_info, self.pressure,
                                     self.observation.latitude[i])
            res.append(chan_alt)
        return res

    @cached_property
    def altitude_refractor(self):
        res = []
        for i in range(self.num_channels):
            chan_alt = rf.AltitudeHydrostatic(
                self.pressure,
                self.temperature,
                rf.DoubleWithUnit(self.observation.latitude[i], "deg"),
                rf.DoubleWithUnit(self.observation.surface_height[i], "m"))
            res.append(chan_alt)
        return res

    @property
    def altitude(self):
        if(self.match_py_retrieve):
            return self.altitude_muses
        else:
            return self.altitude_refractor

    def alt_vec(self):
        res = rf.vector_altitude()
        for alt in self.altitude:
            res.push_back(alt)
        return res
    
    @cached_property
    def atmosphere(self):
        atm = rf.AtmosphereStandard(self.absorber, self.pressure,
            self.temperature, self.rayleigh, self.relative_humidity,
            self.ground, self.alt_vec(), self.constants)
        # Atmosphere doesn't directly use state vector elements, but it needs
        # to know when this changes because the number of jacobian variables
        # might change, and we need to know that the cache should be cleared.
        self.fm_sv.add_observer(atm)
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

        rt = rf.PCARt(self.atmosphere, self.primary_absorber,
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


__all__ = ["RefractorFmObjectCreator", ]
