try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from .muses_optical_depth_file import MusesOpticalDepthFile
from .muses_altitude import MusesAltitude
from .muses_spectrum_sampling import MusesSpectrumSampling
from .muses_raman import MusesRaman
from .refractor_uip import RefractorUip
from .muses_forward_model import RefractorForwardModel
import refractor.framework as rf
import os
from pathlib import Path
import logging
import numpy as np
import glob
import abc

from typing import Sequence

logger = logging.getLogger("py-retrieve")

class RefractorFmObjectCreator(object, metaclass=abc.ABCMeta):
    '''There are a lot of interrelated object needed to be created to
    make a ForwardModel.

    This class provides a framework for a lot of the common pieces. It
    is intended that derived classes override certain functionality (e.g.,
    have a difference Pressure object). The changes then ripple through
    the creation - so anything that needs a Pressure object to create itself
    will use whatever the derived class supplies.

    Note that this class is a convenience - for things like CostFuncCreator
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

    def __init__(self, rf_uip : RefractorUip,
                 instrument_name: str, observation : 'MusesObservation',
                 input_dir=None,
                 # Short term, so we can flip between pca vs lidort
                 use_pca=True, use_lrad=False, lrad_second_order=False,
                 use_raman=True,
                 include_bad_sample=True,
                 ):
        '''Constructor. This takes a RefractorUip (so *not* the
        muses-py dictionary, but rather a RefractorUip created from
        that).
        
        The input directory can be given, this is used to read the 
        solar model data (omisol_v003_avg_nshi_backup.h5). If not supplied,
        we use the default directory path.
        '''

        self.input_dir = input_dir

        self.use_pca = use_pca
        self.use_lrad = use_lrad
        self.use_raman = use_raman
        self.instrument_name = instrument_name
        self.lrad_second_order = lrad_second_order
        self.include_bad_sample = include_bad_sample
        self.observation = observation

        self.rf_uip = rf_uip
        # Hopefully we can move away from using rf_uip and use state_info
        # directly. But for now we get this from the rf_uip
        # Short term, support this missing in rf_uip. This is because we
        # have old test data that doesn't have this in it yet.
        if(hasattr(rf_uip, "state_info")):
            self.state_info = rf_uip.state_info
        else:
            self.state_info = None

        if(self.input_dir is None):
            self.input_dir = os.path.realpath(os.path.join(rf_uip.uip_all(self.instrument_name)['L2_OSP_PATH'], "OMI"))

        self.num_channel = len(self.channel_list())

        self.sza = np.array([float(self.rf_uip.solar_zenith(i))
                             for i in self.channel_list() ])
        self.oza = np.array([float(self.rf_uip.observation_zenith(i))
                             for i in self.channel_list() ])
        # For TROPOMI view azimuth angle isn't available. Not sure if
        # that matters, I don't think this gets used for anything (only
        # relative azimuth is used). But go ahead and fill this in if
        # we aren't working with TROPOMI.
        if self.instrument_name != "TROPOMI":
            self.oaz = np.array([float(self.rf_uip.observation_azimuth(i))
                                 for i in self.channel_list() ])
        self.raz = np.array([float(self.rf_uip.relative_azimuth(i))
                             for i in self.channel_list() ])

        self.sza_with_unit = rf.ArrayWithUnit(self.sza, "deg")
        self.oza_with_unit = rf.ArrayWithUnit(self.oza, "deg")
        if False:
            self.oaz_with_unit = rf.ArrayWithUnit(self.oaz, "deg")
        self.raz_with_unit = rf.ArrayWithUnit(self.raz, "deg")
        self.filter_name = [self.rf_uip.filter_name(i) for i in self.channel_list()]

        # This is what OMI currently uses, and TROPOMI band 3. We may put all this
        # together, but right now tropomi_fm_object_creator may replace this.
        self._inner_absorber = O3Absorber(self)

    def channel_list(self):
        '''This is list of microwindows relevant to self.instrument_name

        Note that there are two microwindow indexes floating around. We have
        ii_mw which goes through all the instruments, so for step 7 in
        AIRS+OMI ii_mw goes through 12 values (only 10 and 11 are OMI).
        mw_index (also call fm_idx) is relative to a instrument,
        so if we are working with OMI the first microwindow has ii_mw = 10, but
        mw_index is 0 (UV1, with the second UV2).
        
        The contents of channel_list() are ii_mw (e.g., 10 and 11 in our 
        AIRS+OMI example), and the index into channel_list() is fm_idx
        (also called mw_index). So you might loop with something like:

        for fm_idx, ii_mw in enumerate(self.channel_list()):
             blah blah
        '''
        chan_list = []
        for ii_mw in range(self.rf_uip.number_micro_windows):
            if self.rf_uip.instrument_name(ii_mw) == self.instrument_name:
                chan_list.append(ii_mw)

        return chan_list

    def solar_model(self, mw_index):
        return rf.SolarReferenceSpectrum(self.rf_uip.solar_irradiance(mw_index,
                                            self.instrument_name), None)

    @cached_property
    def spec_win(self):
        t = np.vstack([np.array([self.rf_uip.micro_windows(i).value])
                       for i in self.channel_list()])
        swin= rf.SpectralWindowRange(rf.ArrayWithUnit(t, "nm"))
        if(not self.include_bad_sample):
            for i in range(swin.number_spectrometer):
                swin.bad_sample_mask(self.observation.bad_sample_mask(i), i)
        return swin

    @cached_property
    def spectrum_sampling(self):
        return MusesSpectrumSampling(self.instrument_name, rf_uip=self.rf_uip)

    @cached_property
    def dispersion(self):
        # return creator.instrument.SampleGridSpectralDomain.create_direct(
        #    spectral_domains = self.rf_uip.sample_grid,
        #    desc_band_name = self.channel_names,
        #    num_channels = self.num_channels,
        #    spec_win = self.spec_win)
        # Not currently used
        return None

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
        for fm_idx, ii_mw in enumerate(self.channel_list()):
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
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            sg = rf.SampleGridSpectralDomain(self.rf_uip.sample_grid(fm_idx, ii_mw),
                                             self.filter_name[fm_idx])

            ils_method = self.rf_uip.ils_method(fm_idx, self.rf_uip.instrument_name(ii_mw))
            if ils_method == "FASTCONV":
                ils_params = self.rf_uip.ils_params(fm_idx, self.rf_uip.instrument_name(ii_mw))

                # High res extensions unused by IlsFastApply
                high_res_ext = rf.DoubleWithUnit(0, rf.Unit("nm"))

                ils_obj = rf.IlsFastApply(ils_params["scaled_uh_isrf"].transpose(),
                                          ils_params["svh_isrf_fft_real"].transpose(),
                                          ils_params["svh_isrf_fft_imag"].transpose(),
                                          ils_params["where_extract"],
                                          sg,
                                          high_res_ext,
                                          self.filter_name[fm_idx], self.filter_name[fm_idx])
            elif ils_method == "POSTCONV":
                ils_params = self.rf_uip.ils_params(fm_idx, self.rf_uip.instrument_name(ii_mw))
                # Calculate the wavelength grid first - deltas in wavelength don't translate to wavenumber deltas
                # JLL: I *think* that "central_wavelength" from the UIP ILS parameters will be the wavelengths that the 
                #  ISRF is defined on. Not sure what "central_wavelength_fm" is; in testing, it was identical to central_wavelength.
                response_wavelength = ils_params['central_wavelength'].reshape(-1,1) + ils_params['delta_wavelength']

                # Convert to frequency-ordered wavenumber arrays. The 2D arrays need flipped on both axes since the order reverses in both
                center_wn = np.flip(rf.ArrayWithUnit(ils_params['central_wavelength'], 'nm').convert_wave('cm^-1').value)
                response_wn = np.flip(np.flip(rf.ArrayWithUnit(response_wavelength, 'nm').convert_wave('cm^-1').value, axis=1), axis=0)
                response = np.flip(np.flip(ils_params['isrf'], axis=1), axis=0)

                # Calculate the deltas in wavenumber space
                delta_wn = response_wn - center_wn.reshape(-1,1)

                # Build a table of ILSs at the sampled wavelengths/frequencies
                interp_wavenumber = True
                band_name = self.rf_uip.filter_name(ii_mw)
                ils_func = rf.IlsTableLinear(center_wn, delta_wn, response, band_name, band_name, interp_wavenumber)
                
                # That defines the ILS function, but now we need to get the actual grating object.
                # Technically we've conflating things here; POSTCONV doesn't necessarily need to 
                # mean a grating spectrometer - we could be working with an FTIR. But we'll deal 
                # with that when ReFRACtor has an FTIR ILS object.
                #
                # It also seems to be important that the hwhm be in the same units as the ILS table. In my CO development,
                # when I tried using a HWHM in nm, increasing it from ~0.2 nm to ~0.25 nm make the line widths simulated by
                # ReFRACtor compare worse to measured TROPOMI radiances, but discussion with Matt Thill suggested that the
                # HWHM input to the IlsGrating should just set how wide a window around the central wavelength that component
                # does its calculations over, so a wider window should always produce a more accurate result. Converting
                # HWHM to wavenumber seems to fix that issue; once I did that the 0.2 and 0.25 nm HWHM gave similar results.
                hwhm = self.instrument_hwhm(ii_mw)
                if hwhm.units.name != 'cm^-1':
                    # Don't try to convert non-wavenumber values - remember, this is a delta, and delta wavelengths
                    # can't be simply converted to delta wavenumbers without knowing what wavelength we're working at.
                    raise ValueError('Half width at half max values for POSTCONV ILSes must be given in wavenumbers')
                
                model_wavenumbers = self.rf_uip.sample_grid(fm_idx, ii_mw).convert_wave('cm^-1')
                spec_domain = rf.SpectralDomain(model_wavenumbers)
                sample_grid = rf.SampleGridSpectralDomain(spec_domain, band_name)

                ils_obj = rf.IlsGrating(sample_grid, ils_func, hwhm)
            else:
                ils_obj = rf.IdentityIls(sg)

            ils_vec.append(ils_obj)
        return rf.IlsInstrument(ils_vec, self.instrument_correction)
    
    @abc.abstractmethod
    def instrument_hwhm(self, ii_mw: int) -> rf.DoubleWithUnit:
        '''Grating spectrometers like OMI and TROPOMI require a fixed half 
        width at half max for the IlsGrating object. This can vary from band
        to band. This function must return the HWHM in wavenumbers for the 
        band indicated by `ii_mw`, which will be the index from `self.channel_list()`
        for the current band.'''
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

    @abc.abstractproperty
    def uip_params(self):
        raise NotImplementedError
        
    @cached_property
    def pressure(self):
        cloud_pressure = self.uip_params["cloud_pressure"]
        
        # We sometimes get negative cloud pressure (e.g. -32767), which later shows as bad_alloc errors
        if cloud_pressure < 0:
            raise RuntimeError(f"Invalid cloud pressure: {cloud_pressure}.")

        # Note, there is a bit of a difference between the use of
        # cloud_pressure in py_retrieve vs. ReFRACtor. py_retrieve compares
        # the cloud pressure against the pressure of *layers*, while ReFRACtor
        # does this against the pressure at *levels*. In practice the
        # cloud_pressure is only used to determine what levels/layers are
        # included in cloudy forward model. So we determine an equivalent
        # "cloud pressure level" that gives the same number of layers. We
        # could change ReFRACtor to use layers, but there doesn't seem to be
        # much point.
        ncloud_lay = np.count_nonzero(self.rf_uip.ray_info(self.instrument_name)["pbar"] <= self.uip_params['cloud_pressure'])
        pgrid = self.pressure_fm.pressure_grid().value.value
        if(ncloud_lay+1 < pgrid.shape[0]):
            cloud_pressure_level = (pgrid[ncloud_lay] + pgrid[ncloud_lay+1]) / 2
        else:
            cloud_pressure_level = pgrid[ncloud_lay]
            
        p = rf.PressureWithCloudHandling(self.pressure_fm, cloud_pressure_level)
        return p

    @cached_property
    def temperature(self):
        tlev_fm = self.rf_uip.atmosphere_column("TATM")
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

    @cached_property
    def ground_cloud(self):
        albedo = np.zeros((self.num_channel, 1))
        which_retrieved = np.full((self.num_channel, 1), False, dtype=bool)
        band_reference = np.zeros(self.num_channel)
        band_reference[:] = 1000

        albedo[:,0] = self.uip_params['cloud_Surface_Albedo']

        return rf.GroundLambertian(albedo,
                      rf.ArrayWithUnit(band_reference, "nm"),
                      ["Cloud",] * self.num_channel,
                      rf.StateMappingAtIndexes(np.ravel(which_retrieved)))
    

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
        for i in self.channel_list():
            # chan_alt = rf.AltitudeHydrostatic(self.pressure,
            #     self.temperature, self.rf_uip.latitude_with_unit(i),
            #     self.rf_uip.surface_height_with_unit(i))

            chan_alt = MusesAltitude(self.rf_uip, self.instrument_name,
                        self.pressure, self.rf_uip.latitude_with_unit(i))
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
        a = np.zeros((self.num_channel, 1))
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
        a = np.zeros((self.num_channel, 1))
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
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            per_channel_eff = []
            if(self.use_raman):
                per_channel_eff.append(self.raman_effect[fm_idx])
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

    @abc.abstractproperty
    @cached_property
    def raman_effect(self):
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
        if self._parent.rf_uip.ils_method(0, self._parent.instrument_name) == "FASTCONV":
            return self.absorber_xsec
        else:
            return self.absorber_muses

    @cached_property
    def absorber_muses(self):
        '''Uses MUSES O3 optical files, which are precomputed ahead
        of the forward model. They may include a convolution with the ILS.
        '''

        res = MusesOpticalDepthFile(self._parent.rf_uip, self._parent.instrument_name,
                                    self._parent.pressure,
                                    self._parent.temperature, self._parent.altitude,
                                    self.absorber_vmr, self._parent.num_channel)
        return res

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
