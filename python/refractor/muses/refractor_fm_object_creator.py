try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from .muses_optical_depth_file import MusesOpticalDepthFile
from .muses_altitude import MusesAltitude
from .muses_spectrum_sampling import MusesSpectrumSampling
from .muses_raman import MusesRaman
from .refractor_uip import RefractorUip
import refractor.framework as rf
import os
import math
import logging
import numpy as np
import glob
import abc

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

    An important bug note, round tripping by sending a python object to
    C++ and then getting it back again will result in memory problems. See
    refractor_object_creator_test.py for an example of this. Basically the
    combination of python, directors, and smart pointers is a bit buggy in
    swig (last tested 3.0.12). It is possible that this will get fixed at
    some point in a later version of swig, or perhaps it becomes enough of
    an issue that we figure out how to work around whatever swig is doing
    wrong here.

    But an easy work around is to just keep a copy of the python object from
    before we hand it to C++ and then just use that object. So for example
    if we are using MusesOpticalDepthFile as our absorber, then we can
    get access to that by self.absorber rather than something like
    self.atmosphere.absorber.

    This is annoying, and clearly a bug, but for now we can just live with
    this.
    '''

    def __init__(self, rf_uip : RefractorUip,
                 instrument_name: str, input_dir=None, 
                 # Short term, so we can flip between pca vs lidort
                 use_pca=True, use_lrad=False, lrad_second_order=False,
                 use_raman=True,
                 use_full_state_vector=True,
                 remove_bad_sample=False,
                 ):
        '''Constructor. This takes a RefractorUip (so *not* the
        muses-py dictionary, but rather a RefractorUip created from
        that).

        For the retrieval, we use the "Retrieval State Vector".
        However, for testing it can be useful to use the "Full State Vector".
        See "Tropospheric Emission Spectrometer: Retrieval Method and Error
        Analysis" (IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,
        VOL. 44, NO. 5, MAY 2006) section III.A.1 for a discussion of this.
        Lower level muses-py functions work with the "Full State Vector", so
        it is useful to have the option of supporting this. Set
        use_full_state_vector to True to use the full state vector.

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
        self.use_full_state_vector = use_full_state_vector
        self.remove_bad_sample = remove_bad_sample

        self.rf_uip = rf_uip

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
        if(self.remove_bad_sample):
            for i in range(swin.number_spectrometer):
                swin.bad_sample_mask(self.observation.bad_sample_mask(i), i)
        return swin

    @cached_property
    def spectrum_sampling(self):
        return MusesSpectrumSampling(self.rf_uip, self.instrument_name)

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
        res = rf.vector_vector_instrument_correction()
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            v = rf.vector_instrument_correction()
            res.push_back(v)
        return res

    @cached_property
    def ils_half_width(self):
        # For the "real" ILS
        # return rf.ArrayWithUnit(np.array([0.63/2.0, 0.42/2.0]), "nm")
        return None

    @cached_property
    def instrument_spectral_domain(self):
        res = []
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            full_sd = self.rf_uip.sample_grid(fm_idx, ii_mw)
            mw = self.rf_uip.micro_windows(ii_mw).value[0]
            sd = rf.SpectralDomain(full_sd.data[np.logical_and(full_sd.data>=mw[0],full_sd.data<=mw[1])], full_sd.units)
            res.append(rf.SpectralDomain(sd.wavelength("nm"), rf.Unit("nm")))
        return res

    @cached_property
    def instrument(self):
        ils_vec = rf.vector_ils()
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            sg = rf.SampleGridSpectralDomain(self.instrument_spectral_domain[fm_idx],
                                             self.filter_name[fm_idx])

            if self.rf_uip.ils_method(fm_idx,
                         self.rf_uip.instrument_name(ii_mw)) == "FASTCONV":
                ils_params = self.rf_uip.ils_params(fm_idx)

                # High res extensions unused by IlsFastApply
                high_res_ext = rf.DoubleWithUnit(0, rf.Unit("nm"))

                ils_obj = rf.IlsFastApply(ils_params["scaled_uh_isrf"].transpose(),
                                          ils_params["svh_isrf_fft_real"].transpose(),
                                          ils_params["svh_isrf_fft_imag"].transpose(),
                                          ils_params["where_extract"],
                                          sg,
                                          high_res_ext,
                                          self.filter_name[fm_idx], self.filter_name[fm_idx])
            else:
                ils_obj = rf.IdentityIls(sg)

            ils_vec.push_back(ils_obj)
        return rf.IlsInstrument(ils_vec, self.instrument_correction)

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

    def find_absco_fname(self, pattern):
        absco_base_path = os.environ['ABSCO_PATH'] + "/"
        fname_pat = absco_base_path + pattern
        flist = glob.glob(fname_pat)
        if(len(flist) > 1):
            raise RuntimeError("Found more than one ABSCO file at " +
                               fname_pat)
        if(len(flist) == 0):
            raise RuntimeError("No ABSCO files found at " + fname_pat)
        return flist[0]

    @cached_property
    def absorber_vmr(self):
        vmrs = rf.vector_absorber_vmr()
        nlevel = len(self.rf_uip.atmosphere_column("O3"))

        # Log mapping must come first to convert state vector elements from log first
        # before mapping to a different number of levels
        mappings = rf.vector_state_mapping()
        if(not self.use_full_state_vector):
            basis_matrix = self.rf_uip.atmosphere_basis_matrix("O3").transpose()
            if(len(basis_matrix) > 0):
                mappings.push_back(rf.StateMappingBasisMatrix(basis_matrix))
        mappings.push_back(rf.StateMappingLog())

        smap = rf.StateMappingComposite(mappings)

        vmrs.push_back(rf.AbsorberVmrLevel(self.pressure_fm,
                                           self.rf_uip.atmosphere_column("O3"),
                                           "O3",
                                           smap))
        return vmrs

    @cached_property
    def absorber(self):
        '''Absorber to use. This just gives us a simple place to switch
        between absco and cross section.'''

        # Use higher resolution xsec when using FASTCONV
        if self.rf_uip.ils_method(0, self.instrument_name) == "FASTCONV":
            return self.absorber_xsec
        else:
            return self.absorber_muses

    @cached_property
    def absorber_muses(self):
        '''Uses MUSES O3 optical files, which are precomputed ahead
        of the forward model. They may include a convolution with the ILS.
        '''

        res = MusesOpticalDepthFile(self.rf_uip, self.instrument_name,
                                    self.pressure,
                                    self.temperature, self.altitude,
                                    self.absorber_vmr, self.num_channel)
        return res

    @cached_property
    def absorber_xsec(self):
        '''Use the O3 cross section files for calculation absorption.
        This does not include the ILS at the absorption calculation level,
        so to get good results we should include an ILS with our forward
        model.'''
        xsectable = rf.vector_xsec_table()
        for gas in ["O3", ]:
            xsec_data = np.loadtxt(rf.cross_section_filenames[gas])
            cfac = rf.cross_section_file_conversion_factors.get(gas, 1.0)
            spec_grid = rf.ArrayWithUnit(xsec_data[:, 0], "nm")
            xsec_values = rf.ArrayWithUnit(xsec_data[:, 1:], "cm^2")
            if xsec_data.shape[1] >= 4:
                xsectable.push_back(rf.XSecTableTempDep(spec_grid, xsec_values,
                                                        cfac))
            else:
                xsectable.push_back(rf.XSecTableSimple(spec_grid, xsec_values,
                                                       cfac))
        return rf.AbsorberXSec(self.absorber_vmr, self.pressure,
                               self.temperature, self.altitude,
                               xsectable)

    @cached_property
    def absorber_absco(self):
        '''Use ABSCO tables to calculation absorption.'''
        absorptions = rf.vector_gas_absorption()
        absco_filename = self.find_absco_fname("O3_*_v0.0_init.nc")
        absorptions.push_back(rf.AbscoAer(absco_filename, 1.0, 5000,
                               rf.AbscoAer.NEAREST_NEIGHBOR_WN))
        return rf.AbsorberAbsco(self.absorber_vmr, self.pressure,
                                self.temperature,
                                self.altitude, absorptions, self.constants)

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
        res = rf.vector_altitude()
        for i in self.channel_list():
            # chan_alt = rf.AltitudeHydrostatic(self.pressure,
            #     self.temperature, self.rf_uip.latitude_with_unit(i),
            #     self.rf_uip.surface_height_with_unit(i))

            chan_alt = MusesAltitude(self.rf_uip, self.instrument_name,
                        self.pressure, self.rf_uip.latitude_with_unit(i))
            res.push_back(chan_alt)
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
        primary_absorber = "O3"
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
        res = rf.vector_vector_spectrum_effect()
        for fm_idx, ii_mw in enumerate(self.channel_list()):
            per_channel_eff = rf.vector_spectrum_effect()
            if(self.use_raman):
                per_channel_eff.push_back(self.raman_effect[fm_idx])
            res.push_back(per_channel_eff)
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
        return res

    @abc.abstractproperty
    @cached_property
    def raman_effect(self):
        raise NotImplementedError

__all__ = ["RefractorFmObjectCreator", ]
