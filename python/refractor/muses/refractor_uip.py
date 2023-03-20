from . import muses_py as mpy
from .replace_function_helper import register_replacement_function_in_block
from .refractor_capture_directory import RefractorCaptureDirectory
import refractor.framework as rf
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
import numpy as np
import glob
from weakref import WeakSet
import pickle

if(mpy.have_muses_py):
    class _FakeUipExecption(Exception):
        def __init__(self, uip, ret_info, retrieval_vec):
            self.uip = uip
            self.ret_info = ret_info
            self.retrieval_vec = retrieval_vec
        
    class _CaptureUip(mpy.ReplaceFunctionObject):
        def __init__(self, func_count=1):
            self.func_count = func_count

        def should_replace_function(self, func_name, parms):
            self.func_count -= 1
            if self.func_count <= 0:
                return True
            return False
            
        def replace_function(self, func_name, parms):
            # The UIP passed in is *before* updating with xInit. We
            # want this after the update, so call that before passing
            # value back.
            o_x_vector = parms['xInit']
            uip = parms['uip']
            ret_info = parms['ret_info']
            (uip, o_x_vector) = mpy.update_uip(uip, ret_info, o_x_vector)
            raise _FakeUipExecption(uip, ret_info, o_x_vector)

@contextmanager
def _all_output_disabled():
    '''Suppress stdout, stderr, and logging'''
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()) as sout:
            with redirect_stderr(io.StringIO()) as serr:
                yield
    finally:
        logging.disable(previous_level)

# We should perhaps add singleton/multiple observer attachment. But as
# a short term work around, we use a singleton pattern here and just keep
# a list of objects to notify
class WatchUipUpdate(mpy.ObserveFunctionObject if mpy.have_muses_py else object):
    '''Helper object to watch calls to UtilUIP.UtilUIP. This is basically
    the muses-py equivalent of StateVector.update_state in ReFRACtor.
    Unfortunately this doesn't get passed down to the forward model
    call, so we just intercept it here.  

    This object just forwards the calls to the object in the notify_set'''
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.notify_set = WeakSet()
        return cls._instance

    def add_notify_object(self, obj):
        self.notify_set.add(obj)

    def remove_notify_object(self, obj):
        self.notify_set.remove(obj)
        
    def notify_function_call(self, func_name, parms):
        fm_vec = np.matmul(parms["i_retrieval_vec"],
                           parms["i_ret_info"]["basis_matrix"])
        for obj in self.notify_set:
            obj.update_state(fm_vec, parms=parms)

class WatchUipCreation(mpy.ObserveFunctionObject if mpy.have_muses_py else object):
    '''Helper object to watch calls to make_uip_master. This gets called for
    each strategy step, and basically invalidates any cached forward 
    model we have. On the next call for RefractorFm, we'll need to create
    a new ForwardModel.

    This object just forwards the calls to the object passed in when it was
    created.'''
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.notify_set = WeakSet()
        return cls._instance

    def add_notify_object(self, obj):
        self.notify_set.add(obj)

    def remove_notify_object(self, obj):
        self.notify_set.remove(obj)
        
    def notify_function_call(self, func_name, parms):
        for obj in self.notify_set:
            obj.invalidate_cache()

# TODO Clean this up by allowing multiple observer functions in
# muses-py
if(mpy.have_muses_py):
    mpy.register_observer_function("update_uip", WatchUipUpdate())
    mpy.register_observer_function("make_uip_master", WatchUipCreation())

class RefractorUip:
    '''The 'uip' is a central variable in muses-py. It is a python dict
    object, which contains all the input data need to generate the 
    forward model. It is largely read only, but it is updated by the
    state vector (in update_uip).

    This is a light wrapper in ReFRACtor for working with the UIP.

    We give a number of access routines to various pieces of the UIP we 
    are interested in. This 1) gives a cleaner interface and 2) protects
    somewhat from changes to the uip (so changed names just need to be 
    updated here).

    Note that although some thing need access to muses_py (e.g., 
    create_from_table), a lot of this functionality doesn't actually 
    depend on muses-py. So if we have a pickled version of this object
    or the original uip, you can do things with it w/o muses-py. This
    can be useful for example for having pytest tests that don't depend
    on having muses-py available.

    Note that there are two microwindow indexes floating around. We have
    ii_mw which goes through all the instruments, so for step 7 in
    AIRS+OMI ii_mw goes through 12 values (only 10 and 11 are OMI).
    mw_index (also call fm_idx) is relative to a instrument,
    so if we are working with OMI the first microwindow has ii_mw = 10, but
    mw_index is 0 (UV1, with the second UV2).

    It isn't 100% clear what the right interface is here, so we may modify
    this class a bit in the future.'''

    def __init__(self, uip = None, strategy_table = None, ret_info = None,
                 retrieval_vec = None):
        '''Constructor. This takes the uip structure (the muses-py dictionary)
        and/or the strategy_table file name'''
        # Depending on where this is called from, uip may be a dict or
        # an ObjectView. Just to make things simpler, we always store this
        # as a dict.
        if(hasattr(uip, 'as_dict')): 
            self.uip = uip.as_dict(uip)
        else:
            self.uip = uip
        self.ret_info = ret_info
        self.retrieval_vec = retrieval_vec
        self.strategy_table = strategy_table
        self.capture_directory = RefractorCaptureDirectory()
        self.rundir = "."
        # Depending on where this comes from, it may or may not have the
        # uip_OMI stuff included. If not, add this in. 'jacobians' happens
        # to be something not in the original uip, that gets added with omi
        # This duplicates what py_retrieve does in fm_wrapper
        if('jacobians' not in self.uip and 'uip_OMI' in self.uip):
            self.uip_all = mpy.struct_combine(self.uip, self.uip['uip_OMI'])
        elif('jacobians' not in self.uip and 'uip_TROPOMI' in self.uip):
            self.uip_all = mpy.struct_combine(self.uip, self.uip['uip_TROPOMI'])
        else:
            self.uip_all = self.uip
           
    @classmethod
    def create_from_table(cls, strategy_table, step=1, capture_directory=False,
              save_pickle_file=None,
              vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli"):
        '''This creates a UIP from a run directory (e.g., created
        by earlier steps of amuse-me).  The table is passed in that
        points to everything, usually this is called 'Table.asc' in
        the run directory (e.g. ~/output_py/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_23/Table.asc).

        In addition to a uip, the muses-py code requires a number of files
        in a directory. To allow running the MusesPyForwardModel, we can
        also capture information form the directory the strategy_table is
        located at. This is only needed for MusesPyForwardModel, the 
        ReFRACtor forward model doesn't need this. You can set 
        capture_directory to True if you intend on using the UIP to run
        MusesPyForwardModel.

        Because it is common to do, you can optionally supply a pickle file
        name and we'll save the uip to the pickle file after creating it.

        Note I'm not exactly sure how to extract steps other than by
        doing a full run. What we currently do is run the retrieval
        until we get to the requested step, which can potentially be
        slow. So if you request a step other than 1, be aware that it
        might take a while to generate.  But for doing things like
        generating test data this should be fine, just pickle the
        object or otherwise save it for use later. We can probably
        work out a way to do this more directly.

        '''

        strategy_table = os.path.abspath(strategy_table)
        # We would like to just call a function in muses-py to generate
        # the UIP. Unfortunately, one doesn't exist. Instead this is
        # created inline as the processing is set up.
        #
        # We could duplicate this functionality here, but then any
        # updates to the muses-py code wouldn't show up here.
        #
        # So instead, we do a trick. We pretend like we are doing
        # a retrieval, but once the call is made to levmar_nllsq_elanor
        # we intercept this, grab the uip, and then force a return
        # by throwing an exception. This is pretty evil, an exception
        # shouldn't be used for controlling execution. But this is a
        # special case, where breaking the normal rule is the right thing.
        #
        # A better long term solution it to get muses-py to add a function
        # call.
        curdir = os.getcwd()
        old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
        mpy.cli_options.vlidort_cli=vlidort_cli
        try:
            os.environ["MUSES_DEFAULT_RUN_DIR"] = os.path.dirname(strategy_table)
            os.chdir(os.path.dirname(strategy_table))
            with register_replacement_function_in_block("levmar_nllsq_elanor",
                                 _CaptureUip(func_count=step)):
                # This is pretty noisy, so suppress printing. We can revisit
                # this if needed, but I think this is a good idea
                with _all_output_disabled() as f:
                    mpy.script_retrieval_ms(os.path.basename(strategy_table))
        except _FakeUipExecption as e:
            res = cls(uip=e.uip,strategy_table=strategy_table,
                      ret_info=e.ret_info, retrieval_vec=e.retrieval_vec)
        finally:
            if(old_run_dir):
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            else:
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            os.chdir(curdir)
        if(capture_directory):
            res.tar_directory()
        if(save_pickle_file is not None):
            pickle.dump(res, open(save_pickle_file, "wb"))
        return res

    def tar_directory(self):
        vlidort_input = None
        if("uip_OMI" in self.uip):
            vlidort_input = self.uip['uip_OMI']["vlidort_input"]
        if("uip_TROPOMI" in self.uip):
            vlidort_input = self.uip['uip_TROPOMI']["vlidort_input"]
        self.capture_directory.save_directory(os.path.dirname(self.strategy_table), vlidort_input)
        

    @property
    def vlidort_input(self):
        if(self.uip_omi):
            return self.uip_omi["vlidort_input"]
        elif(self.uip_tropomi):
            return self.uip_tropomi["vlidort_input"]
        else:
            raise RuntimeError("Only support omi and tropomi")

    @property
    def vlidort_output(self):
        if(self.uip_omi):
            return self.uip_omi["vlidort_output"]
        elif(self.uip_tropomi):
            return self.uip_tropomi["vlidort_output"]
        else:
            raise RuntimeError("Only support omi and tropomi")
        
    @classmethod
    def load_uip(cls, save_pickle_file, path=".", change_to_dir = False,
                 osp_dir=None, gmao_dir=None):
        '''This is the pair to create_from_table, it loads a RefractorUip
        from a pickle file, extracts the saved directory, and optionally
        changes to that directory.'''
        uip = pickle.load(open(save_pickle_file, "rb"))
        uip.capture_directory.extract_directory(path=path,
                              change_to_dir=change_to_dir, osp_dir=osp_dir,
                              gmao_dir=gmao_dir)
        uip.rundir = uip.capture_directory.rundir
        return uip
    
    def atmosphere_column(self, param_name):
        '''Return the atmospheric column. Note that MUSES use 
        a decreasing pressure order (to surface to TOA). This is
        the opposite of the normal ReFRACtor convention. This is
        handled by marking the pressure levels as 
        PREFER_DECREASING_PRESSURE, so this difference is handled by
        the forward model. But be aware of the difference if you
        are looking at the data directly.'''
        param_list = [ n.lower() for n in self.uip['atmosphere_params'] ]
        param_index = param_list.index(param_name.lower())
        return self.uip['atmosphere'][param_index,:]

    # I'm not positive about the design here. But for a lot of purposes
    # we really do want to separate omi from tropomi. So we have specific
    # properties here rather than indexing by the instrument type
    
    @property
    def uip_omi(self):
        '''Short cut to uip_OMI'''
        return self.uip.get('uip_OMI')

    @property
    def omi_params(self):
        '''Short cut for omiPars'''
        return self.uip.get('omiPars')

    @property
    def uip_tropomi(self):
        '''Short cut to uip_TROPOMI'''
        return self.uip.get('uip_TROPOMI')

    @property
    def tropomi_params(self):
        '''Short cut for tropomiPars'''
        return self.uip.get('tropomiPars')
    
    def measured_radiance(self, instrument_name):
        '''Note muses-py handles the radiance data in pretty much the reverse
        way that ReFRACtor does.

        For a traditional ReFRACtor retrieval, we take the reflectance and
        multiple this with the solar model to give radiance that has units.
        We then compare this against the measured radiance in our cost 
        function.

        muses-py on the other hand scales the measured radiance by the
        solar model. The cost function is then the difference between
        reflectance like values (so unitless).

        This means that the "omi_measured_radiance" here depends on the
        solar model, which in turn depends on the state vector.

        There is nothing wrong with this, but the ReFRACtor ForwardModel
        isn't currently set up to work this way. So we need to track the
        solar model state vector elements/jacobians separate from the
        ReFRACtor ForwardModel.
        '''
        if(instrument_name == "OMI"):
            rad = mpy.get_omi_radiance(self.omi_params)
            freqindex = self.uip_omi['freqIndex']
        elif(instrument_name == "TROPOMI"):
            rad = mpy.get_tropomi_radiance(self.tropomi_params)
            freqindex = self.uip_tropomi['freqIndex']
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return {
            'measured_radiance_field': rad['normalized_rad'][freqindex],  
            'measured_nesr': rad['nesr'][freqindex],
            'normwav_jac': rad['normwav_jac'][freqindex],
            'odwav_jac': rad['odwav_jac'][freqindex],
            'odwav_slope_jac': rad['odwav_slope_jac'][freqindex],
        }

    def nfreq_mw(self, mw_index, instrument_name):
        '''Number of frequencies for microwindow.'''
        if(instrument_name == "OMI"):
            # It is a bit odd that mw_index get used twice here, but this
            # really is how this is set up. So although this looks odd, it
            # is correct
            startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw"][mw_index]
        elif(instrument_name == "TROPOMI"):
            startmw_fm = self.uip_tropomi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw"][mw_index]
            
        return endmw_fm - startmw_fm + 1

    @property
    def atm_params(self):
        pangle_original = self.uip_all['obs_table']['pointing_angle']
        self.uip_all['obs_table']['pointing_angle'] = 0.0
        res = mpy.atmosphere_level(self.uip_all)
        self.uip_all['obs_table']['pointing_angle'] = pangle_original
        return res

    @property
    def ray_info(self):
        pangle_original = self.uip_all['obs_table']['pointing_angle']
        self.uip_all['obs_table']['pointing_angle'] = 0.0
        res = mpy.raylayer_nadir(mpy.ObjectView(self.uip_all),
                                 mpy.ObjectView(self.atm_params))
        self.uip_all['obs_table']['pointing_angle'] = pangle_original
        return res
    
    @property
    def omi_cloud_fraction(self):
        '''Cloud fraction for OMI'''
        return self.omi_params['cloud_fraction']

    @property
    def tropomi_cloud_fraction(self):
        '''Cloud fraction for TROPOMI'''
        return self.tropomi_params['cloud_fraction']
    
    @property
    def omi_obs_table(self):
        '''Short cut to omi_obs_table'''
        if(self.uip_omi is not None):
            return self.uip_omi["omi_obs_table"]
        return None

    @property
    def tropomi_obs_table(self):
        '''Short cut to tropomi_obs_table'''
        if(self.uip_tropomi):
            return self.uip_tropomi["tropomi_obs_table"]
        return None
    
    @property
    def number_micro_windows(self):
        '''Total number of microwindows. This is like a channel_index,
        except muses-py can retrieve multiple instruments.'''
        return len(self.uip['microwindows_all'])

    def instrument_name(self, ii_mw):
        '''Instrument name for the micro_window index ii_mw'''
        return self.uip['microwindows_all'][ii_mw]['instrument']
    
    def micro_windows(self, ii_mw):
        '''Return start and end of microwindow'''
        return rf.ArrayWithUnit(np.array([[
            self.uip['microwindows_all'][ii_mw]['start'], 
            self.uip['microwindows_all'][ii_mw]['endd']
            ],]), "nm")

    @property
    def state_vector_params(self):
        '''List of parameter types to include in the state vector.'''
        if(self.uip_omi is not None):
            return self.uip_omi['jacobians']
        elif(self.uip_tropomi is not None):
            return self.uip_tropomi['jacobians']
        return []
        

    @property
    def state_vector_names(self):
        '''Full list of the name for each state vector list item'''
        sv_list = []
        for jac_name in self.uip['speciesListFM']:
            if jac_name in self.state_vector_params:
                sv_list.append(jac_name)
        return sv_list

    @property
    def state_vector_update_indexes(self):
        '''Indexes for this instrument's state vector element updates from the full update vector'''

        sv_extract_index = []
        for full_idx, jac_name in enumerate(self.uip['speciesListFM']):
            if jac_name in self.state_vector_params:
                sv_extract_index.append(full_idx)

        return np.array(sv_extract_index)

    @property
    def earth_sun_distance(self):
        '''Earth sun distance, in meters. Right now this is OMI specific'''
        # Same value for all the bands, so just grab the first one
        if(self.omi_obs_table is not None):
            return self.omi_obs_table['EarthSunDistance'][0]
        elif(self.tropomi_obs_table is not None):
            return self.tropomi_obs_table['EarthSunDistance'][0]
        else:
            RuntimeError("Didn't find a observation table")
        
    def sample_grid(self, mw_index, ii_mw):
        '''This is the full set of samples. We only actually use a subset of
        these, but these are the values before the microwindow gets applied.

        Right now this is omi specific.'''

        if self.ils_method(mw_index,self.instrument_name(ii_mw)) == "FASTCONV":
            ils_uip_info = self.ils_params(mw_index)

            return rf.SpectralDomain(ils_uip_info["central_wavelength"], rf.Unit("nm"))
        else:
            if(self.instrument_name(ii_mw) == "OMI"):
                all_freq = self.uip_omi['fullbandfrequency']
                filt_loc = np.array(self.uip_omi['frequencyfilterlist'])
            elif(self.instrument_name(ii_mw)):
                all_freq = self.uip_tropomi['fullbandfrequency']
                filt_loc = np.array(self.uip_tropomi['frequencyfilterlist'])
            else:
                raise RuntimeError(f"Invalid instrument {self.instrument_name(ii_mw)}")
            return rf.SpectralDomain(all_freq[np.where(filt_loc == self.filter_name(ii_mw))], rf.Unit("nm"))

    def muses_fm_spectral_domain(self, mw_index):
        '''
        NOTE - The logic here has changed in py-retrieve. We only ever
        used this for the Raman calculation, and it isn't used there 
        anymore. Leave this is place for now, but we should be able to
        remove this once all the old code gets updated. Look to
        raman_spectral_domain instead.

        Wavelengths do to do the forward model on. This is read from
        the ILS. Not sure how this compares to what we already get from
        the base_config or OmiForwardModel, but for now we give separate
        access to this.

        Right now this is omi specific.'''

        ils_uip_info = self.ils_params(mw_index)

        if self.ils_method(mw_index) == "FASTCONV":
            return rf.SpectralDomain(ils_uip_info["monochromgrid"], rf.Unit("nm"))
        else:
            return rf.SpectralDomain(ils_uip_info["X0_fm"], rf.Unit("nm"))

    def ils_params(self, mw_index, instrument_name):
        '''Returns ILS information for the given microwindow'''
        if(instrument_name == "OMI"):
            return self.uip_omi["ils_%02d" % (mw_index+1)]
        elif(instrument_name == "TROPOMI"):
            return self.uip_tropomi["ils_%02d" % (mw_index+1)]
        else: 
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def ils_method(self, mw_index, instrument_name):
        '''Returns a string describing the ILS method configured by MUSES'''
        if(instrument_name == "OMI"):
            return self.uip_omi['ils_omi_xsection']
        elif(instrument_name == "TROPOMI"):
            return self.uip_tropomi['ils_tropomi_xsection']
        else: 
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def radiance_info(self, mw_index, instrument_name):
        '''This is a bit convoluted. It comes from a python pickle file that
        gets created before the retrieval starts. So this is 
        "control coupling". On the other hand, most of the UIP is sort of 
        control coupling, so for now we'll just live with this.

        We 1) want to just directly evaluate this using ReFRACtor code or 
        2) track down what exactly py-retrieve is doing to create this and
        do it directly. 
        '''
        input_directory = f"{self.rundir}/Input/"
        if(not os.path.exists(input_directory)):
            raise RuntimeError(f"Input directory {input_directory} not found.")
        if(instrument_name == "OMI"):
            fname = glob.glob(input_directory + "Radiance_OMI*.pkl")[0]
        elif(instrument_name == "TROPOMI"):
            fname = glob.glob(input_directory + "Radiance_TROPOMI*.pkl")[0]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return pickle.load(open(fname, "rb"))

    def mw_fm_slice(self, mw_index, instrument_name):
        '''This is the portion of the full microwindow frequencies that we are
        using in calculations such as RamanSioris. This is a bit
        bigger than the instrument_spectral_domain in
        RefractorObjectCreator, which is the range fitted in the
        retrieval. This has extra padding for things like the
        RamanSioris calculation'''
        if(instrument_name == "OMI"):
            startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw_fm"][mw_index]
            endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw_fm"][mw_index]
        elif(instrument_name == "TROPOMI"):
            startmw_fm = self.uip_tropomi["microwindows"][mw_index]["startmw_fm"][mw_index]
            endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw_fm"][mw_index]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return slice(startmw_fm, endmw_fm+1)
        
    def full_band_frequency(self, instrument_name):
        '''This is the full frequency range for the instrument. I believe
        this is the same as the wavelengths found in the radiance pickle
        file (self.radiance_info), but this comes for a different source in
        the UIP object so we have this in case this is somehow different.'''
        if(instrument_name == "OMI"):
            return self.uip_omi["fullbandfrequency"]
        elif(instrument_name == "TROPOMI"):
            return self.uip_tropomi["fullbandfrequency"]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
            
    def rad_wavelength(self, mw_index, instrument_name):

        '''This is the wavelengths that the L1B data was measured at, truncated
        to fit our microwindow'''
        slc = self.mw_fm_slice(mw_index, instrument_name)
        rad_info = self.radiance_info(mw_index, instrument_name)
        return rf.SpectralDomain(rad_info['Earth_Radiance']['Wavelength'][slc],
                                 rf.Unit("nm"))        

    def solar_irradiance(self, mw_index, instrument_name):
        '''This is currently just used for the Raman calculation of the 
        RefractorRtfOmi class. This has been adjusted for the 
        '''
        slc = self.mw_fm_slice(mw_index, instrument_name)
        rad_info = self.radiance_info(mw_index, instrument_name)

        # Note this looks wrong (why not use Solar_Radiance Wavelength here?),
        # but is actually correct. The solar data has already been interpolated
        # to the same wavelengths as  the Earth_Radiance, this happens in
        # daily_tropomi_irad for TROPOMI, and similarly for OMI. Not sure
        # why the original wavelengths are left in rad_info['Solar_Radiance'],
        # that is actually misleading.
        sol_domain = rf.SpectralDomain(rad_info['Earth_Radiance']['Wavelength'][slc],
                                       rf.Unit("nm"))
        sol_range = rf.SpectralRange(rad_info['Solar_Radiance']['AdjustedSolarRadiance'][slc],
                                     rf.Unit("ph / nm / s"))
        return rf.Spectrum(sol_domain, sol_range)

    def filter_name(self, ii_mw):
        '''The filter name (e.g., UV1 or UV2)'''
        return self.uip['microwindows_all'][ii_mw]['filter']
        
    def channel_indexes(self, ii_mw):
        '''Determine the channel indexes that we are processing.
        '''
        # You would think this would just be an argument, but it
        # isn't. We need to get the filter name from one place, and
        # use that to look up the channel index in another.
        if(self.instrument_name(ii_mw) == "OMI"):
            return np.where(np.asarray(self.omi_obs_table["Filter_Band_Name"])
                            == self.filter_name(ii_mw))[0]
        if(self.instrument_name(ii_mw) == "TROPOMI"):
            return np.where(np.asarray(self.tropomi_obs_table["Filter_Band_Name"])
                            == self.filter_name(ii_mw))[0]
        else:
            raise RuntimeError("Don't know how to find observation table")

    def _avg_obs(self, nm, ii_mw):
        '''Average values that match the self.channel_indexes. 

        Not sure if this makes sense or not, but it is what py_retrieve 
        does.

        Right now this is omi specific'''
        if(self.omi_obs_table):
            return np.mean(np.asarray(self.omi_obs_table[nm])[self.channel_indexes(ii_mw)])
        if(self.tropomi_obs_table):
            return np.mean(np.asarray(self.tropomi_obs_table[nm])[self.channel_indexes(ii_mw)])
        raise RuntimeError("Don't know how to find observation table")
      
    def observation_zenith(self, ii_mw):
        '''Observation zenith angle for the microwindow index ii_mw'''
        return self._avg_obs("ViewingZenithAngle", ii_mw)

    def observation_zenith_with_unit(self, ii_mw):
        '''Observation zenith angle for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.observation_zenith(ii_mw)), "deg")

    def observation_azimuth(self, ii_mw):
        '''Observation azimuth angle for the microwindow index ii_mw'''
        return self._avg_obs("ViewingAzimuthAngle", ii_mw)

    def observation_azimuth_with_unit(self, ii_mw):
        '''Observation azimuth angle for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.observation_azimuth(ii_mw)), "deg")

    def solar_azimuth(self, ii_mw):
        '''Solar azimuth angle for the microwindow index ii_mw'''
        return self._avg_obs("SolarAzimuthAngle", ii_mw)

    def solar_azimuth_with_unit(self, ii_mw):
        '''Solar azimuth angle for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.solar_azimuth(ii_mw)), "deg")

    def solar_zenith(self, ii_mw):
        '''Solar zenith angle for the microwindow index ii_mw'''
        return self._avg_obs("SolarZenithAngle", ii_mw)

    def solar_zenith_with_unit(self, ii_mw):
        '''Solar zenith angle for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.solar_zenith(ii_mw)), "deg")

    def relative_azimuth(self, ii_mw):
        '''Relative azimuth angle for the microwindow index ii_mw'''
        return self._avg_obs("RelativeAzimuthAngle", ii_mw)

    def relative_azimuth_with_unit(self, ii_mw):
        '''Relative azimuth angle for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.relative_azimuth(ii_mw)), "deg")
    
    def latitude(self, ii_mw):
        '''Latitude for the microwindow index ii_mw'''
        return self._avg_obs("Latitude", ii_mw)

    def latitude_with_unit(self, ii_mw):
        '''Latitude for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.latitude(ii_mw)), "deg")
    
    def longitude(self, ii_mw):
        '''Longitude for the microwindow index ii_mw'''
        return self._avg_obs("Longitude", ii_mw)

    def longitude_with_unit(self, ii_mw):
        '''Longitude for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.longitude(ii_mw)), "deg")

    def surface_height(self, ii_mw):
        '''Surface height for the microwindow index ii_mw'''
        return self._avg_obs("TerrainHeight", ii_mw)

    def surface_height_with_unit(self, ii_mw):
        '''Surface height for the microwindow index ii_mw'''
        return rf.DoubleWithUnit(float(self.surface_height(ii_mw)), "m")
    
    def across_track_indexes(self, ii_mw):
        '''Across track indexes for the microwindow index ii_mw.

        Right now this is omi specific'''
        # Can't really average these to have anything that makes sense.
        # So for now we just pick the first one that matches
        if(self.omi_obs_table):
            return np.asarray(self.omi_obs_table["XTRACK"])[self.channel_indexes(ii_mw)]
        if(self.tropomi_obs_table):
            return np.asarray(self.tropomi_obs_table["XTRACK"])[self.channel_indexes(ii_mw)]
        raise RuntimeError("Don't know how to find observation table")

    def update_uip(self, retrieval_vec):
        '''This updates the underlying UIP with the new retrieval_vec, e.g., this is
        the py-retrieve equivalent up updating the StateVector in ReFRACtor.

        Note that this is the retrieval vector, not the state vector.'''
        self.retrieval_vec = np.copy(retrieval_vec)
        self.uip, _ = mpy.update_uip(self.uip, self.ret_info, retrieval_vec)
        if(hasattr(self.uip, 'as_dict')): 
            self.uip = self.uip.as_dict(self.uip)
        if('jacobians' not in self.uip and 'uip_OMI' in self.uip):
            self.uip_all = mpy.struct_combine(self.uip, self.uip['uip_OMI'])
        elif('jacobians' not in self.uip and 'uip_TROPOMI' in self.uip):
            self.uip_all = mpy.struct_combine(self.uip, self.uip['uip_TROPOMI'])
        else:
            self.uip_all = self.uip


__all__ = ["RefractorUip", "WatchUipCreation", "WatchUipUpdate"]            
