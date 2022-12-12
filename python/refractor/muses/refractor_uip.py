from . import muses_py as mpy
from .replace_function_helper import register_replacement_function_in_block
import refractor.framework as rf
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
import numpy as np
import glob
from weakref import WeakSet
import pickle
import tarfile

if(mpy.have_muses_py):
    class _FakeUipExecption(Exception):
        def __init__(self, uip):
            self.uip = uip
        
    class _CaptureUip(mpy.ReplaceFunctionObject):
        def __init__(self, func_count=1):
            self.func_count = func_count

        def should_replace_function(self, func_name, parms):
            self.func_count -= 1
            if self.func_count <= 0:
                return True
            return False
            
        def replace_function(self, func_name, parms):
            raise _FakeUipExecption(parms['uip'])

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

    Right now a number of things are OMI specific. As we get more experience
    with the uip hopefully we can relax some of this.

    Note that although some thing need access to muses_py (e.g., 
    create_from_table), a lot of this functionality doesn't actually 
    depend on muses-py. So if we have a pickled version of this object
    or the original uip, you can do things with it w/o muses-py. This
    can be useful for example for having pytest tests that don't depend
    on having muses-py available.

    It isn't 100% clear what the right interface is here, so we may modify
    this class a bit in the future.'''

    def __init__(self, uip = None, strategy_table = None):
        '''Constructor. This takes the uip structure (the muses-py dictionary)
        and/or the strategy_table file name'''
        # Depending on where this is called from, uip may be a dict or
        # an ObjectView. Just to make things simpler, we always store this
        # as a dict.
        if(hasattr(uip, 'as_dict')) :
            self.uip = uip.as_dict()
        else:
            self.uip = uip
        self.strategy_table = strategy_table
        self.capture_directory = None
        # Depending on where this comes from, it may or may not have the
        # uip_OMI stuff included. If not, add this in. 'obs_table' happens
        # to be something not in the original uip, that gets added with omi
        if('jacobians' not in self.uip):
            self.uip_all = mpy.struct_combine(self.uip, self.uip['uip_OMI'])
        else:
            self.uip_all = self.uip
           

    def tar_directory(self):
        '''Capture information from the directory that self.strategy_table is 
        in so we can recreate the directory later. This is only needed by
        muses-py which uses a lot of files as "hidden" arguments to functions.
        ReFRACtor doesn't need this.'''
        fh = io.BytesIO()
        dirbase = os.path.dirname(self.strategy_table)
        relpath = "./" + os.path.basename(dirbase)
        relpath2 = "./OSP/OMI"
        with tarfile.open(fileobj=fh, mode="x:bz2") as tar:
            for f in ("RamanInputs", "Input", "StepFM/VLIDORTInput"):
                tar.add(f"{dirbase}/{f}", f"{relpath}/{f}")
            for f in ("omi_rtm_driver", "ring"):
                tar.add(f"{dirbase}/{f}", f"{relpath2}/{f}")
        self.capture_directory = fh.getvalue()

    def extract_directory(self, path=".", change_to_dir = False,
                          osp_dir=None):
        '''Extract a directory that has been previously saved.
        This gets extracted into the directory passed in the path. You can
        optionally change into the run directory.

        For pretty much everything below run_retrieval, the small OSP content
        we have stashed is sufficient to run. But for higher level functions,
        you need the full OSP directory. We don't carry this in this class,
        but if you supply a osp_dir we use that instead of the OSP we have
        stashed.'''
        if(self.capture_directory is None):
            raise RuntimeError("extract_directory can only be called if this object previously captured a directory")
        fh = io.BytesIO(self.capture_directory)
        with tarfile.open(fileobj=fh, mode="r:bz2") as tar:
            tar.extractall(path=path)
        if(osp_dir is not None):
            os.rename(path + "/OSP", path + "/OSP_not_used")
            os.symlink(osp_dir, path + "/OSP")
        if(change_to_dir):
            runbase = os.path.basename(os.path.dirname(self.strategy_table))
            rundir = os.path.abspath(path + "/" + runbase)
            os.environ["MUSES_DEFAULT_RUN_DIR"] = rundir
            os.chdir(rundir)
                
    @classmethod
    def create_from_table(cls, strategy_table, step=1, capture_directory=False):
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
            res = cls(uip=e.uip,strategy_table=strategy_table)
        finally:
            if(old_run_dir):
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            else:
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            os.chdir(curdir)
        if(capture_directory):
            res.tar_directory()
        return res

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

    @property
    def uip_omi(self):
        '''Short cut to uip_OMI'''
        return self.uip['uip_OMI']

    @property
    def omi_params(self):
        '''Short cut for omiPars'''
        return self.uip['omiPars']

    @property
    def omi_radiance(self):
        '''Results of mpy.get_omi_radiance'''
        return mpy.get_omi_radiance(self.omi_params)

    @property
    def omi_measured_radiance(self):
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
        omirad = self.omi_radiance
        return {
            'measured_radiance_field': omirad['normalized_rad'][self.uip_omi['freqIndex']],  
            'measured_nesr': omirad['nesr'][self.uip_omi['freqIndex']],
            'normwav_jac': omirad['normwav_jac'][self.uip_omi['freqIndex']],
            'odwav_jac': omirad['odwav_jac'][self.uip_omi['freqIndex']],
            'odwav_slope_jac': omirad['odwav_slope_jac'][self.uip_omi['freqIndex']],
        }

    def nfreq_mw(self, ii_mw):
        '''Number of frequencies for microwindow.'''
        startmw_fm = self.uip_omi["microwindows"][ii_mw]["startmw"][ii_mw]
        endmw_fm = self.uip_omi["microwindows"][ii_mw]["enddmw"][ii_mw]
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
    def omi_obs_table(self):
        '''Short cut to omi_obs_table'''
        return self.uip_omi["omi_obs_table"]

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
    def state_vector_list(self):
        '''List of state vector elements to include'''
        jacall = self.uip["jacobians_all"]
        sv_list = [jacall[i] for i in range(len(jacall)) if
                   i == 0 or (jacall[i] != jacall[i-1])]
        return sv_list
        
    @property
    def earth_sun_distance(self):
        '''Earth sun distance, in meters. Right now this is OMI specific'''
        # Same value for all the bands, so just grab the first one
        return self.omi_obs_table['EarthSunDistance'][0]
        
    def sample_grid(self, ii_mw):
        '''This is the full set of samples. We only actually use a subset of
        these, but these are the values before the microwindow gets applied.

        Right now this is omi specific.'''
        res = []
        all_freq = self.uip_omi['fullbandfrequency']
        filt_loc = np.array(self.uip_omi['frequencyfilterlist'])
        return rf.SpectralDomain(all_freq[np.where(filt_loc == self.filter_name(ii_mw))], rf.Unit("nm"))

    def muses_fm_spectral_domain(self, ii_mw):
        '''Wavelengths do to do the forward model on. This is read from
        the ILS. Not sure how this compares to what we already get from
        the base_config or OmiForwardModel, but for now we give separate
        access to this.

        Right now this is omi specific.'''
        return rf.SpectralDomain(
            self.uip_omi["ils_%02d" % (ii_mw+1)]["X0_fm"], rf.Unit("nm"))

    def solar_irradiance(self, ii_mw, input_directory):
        '''This is a bit convoluted. It comes from a python pickle file that
        gets created before the retrieval starts. So this is 
        "control coupling". On the other hand, most of the UIP is sort of 
        control coupling, so for now we'll just live with this.

        We 1) want to just directly evaluate this using ReFRACtor code or 
        2) track down what exactly py-retrieve is doing to create this and
        do it directly. 

        This is currently just used for the Raman calculation of the 
        RefractorRtfOmi class
        '''
        fname = glob.glob(input_directory + "Radiance_OMI*.pkl")[0]
        omi_info = pickle.load(open(fname, "rb"))
        startmw_fm = self.uip_omi["microwindows"][ii_mw]["startmw_fm"][ii_mw]
        endmw_fm = self.uip_omi["microwindows"][ii_mw]["enddmw_fm"][ii_mw]
        return omi_info['Solar_Radiance']['AdjustedSolarRadiance'][slice(startmw_fm, endmw_fm+1)]

    def filter_name(self, ii_mw):
        '''The filter name (e.g., UV1 or UV2)'''
        return self.uip['microwindows_all'][ii_mw]['filter']
        
    def channel_indexes(self, ii_mw):
        '''Determine the channel indexes that we are processing.

        Right now this is omi specific.
        '''
        # You would think this would just be an argument, but it
        # isn't. We need to get the filter name from one place, and
        # use that to look up the channel index in another.
        return np.where(np.asarray(self.omi_obs_table["Filter_Band_Name"])
                        == self.filter_name(ii_mw))[0]

    def _avg_obs(self, nm, ii_mw):
        '''Average values that match the self.channel_indexes. 

        Not sure if this makes sense or not, but it is what py_retrieve 
        does.

        Right now this is omi specific'''
        return np.mean(np.asarray(self.omi_obs_table[nm])[self.channel_indexes(ii_mw)])

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
        return rf.DoubleWithUnit(float(self.observation_azimuth(ii_mw)), "deg")
        '''Observation azimuth angle for the microwindow index ii_mw'''

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
        return np.asarray(self.omi_obs_table["XTRACK"])[self.channel_indexes(ii_mw)]


__all__ = ["RefractorUip", "WatchUipCreation", "WatchUipUpdate"]            
