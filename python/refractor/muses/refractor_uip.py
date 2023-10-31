from . import muses_py as mpy
from .replace_function_helper import register_replacement_function_in_block
from .refractor_capture_directory import RefractorCaptureDirectory
from .constant_dict import ConstantDict
from .osswrapper import osswrapper
import refractor.framework as rf
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
import numpy as np
import glob
import pickle
from collections import UserDict, defaultdict
import copy

if(mpy.have_muses_py):
    class _FakeUipExecption(Exception):
        def __init__(self, uip, ret_info, retrieval_vec):
            self.uip = uip
            self.ret_info = ret_info
            self.retrieval_vec = retrieval_vec
        
    class _CaptureUip(mpy.ReplaceFunctionObject):
        '''Note a complication. For CrIS-TROPOMI we have some steps that
        don't actually call levmar_nllsq_elanor. So we get the registered
        twice, once to run_retrieval and once to levmar_nllsq_elanor. We then
        count the number of run_retrieval calls, but once the total is reached
        we replace only the *next* levmar_nllsq_elanor call.'''
        def __init__(self, func_count=1):
            self.func_count = func_count

        def should_replace_function(self, func_name, parms):
            if func_name == "run_retrieval":
                self.func_count -= 1
                print(f"In run_retrieval, func_count is {self.func_count}")
            if self.func_count <= 0 and func_name == "levmar_nllsq_elanor":
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

class RefractorCache(UserDict):
    '''We ran into an issue where run_retrieval in muses-py tries to
    do a deepcopy of the final UIP and we had a failure because
    refractor_cache can't be deepcopied (some of the object can't be
    pickled, which is what deepcopy does).

    So we use a cache object that is just dict but returns an empty
    cache when we do a deepcopy.

    Even if we fix the pickle issue, this is probably what we want to
    do anyways - the cache really is just that. If we don't have an object
    already in the cache our code just recreates it, we just use the cache
    to be able to reuse objects between calls to ReFRACtor from muses-py.

    '''
    def __deepcopy__(self, memo):
        return RefractorCache()
    
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

    Note that the UIP doesn't include the basis matrix needed to map
    from the retrieval vector to the full model vector ('z' and 'x' in
    the notation of the TES paper “Tropospheric Emission Spectrometer:
    Retrieval Method and Error Analysis” (IEEE TRANSACTIONS ON
    GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY 2006). There
    really isn't another natural place to put this, so we stash this
    matrix into this class. Depending on the call chain, this may or
    may not be available, so code should check if this is None and somehow
    handle this (including throwing an exception. We store this
    as basis_matrix

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
    this class a bit in the future.

    '''

    def __init__(self, uip = None, basis_matrix = None):
        '''Constructor. This takes the uip structure (the muses-py dictionary)
        and a basis_matrix if available
        '''
        # Depending on where this is called from, uip may be a dict or
        # an ObjectView. Just to make things simpler, we always store this
        # as a dict.
        if(hasattr(uip, 'as_dict')): 
            self.uip = uip.as_dict(uip)
        else:
            self.uip = uip
        # Thought this would be useful to make sure UIP isn't changed
        # behind the scenes. But this turns out to break a bunch of
        # stuff. We could probably eventually sort this out, but this
        # seems like a lot or work for little gain. Instead, we can
        # just check by inspection if the UIP is changed in any
        # muses-py code self.uip = ConstantDict(self.uip)
        self.basis_matrix = basis_matrix
        self.capture_directory = RefractorCaptureDirectory()

    def __getstate__(self):
        '''Pickling grabs attributes, which includes properties.
        We don't actually want that, so just explicitly list what
        we want saved.'''
        return {'uip' : self.uip, 'basis_matrix' : self.basis_matrix,
                'capture_directory' : self.capture_directory}
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def uip_all(self, instrument_name):
        '''Add in the stuff for the given instrument name. This is
        used in a number of places in muses-py calls.'''
        # Depending on where this comes from, it may or may not have the
        # uip_all stuff included.
        # 'jacobians' happens
        # to be something not in the original uip, that gets added with
        # uip_all
        if('jacobians' in self.uip):
            return self.uip
        return mpy.struct_combine(self.uip, self.uip[f'uip_{instrument_name}']) 

    @property
    def run_dir(self):
        '''Return run_dir for capture_directory. Note this defaults to
        "." if RefractorCaptureDirectory hasn't changed this.'''
        return self.capture_directory.rundir

    @run_dir.setter
    def run_dir(self, v):
        self.capture_directory.rundir = v
    
    @property
    def refractor_cache(self):
        '''Return a simple dict we use for caching values. Note that
        by design this really is a cache, if this is missing or
        anything in it is then we just create on first use. Note this
        is the equivalent of a "mutable" in C++ - we allow things to
        get updated in the cache in places that should otherwise want
        the UIP to be held constant.
        '''
        if("refractor_cache" not in self.uip or
           self.uip['refractor_cache'] is None):
            self.uip["refractor_cache"] = RefractorCache()
        return self.uip["refractor_cache"]
    
    @classmethod
    def create_from_table(cls, strategy_table, step=1, capture_directory=False,
              save_pickle_file=None,
              vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
              suppress_noisy_output=True):
        '''This creates a UIP from a run directory (e.g., created
        by earlier steps of amuse-me).  The table is passed in that
        points to everything, usually this is called 'Table.asc' in
        the run directory (e.g. ~/output_py/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_23/Table.asc).

        In addition to a uip, the muses-py code requires a number of
        files in a directory. To allow running the
        e.g. MusesTropomiForwardModell, we can also capture
        information form the directory the strategy_table is located
        at. This is only needed for muses-py code, the ReFRACtor
        forward model doesn't need this. You can set capture_directory
        to True if you intend on using the UIP to run muses-py code.

        Because it is common to do, you can optionally supply a pickle file
        name and we'll save the uip to the pickle file after creating it.

        Note I'm not exactly sure how to extract steps other than by
        doing a full run. What we currently do is run the retrieval
        until we get to the requested step, which can potentially be
        slow. So if you request a step other than 1, be aware that it
        might take a while to generate.  But for doing things like
        generating test data this should be fine, just pickle the
        object or otherwise save it for use later. We can probably
        work out a way to do this more directly if it becomes important.
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
            cfun = _CaptureUip(func_count=step)
            with register_replacement_function_in_block("run_retrieval", cfun):
                with register_replacement_function_in_block("levmar_nllsq_elanor", cfun):
                    # This is pretty noisy, so suppress printing. We can revisit
                    # this if needed, but I think this is a good idea
                    if(suppress_noisy_output):
                        with _all_output_disabled() as f:
                            mpy.script_retrieval_ms(os.path.basename(strategy_table))
                    else:
                        mpy.script_retrieval_ms(os.path.basename(strategy_table))
        except _FakeUipExecption as e:
            res = cls(uip=e.uip,
                      basis_matrix = e.ret_info["basis_matrix"])
        finally:
            if(old_run_dir):
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            else:
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            os.chdir(curdir)
        if(capture_directory):
            res.tar_directory(strategy_table)
        if(save_pickle_file is not None):
            pickle.dump(res, open(save_pickle_file, "wb"))
        return res

    def tar_directory(self, strategy_table):
        vlidort_input = None
        if("uip_OMI" in self.uip):
            vlidort_input = self.uip['uip_OMI']["vlidort_input"]
        if("uip_TROPOMI" in self.uip):
            vlidort_input = self.uip['uip_TROPOMI']["vlidort_input"]
        self.capture_directory.save_directory(os.path.dirname(strategy_table), vlidort_input)
        
    @property
    def current_state_x(self):
        '''Return the current guess. This is the same thing as retrieval_vec,
        update_uip sets this so we know this.'''
        return self.uip["currentGuessList"]

    @property
    def current_state_x_fm(self):
        '''Return the current guess for the full state model (called fm_vec
        in some places) This is the same thing as retrieval_vec @ basis_matrix
        update_uip sets this so we know this.'''
        return self.uip["currentGuessListFM"]
    
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
        return uip

    def instrument_sub_basis_matrix(self, instrument_name, use_full_state_vector=False):
        '''Return the portion of the basis matrix that includes jacobians
        for the given instrument. This is what the various muses-py forward
        models return - only the subset of jacobians actually relevant for
        that instrument.
        '''
        if(not use_full_state_vector):
            return self.basis_matrix[:,[t in list(self.state_vector_params(instrument_name)) for t in self.uip["speciesListFM"]]]
        bmatrix = np.eye(len(self.uip["speciesListFM"]))
        return bmatrix[:,[t in list(self.state_vector_params(instrument_name)) for t in self.uip["speciesListFM"]]]

    @property
    def is_bt_retrieval(self):
        '''For BT retrievals, the species aren't set. This means we
        need to do special handling in some cases. Determine if we are
        doing a BT retrieval and return True if we are.'''
        # Note the logic is a bit obscure here, but this matches what
        # fm_wrapper does. If the speciesListFM is ['',] then we just
        # "know" that this is a BT retrieval
        return (len(self.uip["speciesListFM"])  == 0 or
                (len(self.uip["speciesListFM"]) == 1 and
                 self.uip["speciesListFM"] == ['',]))
    
    def species_basis_matrix(self, species_name):
        '''Muses does the retrieval on a subset of the full forward model
        grid. The mapping between the two sets is handled by the
        basis_matrix. We subset this for just this particular species_name
        (e.g, O3).'''
        t1 = np.array(self.uip['speciesList']) == species_name
        t2 = np.array(self.uip['speciesListFM']) == species_name
        return self.basis_matrix[t1,:][:,t2]

    def species_basis_matrix_calc(self, species_name):
        '''Rather than return the basis matrix in ret_info, calculate
        this like get_species_information does in muses-py.

        Note that this is a bit circular, we use
        species_retrieval_level_subset which depends on self.basis_matrix
        (because we don't have this information available at this level of
        the processing tree).

        But go ahead and have this function, it is a nice documentation
        of how we would possibly move this calculation into refractor, and
        that our data is consistent.'''
        # Note this is in Pa rather than hPa. make_maps expects this, so
        # it is consistent. But this is different than what refractor uses
        # elsewhere.
        plev = self.atmosphere_column("pressure")
        # +1 here is because make_maps is expecting 1 based levels rather
        # the 0 based we return from species_retrieval_level_subset.
        return mpy.make_maps(plev, self.species_retrieval_level_subset(species_name)+1)['toState']
    
    def species_retrieval_level_subset(self, species_name):
        '''This is the levels of the forward model grid that we do
        the retrieval on.

        It would be nice to get this directly, this is a value
        determined by get_species_information, which is called by
        script_retrieval_ms. But for now, we can indirectly back
        out this information by looking at the structure of the
        basis_matrix.

        Note that this is 0 based, although the py_retrieve function is
        in terms of 1 based. 
        '''
        i_levels = np.any(self.species_basis_matrix(species_name) == 1,
                          axis=0).nonzero()[0]
        return i_levels
    
    def atmosphere_column(self, species_name):
        '''Return the atmospheric column. Note that MUSES use 
        a decreasing pressure order (to surface to TOA). This is
        the opposite of the normal ReFRACtor convention. This is
        handled by marking the pressure levels as 
        PREFER_DECREASING_PRESSURE, so this difference is handled by
        the forward model. But be aware of the difference if you
        are looking at the data directly.'''
        param_list = [ n.lower() for n in self.uip['atmosphere_params'] ]
        param_index = param_list.index(species_name.lower())
        return self.uip['atmosphere'][param_index,:]

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

    def frequency_list(self, instrument_name):
        return self.uip[f"uip_{instrument_name}"]["frequencyList"]

    @property
    def instrument_list(self):
        '''List of all the radiance data we are generating, identifying
        which instrument fills in that particular index'''
        return self.uip["instrumentList"]

    @property
    def instrument(self):
        '''List of instruments that are part of the UIP'''
        return self.uip["instruments"]
    
    def freq_index(self, instrument_name):
        '''Return frequency index for given instrument'''
        if(instrument_name == "OMI"):
            return self.uip_omi['freqIndex']
        elif(instrument_name == "TROPOMI"):
            return self.uip_tropomi['freqIndex']
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
    
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
        elif(instrument_name == "TROPOMI"):
            rad = mpy.get_tropomi_radiance(self.tropomi_params)
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        freqindex = self.freq_index(instrument_name)
        return {
            'wavelength'  : rad['wavelength'][freqindex],
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

    def atm_params(self, instrument_name, set_pointing_angle_zero=True):
        uall = self.uip_all(instrument_name)
        # tropomi_fm and omi_fm set this to zero before calling raylayer_nadir.
        # I'm not sure if always want to do this or not. Note that uall
        # is a copy of uip, so no need to set this back.
        if(set_pointing_angle_zero):
            uall['obs_table']['pointing_angle'] = 0.0
        return mpy.atmosphere_level(uall)

    def ray_info(self, instrument_name, set_pointing_angle_zero=True):
        uall = self.uip_all(instrument_name)
        # tropomi_fm and omi_fm set this to zero before calling raylayer_nadir.
        # I'm not sure if always want to do this or not. Note that uall
        # is a copy of uip, so no need to set this back.
        if(set_pointing_angle_zero):
            uall['obs_table']['pointing_angle'] = 0.0
        return mpy.raylayer_nadir(mpy.ObjectView(uall),
                         mpy.ObjectView(mpy.atmosphere_level(uall)))
    
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
    def jacobian_all(self):
        '''List of jacobians we are including in the state vector'''
        return self.uip['jacobians_all']
    
    def state_vector_species_index(self, species_name,
                                   use_full_state_vector=False):
        '''Index and length for the location of the species_name in
        our state vector. We either do this for the retrieval state vector
        or the full state vector.'''
        if(self.is_bt_retrieval):
            # Special handling for BT retrieval. BTW, this is really just
            # sort of a "magic" logic in fm_wrapper, there is nothing that
            # indicates the length is 1 here except the hard coded logic
            # in fm_wrapper.
            pstart = 0
            plen = 1
        elif(not use_full_state_vector):
            pstart = list(self.uip["speciesList"]).index(species_name)
            plen = list(self.uip["speciesList"]).count(species_name)
        else:
            pstart = list(self.uip["speciesListFM"]).index(species_name)
            plen = list(self.uip["speciesListFM"]).count(species_name)
        return pstart, plen
        

    def state_vector_params(self, instrument_name):
        '''List of parameter types to include in the state vector.'''
        return self.uip[f"uip_{instrument_name}"]["jacobians"]

    def state_vector_names(self, instrument_name):
        '''Full list of the name for each state vector list item'''
        sv_list = []
        for jac_name in self.uip['speciesListFM']:
            if jac_name in self.state_vector_params(instrument_name):
                sv_list.append(jac_name)
        return sv_list

    def state_vector_update_indexes(self, instrument_name):
        '''Indexes for this instrument's state vector element updates from the full update vector'''
        sv_extract_index = []
        for full_idx, jac_name in enumerate(self.uip['speciesListFM']):
            if jac_name in self.state_vector_params(instrument_name):
                sv_extract_index.append(full_idx)

        return np.array(sv_extract_index)

    def species_lin_log_mapping(self, specie_name: str) -> str:
        output_map = None

        # JLL: figured we might as well go through this process of checking the UIP each call rather than doing any caching,
        # that way if the UIP changes we get the updated map type.
        for fm_spec, fm_map in zip(self.uip['speciesListFM'], self.uip['mapTypeListFM']):
            if fm_spec == specie_name and output_map is None:
                output_map = fm_map
            elif fm_spec == specie_name and output_map != fm_map:
                raise RuntimeError(f'There were at least two different FM map types in the UIP for specie {specie_name}: {output_map} and {fm_map}')

        if output_map is None:
            raise ValueError(f'Specie {specie_name} was not present in the FM species list, so could not find its map type')
        else:
            return output_map
            

    def earth_sun_distance(self, instrument_name):
        '''Earth sun distance, in meters. Right now this is OMI specific'''
        # Same value for all the bands, so just grab the first one
        if(instrument_name == "OMI"):
            return self.omi_obs_table['EarthSunDistance'][0]
        elif(instrument_name == "TROPOMI"):
            return self.tropomi_obs_table['EarthSunDistance'][0]
        else:
            RuntimeError("Didn't find a observation table")
        
    def sample_grid(self, mw_index, ii_mw):
        '''This is the full set of samples. We only actually use a subset of
        these, but these are the values before the microwindow gets applied.

        Right now this is omi specific.'''

        if self.ils_method(mw_index,self.instrument_name(ii_mw)) == "FASTCONV":
            ils_uip_info = self.ils_params(mw_index, self.instrument_name(ii_mw))

            return rf.SpectralDomain(ils_uip_info["central_wavelength"], rf.Unit("nm"))
        else:
            if(self.instrument_name(ii_mw) == "OMI"):
                all_freq = self.uip_omi['fullbandfrequency']
                filt_loc = np.array(self.uip_omi['frequencyfilterlist'])
            elif(self.instrument_name(ii_mw) == "TROPOMI"):
                all_freq = self.uip_tropomi['tropomiInfo']["Earth_Radiance"]["Wavelength"]
                filt_loc = np.array(self.uip_tropomi['tropomiInfo']["Earth_Radiance"]["EarthWavelength_Filter"])
            else:
                raise RuntimeError(f"Invalid instrument {self.instrument_name(ii_mw)}")
            return rf.SpectralDomain(all_freq[np.where(filt_loc == self.filter_name(ii_mw))], rf.Unit("nm"))

    def ils_params(self, mw_index, instrument_name):
        '''Returns ILS information for the given microwindow'''
        if(instrument_name == "OMI"):
            return self.uip_omi["ils_%02d" % (mw_index+1)]
        elif(instrument_name == "TROPOMI"):
            # JLL: the TROPOMI UIP seems to use a different naming convention than the OMI UIP
            # (ils_mw_II, where II is the zero-based index - see end of make_uip_tropomi).
            return self.uip_tropomi["ils_mw_%02d" % (mw_index)]
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
        2) track down what exactly muses-py is doing to create this and
        do it directly. 
        '''
        input_directory = f"{self.run_dir}/Input/"
        if(not os.path.exists(input_directory)):
            raise RuntimeError(f"Input directory {input_directory} not found.")
        if(instrument_name == "OMI"):
            fname = glob.glob(input_directory + "Radiance_OMI*.pkl")[0]
        elif(instrument_name == "TROPOMI"):
            fname = glob.glob(input_directory + "Radiance_TROPOMI*.pkl")[0]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return pickle.load(open(fname, "rb"))

    def mw_slice(self, mw_index, instrument_name):
        '''Variation of mw_slice that uses startmw and endmw. I think these are
        the same if we aren't doing an ILS, but different if we are. Should track
        this through, but for now just try this out'''
        if(instrument_name == "OMI"):
            startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw"][mw_index]
        elif(instrument_name == "TROPOMI"):
            startmw_fm = self.uip_tropomi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw"][mw_index]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return slice(startmw_fm, endmw_fm+1)

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
        slc = self.mw_slice(mw_index, instrument_name)
        rad_info = self.radiance_info(mw_index, instrument_name)
        return rf.SpectralDomain(rad_info['Earth_Radiance']['Wavelength'][slc],
                                 rf.Unit("nm"))        

    def solar_irradiance(self, mw_index, instrument_name):
        '''This is currently just used for the Raman calculation of the 
        RefractorRtfOmi class. This has been adjusted for the 
        '''
        slc = self.mw_slice(mw_index, instrument_name)
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
    
    def across_track_indexes(self, ii_mw, instrument_name):
        '''Across track indexes for the microwindow index ii_mw.

        Right now this is omi specific'''
        # Can't really average these to have anything that makes sense.
        # So for now we just pick the first one that matches
        if(instrument_name == "OMI"):
            return np.asarray(self.omi_obs_table["XTRACK"])[self.channel_indexes(ii_mw)]
        if(instrument_name == "TROPOMI"):
            return np.asarray(self.tropomi_obs_table["XTRACK"])[self.channel_indexes(ii_mw)]
        raise RuntimeError("Don't know how to find observation table")

    def update_uip(self, retrieval_vec):
        '''This updates the underlying UIP with the new retrieval_vec,
        e.g., this is the muses-py equivalent up updating the
        StateVector in ReFRACtor.

        Note that this is the retrieval vector, not the full state vector.
        '''
        # Fake the ret_info structure. update_uip only uses the basis
        # matrix
        ret_info = {'basis_matrix' : self.basis_matrix}
        self.uip, _ = mpy.update_uip(self.uip, ret_info, retrieval_vec)
        if(hasattr(self.uip, 'as_dict')): 
            self.uip = self.uip.as_dict(self.uip)

    @classmethod
    def create_uip(cls, i_stateInfo, i_table, i_windows,     
                   i_retrievalInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi,
                   i_oco2, jacobian_speciesIn=None):
        '''We duplicate what mpy.run_retrieval does to make the uip.'''
        i_state = copy.deepcopy(i_stateInfo)
        i_windows = copy.deepcopy(i_windows)
        i_retrievalInfo = copy.deepcopy(i_retrievalInfo)
        if(isinstance(i_state, dict)):
            i_state = mpy.ObjectView(i_state)
        if(isinstance(i_retrievalInfo, dict)):
            i_retrievalInfo = mpy.ObjectView(i_retrievalInfo)
        if(jacobian_speciesIn):
            jacobian_speciesNames=jacobian_speciesIn
        else:
            jacobian_speciesNames = i_retrievalInfo.species[0:i_retrievalInfo.n_species]
        uip = mpy.make_uip_master(i_state, i_state.current, i_table,
                                  i_windows, jacobian_speciesNames,
                                  i_cloudIndex=0, 
                                  i_modifyCloudFreq=True)
        # run_forward_model doesn't have mapType, not really sure why. It
        # just puts an empty list here. Similarly no n_totalParameters.
        if ('mapType' in i_retrievalInfo.__dict__
            and i_retrievalInfo.n_totalParameters > 0):
            uip['jacobiansLinear'] = [i_retrievalInfo.species[i] for i in range(len(i_retrievalInfo.mapType)) if i_retrievalInfo.mapType[i] == 'linear' and i_retrievalInfo.species[i] not in ('EMIS', 'TSUR', 'TATM') ]
            uip['speciesList'] = copy.deepcopy(i_retrievalInfo.speciesList[0:i_retrievalInfo.n_totalParameters])
            uip['speciesListFM'] = copy.deepcopy(i_retrievalInfo.speciesListFM[0:i_retrievalInfo.n_totalParametersFM])
            uip['mapTypeListFM'] = copy.deepcopy(i_retrievalInfo.mapTypeListFM[0:i_retrievalInfo.n_totalParametersFM])
            uip['initialGuessListFM'] = copy.deepcopy(i_retrievalInfo.initialGuessListFM[0:i_retrievalInfo.n_totalParametersFM])
            uip['constraintVectorListFM'] = copy.deepcopy(i_retrievalInfo.constraintVectorListFM[0:i_retrievalInfo.n_totalParametersFM]) # only needed for PCA map type.
        else:
            uip['jacobiansLinear'] = ['']
            uip['speciesList'] = copy.deepcopy(i_retrievalInfo.speciesList)
            uip['speciesListFM'] = copy.deepcopy(i_retrievalInfo.speciesListFM)
            uip['mapTypeListFM'] = copy.deepcopy(i_retrievalInfo.mapTypeListFM[0:i_retrievalInfo.n_totalParametersFM])
            uip['initialGuessListFM'] = copy.deepcopy(i_retrievalInfo.initialGuessListFM[0:i_retrievalInfo.n_totalParametersFM])
            uip['constraintVectorListFM'] = copy.deepcopy(i_retrievalInfo.constraintVectorListFM[0:i_retrievalInfo.n_totalParametersFM]) # only needed for PCA map type.
        uip['microwindows_all'] = i_windows
        # Basis matrix if available, this isn't in run_forward_model.
        if ('mapToState' in i_retrievalInfo.__dict__ and
            i_retrievalInfo.n_totalParameters > 0):
            mmm = i_retrievalInfo.n_totalParameters
            nnn = i_retrievalInfo.n_totalParametersFM
            basis_matrix = i_retrievalInfo.mapToState[0:mmm, 0:nnn]
        else:
            basis_matrix = None
        rf_uip = cls(uip, basis_matrix)
        
        # Group windows by instrument
        inst_to_window = defaultdict(list)
        for w in i_windows:
            inst_to_window[w['instrument']].append(w)
        if 'AIRS' in inst_to_window:
            # For who knows what bizarre reason. the arguments are
            # different here if we are calling from run_forward_model. We
            # trigger off having jacobian_speciesIn. I think this might
            # have something to do with the BT retrieval handling, which seems
            # to have been crammed in breaking stuff. We need to conform
            # to the existing code
            if(jacobian_speciesIn is None):
                uip['uip_AIRS'] = mpy.make_uip_airs(i_state, i_state.current,
                                                i_table, inst_to_window['AIRS'],
                                                uip['jacobians_all'],
                                                uip['speciesListFM'],
                                                None, i_airs['radiance'],
                                                i_modifyCloudFreq=True)
            else:
                uip['uip_AIRS'] = mpy.make_uip_airs(i_state, i_state.current,
                                                i_table, inst_to_window['AIRS'],
                                                '',
                                                uip['jacobians_all'],
                                                None, i_airs['radiance'],
                                                i_modifyCloudFreq=True)
                
        if 'CRIS' in inst_to_window:
            # For who knows what bizarre reason. the arguments are
            # different here if we are calling from run_forward_model. We
            # trigger off having jacobian_speciesIn.I think this might
            # have something to do with the BT retrieval handling, which seems
            # to have been crammed in breaking stuff. We need to conform
            # to the existing code
            if(jacobian_speciesIn is None):
                uip['uip_CRIS'] = mpy.make_uip_cris(i_state, i_state.current,
                                               i_table, inst_to_window['CRIS'],
                                               uip['jacobians_all'],
                                               uip['speciesListFM'],
                                               i_cris['radianceStruct'.upper()],
                                               i_modifyCloudFreq=True)
            else:
                uip['uip_CRIS'] = mpy.make_uip_cris(i_state, i_state.current,
                                               i_table, inst_to_window['CRIS'],
                                               '',
                                               uip['jacobians_all'],
                                               i_cris['radianceStruct'.upper()],
                                               i_modifyCloudFreq=True)
                
        if 'TES' in inst_to_window:
            raise RuntimeError("TES is not implemented yet")
        if "OMI" in inst_to_window:
            uip["uip_OMI"] = mpy.make_uip_omi(i_state, i_state.current,
                                              i_table,
                                              inst_to_window["OMI"],
                                              uip['jacobians_all'],
                                              uip,
                                              i_omi)
        if "TROPOMI" in inst_to_window:
            uip["uip_TROPOMI"] = mpy.make_uip_tropomi(i_state, i_state.current,
                                              i_table,
                                              inst_to_window["TROPOMI"],
                                              uip['jacobians_all'],
                                              uip,
                                              i_tropomi)
        if "OCO2" in inst_to_window:
            uip["uip_OCO2"] = mpy.make_uip_oco2(i_state, i_state.current,
                                                i_table,
                                                inst_to_window["OCO2"],
                                                uip['jacobians_all'],
                                                uip,
                                                i_oco2)

        # Correct surface pointing angle. Not sure why this needs to be
        # done, but this matches what run_retrieval does
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if f'uip_{k}' in uip:
                uip[f'uip_{k}']["obs_table"]["pointing_angle_surface"] = \
                    rf_uip.ray_info(k, set_pointing_angle_zero=False)["ray_angle_surface"]

        # Make jacobians entry only have unique element.
        #
        # Note that starting with python 3.7 dict preserves insertion order
        # (guaranteed, 3.6 actually did this also but it was just an
        # implementation detail rather than guaranteed),
        # so list(dict.fromkeys(v)) will have a list of unique elements in the
        # order that the first of each item appears
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if f'uip_{k}' in uip:
                uip[f'uip_{k}']["jacobians"] = np.array(list(dict.fromkeys(uip[f'uip_{k}']["jacobians"])))

        # Copy some of the oss stuff to the top level
        with osswrapper(uip) as owrap:
            if(owrap.oss_dir_lut is not None):
                uip['oss_dir_lut'] = owrap.oss_dir_lut
                uip['oss_jacobianList'] = owrap.oss_jacobianList
                uip['oss_frequencyList'] = owrap.oss_frequencyList
                uip['oss_frequencyListFull'] = owrap.oss_frequencyListFull
        
        # Create instrument list
        uip['instrumentList'] = []
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if f'uip_{k}' in uip:
                uip['instrumentList'].extend([k,] * len(uip[f'uip_{k}']["frequencyList"]))
                
        # Add extra pieces to the microwindows.
        for w in i_windows:
            for k in ('enddmw_fm', 'enddmw', 'startmw_fm', 'startmw'):
                if k not in w:
                    w[k] = 0
                    
        if(basis_matrix is not None and
           i_retrievalInfo.n_totalParameters > 0):
            xig = i_retrievalInfo.initialGuessList[0:i_retrievalInfo.n_totalParameters]
            rf_uip.update_uip(xig)
        else:
            uip['currentGuessList'] = i_retrievalInfo.initialGuessList
            uip['currentGuessListFM'] = i_retrievalInfo.initialGuessListFM

        return rf_uip
                                  
                                  


__all__ = ["RefractorUip"]            
