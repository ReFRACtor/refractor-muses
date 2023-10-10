from .refractor_capture_directory import (RefractorCaptureDirectory,
                                          muses_py_call)
from .retrieval_output import (RetrievalJacobianOutput,
                               RetrievalRadianceOutput, RetrievalL2Output)
from .retrieval_debug_output import (RetrievalInputOutput, RetrievalPickleResult,
                                     RetrievalPlotRadiance, RetrievalPlotResult)
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .fm_obs_creator import FmObsCreator
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
import logging
import refractor.muses.muses_py as mpy
import os
import copy
import numpy as np
import pickle
from pathlib import Path
from pprint import pformat, pprint
import time
from contextlib import contextmanager
from .refractor_retrieval_info import RefractorRetrievalInfo
from .refractor_state_info import RefractorStateInfo
logger = logging.getLogger("py-retrieve")

# We could make this an rf.Observable, but no real reason to push this to a C++
# level. So we just have a simple observation set here
class RetrievalStrategy:
    '''This is an attempt to make the muses-py script_retrieval_ms
    more like our JointRetrieval stuff (pretty dated, but
    https://github.jpl.nasa.gov/refractor/joint_retrieval)

    This is a replacement for script_retrieval_ms, that tries to do a
    few things:

    1. Simplifies the core code, the script_retrieval_ms is really
        pretty long and is a sequence of "do one thing, then another,
        then aother). We do this by:

    2. Moving output out of this class, and having separate classes
       handle this. We use the standard ReFRACtor approach of having
       observers. This tend to give a much cleaner interface with
       clear seperation.

    3. Adopt a extensively, configurable way to handle the initial
       guess (similiar to the OCO-2 InitialGuess structure)

    4. Handle species information as a separate class, which allows us
       to easily extend the list of jacobian parameters (e.g, add
       EOFs). The existing code uses long lists of hardcoded values,
       this attempts to be a more adaptable.

    This has a number of advantages, for example having InitialGuess
    separated out allows us to do unit testing in ways that don't
    require updating the OSP directories with new covariance stuff,
    for example.

    '''
    # TODO Add handling of writeOutput, writePlots, debug. I think we
    # can probably do that by just adding Observers
    def __init__(self, filename, vlidort_cli=None, writeOutput=False, writePlots=False,
                 **kwargs):
        logger.info(f"Strategy table filename {filename}")
        self.capture_directory = RefractorCaptureDirectory()
        self._observers = set()
        self.filename = os.path.abspath(filename)
        self.run_dir = os.path.dirname(self.filename)
        self.vlidort_cli = vlidort_cli
        self._table_step = -1

        self.retrieval_strategy_step_set  = copy.deepcopy(RetrievalStrategyStepSet.default_handle_set())
        self.fm_obs_creator = FmObsCreator()
        self.instrument_handle_set = self.fm_obs_creator.instrument_handle_set
        self.kwargs = kwargs
        self.kwargs["vlidort_cli"] = vlidort_cli
        
        with self.chdir_run_dir():
            _, self.strategy_table = mpy.table_read(self.filename)
            self.strategy_table = self.strategy_table.__dict__
        # Right now, we hardcode the output observers. Probably want to
        # rework this
        self.add_observer(RetrievalJacobianOutput())
        self.add_observer(RetrievalRadianceOutput())
        self.add_observer(RetrievalL2Output())
        # Similarly logic here is hardcoded
        if(writeOutput):
            self.add_observer(RetrievalInputOutput())
            self.add_observer(RetrievalPickleResult())
            if(writePlots):
                self.add_observer(RetrievalPlotResult())
                self.add_observer(RetrievalPlotRadiance())

    def add_observer(self, obs):
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._observers.add(obs)
        if(hasattr(obs, "notify_add")):
            obs.notify_add(self)

    def remove_observer(self, obs):
        self._observers.discard(obs)
        if(hasattr(obs, "notify_remove")):
            obs.notify_remove(self)

    def clear_observers(self):
        # We change self._observers, in our loop so grab a copy of the list
        # before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)
        
    def notify_update(self, location, **kwargs):
        for obs in self._observers:
            obs.notify_update(self, location, **kwargs)

    def retrieval_ms(self):
        '''This is script_retrieval_ms in muses-py'''
        # Wrapper around calling mpy. We can perhaps pull some this out, but
        # for now we'll do that.
        with muses_py_call(self.run_dir,
                           vlidort_cli=self.vlidort_cli):
            self.retrieval_ms_body()

    def retrieval_ms_body(self):
        start_date = time.strftime("%c")
        start_time = time.time()
        # Might be good to wrap these in classes
        (self.o_airs, self.o_cris, self.o_omi, self.o_tropomi, self.o_tes, self.o_oco2,
         self.stateInfo) = mpy.script_retrieval_setup_ms(self.strategy_table, False)
        self.create_windows(all_step=True)
        self.stateInfo = RefractorStateInfo(self.stateInfo)
        self.create_radiance()
        self.stateInfo.state_info_dict = mpy.states_initial_update(self.stateInfo.state_info_dict,
                                                   self.strategy_table,
                                                   self.radiance, self.instruments)
        self.notify_update("initial set up done")

        self.errorInitial = None
        self.errorCurrent = None
        self.retrievalInfo = None
        
        # Go through all the steps once, to make sure we can get all the information
        # we need. This way we fail up front, rather than after multiple retrieval
        # steps
        for stp in range(self.number_table_step):
            self.table_step = stp
            self.get_initial_guess()
        self.stateInfo.copy_current_initialInitial()
        # Now go back through and actually do retrievals.
        # Note that a BT step might change the number of steps we have, it
        # modifies the strategy table. So we can't use a normal for
        # loop here, we need to recalculate self.number_table_step each time.
        # So we use a while loop
        stp = -1
        while stp < self.number_table_step - 1:
            stp += 1
            self.table_step = stp
            self.stateInfo.copy_current_initial()
            self.stateOneNext = copy.deepcopy(self.stateInfo.state_info_dict["current"])
            # TODO May be able to remove this
            self.results = None
            self.radianceStep = None
            logger.info(f'\n---')
            logger.info(f"Step: {self.table_step}, Step Name: {self.step_name}, Total Steps: {self.number_table_step}")
            logger.info(f'\n---')
            self.get_initial_guess()
            self.create_windows(all_step=False)
            self.create_radiance_step()
            logger.info(f"Step: {self.table_step}, Retrieval Type {self.retrieval_type}")
            self.retrieval_strategy_step_set.retrieval_step(self.retrieval_type, self)
            self.stateInfo.copy_state_one_next(self.stateOneNext)
            logger.info(f"Done with step {self.table_step}")

        stop_date = time.strftime("%c")
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        elapsed_time_seconds = stop_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60.0
        logger.info('\n---')    
        logger.info(f"start_date {start_date}")
        logger.info(f"stop_date {stop_date}")
        logger.info(f"elapsed_time {elapsed_time}")
        logger.info(f"elapsed_time_seconds {elapsed_time_seconds}")
        logger.info(f"elapsed_time_minutes {elapsed_time_minutes}")
        logger.info(f"Done")
        
        exitcode = 37
        logger.info('\n---')    
        logger.info(f"signaling successful completion w/ exit code {exitcode}")
        logger.info('\n---')    
        logger.info('\n---')    
        return exitcode
        
    @property
    def press_list(self):
        return [float(mpy.table_get_pref(self.strategy_table, "plotMaximumPressure")), 
                float(mpy.table_get_pref(self.strategy_table, "plotMinimumPressure"))]

    @property
    def quality_name(self):
        res = mpy.table_get_spectral_filename(self.strategy_table, self.table_step)
        res = os.path.basename(res)
        res = res.replace("Microwindows_", "QualityFlag_Spec_")
        res = res.replace("Windows_", "QualityFlag_Spec_")
        res = mpy.table_get_pref(self.strategy_table, "QualityFlagDirectory") + res
            
        # if this does not exist use generic nadir / limb quality flag
        if not os.path.isfile(res):
            logger.warning(f'Could not find quality flag file: {res}')
            viewingMode = mpy.table_get_pref(self.strategy_table, "viewingMode")
            viewMode = viewingMode.lower().capitalize()

            res = f"{os.path.dirname(res)}/QualityFlag_Spec_{viewMode}.asc"
            logger.warning(f"Using generic quality flag file: {res}")
            # One last check.
            if not os.path.isfile(res):
                raise RuntimeError(f"Quality flag filename not found: {res}")
        return res
    
    def create_radiance_step(self):
        # Note, I think we might replace this just with our SpectralWindow stuff,
        # along with an Observation class
        self.radianceStepIn = self.radiance
        self.radianceStepIn = mpy.radiance_set_windows(self.radianceStepIn, self.windows)

        if np.all(np.isfinite(self.radianceStepIn['radiance'])) == False:
            raise RuntimeError('ERROR! radiance NOT FINITE!')

        if np.all(np.isfinite(self.radianceStepIn['NESR'])) == False:
            raise RuntimeError('ERROR! radiance error NOT FINITE!')
        if 'OCO2' in self.instruments:
            ind = np.where(np.array(mpy.radiance_get_instrument_array(self.radianceStepIn)) == 'OCO2')[0]
            sample_indexes_step = mpy.oco2_sample_indexes(self.o_oco2['radianceStruct']['frequency'], self.radianceStepIn['frequency'][ind], selfpo_oco2['sample_indexes'])

            # make oco2_step, containing step information
            self.oco2_step = copy.deepcopy(self.o_oco2)
            self.oco2_step['sample_indexes'] = sample_indexes_step

            # OCO-2 step radiance
            radiancex = copy.deepcopy(self.radianceStepIn)
            radiancex = mpy.radiance_set_instrument(self.radiancex, 'OCO2')

            self.oco2_step['radianceStruct'] = radiancex
        else:
            self.oco2_step = None
        # update with omi pars to omi measured radiance
        my_instruments = np.asarray(mpy.radiance_get_instrument_array(self.radianceStepIn))
        ind = np.where(my_instruments == 'OMI')[0]
        if len(ind) > 0:
            # Get windows that are 'OMI' specific.
            ind2 = []
            for ii in range(0, len(self.windows)):
                if self.windows[ii]['instrument'] == 'OMI':
                    ind2.append(ii)
            ind2 = np.asarray(ind2)  # Convert to array so we can use it as an index.

            # The type of stateInfo is sometimes dict and sometimes ObjectView.

            result = mpy.get_omi_radiance(self.stateInfo.state_info_dict["current"]['omi'], copy.deepcopy(self.o_omi))

            obsrad = mpy.radiance_data(result['normalized_rad'], result['nesr'], [-1], result['wavelength'], result['filter'], "OMI")

            # reduce to omi step windows 
            obsrad = mpy.radiance_set_windows(obsrad, np.asarray(self.windows)[ind2])

            # put into omi part of step windows 
            self.radianceStepIn['radiance'][ind] = copy.deepcopy(obsrad['radiance'])
            if np.all(np.isfinite(self.radianceStepIn['radiance'])) == False:
                raise RuntimeError('ERROR! radiance NOT FINITE!')

        indT = np.where(my_instruments == 'TROPOMI')[0]
        if len(indT) > 0:
            # Get windows that are 'TROPOMI' specific.
            ind2 = []
            for ii in range(0, len(self.windows)):
                if self.windows[ii]['instrument'] == 'TROPOMI':
                    ind2.append(ii)
            ind2 = np.asarray(ind2)  # Convert to array so we can use it as an index.

            # The type of stateInfo is sometimes dict and sometimes ObjectView.

            result = mpy.get_tropomi_radiance(self.stateInfo.state_info_dict["current"]['tropomi'],
                                              copy.deepcopy(self.o_tropomi))

            obsrad = mpy.radiance_data(result['normalized_rad'], result['nesr'], [-1], result['wavelength'], result['filter'], "TROPOMI")

            # reduce to omi step windows 
            obsrad = mpy.radiance_set_windows(obsrad, np.asarray(self.windows)[ind2])

            # put into omi part of step windows 
            self.radianceStepIn['radiance'][indT] = copy.deepcopy(obsrad['radiance'])
            if np.all(np.isfinite(self.radianceStepIn['radiance'])) == False:
                raise RuntimeError('ERROR! radiance NOT FINITE!')
        # end: if len(indT) > 0:

    def create_radiance(self):
        '''Read the radiance data. We can  perhaps move this into a Observation class
        by instrument.

        Note that this also creates the magic files Radiance_OMI_.pkl and
        Radiance_TROPOMI_.pkl. It would be nice if can rework that.
        '''
        logger.info(f"Instruments: {len(self.instruments)} {self.instruments}")
        obsrad = None
        for instrument_name in self.instruments:
            logger.info(f"Reading radiance: {instrument_name}")
            if instrument_name == 'OMI':
                result = mpy.get_omi_radiance(self.stateInfo.state_info_dict['current']['omi'], copy.deepcopy(self.o_omi))
                radiance = result['normalized_rad']
                nesr = result['nesr']
                my_filter = result['filter']
                frequency = result['wavelength']
                fname = f'{self.run_dir}/Input/Radiance_OMI_.pkl'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                pickle.dump(self.o_omi, open(fname, "wb"))
            if instrument_name == 'TROPOMI':
                result = mpy.get_tropomi_radiance(self.stateInfo.state_info_dict['current']['tropomi'], copy.deepcopy(self.o_tropomi))

                radiance = result['normalized_rad']
                nesr = result['nesr']
                my_filter = result['filter']
                frequency = result['wavelength']
                fname = f'{self.run_dir}/Input/Radiance_TROPOMI_.pkl'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                pickle.dump(self.o_tropomi, open(fname, "wb"))

            if instrument_name == 'AIRS':
                radiance = self.o_airs['radiance']['radiance']
                frequency = self.o_airs['radiance']['frequency']
                nesr = self.o_airs['radiance']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_airs['radiance'])
            if instrument_name == 'CRIS':
                # The o_cris dictionary uses all uppercase keys.
                radiance = self.o_cris['radiance'.upper()]
                frequency = self.o_cris['frequency'.upper()]
                nesr = self.o_cris['nesr'.upper()]
                my_filter = mpy.radiance_get_filter_array(self.o_cris['radianceStruct'.upper()])
            if instrument_name == 'OCO2':
                radiance = self.o_oco2['radianceStruct']['radiance']
                frequency = self.o_oco2['radianceStruct']['frequency']
                nesr = self.o_oco2['radianceStruct']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_oco2['radianceStruct'])

            if instrument_name == 'TES':
                radiance = self.o_tes['radianceStruct']['radiance']
                frequency = self.o_tes['radianceStruct']['frequency']
                nesr = self.o_tes['radianceStruct']['NESR']
                my_filter = mpy.radiance_get_filter_array(self.o_tes['radianceStruct'])

            # Add the first radiance if this is the first time in the loop.
            if(obsrad is None):
                obsrad = mpy.radiance_data(radiance, nesr, [-1], frequency, my_filter, instrument_name, None)
            else:
                filtersIn = np.asarray(['' for ii in range(0, len(frequency))])
                obsrad = mpy.radiance_add_filter(obsrad, radiance, nesr, [-1], frequency, my_filter, instrument_name)
        self.radiance = obsrad
        
    def create_windows(self, all_step=False):
        # We should rework this a bit, it is just a string of magic code. Perhaps
        # we should have each instrument have a function for this?
        if(all_step):
            self.windows = mpy.new_mw_from_table_all_steps(self.strategy_table)
        else:
            self.windows = mpy.new_mw_from_table(self.strategy_table, self.table_step)
            
        self.instruments = mpy.get_unique_windows(self.windows)

        # This magic adjustment should go somewhere else
        if 'CRIS' in self.instruments:
            for tempind, win in enumerate(self.windows):
                if win['instrument'] == 'CRIS': # EM - Necessary for joint retrievals
                    con1 = self.o_cris['FREQUENCY'] >= win['start']
                    con2 = self.o_cris['FREQUENCY'] <= win['endd']

                    tempind = np.where(np.logical_and(con1 == True, con2 == True))[0]
        
                    MAXOPD = np.unique(self.o_cris['MAXOPD'][tempind])
                    SPACING = np.unique(self.o_cris['SPACING'][tempind])

                    if len(MAXOPD) > 1 or len(SPACING) > 1:
                        raise Running('ERROR!!! Microwindowds across CrIS filter bands leading to spacing and OPD does not uniform in this MW!')

                    win['maxopd'] = np.float32(MAXOPD[0])
                    win['spacing'] = np.float32(SPACING[0])
                    win['monoextend'] = np.float32(SPACING[0]) * 4.0

        if(all_step):
            # muses-py does this only for the first all_step=True. I'm
            # not sure if that matters, or is even correct. But well
            # have that for now
            self.windows = mpy.mw_combine_overlapping(self.windows,
                                                      self.threshold)

    def get_initial_guess(self):
        '''Set retrievalInfo, errorInitial and errorCurrent for the current step.'''
        (self.retrievalInfo, self.errorInitial, self.errorCurrent) = \
            mpy.get_species_information(self.strategy_table,
                                        self.stateInfo.state_info_obj,
                                        self.errorInitial, self.errorCurrent)
        
        #self.retrievalInfo = mpy.ObjectView.as_object(self.retrievalInfo)
        self.retrievalInfo = RefractorRetrievalInfo(self.retrievalInfo.__dict__)

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        # AT_LINE 319 Script_Retrieval_ms.pro
        nn = self.retrievalInfo.n_totalParameters
        logger.info(f"Step: {self.table_step}, Total Parameters: {nn}")

        # AT_LINE 320 Script_Retrieval_ms.pro
        if nn > 0:
            xig = self.retrievalInfo.initialGuessList[0:nn]

            # Note that we do not pass in stateOneNext (None) and do not get back stateOneNext on the left handside as donotcare_stateOneNext.
            (self.stateInfo.state_info_dict, _, _) = \
                mpy.update_state(self.stateInfo.state_info_dict,
                                 self.retrievalInfo.retrieval_info_obj,
                                 xig, self.cloud_prefs, self.table_step, [], None)
            self.stateInfo.state_info_dict = self.stateInfo.state_info_dict.__dict__

    @property
    def threshold(self):
        res = mpy.table_get_pref(self.strategy_table,
                                 "apodizationWindowCombineThreshold")
        return int(res.split()[0])

    @property
    def cloud_prefs(self):
        filename = self.strategy_table['preferences']['CloudParameterFilename']
        (_, fileID) = mpy.read_all_tes_cache(filename)
        cloudPrefs = fileID['preferences']
        return cloudPrefs
        
    @property
    def table_step(self):
        return self._table_step

    @table_step.setter
    def table_step(self, v):
        self._table_step = v
        mpy.table_set_step(self.strategy_table, self._table_step)

    @property
    def number_table_step(self):
        return self.strategy_table["numRows"]

    @property
    def step_name(self):
        return mpy.table_get_entry(self.strategy_table, self.table_step, "stepName")

    @property
    def retrieval_type(self):
        return self.retrievalInfo.type.lower()

    @contextmanager
    def chdir_run_dir(self):
        '''We do this in a number of places, so pull out into a function. We
        temporarily change into self.run_dir, and after the function finishes
        changes back to the current directory. This should be used in a
        "with" block:
        
        with self.chdir_run_dir():
           blah blah
        '''
        curdir = os.getcwd()
        try:
            os.chdir(self.run_dir)
            yield
        finally:
            os.chdir(curdir)

    @property
    def output_directory(self):
        '''Get the output directory from the strategy_table. Note that unlike
        muses-py this doesn't require that we are actually in the run directory,
        we handle this.'''
        with self.chdir_run_dir():
            return os.path.abspath(self.strategy_table['outputDirectory'])

    def save_pickle(self, save_pickle_file):
        '''Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy.'''
        self.capture_directory.save_directory(self.run_dir, vlidort_input=None)
        pickle.dump(self, open(save_pickle_file, "wb"))

    @classmethod
    def load_retrieval_strategy(cls, save_pickle_file, path=".",
                                change_to_dir = False,
                                osp_dir=None, gmao_dir=None,
                                vlidort_cli=None):
        '''This pairs with save_pickle.'''
        res = pickle.load(open(save_pickle_file, "rb"))
        res.run_dir = f"{os.path.abspath(path)}/{res.capture_directory.runbase}"
        res.capture_directory.extract_directory(path=path,
                              change_to_dir=change_to_dir, osp_dir=osp_dir,
                              gmao_dir=gmao_dir)
        if(vlidort_cli is not None):
            res.vlidort_cli = vlidort_cli
        return res

class RetrievalStrategyCaptureObserver:
    '''Helper class, pickles RetrievalStrategy at each time notify_update is
    called. Intended for unit tests and other kinds of debugging.'''
    def __init__(self, basefname, location_to_capture):
        self.basefname = basefname
        self.location_to_capture = location_to_capture

    def notify_update(self, retrieval_strategy, location):
        if(location != self.location_to_capture):
            return
        fname = f"{self.basefname}_{retrieval_strategy.table_step}.pkl"
        retrieval_strategy.save_pickle(fname)
        
__all__ = ["RetrievalStrategy", "RetrievalStrategyCaptureObserver"]    

    
