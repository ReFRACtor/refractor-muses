from .refractor_capture_directory import (RefractorCaptureDirectory,
                                          muses_py_call)
from .retrieval_output import (RetrievalJacobianOutput,
                               RetrievalRadianceOutput, RetrievalL2Output)
from .retrieval_debug_output import (RetrievalInputOutput, RetrievalPickleResult,
                                     RetrievalPlotRadiance, RetrievalPlotResult)
from .refractor_uip import RefractorUip
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .fm_obs_creator import FmObsCreator
from .cost_function import CostFunction
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
        # Will probably rework this
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
        
    def notify_update(self, location):
        for obs in self._observers:
            obs.notify_update(self, location)

    def retrieval_ms(self):
        '''This is script_retrieval_ms in muses-py'''
        # Wrapper around calling mpy. We can perhaps pull some this out, but
        # for now we'll do that.
        with muses_py_call(self.run_dir,
                           vlidort_cli=self.vlidort_cli):
            self.retrieval_ms_body2()

    def retrieval_ms_body(self):
        mpy.script_retrieval_ms(self.filename)

    def retrieval_ms_body2(self):
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
        # Not really sure what this is
        self.BTstruct = [{'diff':0.0, 'obs':0.0, 'fit':0.0} for i in range(100)]

        self.errorInitial = None
        self.errorCurrent = None
        self.retrievalInfo = None
        self.propagatedTATMQA = 1
        self.propagatedH2OQA = 1
        self.propagatedO3QA = 1
        
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
        
    def create_cost_function(self):
        '''Create CostFunction. As a convenience, we do this even when we aren't
        do a full retrieval - just because is has all the pieces needed for
        the Observation and ForwardModel. We could break this up a bit if needed,
        for example we don't actually need the full CostFunction for doing the
        BT forward model. But for now, just do everything'''
        # We'd like to get away from the UIP - it is just a reshuffling of a
        # bunch of data we already have. But for now put this in place. We can
        # perhaps push this down to just the MusesForwardModel classes.
        self.rf_uip = RefractorUip.create_uip(self.stateInfo, self.strategy_table,
                                         self.windows, self.retrievalInfo,
                                         self.o_airs, self.o_tes,
                                         self.o_cris, self.o_omi, self.o_tropomi,
                                         self.o_oco2)
        # Used by levmar_nllsq_elanor also, so save this. Would be good if we
        # could get this push down a but farther
        self.ret_info = { 
            'obs_rad': self.radianceStepIn["radiance"],
            'meas_err': self.radianceStepIn["NESR"],            
            'CostFunction': 'MAX_APOSTERIORI',   
            'basis_matrix': self.rf_uip.basis_matrix,   
            'sqrt_constraint': (mpy.sqrt_matrix(self.retrievalInfo.apriori_cov)).transpose(),
            'const_vec': self.retrievalInfo.apriori,
            'minimumList':self.retrievalInfo.minimumList[0:self.retrievalInfo.n_totalParameters],
            'maximumList':self.retrievalInfo.maximumList[0:self.retrievalInfo.n_totalParameters],
            'maximumChangeList':self.retrievalInfo.maximumChangeList[0:self.retrievalInfo.n_totalParameters]
        }
        # Create a cost function, and use to implement residual_fm_jacobian
        # when we call levmar_nllsq_elanor
        self.cost_function = CostFunction(*self.fm_obs_creator.fm_and_obs(self.rf_uip,
                                self.ret_info, **self.kwargs))
        self.cost_function.parameters = self.rf_uip.current_state_x
        
    def systematic_jacobian(self):
        if(self.retrievalInfo.n_speciesSys <= 0):
            return
        # make a temporary retrieval structure for sys info
        retrieval_info = self.retrievalInfo.retrieval_info_obj
        retrieval_info_temp = mpy.ObjectView({
            'parameterStartFM': retrieval_info.parameterStartSys,
            'parameterEndFM' : retrieval_info.parameterEndSys,
            'species': retrieval_info.speciesSys,
            'n_species': retrieval_info.n_speciesSys,
            'speciesList': retrieval_info.speciesListSys,
            'speciesListFM': retrieval_info.speciesListSys,
            'mapTypeListFM': mpy.constraint_get_maptype(self.errorCurrent, retrieval_info.speciesListSys),
            'initialGuessListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
            'constraintVectorListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
            'initialGuessList': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
            'n_totalParametersFM': retrieval_info.n_totalParametersSys
        })
        # This is a list of unique species.
        jacobian_speciesNames = retrieval_info.speciesSys[0:retrieval_info.n_speciesSys]
        # This is a list of species (not unique).
        jacobian_specieslist = retrieval_info.speciesListSys[0:retrieval_info.n_totalParametersSys]  
        
        logger.info("Running run_forward_model for systematic jacobians ...")
        # TODO - Should have logic here to skip running forward model if we
        # don't have any species in the jacobian of it - so for example some
        # of the Cris/Tropomi systematic jacobian don't actually use any tropomi
        # stuff. But right now, both forward models are run even though only one
        # contributes to the jacobian.
        _, jacobianSys = self.run_forward_model(
            self.strategy_table, self.stateInfo, self.windows, retrieval_info_temp,
            jacobian_speciesNames, jacobian_specieslist,
            self.radianceStepIn,
            self.o_airs, self.o_cris, self.o_tes, self.o_omi, self.o_tropomi,
            self.oco2_step,
            mytiming=None, 
            writeOutputFlag=False, 
            RJFlag=True)

        if jacobianSys["jacobian_data"].shape[1] != retrieval_info.n_totalParametersSys:
            raise RuntimeError("sys species does not match the # of species in sys Jacobians\nThis can happen if one of the species is not used in selected windows\n")

        self.results.jacobianSys = jacobianSys["jacobian_data"]
        
    def error_analysis(self):
        if(self.results is None):
            return
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radianceNoise = {"radiance" : np.zeros_like(self.radianceStep["radiance"]) }
        (self.results, self.errorCurrent) = mpy.error_analysis_wrapper(
            self.table_step,
            self.strategy_table["dirAnalysis"],
            self.radianceStep,
            radianceNoise,
            self.retrievalInfo.retrieval_info_obj,
            self.stateInfo.state_info_obj,
            self.errorInitial,
            self.errorCurrent,
            self.windows,
            self.results
            )
        
    def update_retrieval_summary(self):
        '''Calculate various summary statistics for retrieval'''
        # Only the branch of code with results set should get retrieval_summary
        # done
        if(self.results is None):
            return
        # Note that despite the name "write_retrieval_summary" this calculates
        # summary parameters. It doesn't actually write any output unless we
        # have writeOutput. This is just a glob of code, it really should get
        # refractored into some set of classes, e.g. self.results can calculate
        # various pieces
        # TODO check on dirAnalysis, may want to do the same thing we did with
        # output_directory
        self.results = mpy.write_retrieval_summary(
            self.strategy_table["dirAnalysis"],
            self.retrievalInfo.retrieval_info_obj,
            self.stateInfo.state_info_obj,
            None,
            self.results,
            self.windows,
            self.press_list,
            self.quality_name, 
            self.table_step, 
            self.errorCurrent, 
            writeOutputFlag=False, 
            errorInitial=self.errorInitial
        )
        if 'TATM' in self.retrievalInfo.species_names:
            self.propagatedTATMQA = self.results.masterQuality
        if 'O3' in self.retrievalInfo.species_names:
            self.propagatedO3QA = self.results.masterQuality
        if 'H2O' in self.retrievalInfo.species_names:
            self.propagatedH2OQA = self.results.masterQuality
        

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

            self.myobsrad = mpy.radiance_data(result['normalized_rad'], result['nesr'], [-1], result['wavelength'], result['filter'], "OMI")

            # reduce to omi step windows 
            self.myobsrad = mpy.radiance_set_windows(self.myobsrad, np.asarray(self.windows)[ind2])

            # put into omi part of step windows 
            self.radianceStepIn['radiance'][ind] = copy.deepcopy(self.myobsrad['radiance'])
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

            self.myobsrad = mpy.radiance_data(result['normalized_rad'], result['nesr'], [-1], result['wavelength'], result['filter'], "TROPOMI")

            # reduce to omi step windows 
            self.myobsrad = mpy.radiance_set_windows(self.myobsrad, np.asarray(self.windows)[ind2])

            # put into omi part of step windows 
            self.radianceStepIn['radiance'][indT] = copy.deepcopy(self.myobsrad['radiance'])
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
        self.myobsrad = None
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
            if(self.myobsrad is None):
                self.myobsrad = mpy.radiance_data(radiance, nesr, [-1], frequency, my_filter, instrument_name, None)
            else:
                filtersIn = np.asarray(['' for ii in range(0, len(frequency))])
                self.myobsrad = mpy.radiance_add_filter(self.myobsrad, radiance, nesr, [-1], frequency, my_filter, instrument_name)
        self.radiance = self.myobsrad
        
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

    # Temp, we'll copy this over for now but should have lots of
    # changes to this

    def run_retrieval_zero_iterations(self):
        '''run_retrieval when maxIter is 0, pulled out just to simplify
        the code'''
        jacobian_speciesNames = self.retrievalInfo.species_names
        jacobian_speciesList = self.retrievalInfo.species_list_fm
        (radianceOut, jac_fm, _, _, _) = self.cost_function.fm_wrapper({"currentGuessListFM" : self.cost_function.parameters}, None, {})
        jacobianOut = mpy.jacobian_data(jac_fm, radianceOut['detectors'],
                                        radianceOut['frequency'],
                                        self.retrievalInfo.species_list_fm)
        self.radianceStep = self.cost_function.radianceStep(self.radianceStepIn)
        if jacobian_speciesList[0] != '':
            xret = self.retrievalInfo.apriori
        else:
            jacobianOut = 0
            xret = 0
        o_retrievalResults = {
            'bestIteration': 0,                                  
            'xretIterations': xret,                               
            'num_iterations': 0,                                  
            'residualRMS': np.asarray([0]),                    
            'stopCode': -1,                                 
            'stopCriteria': np.zeros(shape=(1, 3), dtype=np.int),
            'resdiag': np.zeros(shape=(1, 5), dtype=np.int),
            'xret': xret,
            'xretFM': self.retrievalInfo.initialGuessListFM,
            'radiance': radianceOut,
            'jacobian': jacobianOut,
            'delta': 0,
            'rho': 0,
            'lambda': 0            
        }
        return o_retrievalResults

    def run_retrieval(self):
        '''run_retrieval

        Note despite the name i_radianceInfo get updated with any radiance
        changes from tropomi/omi.'''
        self.create_cost_function()
        maxIter = int(mpy.table_get_entry(self.strategy_table, self.strategy_table["step"], "maxNumIterations"))
        if maxIter == 0:
            self.radianceStep = copy.deepcopy(self.radianceStepIn)
            return self.run_retrieval_zero_iterations()
        
        # Various thresholds from the input table
        ConvTolerance_CostThresh = np.float32(mpy.table_get_pref(self.strategy_table, "ConvTolerance_CostThresh"))
        ConvTolerance_pThresh = np.float32(mpy.table_get_pref(self.strategy_table, "ConvTolerance_pThresh"))     
        ConvTolerance_JacThresh = np.float32(mpy.table_get_pref(self.strategy_table, "ConvTolerance_JacThresh"))
        Chi2Tolerance = 2.0 / len(self.radianceStepIn["NESR"]) # theoretical value for tolerance
        retrievalType = mpy.table_get_entry(self.strategy_table, self.strategy_table["step"], "retrievalType").lower()
        if retrievalType == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = mpy.table_get_pref(self.strategy_table, 'LMDelta') # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc
            
        # TODO Make this look like a ReFRACtor solver
        with register_replacement_function_in_block("residual_fm_jacobian",
                                                    self.cost_function):
            (xret, diag_lambda_rho_delta, stopcrit, resdiag, 
             x_iter, res_iter, radiance_fm, radiance_iter, jacobian_fm, 
             iterNum, stopCode, success_flag) =  mpy.levmar_nllsq_elanor(  
                    self.rf_uip.current_state_x, 
                    self.table_step, 
                    self.rf_uip.uip, 
                    self.ret_info, 
                    maxIter, 
                    verbose=False, 
                    delta_value=delta_value, 
                    ConvTolerance=ConvTolerance,   
                    Chi2Tolerance=Chi2Tolerance
                    )
        if success_flag == 0:
            raise RuntimeError("----- script_retrieval_ms: Error -----")

        
        self.radianceStep = self.cost_function.radianceStep(self.radianceStepIn)
        
        # Find iteration used, only keep the best iteration
        rms = np.array([np.sqrt(np.sum(res_iter[i,:]*res_iter[i,:])/
                       res_iter.shape[1]) for i in range(iterNum+1)])
        bestIter = np.argmin(rms)
        residualRMS = rms
        
        # Take results and put into the expected output structures.
        
        radianceOut2 = copy.deepcopy(self.radianceStepIn)
        radianceOut2['NESR'][:] = 0
        radianceOut2['radiance'][:] = radiance_fm

        detectors = [0]
        speciesFM = self.retrievalInfo.species_list_fm
        jacobianOut2 = mpy.jacobian_data(jacobian_fm, detectors,
                                         self.radianceStepIn['frequency'], speciesFM)
        radianceOutIter = radiance_iter[:,np.newaxis,:]
        
        o_retrievalResults = {
            'bestIteration'  : int(bestIter), 
            'num_iterations' : iterNum, 
            'stopCode'       : stopCode, 
            'xret'           : xret, 
            'xretFM': self.rf_uip.current_state_x_fm,
            'radiance'       : radianceOut2, 
            'jacobian'       : jacobianOut2, 
            'radianceIterations': radianceOutIter, 
            'xretIterations' : x_iter, 
            'stopCriteria'   : np.copy(stopcrit), 
            'resdiag'        : np.copy(resdiag), 
            'residualRMS'    : residualRMS,
            'delta': diag_lambda_rho_delta[:, 2],
            'rho': diag_lambda_rho_delta[:, 1],
            'lambda': diag_lambda_rho_delta[:, 0],        
        }
        return o_retrievalResults

    def run_forward_model(self, i_table, i_stateInfo, i_windows,
                          i_retrievalInfo, jacobian_speciesIn,
                          jacobian_speciesListIn,
                          uip, airs, cris, tes, omi, tropomi, oco2,
                          mytiming, writeOutputFlag=False, trueFlag=False,
                          RJFlag=False, rayTracingFlag=False):
        rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,
                                         i_retrievalInfo, airs, tes, cris,
                                         omi, tropomi, oco2,
                                         jacobian_speciesIn=jacobian_speciesIn)
        # As a convenience, we use the CostFunction to stitch stuff together.
        # We don't have the Observation for this, but we don't actually use
        # it in fm_wrapper. So we just fake it so we have the proper fields
        # for CostFunction.
        cfunc = CostFunction(*self.fm_obs_creator.fm_and_fake_obs(rf_uip,
                             **self.kwargs, use_full_state_vector=True,
                             include_bad_sample=True))
        (o_radiance, jac_fm, _, _, _) = cfunc.fm_wrapper(rf_uip.uip, None, {})
        o_jacobian = mpy.jacobian_data(jac_fm, o_radiance['detectors'],
                                       o_radiance['frequency'],
                                       rf_uip.uip['speciesListFM'])
        return (o_radiance, o_jacobian)


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

    
