from .refractor_capture_directory import (RefractorCaptureDirectory,
                                          muses_py_call)
from .retrieval_l2_output import RetrievalL2Output
from .retrieval_radiance_output import RetrievalRadianceOutput
from .retrieval_jacobian_output import RetrievalJacobianOutput
from .retrieval_debug_output import (RetrievalInputOutput, RetrievalPickleResult,
                                     RetrievalPlotRadiance, RetrievalPlotResult)
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .strategy_table import StrategyTable
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
        self.fm_obs_creator = FmObsCreator(rs=self)
        self.instrument_handle_set = self.fm_obs_creator.instrument_handle_set
        self.kwargs = kwargs
        self.kwargs["vlidort_cli"] = vlidort_cli

        self.strategy_table = StrategyTable(self.filename)
        
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

    @property
    def run_dir(self):
        return self.capture_directory.rundir

    @run_dir.setter
    def run_dir(self, v):
        self.capture_directory.rundir = v
        
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
         self.state_info) = mpy.script_retrieval_setup_ms(self.strategy_table.strategy_table_dict, False)
        self.create_windows(all_step=True)
        # Instruments is normally the instruments for a particular retrieval step.
        # But because of the "all_step" at this point it is all in the instruments
        # in all steps. Grab a copy of this so we have the full list.
        self.instruments_all = copy.deepcopy(self.instruments)
        self.state_info = RefractorStateInfo(self.state_info)
        self.state_info.state_info_dict = mpy.states_initial_update(
            self.state_info.state_info_dict, self.strategy_table.strategy_table_dict,
            self.fm_obs_creator.radiance(), self.instruments)
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
        self.state_info.copy_current_initialInitial()
        # Now go back through and actually do retrievals.
        # Note that a BT step might change the number of steps we have, it
        # modifies the strategy table. So we can't use a normal for
        # loop here, we need to recalculate self.number_table_step each time.
        # So we use a while loop
        stp = -1
        while stp < self.number_table_step - 1:
            stp += 1
            self.table_step = stp
            self.state_info.copy_current_initial()
            self.stateOneNext = copy.deepcopy(self.state_info.state_info_dict["current"])
            logger.info(f'\n---')
            logger.info(f"Step: {self.table_step}, Step Name: {self.step_name}, Total Steps: {self.number_table_step}")
            logger.info(f'\n---')
            self.get_initial_guess()
            self.create_windows(all_step=False)
            logger.info(f"Step: {self.table_step}, Retrieval Type {self.retrieval_type}")
            self.retrieval_strategy_step_set.retrieval_step(self.retrieval_type, self)
            self.state_info.copy_state_one_next(self.stateOneNext)
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
        return [float(self.strategy_table.preferences["plotMaximumPressure"]),
                float(self.strategy_table.preferences["plotMinimumPressure"])]

    @property
    def quality_name(self):
        res = os.path.basename(self.strategy_table.spectral_filename)
        res = res.replace("Microwindows_", "QualityFlag_Spec_")
        res = res.replace("Windows_", "QualityFlag_Spec_")
        res = self.strategy_table.preferences["QualityFlagDirectory"] + res
            
        # if this does not exist use generic nadir / limb quality flag
        if not os.path.isfile(res):
            logger.warning(f'Could not find quality flag file: {res}')
            viewingMode = self.strategy_table.preferences["viewingMode"]
            viewMode = viewingMode.lower().capitalize()

            res = f"{os.path.dirname(res)}/QualityFlag_Spec_{viewMode}.asc"
            logger.warning(f"Using generic quality flag file: {res}")
            # One last check.
            if not os.path.isfile(res):
                raise RuntimeError(f"Quality flag filename not found: {res}")
        return res
    
    def create_windows(self, all_step=False):
        # We should rework this a bit, it is just a string of magic code. Perhaps
        # we should have each instrument have a function for this?
        if(all_step):
            self.windows = mpy.new_mw_from_table_all_steps(self.strategy_table.strategy_table_dict)
        else:
            self.windows = mpy.new_mw_from_table(self.strategy_table.strategy_table_dict, self.table_step)
            
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
            mpy.get_species_information(self.strategy_table.strategy_table_dict,
                                        self.state_info.state_info_obj,
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
            (self.state_info.state_info_dict, _, _) = \
                mpy.update_state(self.state_info.state_info_dict,
                                 self.retrievalInfo.retrieval_info_obj,
                                 xig, self.cloud_prefs, self.table_step, [], None)
            self.state_info.state_info_dict = self.state_info.state_info_dict.__dict__

    @property
    def threshold(self):
        res = self.strategy_table.preferences["apodizationWindowCombineThreshold"]
        return int(res.split()[0])

    @property
    def cloud_prefs(self):
        (_, fileID) = mpy.read_all_tes_cache(self.strategy_table.cloud_parameters_filename)
        cloudPrefs = fileID['preferences']
        return cloudPrefs
        
    @property
    def table_step(self):
        return self.strategy_table.table_step

    @table_step.setter
    def table_step(self, v):
        self.strategy_table.table_step = v

    @property
    def number_table_step(self):
        return self.strategy_table.number_table_step

    @property
    def step_name(self):
        return self.strategy_table.step_name

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
        return self.strategy_table.output_directory

    def retrieval_setup(self):
        # Not sure we actually want this here, but short term we'll place this here
        # and ideally clean up and separate.

        o_airs = None
        o_cris = None
        o_omi = None
        o_tropomi = None
        o_tes = None
        o_oco2 = None
        o_state_info = None
        instrument_file_name = 'Measurement_ID.asc'

        (_, o_file_content) = read_all_tes(instrument_file_name)
        file_id = tes_file_get_struct(o_file_content)

        if 'oceanFlag' in file_id['preferences']:
            oceanFlag = int(file_id['preferences']['oceanFlag'])
        elif 'OCEANFLAG' in file_id['preferences']:
            oceanFlag = int(file_id['preferences']['OCEANFLAG'])
        else:
            raise RuntimeError(f"ERROR: Could not find 'oceanflag' or 'OCEANFLAG' from preferences: {file_id['preferences']}")
    
        # TODO Set up for this to get overriden
        gmao_path = '../GMAO/'
        gmao_type = ''
        if 'GMAO' in file_id['preferences'] and 'GMAO_TYPE' in file_id['preferences']:
            gmao_path = str(file_id['preferences']['GMAO'])
            gmao_type = str(file_id['preferences']['GMAO_TYPE'])

        my_key = file_id['preferences']['key'] 
        directoryIG = table_get_pref(i_table_struct, "initialGuessDirectory")
        directoryConstraint = table_get_pref(i_table_struct, "constraintVectorDirectory") # where state goes is specified by the Table

        # First see what instruments are used in the retrieval.  Read through
        # all windows files and get instrument list
        # Get micro windows from strategy table for all retrieval steps.
        windows = new_mw_from_table_all_steps(i_table_struct)

        # There may be more than one instruments in windows list.
        instruments = []
        for one_window in windows:
            if one_window['instrument'] not in instruments:
                instruments.append(one_window['instrument'])

        instrument_name = 'DUMMY_INSTRUMENT_NAME'

        # Get lat/lon/time from appropriate instrument
        instrument_name = None
        for i in ('TES', 'AIRS', 'CRIS', 'OMI', 'TROPOMI', 'OCO2'):
            if i in instruments:
                instrument_name = i
                break
        if(instrument_name is None):
            raise RuntimeError('Unknown instrument.  Must have TES, AIRS, CRIS, OMI, TROPOMI, OCO2 specified as instrument in windows files')

        o_latitude = 0
        o_longitude = 0
        o_dateStruct = {}

        try:
            dateStruct = utc_from_string(file_id['preferences'][f'{instrument_name}_utcTime'])
            o_dateStruct['dateStruct'] = dateStruct
            o_dateStruct['year'] = dateStruct['utctime'].year
            o_dateStruct['month'] = dateStruct['utctime'].month
            o_dateStruct['day'] = dateStruct['utctime'].day
            o_dateStruct['hour'] = dateStruct['utctime'].hour
            o_dateStruct['minute'] = dateStruct['utctime'].minute
            o_dateStruct['second'] = dateStruct['utctime'].second
        except:
            o_dateStruct = tai(np.float64(file_id['preferences'][f'{instrument_name}_time']))
            file_id['preferences'][f'{instrument_name}_utcTime'] = utc(o_dateStruct, True)

        try:
            o_latitude = float(file_id['preferences'][f'{instrument_name}_latitude'])
            o_longitude = float(file_id['preferences'][f'{instrument_name}_longitude'])
        except:
            o_latitude = float(file_id['preferences'][f'{instrument_name}_Latitude'])
            o_longitude = float(file_id['preferences'][f'{instrument_name}_Longitude'])
            file_id['preferences'][f'{instrument_name}_latitude'] = o_latitude
            file_id['preferences'][f'{instrument_name}_longitude'] = o_longitude

        if table_get_pref(i_table_struct, 'radianceSource') == 'Synthetic':
            raise RuntimeError("Don't support synthetic input yet")
        if 'TES' in instruments:
            o_tes = read_tes_l1b(file_id, windows)

            # apodize here if specified
            apodizationMethodObs = table_get_pref(i_table_struct, 'apodizationMethodObs')
            apodizationMethodFit = table_get_pref(i_table_struct, 'apodizationMethodFit')
            apodizationWindowCombineThreshold = table_get_pref(i_table_struct, 'apodizationWindowCombineThreshold')
            apodStrength = table_get_pref(i_table_struct, 'NortonBeerApodizationStrength')
            apodizationFunction = table_get_pref(i_table_struct, 'apodizationFunction')

            if apodizationFunction == 'NORTON_BEER':
                status, file = read_all_tes(table_get_pref(i_table_struct, 'defaultSpectralWindowsDefinitionFilename'),'asc')
                maxOPD = np.array(tes_file_get_column(file, 'MAXOPD'))
                filter = np.array(tes_file_get_column(file, 'FILTER'))
                spacing = np.array(tes_file_get_column(file, 'RET_FRQ_SPC'))

                # apodization radiance and NESR
                # if this is synthetic data, need to modify additive noise also.
                radianceStruct = radiance_apodize(o_tes['radianceStruct'], apodStrength, filter, maxOPD, spacing)

                o_tes['radianceStruct'] = radianceStruct
        # end: if 'TES' in instruments:
        
        if 'AIRS' in instruments:
            o_airs = read_airs(file_id, windows)
            
        if 'CRIS' in instruments:
            filename = file_id['preferences']['CRIS_filename']
            logger.info(f"CRIS_filename: {filename}")
            if 'nasa_nsr' in filename:
                cris_type = 'suomi_nasa_nsr' #0
                # uses fsr reader, same content format as fsr
                o_cris = read_nasa_cris_fsr(file_id['preferences'])
            elif 'nasa_fsr' in filename:
                cris_type = 'suomi_nasa_fsr' #1
                # add in type 2 with nomw based on date.
                # CrIS Suomi NPP (Full Spectral Resolution), L1B data is generated by NASA
                # /project/muses/input/cris/nasa_fsr
                o_cris = read_nasa_cris_fsr(file_id['preferences'])
            elif 'jpss_1_fsr' in filename:
                cris_type = 'jpss1_nasa_fsr' #3
                # CrIS JPSS-1 / NOAA-20 (Full Spectral Resolution), L1B data is generated by NASA
                # /project/muses/input/cris/jpss_1_fsr    
                o_cris = read_nasa_cris_fsr(file_id['preferences'])
            elif 'snpp_fsr' in filename:
                cris_type = 'suomi_cspp_fsr' #4
                # CrIS Suomi NPP, L1B data is generated by CSPP (Community Satellite Processing Package)    
                # /project/muses/input/cris_cspp/snpp_fsr    
                o_cris = read_noaa_cris_fsr(file_id['preferences'])
            elif 'noaa_fsr' in filename:
                cris_type = 'suomi_noaa_fsr' #5
                # CrIS JPSS-1 / NOAA-20 (Full Spectral Resolution), L1B data is generated by CSPP (Community Satellite Processing Package)
                # /project/muses/input/cris_cspp/noaa_fsr    
                o_cris = read_noaa_cris_fsr(file_id['preferences'])
            else:
                raise RuntimeError(f"Need to Add case for CRIS type {filename}")
            
            radiance = o_cris['radiance'.upper()]
            frequency = o_cris['frequency'.upper()]
            nesr = o_cris['nesr'.upper()]

            filters = np.array(['CrIS-fsr-lw' for ii in range(len(nesr))])
            ind_arr = np.where(frequency > 1200)[0]
            if len(ind_arr) > 0:
                filters[ind_arr] = 'CrIS-fsr-mw'
            ind_arr = np.where(frequency > 2145)[0]
            if len(ind_arr) > 0:
                filters[ind_arr] = 'CrIS-fsr-sw'
            o_cris['radianceStruct'.upper()] = radiance_data(radiance, nesr, [0], frequency, filters, 'CRIS')
        # end instrument_name == 'CRIS':

        if 'OMI' in instruments:
            OMI_RAD_CALRUN_FLAG = 0
            if 'OMI_Rad_calRun_flag' in file_id['preferences']:
                OMI_RAD_CALRUN_FLAG = int(float(file_id['preferences']['OMI_Rad_calRun_flag']))

            calibrationFilename = table_get_pref(i_table_struct, 'omi_calibrationFilename')
            filename = file_id['preferences']['OMI_filename']

            if OMI_RAD_CALRUN_FLAG == 0:
                cldFilename = None
                if 'OMI_Cloud_filename' in file_id['preferences']:
                    cldFilename = file_id['preferences']['OMI_Cloud_filename']
                o_omi = read_omi(
                    filename, 
                    int(file_id['preferences']['OMI_XTrack_UV2_Index']), 
                    int(file_id['preferences']['OMI_ATrack_Index']), 
                    file_id['preferences']['OMI_utcTime'], 
                    calibrationFilename,
                    cldFilename=cldFilename
                )
            else:
                raise RuntimeError("Read_OMI_without_RadCal is not implemented yet")
                # end if OMI_RAD_CALRUN_FLAG == 0:

            # modify the OMI NESR if year >= 2010
            if o_dateStruct['year'] >= 2010:
                ind = np.where(o_omi['Earth_Radiance']['EarthRadianceNESR'] > 0)[0]
                o_omi['Earth_Radiance']['EarthRadianceNESR'][ind] = o_omi['Earth_Radiance']['EarthRadianceNESR'][ind] * 2
                
        ################   TROPOMI
        if 'TROPOMI' in instruments:
            # EM NOTE Retrievals must be set as 'fullfilter' if retrieving O3 or any other gas from multiple bands seperately
            # This invokes the defaults spectral windows, which are specified in the OSP/Strategy_Tables/Defaults, with the TROPOMI bands defined in the
            # *_CrIS_TROPOMI.asc and the *_TROPOMI.asc file.

            ### Here we need soft calibration for UV1, UV2 and UVIS for O3 retrieval, unclear if necessary
            ### for other bands. 
            #calibrationFilename = table_get_pref(i_table_struct,  'tropomi_calibrationFilename')

            outputFilenameTROPOMI = './Input/Radiance_TROPOMI_' + my_key + '.pkl'
            
            # EM - Depending on the type of retrieval for TROPOMI, the number of filenames will vary, therefore
            # all filenames are fed in as a list, comma seperated. This is split in 'read_tropomi'.
            filename = []
            XTrack = []
            for ii in windows: # There are 8 TROPOMI bands, check which windows are invoked
                if ii['instrument'] == 'TROPOMI': # EM Need this check for duel band purposes
                    if file_id['preferences'][f"TROPOMI_filename_{ii['filter']}"] is None: 
                        raise RuntimeError(f"TROPOMI L1B file for BAND not found {ii['filter']}")
                    else:
                        filename.append(file_id['preferences'][f"TROPOMI_filename_{ii['filter']}"])
                        XTrack.append(file_id['preferences'][f"TROPOMI_XTrack_Index_{ii['filter']}"])

            irrFilename = file_id['preferences']['TROPOMI_IRR_filename']
            cldFilename = file_id['preferences']['TROPOMI_Cloud_filename']

            
            if not os.path.isfile(outputFilenameTROPOMI):   # THIS SHOULD BE IF NOT, REMOVED NOT SO PICKLE FILE ISN'T READ FOR TIME BEING
                TROPOMI_RAD_CALRUN_FLAG = 0
                if 'TROPOMI_Rad_calRun_flag' in file_id['preferences']:
                    TROPOMI_RAD_CALRUN_FLAG = int(file_id['preferences']['TROPOMI_Rad_calRun_flag'])
                
                if TROPOMI_RAD_CALRUN_FLAG == 0:
                    o_tropomi = read_tropomi(
                        filename,
                        irrFilename,
                        cldFilename,
                        XTrack,  # EM - Like filename, xtrack will be variable, so fed in as list  
                        int(file_id['preferences']['TROPOMI_ATrack_Index']),
                        file_id['preferences']['TROPOMI_utcTime'],
                        windows,
                        calibrationFilename  # EM - Calibration to be implemented 
                    )
                else:
                    o_tropomi = read_tropomi(
                        filename,
                        irrFilename,
                        cldFilename,
                        XTrack,  # EM - Like filename, xtrack will be variable, so fed in as list  
                        int(file_id['preferences']['TROPOMI_ATrack_Index']),
                        file_id['preferences']['TROPOMI_utcTime'],
                        windows
                    )
                # end if OMI_RAD_CALRUN_FLAG == 0:
                
            else:
                logger.info(f"UNPICKLE_ME {outputFilenameTROPOMI}")

                with open(outputFilenameTROPOMI, 'rb') as pickle_handle:
                    o_tropomi = pickle.load(pickle_handle)
            # end else portion of if not (os.path.isfile(outputFilenameTROPOMI)):

        if 'OCO2' in instruments:
            outputFilenameOCO2 = './Input/Radiance_OCO2_' + my_key + '.nc'

            if not os.path.isfile(outputFilenameOCO2):
                filename = file_id['preferences']['OCO2_filename']
                radianceSource = table_get_pref(i_table_struct, "radianceSource")
                o_oco2 = oco2_read_radiance(file_id['preferences'], radianceSource)
            else:
                raise RuntimeError(f"Not implemented {outputFilenameCRIS}")


        # get surface altitude IN METERS
        if 'TES' in instruments:
            surfaceAltitude = o_tes['surfaceElevation']
        elif 'AIRS' in instruments:
            surfaceAltitude = o_airs['surfaceAltitude']
        elif 'CRIS' in instruments:
            # A note about case of keys in o_cris.  All cases are upper to be less confusing.
            surfaceAltitude = o_cris['surfaceAltitude'.upper()]
            if surfaceAltitude < 0.0:
                surfaceAltitude = 0
        elif 'OMI' in instruments:
            surfaceAltitude = o_omi['Earth_Radiance']['ObservationTable']['TerrainHeight'][0]
        elif 'TROPOMI' in instruments:
            for i in range(0, len(o_tropomi['Earth_Radiance']['ObservationTable']['ATRACK'])):        
                surfaceAltitude = read_tropomi_surface_altitude(
                    o_tropomi['Earth_Radiance']['ObservationTable']['Latitude'][i], 
                    o_tropomi['Earth_Radiance']['ObservationTable']['Longitude'][i])
                o_tropomi['Earth_Radiance']['ObservationTable']['TerrainHeight'][i] = surfaceAltitude
        elif 'OCO2' in instruments:
            surfaceAltitude = np.mean(o_oco2['surface_altitude_m'])
        else:
            raise RuntimeError('instruments ~= TES, AIRS, OMI, CRIS, TROPOMI, OCO2\nNeed surface altitude')

        # initial guess 
        ospDirectory = '../OSP'
        setupFilename = table_get_pref(i_table_struct, "initialGuessSetupDirectory") + '/L2_Setup_Control_Initial.asc'

        constraintFlag = False
        stateInitial = get_state_initial(
            setupFilename, ospDirectory, o_latitude, o_longitude, o_dateStruct, file_id['preferences'], 
            constraintFlag, surfaceAltitude, oceanFlag, gmao_path, gmao_type
        )

        # AT_LINE 479 src_ms-2019-05-29/script_retrieval_setup_ms.pro
        # constraint vector
        ospDirectory = '../OSP'
        setupFilename = table_get_pref(i_table_struct, "initialGuessSetupDirectory") + '/L2_Setup_Control_Constraint.asc'

        # state for constraint
        constraintFlag = True
        stateConstraint = get_state_initial(
            setupFilename, ospDirectory, o_latitude, o_longitude, o_dateStruct, file_id['preferences'],
            constraintFlag, surfaceAltitude, oceanFlag, gmao_path, gmao_type
        )

        # AT_LINE 489 src_ms-2019-05-29/script_retrieval_setup_ms.pro
        # get tes, omi, airs pars... 
        # tes: boresight angle.  OMI ring, cloud, albedo. 
        # airs: angle

        # AT_LINE 492 src_ms-2019-05-29/script_retrieval_setup_ms.pro
        if 'TES' in instruments:
            # note:  o_tes should contain all the fields in stateInitial['current']['tes']
            # boresightNadirRadians, orbitInclinationAngle, viewMode, instrumentAzimuth, instrumentLatitude, geoPointing, targetRadius, instrumentRadius, orbitAscending
            (stateInitial, stateConstraint) = _clean_up_dictionaries(instrument_name.lower(), stateInitial, stateConstraint, o_tes)

        # AT_LINE 518 src_ms-2019-05-29/script_retrieval_setup_ms.pro
        if 'CRIS' in instruments:
            o_cris['l1bType'] = cris_type
            (stateInitial, stateConstraint) = _clean_up_dictionaries(instrument_name.lower(), stateInitial, stateConstraint, o_cris)

        # AT_LINE 533 src_ms-2019-05-29/script_retrieval_setup_ms.pro
        if 'AIRS' in instruments:
            (stateInitial, stateConstraint) = _clean_up_dictionaries(instrument_name.lower(), stateInitial, stateConstraint, o_airs)
        
        # AT_LINE 548 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        if 'OMI' in instruments:
            # start parameters
            temp_end = 2 # Dejian revised Nov 22, 2016 for adding the functionality of radiance calibration run
            if int(file_id['preferences']['OMI_Rad_calRun_flag']) == 1:
                temp_end = 1

            # PYTHON_NOTE: The keys in omi_pars should closely match the keys of omi dictionary in new_state_structures.py
            omi_pars = {
                'surface_albedo_uv1': o_omi['SurfaceAlbedo']['MonthlyMinimumSurfaceReflectance'],                 
                'surface_albedo_uv2': o_omi['SurfaceAlbedo']['MonthlyMinimumSurfaceReflectance'],                 
                'surface_albedo_slope_uv2': np.float64(0.0),                                                            
                'nradwav_uv1': np.float64(0.0),                                                            
                'nradwav_uv2': np.float64(0.0),                                                            
                'odwav_uv1': np.float64(0.0),                                                           
                'odwav_uv2': np.float64(0.0),                                                           
                'odwav_slope_uv1': np.float64(0.0),                                                            
                'odwav_slope_uv2': np.float64(0.0),                                                            
                'ring_sf_uv1': np.float64(1.9),                                                            
                'ring_sf_uv2': np.float64(1.9),                                                            
                'cloud_fraction': o_omi['Cloud']['CloudFraction'],                                             
                'cloud_pressure': o_omi['Cloud']['CloudPressure'],                                             
                'cloud_Surface_Albedo': 0.8, # Same key in 'omi' dict as in new_state_structures.py
                'xsecscaling': np.float64(1.0),                                                            
                'resscale_uv1': np.float64(0.0)-999,                                                        
                'resscale_uv2': np.float64(0.0)-999,                                                        
                'SPACECRAFTALTITUDE': np.mean(o_omi['Earth_Radiance']['ObservationTable']['SpacecraftAltitude']),  # Same key in 'omi' dict as in new_state_structures.py
                'sza_uv1': o_omi['Earth_Radiance']['ObservationTable']['SolarZenithAngle'][0],          
                'raz_uv1': o_omi['Earth_Radiance']['ObservationTable']['RelativeAzimuthAngle'][0],      
                'vza_uv1': o_omi['Earth_Radiance']['ObservationTable']['ViewingZenithAngle'][0],        
                'sca_uv1': o_omi['Earth_Radiance']['ObservationTable']['ScatteringAngle'][0],           
                'sza_uv2': np.mean(o_omi['Earth_Radiance']['ObservationTable']['SolarZenithAngle'][1:temp_end+1]),     
                'raz_uv2': np.mean(o_omi['Earth_Radiance']['ObservationTable']['RelativeAzimuthAngle'][1:temp_end+1]), 
                'vza_uv2': np.mean(o_omi['Earth_Radiance']['ObservationTable']['ViewingZenithAngle'][1:temp_end+1]),   
                'sca_uv2': np.mean(o_omi['Earth_Radiance']['ObservationTable']['ScatteringAngle'][1:temp_end+1])       
            }

            ttState = []
            if stateConstraint is not None:
                ttState = list(stateConstraint['current']['omi'].keys())
            
            if stateInitial is not None:
                ttState = list(stateInitial['current']['omi'].keys())

            for ii in range(0, len(ttState)):
                if stateConstraint is not None:
                    stateConstraint['current']['omi'][ttState[ii]] = omi_pars[ttState[ii]]

                if stateInitial is not None:
                    stateInitial['current']['omi'][ttState[ii]] = omi_pars[ttState[ii]]

                    
            # end for ii in range(0, len(ttState)):
        # end if 'OMI' in instruments:

        if 'TROPOMI' in instruments:
            # start parameters, for TROPOMI, declaring them, and then fill them depending on how many
            # bands are used. This is based on the OMI setup, but assuming variable number of bands
            tropomi_pars = {
                'surface_albedo_BAND1': np.float64(0.0),                                            
                'surface_albedo_BAND2': np.float64(0.0),    
                'surface_albedo_BAND3': np.float64(0.0),
                'surface_albedo_BAND7': np.float64(0.0),
                'surface_albedo_slope_BAND1': np.float64(0.0),        
                'surface_albedo_slope_BAND2': np.float64(0.0),    
                'surface_albedo_slope_BAND3': np.float64(0.0),
                'surface_albedo_slope_BAND7': np.float64(0.0),
                'surface_albedo_slope_order2_BAND2': np.float64(0.0),   
                'surface_albedo_slope_order2_BAND3': np.float64(0.0),   
                'surface_albedo_slope_order2_BAND7': np.float64(0.0),   
                'solarshift_BAND1': np.float64(0.0),    
                'solarshift_BAND2': np.float64(0.0),    
                'solarshift_BAND3': np.float64(0.0),    
                'solarshift_BAND7': np.float64(0.0),    
                'radianceshift_BAND1': np.float64(0.0),    
                'radianceshift_BAND2': np.float64(0.0),    
                'radianceshift_BAND3': np.float64(0.0),    
                'radianceshift_BAND7': np.float64(0.0),    
                'radsqueeze_BAND1': np.float64(0.0),    
                'radsqueeze_BAND2': np.float64(0.0),    
                'radsqueeze_BAND3':np.float64(0.0),    
                'radsqueeze_BAND7':np.float64(0.0),    
                'ring_sf_BAND1': np.float64(0.0),    
                'ring_sf_BAND2': np.float64(0.0),    
                'ring_sf_BAND3': np.float64(0.0),
                'ring_sf_BAND7': np.float64(0.0),
                'temp_shift_BAND3' : np.float64(0.0),    
                'temp_shift_BAND7' : np.float64(0.0),    
                'cloud_fraction': np.float64(0.0),    
                'cloud_pressure': np.float64(0.0),    
                'cloud_Surface_Albedo': np.float64(0.0),    
                'xsecscaling': np.float64(0.0),    
                'resscale_O0_BAND1': np.float64(0.0),
                'resscale_O1_BAND1': np.float64(0.0),  
                'resscale_O2_BAND1': np.float64(0.0),  
                'resscale_O0_BAND2': np.float64(0.0),  
                'resscale_O1_BAND2': np.float64(0.0),      
                'resscale_O2_BAND2': np.float64(0.0),    
                'resscale_O0_BAND3': np.float64(0.0),
                'resscale_O1_BAND3': np.float64(0.0),
                'resscale_O2_BAND3': np.float64(0.0),        
                'resscale_O0_BAND7': np.float64(0.0),
                'resscale_O1_BAND7': np.float64(0.0),
                'resscale_O2_BAND7': np.float64(0.0),        
                'sza_BAND1': np.float64(0.0),    
                'raz_BAND1': np.float64(0.0),    
                'vza_BAND1': np.float64(0.0),    
                'sca_BAND1': np.float64(0.0),    
                'sza_BAND2': np.float64(0.0),    
                'raz_BAND2': np.float64(0.0),    
                'vza_BAND2': np.float64(0.0),    
                'sca_BAND2': np.float64(0.0),    
                'sza_BAND3': np.float64(0.0),    
                'raz_BAND3': np.float64(0.0),    
                'vza_BAND3': np.float64(0.0),    
                'sca_BAND3': np.float64(0.0),    
                'sza_BAND7': np.float64(0.0),    
                'raz_BAND7': np.float64(0.0),    
                'vza_BAND7': np.float64(0.0),    
                'sca_BAND7': np.float64(0.0),    
                'SPACECRAFTALTITUDE': np.float64(0.0)
            }

            # PYTHON_NOTE: The keys in tropomi_pars should closely match the keys of tropomi dictionary in new_state_structures.py 
            # These parameters are not band specific
            if o_tropomi['Cloud']['CloudFraction'] == 0.0:
                tropomi_pars['cloud_fraction'] = 0.01 # EM NOTE - So we can fit Cloud albedo, due to calibration errors.
            else:
                tropomi_pars['cloud_fraction'] = (o_tropomi['Cloud']['CloudFraction'])
            tropomi_pars['cloud_pressure'] = (o_tropomi['Cloud']['CloudPressure'])
            tropomi_pars['cloud_Surface_Albedo'] = 0.8 #(o_tropomi['Cloud']['CloudAlbedo']) # Same key in 'tropomi' dict as in new_state_structures.py
            tropomi_pars['SPACECRAFTALTITUDE'] = (np.mean(o_tropomi['Earth_Radiance']['ObservationTable']['SpacecraftAltitude']))
            tropomi_pars['xsecscaling'] = (np.float64(1.0))
            
            current_band = []
            for ii, band in enumerate(windows):  # 8 bands in TROPOMI
                # Assuming that values are appended to o_tropomi from UV to higher wavelengths
                if band['instrument'] == 'TROPOMI':  # EM - Necessary for dual band retrievals
                    if current_band != band['filter']:
                        current_band = band['filter']
                        tropomi_pars[f"surface_albedo_{band['filter']}"] = (o_tropomi['SurfaceAlbedo']['MonthlyMinimumSurfaceReflectance'])
                        tropomi_pars[f"surface_albedo_slope_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"surface_albedo_slope_order2_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"solarshift_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"radianceshift_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"radsqueeze_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"temp_shift_{band['filter']}"] = (np.float64(1.0))
                        tropomi_pars[f"ring_sf_{band['filter']}"] = (np.float64(1.9))
                        tropomi_pars[f"resscale_O0_{band['filter']}"] = (np.float64(1.0))
                        tropomi_pars[f"resscale_O1_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"resscale_O2_{band['filter']}"] = (np.float64(0.0))
                        tropomi_pars[f"sza_{band['filter']}"] = (o_tropomi['Earth_Radiance']['ObservationTable']['SolarZenithAngle'][ii])
                        tropomi_pars[f"rza_{band['filter']}"] = (o_tropomi['Earth_Radiance']['ObservationTable']['RelativeAzimuthAngle'][ii])
                        tropomi_pars[f"vza_{band['filter']}"] = (o_tropomi['Earth_Radiance']['ObservationTable']['ViewingZenithAngle'][ii])
                        tropomi_pars[f"sca_{band['filter']}"] = (o_tropomi['Earth_Radiance']['ObservationTable']['ScatteringAngle'][ii])
                    else:
                        current_band = band['filter']
                        continue
                else:
                    continue  
                        

            ttState = []
            if stateConstraint is not None:
                ttState = list(stateConstraint['current']['tropomi'].keys())
            
            if stateInitial is not None:
                ttState = list(stateInitial['current']['tropomi'].keys())

            for ii in range(0, len(ttState)):
                if stateConstraint is not None:
                    stateConstraint['current']['tropomi'][ttState[ii]] = tropomi_pars[ttState[ii]]

                if stateInitial is not None:
                    stateInitial['current']['tropomi'][ttState[ii]] = tropomi_pars[ttState[ii]]
            # end for ii in range(0, len(ttState)):
        # end if 'TROPOMI' in instruments:


        # OCO-2 
        if 'OCO2' in instruments:

            ttState = []
            if stateConstraint is not None:
                ttState = list(stateConstraint['current']['oco2'].keys())
            
            if stateInitial is not None:
                ttState = list(stateInitial['current']['oco2'].keys())

            for ii in range(0, len(ttState)):
                if stateConstraint is not None:
                    stateConstraint['current']['oco2'][ttState[ii]] = o_oco2[ttState[ii]]

                if stateInitial is not None:
                    stateInitial['current']['oco2'][ttState[ii]] = o_oco2[ttState[ii]]


            #PRINT, 'Get OCO pars' 
            #; get parameters specific to nir/3/...
            #nir_pars = {footprint:fileid.oco2_footprint} # not sure what need yet

            # copy angles/geolocation info to structures
            # IF N_ELEMENTS(stateConstraint) GT 0 THEN ttState = tag_names(stateConstraint.current.nir)
            # IF N_ELEMENTS(stateInitial) GT 0 THEN ttState = tag_names(stateInitial.current.nir)
            # ttnir = tag_names(nir_pars)
            # FOR ii = 0, N_ELEMENTS(ttState)-1 DO BEGIN
            #     indnir = where(ttnir EQ ttState[ii])
            #     IF N_ELEMENTS(stateConstraint) GT 0 and indnir[0] GE 0 THEN stateConstraint.current.nir.(ii) = nir_pars.(indnir)
            #     IF N_ELEMENTS(stateInitial) GT 0 and indnir[0] GE 0 THEN stateInitial.current.nir.(ii) = nir_pars.(indnir)
            # ENDFOR

            # set table.pressurefm to stateConstraint.pressure because OCO-2 is on sigma levels
            i_table_struct['pressureFM'] = stateInitial['current']['pressure']
            
            #stateConstraint['current']['cloudPars']['use'] = 'no'
            #stateInitial['current']['cloudPars']['use'] = 'no'
        #ENDIF


        # AT_LINE 593 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
        if len(stateInitial) > 0:
            # set surface type for products
            oceanString = ['Land', 'Ocean']
            stateInitial['surfaceType'] = oceanString[oceanFlag]    # The type of oceanFlag should be int here so we can use it as an index.
            stateInitial['current']['surfaceType'] = oceanString[oceanFlag]
        
            if i_writeOutput:
                # Because the write_state function modify the 'current' fields of stateInitial structure, we give it a copy.
                stateCurrentCopy = deepcopy(stateInitial['current']) 
                write_state(directoryIG, ObjectView(stateInitial), ObjectView(stateCurrentCopy), my_key="_" + my_key, writeAltitudes=0)
                del stateCurrentCopy # Delete the temporary object.

        # AT_LINE 577 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        # AT_LINE 610 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
        if len(stateConstraint) > 0:
            #  set surface type for products
            oceanString = ['Land', 'Ocean']
            stateConstraint['surfaceType'] = oceanString[oceanFlag]
            stateConstraint['current']['surfaceType'] = oceanString[oceanFlag]

            if i_writeOutput:
                # Because the write_state function modify the 'current' fields of stateConstraint structure, we give it a copy.
                stateCurrentCopy = deepcopy(stateConstraint['current'])
                write_state(directoryConstraint, ObjectView(stateConstraint), ObjectView(stateCurrentCopy), my_key="_" + my_key, writeAltitudes=0)
                del stateCurrentCopy

        
        # AT_LINE 595 script_retrieval_setup_ms.pro script_retrieval_setup_ms
        # AT_LINE 628 src_ms-2019-05-29/script_retrieval_setup_ms.pro script_retrieval_setup_ms
        # set up state


        o_state_info = stateInitial

        # types are from a priori not initial.  ch3ohtype is needed for constraint selection
        o_state_info['ch3ohtype'] = stateConstraint['ch3ohtype']

        # get type from 

        #  set surface type for products
        oceanString = ['Land', 'Ocean']
        o_state_info['surfaceType'] = oceanString[oceanFlag]


        # Make a deepcopy of stateInitial['current'] and stateConstraint['current'] to o_state_info so each will have its own memory.

        o_state_info['initialInitial'] = deepcopy(stateInitial['current'])
        o_state_info['initial'] = deepcopy(stateInitial['current'])
        o_state_info['current'] = deepcopy(stateInitial['current'])
        o_state_info['constraint'] = deepcopy(stateConstraint['current'])


        return (o_airs, o_cris, o_omi, o_tropomi, o_tes, o_oco2, o_state_info) # More instrument data later.
        

    def save_pickle(self, save_pickle_file, **kwargs):
        '''Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy.'''
        self.capture_directory.save_directory(self.run_dir, vlidort_input=None)
        pickle.dump([self, kwargs], open(save_pickle_file, "wb"))

    @classmethod
    def load_retrieval_strategy(cls, save_pickle_file, path=".",
                                change_to_dir = False,
                                osp_dir=None, gmao_dir=None,
                                vlidort_cli=None):
        '''This pairs with save_pickle.'''
        res, kwargs = pickle.load(open(save_pickle_file, "rb"))
        res.run_dir = f"{os.path.abspath(path)}/{res.capture_directory.runbase}"
        res.strategy_table.filename = f"{res.run_dir}/{os.path.basename(res.strategy_table.filename)}"
        res.capture_directory.extract_directory(path=path,
                              change_to_dir=change_to_dir, osp_dir=osp_dir,
                              gmao_dir=gmao_dir)
        if(vlidort_cli is not None):
            res.vlidort_cli = vlidort_cli
        return res, kwargs

class RetrievalStrategyCaptureObserver:
    '''Helper class, pickles RetrievalStrategy at each time notify_update is
    called. Intended for unit tests and other kinds of debugging.'''
    def __init__(self, basefname, location_to_capture):
        self.basefname = basefname
        self.location_to_capture = location_to_capture

    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location != self.location_to_capture):
            return
        fname = f"{self.basefname}_{retrieval_strategy.table_step}.pkl"
        retrieval_strategy.save_pickle(fname, **kwargs)
        
__all__ = ["RetrievalStrategy", "RetrievalStrategyCaptureObserver"]    

    
