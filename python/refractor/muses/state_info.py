from __future__ import annotations # We can remove this when we upgrade to python 3.9
import abc
from .priority_handle_set import PriorityHandleSet
import refractor.muses.muses_py as mpy
import copy
import refractor.framework as rf
import numpy as np
import numbers

class StateElement(object, metaclass=abc.ABCMeta):
    '''Muses-py tends to call everything in its state "species",
    although in a few places things things are called
    "parameters". These should really be thought of as "things that go
    into a StateVector". So we refer to these as StateElements, which also
    parallels the species we retrieve on referred to as RetrievalElements.

    We try to treat all these things as identical at some level, but
    there is some behavior that is species dependent. We'll sort that
    out, and try to figure out the right design here.

    These get referenced by a "name", usually called a "species_name"
    in the muses-py code. The StateInfo can be used to look these
    up.
    '''
    def __init__(self, state_info : "StateInfo", name : str):
        self._name = name
        self.state_info = state_info
        
    @property
    def name(self):
        return self._name
    
    @property
    @abc.abstractmethod
    def value(self):
        raise NotImplementedError

class RetrievableStateElement(StateElement):
    '''This has additional functionality to have a StateElement be retrievable,
    so things like having a priori and initial guess needed in a retrieval. Most
    StateElements are retrievable, but not all - so we separate out the
    functionality.'''
    @abc.abstractmethod
    def update_initial_guess(self, strategy_table : StrategyTable):
        '''Create/update a initial guess. This currently fills in a number
        of member variables. I'm not sure that all of this is actually needed,
        we may clean up this list. But right now RetrievalInfo needs all these
        values. We'll perhaps clean up RetrievalInfo, and then in turn clean this
        up.

        The list of variables filled in are:
        
        self.mapType
        self.pressureList
        self.altitudeList
        self.constraintVector
        self.initialGuessList
        self.trueParameterList
        self.pressureListFM
        self.altitudeListFM
        self.constraintVectorFM
        self.initialGuessListFM
        self.trueParameterListFM
        self.minimum
        self.maximum
        self.maximum_change
        self.mapToState
        self.mapToParameters
        self.constraintMatrix
        '''
        raise NotImplementedError
    
class StateElementHandle(object, metaclass=abc.ABCMeta):
    '''Return 3 StateElement objects, for initialInitial, initial and
    current state. For many classes, this will just be a deepcopy of the same class,
    but at least for now the older muses-py code stores these in different places.
    We can perhaps change this interface in the future as we move out the old muses-py
    stuff, it is sort of odd to create these 3 things at once. But we'll match
    what muses-py does for now
    '''
    @abc.abstractmethod
    def state_element_object(self, state_info : "StateInfo",
                             name : str) -> \
            tuple[bool, tuple[StateElement, StateElement, StateElement] | None]:
        raise NotImplementedError

class StateElementHandleSet(PriorityHandleSet):
    '''This maps a species name to the SpeciesOrParametersState object that handles
    it.'''
    def state_element_object(self, state_info : StateInfo, name : str) -> \
            tuple[StateElement, StateElement, StateElement]:
        return self.handle(state_info, name)

    def handle_h(self, h : StateElementHandle,
                 state_info : StateInfo, name : str)  -> \
            tuple[bool, tuple[StateElement, StateElement, StateElement] | None]:
        return h.state_element_object(state_info, name)
    
class SoundingMetadata:
    '''Not really clear that this belongs in the StateInfo, but the muses-py seems
    to at least allow the possibility of this changing from one step to the next.
    I'm not sure if that actually can happen, but there isn't another obvious place
    to put this metadata so we'll go ahead and keep this here.'''
    def __init__(self, state_info, step="current"):
        if(step not in ("current", "initial", "initialInitial")):
            raise RuntimeError("Don't support anything other than the current, initial, or initialInitial step")
        self._latitude = rf.DoubleWithUnit(state_info.state_info_dict[step]["latitude"], "deg")
        self._longitude = rf.DoubleWithUnit(state_info.state_info_dict[step]["longitude"], "deg")
        self._surface_altitude = rf.DoubleWithUnit(state_info.state_info_dict[step]["tsa"]["surfaceAltitudeKm"], "km")
        self._height = rf.ArrayWithUnit_double_1(state_info.state_info_dict[step]["heightKm"], "km")
        self._surface_type = state_info.state_info_dict[step]['surfaceType'].upper()
        self._tai_time = state_info._tai_time
        self._sounding_id = state_info._sounding_id
        self._utc_time = state_info._utc_time

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @property
    def surface_altitude(self):
        return self._surface_altitude

    @property
    def height(self):
        return self._height
    
    @property
    def tai_time(self):
        return self._tai_time

    @property
    def utc_time(self):
        return self._utc_time

    @property
    def local_hour(self):
        timestruct = mpy.utc(self.utc_time)
        hour = timestruct['hour'] + self.longitude.convert("deg").value / 180. * 12
        if hour < 0:
            hour += 24
        if hour > 24:
            hour -= 24
        return hour
    
    @property
    def wrong_tai_time(self):
        '''The muses-py function mpy.tai uses the wrong number of leapseconds, it
        doesn't include anything since 2006. To match old data, return the incorrect
        value so we can match the file. This should get fixed actually.'''
        timestruct = mpy.utc(self.utc_time)
        if(timestruct["yearfloat"] >= 2017.0):
            extraleapscond = 4
        elif(timestruct["yearfloat"] >= 2015.5):
            extraleapscond = 3
        elif(timestruct["yearfloat"] >= 2012.5):
            extraleapscond = 2
        elif(timestruct["yearfloat"] >= 2009.0):
            extraleapscond = 1
        else:
            extraleapscond = 0
        return self._tai_time-extraleapscond
    
    @property
    def sounding_id(self):
        return self._sounding_id

    @property
    def surface_type(self):
        return self._surface_type

    @property
    def is_ocean(self):
        return self.surface_type == "OCEAN"

    @property
    def is_land(self):
        return self.surface_type == "LAND"
        
class Level1bAirs:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        info_file = self.rs.info_file
        return {
            "AIRS_GRANULE" : np.int16(info_file['preferences']['AIRS_Granule']),
            "AIRS_ATRACK_INDEX" : np.int16(info_file['preferences']['AIRS_ATrack_Index']),
            "AIRS_XTRACK_INDEX" : np.int16(info_file['preferences']['AIRS_XTrack_Index']),
            "POINTINGANGLE_AIRS" : abs(self.scan_angle(0).value)
        }
            
    def altitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["airs"]["satHeight"]), "km")

    def latitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["latitude"]), "deg")

    def longitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["longitude"]), "deg")

    def scan_angle(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["airs"]["scanAng"]), "deg")

    def surface_altitude(self, ind):
        # Not clear if this is the best place for this, but for now shove here.
        # Framework didn't have this anywhere I don't think.
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["tsa"]["surfaceAltitudeKm"]), "km")

class Level1bTes:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        info_file = self.rs.info_file
        return {
            "TES_RUN" : np.int16(info_file['preferences']['TES_run']),
            "TES_SEQUENCE" : np.int16(info_file['preferences']['TES_sequence']),
            "TES_SCAN" : np.int16(info_file['preferences']['TES_scan']),
            "POINTINGANGLE_TES" : self.boresight_angle.convert("deg").value
        }

    @property
    def boresight_angle(self):
        return rf.DoubleWithUnit(self.rs.state_info_dict["current"]["boresightNadirRadians"], "rad")

class Level1bOmi:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        info_file = self.rs.info_file
        return {
            "OMI_ATRACK_INDEX": np.int16(info_file['preferences']['OMI_ATrack_Index']),
            "OMI_XTRACK_INDEX_UV1": np.int16(info_file['preferences']['OMI_XTrack_UV1_Index']),
            "OMI_XTRACK_INDEX_UV2": np.int16(info_file['preferences']['OMI_XTrack_UV2_Index']),
            "POINTINGANGLE_OMI" : abs(self.scan_angle(1).value)
        }

    def scan_angle(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["omi"][f"vza_uv{ind+1}"]), "deg")
    

class Level1bCris:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        info_file = self.rs.info_file
        return {
            "CRIS_GRANULE" : np.int16(info_file['preferences']['CRIS_Granule']),
            "CRIS_ATRACK_INDEX" : np.int16(info_file['preferences']['CRIS_ATrack_Index']),
            "CRIS_XTRACK_INDEX" : np.int16(info_file['preferences']['CRIS_XTrack_Index']),
            "CRIS_PIXEL_INDEX" : np.int16(info_file['preferences']['CRIS_Pixel_Index']),
            "POINTINGANGLE_CRIS" : abs(self.scan_angle(0).value),
            "CRIS_L1B_TYPE" : np.int16(self.l1b_type_int)
        }

    @property
    def l1b_type_int(self):
        return ['suomi_nasa_nsr', 'suomi_nasa_fsr', 'suomi_nasa_nomw',
                'jpss1_nasa_fsr', 'suomi_cspp_fsr','jpss1_cspp_fsr',
                'jpss2_cspp_fsr'].index(self.l1b_type)

    @property
    def l1b_type(self):
        return self.rs.state_info_dict["current"]["cris"]["l1bType"]
    
    def scan_angle(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_dict["current"]["cris"]["scanAng"]), "deg")

class Level1bTropomi:
    '''This is like a Level1b class from framework, although right now we won't
    bother making this actually one those. Instead this pulls stuff out of
    StateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from StateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs
        self.info_file = self.rs.info_file

    @property
    def sounding_desc(self):
        '''Different types of instruments have different description of the
        sounding ID. This gets used in retrieval_l2_output for metadata.'''
        res = {
            "TROPOMI_ATRACK_INDEX" : np.int16(self.info_file['preferences']['TROPOMI_ATrack_Index'])
        }
        for b in range(8):
            res[f"TROPOMI_XTRACK_INDEX_BAND{b+1}"] = self.xtrack_index(b)
            if(self.xtrack_index(b) < -998):
                res[f"POINTINGANGLE_TROPOMI_BAND{b+1}"] = -999.0
            else:
                sang = self.scan_angle(b).value
                res[f"POINTINGANGLE_TROPOMI_BAND{b+1}"] = np.abs(sang) if sang > -998 else -999.0
        return res
            
    def xtrack_index(self, ind):
        p = self.info_file["preferences"]
        ky = f'TROPOMI_XTrack_Index_BAND{ind+1}'
        return np.int16(p.get(ky, -999))
    
    def scan_angle(self, ind):
        v = self.rs.state_info_dict["current"]['tropomi'].get(f'vza_BAND{ind+1}', -999)
        return rf.DoubleWithUnit(float(v), "deg")
    
class StateInfo:
    '''
    A few functions seem sort of like member functions, we'll just make a list
    of these to sort out later but not try to get the full interface in place.

    script_retrieval_setup_ms
    states_initial_update - These seem to create the stateInfo
    get_species_information - Seems to read a lot from stateinfo
    update_state
    create_uip - lots of reading here
    modify_from_bt
    write_retrieval_input
    plot_results
    set_retrieval_results
    write_retrieval_summary
    error_analysis_wrapper
    write_products_one_jacobian
    write_products_one_radiance
    write_products_one
    '''
    def __init__(self):
        self.state_info_dict = None
        self._ordered_species_list = mpy.ordered_species_list()
        self.state_element_handle_set = copy.deepcopy(StateElementHandleSet.default_handle_set())
        self.initialInitial = {}
        self.initial = {}
        self.current = {}
        self.next_state = {}
        self.next_state_dict = {}
        
        # Odds and ends that are currently in the StateInfo. Doesn't exactly have
        # to do with the state, but we don't have another place for these.
        # Perhaps SoundingMetadata can migrate into its own thing, and these can
        # get moved over to there.
        self._tai_time = None
        self._utc_time = None
        self.info_file = None
        self._sounding_id = None
        
    def init_state(self, strategy_table : 'StrategyTable',
                 fm_obs_creator : 'FmObsCreator', instruments_all, run_dir : str):
        (_, _, _, _, _, _,
         self.state_info_dict) = mpy.script_retrieval_setup_ms(strategy_table.strategy_table_dict, False)
        self.state_info_dict = mpy.states_initial_update(
            self.state_info_dict, strategy_table.strategy_table_dict,
            fm_obs_creator.radiance(self, instruments_all), instruments_all)

        # Read some metadata that isn't already available
        tai_time = mpy.tes_file_get_preference(
            mpy.read_all_tes(f"{run_dir}/DateTime.asc")[1], "TAI_Time_of_ZPD")
        utc_time = mpy.tes_file_get_preference(
            mpy.read_all_tes(f"{run_dir}/DateTime.asc")[1], "UTC_Time")
        self._tai_time = float(tai_time)
        self._utc_time = utc_time
        self.info_file = mpy.tes_file_get_struct(
            mpy.read_all_tes(f"{run_dir}/Measurement_ID.asc")[1])
        self._sounding_id = self.info_file['preferences']['key']
        self.next_state_dict = None
        
    @property
    def state_info_obj(self):
        return mpy.ObjectView(self.state_info_dict)

    def copy_current_initialInitial(self):
        self.state_info_dict["initialInitial"] = copy.deepcopy(self.state_info_dict["current"])
        # Don't actually want to copy current, since a lot of the StateElements are
        # hardcoded to current. We can perhaps come up with some kind of "clone"
        # or "copy" function that is nothing on the existing muses-py, but copies
        # state information for something that maintains the state
        #self.initialInitial = copy.deepcopy(self.current)

    def copy_current_initial(self):
        self.state_info_dict["initial"] = copy.deepcopy(self.state_info_dict["current"])
        
        # Don't actually want to copy current, since a lot of the StateElements are
        # hardcoded to current. We can perhaps come up with some kind of "clone"
        # or "copy" function that is nothing on the existing muses-py, but copies
        # state information for something that maintains the state
        #self.initial = copy.deepcopy(self.current)

    def next_state_to_current(self):
        # We might have not actually called update, so skip this if we don't
        # have a next state
        if(self.next_state_dict is not None):
            self.state_info_dict["current"] = self.next_state_dict
        self.next_state_dict = None
        #self.current = copy.deepcopy(self.next_state)

    def l1b_file(self, instrument):
        if(instrument == "AIRS"):
            return Level1bAirs(self)
        if(instrument == "CRIS"):
            return Level1bCris(self)
        elif(instrument == "TES"):
            return Level1bTes(self)
        elif(instrument == "OMI"):
            return Level1bOmi(self)
        elif(instrument == "TROPOMI"):
            return Level1bTropomi(self)
        else:
            raise RuntimeError(f"Don't recognize instrument {instrument}")

    def sounding_metadata(self, step="current"):
        return SoundingMetadata(self, step=step)

    def has_true_values(self):
        '''Indicate if we have true values in our state info.'''
        return np.max(self.state_info_dict["true"]["values"]) > 0

    def gmao_tropopause_pressure(self):
        # Not clear how to handle incidental things like this. For now,
        # just make a clear function so we know we need some way of handling this.
        return self.state_info_dict["gmaoTropopausePressure"]

    @property
    def state_element_on_levels(self):
        return self.state_info_dict["species"]

    @property
    def nh3type(self):
        return self.state_info_dict["nh3type"]

    @property
    def hcoohtype(self):
        return self.state_info_dict["hcoohtype"]

    @property
    def ch3ohtype(self):
        return self.state_info_dict["ch3ohtype"]
    
    @property
    def pressure(self):
        # The pressure is kind of like a StateElementOnLevels, but it is a bit of a
        # special case. This is needed to interpret the rest of the data.
        return self.state_info_dict["current"]["pressure"]

    def order_species(self, species_list):
        '''This provides an order to the given species list.

        It isn't really clear why this is needed, but it is in the existing muses-py.
        We use the list order they have for items found in that list, and then we
        just alphabetize any species not found in that list. We could adapt this
        logic if needed (e.g., extend ordered_species_list), but is isn't really
        clear why the ordering is done in the first place.'''
        
        # The use of a tuple here is a standard python "trick" to separate out
        # the values sorted
        # by _ordered_species_list vs. alphabetically. In python
        # False < True, so all the items in _ordered_species_list are put first in
        # the sorted list. The second part of the tuple is only sorted for things
        # that are in the same set for the first test, so using integers from index
        # vs. string comparison is separated nicely.
        return sorted(species_list, key=lambda v:
                      (v not in self._ordered_species_list,
                       self._ordered_species_list.index(v) if
                       v in self._ordered_species_list else v))

    def update_state(self, retrieval_info : "RetrievalInfo",
                     results_list, do_not_update, cloud_prefs, step):
        '''Note this updates the current state, and also creates a "next_state".
        The difference is that current gets all the changes found in the
        results_list, but next_state only gets the elements updated that aren't
        listed in do_not_update. This allows things to be done in a particular
        retrieval step, but not actually propagated to the next retrieval step.
        Call next_state_to_current() to update the current state with the
        next_state (i.e., remove the changes for things listed in do_not_update).'''
        self.next_state_dict = copy.deepcopy(self.state_info_dict["current"])

        do_update_fm = np.zeros(retrieval_info.n_totalParametersFM)

        for state_element_name in retrieval_info.species_names:
            self.update_state_element(state_element_name, self,
                                      retrieval_info, results_list, do_not_update,
                                      cloud_prefs, step, do_update_fm)
        # Update altitude and air density
        indt = self.state_element_on_levels.index("TATM")
        indh = self.state_element_on_levels.index("H2O")
        smeta = self.sounding_metadata()
        (results, _) = mpy.compute_altitude_pge(
            self.pressure,
            self.state_info_dict["current"]["values"][indt, :],
            self.state_info_dict["current"]["values"][indh, :],
            smeta.surface_altitude.convert("m").value,
            smeta.latitude.value,
            None, True)
        self.state_info_dict["current"]['heightKm'] = results["altitude"] / 1000.0
        self.state_info_dict["current"]['airDensity'] = results["airDensity"]
        # Update doUpdateFM in i_retrievalInfo. Note it might be good to move this
        # out of this function, it isn't good to have "side effects". But leave for
        # now
        retrieval_info.retrieval_dict["doUpdateFM"] = do_update_fm

    def update_state_element(self, state_element_name, state_info,
                             retrieval_info, results_list, donotupdate,
                             cloud_prefs, step, do_update_fm):
        updateFlag = True
        ij = retrieval_info.species_names.index(state_element_name)
        i_stateOneNext = mpy.ObjectView(state_info.next_state_dict)
        if len(donotupdate) > 0:
            ind = []
            for nn in range(len(donotupdate)):
                if state_element_name == donotupdate[nn]:
                    ind.append(nn)

            if len(ind) > 0:
                updateFlag = False

        FM_Flag = True
        INITIAL_Flag = True
        TRUE_Flag = False
        CONSTRAINT_Flag = False

        result = mpy.get_vector(results_list, retrieval_info.retrieval_info_obj,
                                state_element_name, FM_Flag, INITIAL_Flag, TRUE_Flag,
                                CONSTRAINT_Flag)

        loc = []
        for ii in range(len(state_info.state_element_on_levels)):
            if state_element_name == state_info.state_element_on_levels[ii]:
                loc.append(ii)

        ind1 = retrieval_info.retrieval_info_obj.parameterStartFM[ij]
        ind2 = retrieval_info.retrieval_info_obj.parameterEndFM[ij]

        # set which parameters are updated in state AND in error
        # analysis... check movement from i.g.
        myinitial = copy.deepcopy(retrieval_info.initialGuessListFM)  # Make a copy so we can work on a copy instead of actually changing the content of i_retrievalInfo.initialGuessListFM.
        mapTypeListFM = (np.char.asarray(retrieval_info.retrieval_info_obj.mapTypeListFM)).lower()

        # For every indices where 'log' is in mapTypeListFM, we take the exponent of myinitial.
        # AT_LINE 49 Update_State.pro
        myinitial[mapTypeListFM == "log"] = np.exp(myinitial[[mapTypeListFM == "log"]])

        # AT_LINE 50 Update_State.pro

        abs_array = np.absolute(result - myinitial[ind1:ind2+1]) / np.absolute(result)

        compare_value = 1.0e-6 
        utilGeneral = mpy.UtilGeneral()
        ind = utilGeneral.WhereGreaterEqualIndices(abs_array, compare_value)

        if ind.size > 0:
            do_update_fm[ind+ind1] = 1
        else:
            # if all at i.g., then must've started at true e.g. for spectral
            # window selection.  Here we want accurate error estimates.
            do_update_fm[:] = 1

        my_map = mpy.get_one_map(retrieval_info.retrieval_info_obj, ij)

        # Get indices influenced by retrieval.
        n = retrieval_info.retrieval_info_obj.n_parametersFM[ij]

        ind = [0 for i in range(n)]
        for ii in range(0, n):
            if abs(np.sum(my_map['toState'][:, ii])) >= 1e-10:
                ind[ii] = 1

        ind = utilGeneral.WhereEqualIndices(ind, 1)

        # Code already interpolates to missing emissivity via the mapping

        # AT_LINE 70 Update_State.pro
        if state_element_name == 'EMIS':
            # only update non-zero emissivities.  Eventually move to
            # emis map
            # ind = where(result NE 0)
            if ind.size > 0:
                state_info.state_info_obj.current['emissivity'][ind] = result[ind]

            # mapping takes care of all interpolation.
            # see get_species_information for that.
            if updateFlag and (i_stateOneNext is not None):
                i_stateOneNext.emissivity = copy.deepcopy(state_info.state_info_obj.current['emissivity'])

        elif state_element_name == 'CLOUDEXT':
            # Note that the variable ind is the list of frequencies that are retrieved
            # AT_LINE 85 Update_State.pro
            if retrieval_info.retrieval_info_obj.type.lower() != 'bt_ig_refine':
                if ind.size > 0: 
                    # AT_LINE 87 Update_State.pro
                    state_info.state_info_obj.current['cloudEffExt'][0, ind] = result[ind]

                    # update all frequencies surrounded by current windows
                    # I think the PGE only updates retrieved frequencies
                    # AT_LINE 91 Update_State.pro

                    # PYTHON_NOTE: Because Python slice does not include the end point, we add 1 to np.amax(ind)
                    interpolated_array = mpy.idl_interpol_1d(
                        np.log(result[ind]),
                        state_info.state_info_obj.cloudPars['frequency'][ind],
                        state_info.state_info_obj.cloudPars['frequency'][np.amin(ind):np.amax(ind)+1]
                    )

                    state_info.state_info_obj.current['cloudEffExt'][0, np.amin(ind):np.amax(ind)+1] = np.exp(interpolated_array)[:]
                else:
                    assert False
            else:
                # IGR step
                # get update preferences
                updateAve = cloud_prefs['CLOUDEXT_IGR_Average'].lower()
                maxAve = float(cloud_prefs['CLOUDEXT_IGR_Max'])
                resetAve = float(cloud_prefs['CLOUDEXT_Reset_Value'])

                # Python note:  During development, we see that it is possible for the value of ind.size to be 0
                #               A side effect of that is we cannot use ind array as indices into other arrays.
                #               So before we can ind, we must check for the size first.
                # average in log-space
                # AT_LINE 105 Update_State.pro
                n = ind.size
                if n < 4:
                    # Sanity check for zero size array.
                    if n > 0:
                        ave = np.exp(np.sum(np.log(result[ind])) / len(result[ind]))
                else:
                    # DONT INCLUDE ENDPOINTS!!!!
                    # AT_LINE 110 Update_State.pro
                    # IDL code has ind0 = ind[1:ind.size-2] so for Python, we subtract 1 instead because Python slices does not include the end point.
                    ind0 = ind[1:ind.size-1]
                    ave = np.exp(np.sum(np.log(result[ind0])) / len(ind0))

                # Set everywhere to ave but keep structure in areas retrieved
                # AT_LINE 115 Update_State.pro
                state_info.state_info_obj.current['cloudEffExt'][:] = ave

                if updateAve == 'no':
                    if n > 0:
                        state_info.state_info_obj.current['cloudEffExt'][0, ind] = result[ind]

                        # update areas surrounded by current windows

                        # PYTHON_NOTE: Because Python slice does not include the end point, we add 1 to np.amax(ind)
                        state_info.state_info_obj.current['cloudEffExt'][0, np.amin(ind):np.amax(ind)+1] = \
                            np.exp(
                                mpy.idl_interpol_1d(
                                    np.log(result[ind]),
                                    state_info.state_info_obj.cloudPars['frequency'][ind],
                                    state_info.state_info_obj.cloudPars['frequency'][np.amin(ind):np.amax(ind)+1]
                                )
                            )
                else:
                    state_info.state_info_obj.current['cloudEffExt'][:] = ave

                # check each value to see if > maxAve
                # don't let get "too large" in refinement step
                # AT_LINE 131 Update_State.pro
                ind = utilGeneral.WhereGreaterEqualIndices(state_info.state_info_obj.current['cloudEffExt'][0, :], maxAve)

                # Sanity check for zero size array.
                if ind.size > 0:
                    state_info.state_info_obj.current['cloudEffExt'][0, ind] = resetAve
            # end part of: if stepType != 'bt_ig_refine':

            if updateFlag and (i_stateOneNext is not None):
                # Something strange here. Sometimes the variable i_stateOneNext is ObjectView, sometimes it is a dictionary.
                if isinstance(i_stateOneNext, dict):
                    i_stateOneNext['cloudEffExt'] = copy.deepcopy(state_info.state_info_obj.current['cloudEffExt'])
                else:
                    i_stateOneNext.cloudEffExt = copy.deepcopy(state_info.state_info_obj.current['cloudEffExt'])

            if state_info.state_info_obj.current['cloudEffExt'][0, 0] == 0.01:
                print(function_name, 'Warning: ', "state_info.state_info_obj.current['cloudEffExt'][0, 0] == 0.01")
        # end elif state_element_name == 'CLOUDEXT'

        elif state_element_name == 'CALSCALE':
            # Sanity check for zero size array.
            if ind.size > 0:
                state_info.state_info_obj.current['calibrationScale'][ind] = result[ind]

            if updateFlag and (i_stateOneNext is not None):
                i_stateOneNext.calibrationScale = copy.deepcopy(state_info.state_info_obj.current['calibrationScale'])

        elif state_element_name == 'CALOFFSET':
            if ind.size > 0:
                state_info.state_info_obj.current['calibrationOffset'][ind] = result[ind]

        elif 'OMI' in state_element_name and 'TROPOMI' not in state_element_name:

            # Not sure if an assignment is bug or not.  Will try to make a copy.
            #state_info.state_info_obj.current['omi']['OMIcloudfraction'] = result;  # Since we know the name of the key, we can use it directly.

            # PYTHON_NOTE: Because within (state_info.state_info_obj.current['omi'] we want to replace all fields with actual value from results.
            #              Using the species_name, 'OMICLOUDFRACTION', we look for 'cloud_fraction' in the keys of state_info.state_info_obj.current['omi'].
            #              So, given OMICLOUDFRACTION, we return the actual_omi_key as 'cloud_fraction'.
            species_name = state_element_name

            omiInfo = mpy.ObjectView(state_info.state_info_obj.current['omi'])

            actual_omi_key = mpy.get_omi_key(omiInfo, species_name)
            state_info.state_info_obj.current['omi'][actual_omi_key] = copy.deepcopy(result)  # Use the actual key and replace the exist key.

            if i_stateOneNext is not None and updateFlag is True:
                # Something strange here.  Sometimes the variable i_stateOneNext is ObjectView, sometimes it is a dictionary.
                if isinstance(i_stateOneNext, mpy.ObjectView):
                    i_stateOneNext.omi[actual_omi_key] = copy.deepcopy(state_info.state_info_obj.current['omi'][actual_omi_key])
                else:
                    i_stateOneNext['omi'][actual_omi_key] = copy.deepcopy(state_info.state_info_obj.current['omi'][actual_omi_key])

        # AT_LINE 175 Update_State.pro
        elif 'TROPOMI' in state_element_name:
            # PYTHON_NOTE: Because within (state_info.state_info_obj.current['tropomi'] we want to replace all fields with actual value from results.
            #              Using the species_name, 'TROPOMICLOUDFRACTION', we look for 'cloud_fraction' in the keys of state_info.state_info_obj.current['tropomi'].
            #              So, given TROPOMICLOUDFRACTION, we return the actual_tropomi_key as 'cloud_fraction'.
            species_name = state_element_name
            tropomiInfo = mpy.ObjectView(state_info.state_info_obj.current['tropomi'])

            actual_tropomi_key = mpy.get_tropomi_key(tropomiInfo, species_name)
            state_info.state_info_obj.current['tropomi'][actual_tropomi_key] = copy.deepcopy(result)  # Use the actual key and replace the exist key.

            if i_stateOneNext is not None and updateFlag is True:
                # Something strange here.  Sometimes the variable i_stateOneNext is ObjectView, sometimes it is a dictionary.
                if isinstance(i_stateOneNext, mpy.ObjectView):
                    i_stateOneNext.tropomi[actual_tropomi_key] = copy.deepcopy(state_info.state_info_obj.current['tropomi'][actual_tropomi_key])
                else:
                    i_stateOneNext['tropomi'][actual_tropomi_key] = copy.deepcopy(state_info.state_info_obj.current['tropomi'][actual_tropomi_key])

        elif 'NIR' in state_element_name[0:3]:
            #tag_names_str = tag_names(state.current.nir)
            #ntag = n_elements(tag_names_str)
            #tag_names_str_new = strarr(ntag)        
            #for tempi = 0,ntag-1 do tag_names_str_new[tempi] = 'NIR'+ replace(tag_names_str[tempi],'_','')
            #indtag = where(tag_names_str_new EQ retrieval.species[ij])
            my_species = state_element_name[3:].lower()
            if my_species == 'alblamb':
                mult = 1
                if state_info.state_info_obj.current['nir']['albtype'] == 2:
                    mult = 1.0/.07
                if state_info.state_info_obj.current['nir']['albtype'] == 3:
                    print(function_name, "Mismatch in albedo type")
                    assert False
                state_info.state_info_obj.current['nir']['albpl'] = result * mult
                my_species = 'albpl'
            elif my_species == 'albbrdf':
                mult = 1
                if state_info.state_info_obj.current['nir']['albtype'] == 1:
                    mult = .07
                if state_info.state_info_obj.current['nir']['albtype'] == 3:
                    print(function_name, "Mismatch in albedo type")
                    assert False
                state_info.state_info_obj.current['nir']['albpl'] = result * mult
                my_species = 'albpl'
            elif my_species == 'albcm':
                mult = 1
                if state_info.state_info_obj.current['nir']['albtype'] != 3:
                    print(function_name, "Mismatch in albedo type")
                    assert False
                state_info.state_info_obj.current['nir']['albpl'] = result * mult
                my_species = 'albpl'
            elif my_species == 'albbrdfpl':
                mult = 1
                if state_info.state_info_obj.current['nir']['albtype'] == 1:
                    mult = .07
                if state_info.state_info_obj.current['nir']['albtype'] == 3:
                    print(function_name, "Mismatch in albedo type")
                    assert False
                state_info.state_info_obj.current['nir']['albpl'] = result * mult
                my_species = 'albpl'
            elif my_species == 'alblambpl':
                mult = 1
                if state_info.state_info_obj.current['nir']['albtype'] == 2:
                    mult = 1/.07
                if state_info.state_info_obj.current['nir']['albtype'] == 3:
                    print(function_name, "Mismatch in albedo type")
                    assert False
                state_info.state_info_obj.current['nir']['albpl'] = result * mult
                my_species = 'albpl'
            elif my_species == 'disp':
                # update only part of the state
                npoly = np.int(len(result)/3)
                state_info.state_info_obj.current['nir']['disp'][:,0:npoly] = np.reshape(result,(3,2))
            elif my_species == 'eof':
                # reshape
                state_info.state_info_obj.current['nir']['eof'][:,:] = np.reshape(result,(3,3)) # checked ordering is good 12/2021
            elif my_species == 'cloud3d':
                # reshape
                state_info.state_info_obj.current['nir']['cloud3d'][:,:] = np.reshape(result,(3,2)) # checked ordering is good 12/2021
            else:
                 state_info.state_info_obj.current['nir'][my_species] = result

            if i_stateOneNext is not None and updateFlag is True:
                if isinstance(i_stateOneNext, mpy.ObjectView):
                    i_stateOneNext.nir[my_species] = state_info.state_info_obj.current['nir'][my_species]
                else:
                    i_stateOneNext['nir'][my_species] = state_info.state_info_obj.current['nir'][my_species]



        # AT_LINE 175 Update_State.pro
        elif state_element_name == 'PCLOUD':
            # Note: Variable result is ndarray (sequence) of size 730
            #       The variable  state_info.state_info_obj.current['PCLOUD'][0] is an element.  We cannot assign an array element with a sequence
            # AT_LINE 284 Update_State.pro
            if isinstance(result, np.ndarray):
                state_info.state_info_obj.current['PCLOUD'][0] = result[0]  # IDL_NOTE: With IDL, we can be sloppy, but in Python, we must use index [0] so we can just get one element.
            else:
                state_info.state_info_obj.current['PCLOUD'][0] = result

            # do bounds checking for refinement step
            if retrieval_info.retrieval_info_obj.type.lower() == 'bt_ig_refine':
                igr_min_reset = float(cloud_prefs['PCLOUD_IGR_Min'])
                resetValue = -1
                if 'PCLOUD_IGR_Reset_Value' in cloud_prefs.keys():
                    resetValue = float(cloud_prefs['PCLOUD_IGR_Reset_Value'])
                if resetValue == -1:
                    resetValue = int(cloud_prefs['PCLOUD_Reset_Value'])

                if state_info.state_info_obj.current['PCLOUD'][0] > state_info.state_info_obj.current['pressure'][0]:
                    state_info.state_info_obj.current['PCLOUD'][0] = state_info.state_info_obj.current['pressure'][1]

                if state_info.state_info_obj.current['PCLOUD'][0] < resetValue:
                    state_info.state_info_obj.current['PCLOUD'][0] = resetValue

            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.PCLOUD[0] = state_info.state_info_obj.current['PCLOUD'][0]

        elif state_element_name == 'TSUR':
            state_info.state_info_obj.current['TSUR'] = result
            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.TSUR = result
        elif state_element_name == 'PSUR':
            # surface pressure
            # update sigma levels

            state_info.state_info_obj.current['pressure'][0] = result

            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.pressure[0] = result
                i_stateOneNext.pressure = pressure_sigma(i_stateOneNext.pressure[0],len(i_stateOneNext.pressure), 'surface')
        elif state_element_name == 'PTGANG':
            state_info.state_info_obj.current['tes']['boresightNadirRadians'] = result
            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.tes['boresightNadirRadians'] = state_info.state_info_obj.current['tes']['boresightNadirRadians']

        elif state_element_name == 'RESSCALE':
            state_info.state_info_obj.current.residualscale[i_step:] = result
            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.residualScale = state_info.state_info_obj.current['residualScale']
        else:
            # AT_LINE 289 Update_State.pro
            max_index = (state_info.state_info_obj.current['values'].shape)[1]  # Get access to the 63 in (1,63)
            state_info.state_info_obj.current['values'][loc, :] = result[0:max_index]
            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.values[loc, :] = state_info.state_info_obj.current['values'][loc, :]

        # end part of if (state_element_name == 'EMIS'):

        locHDO = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'HDO')
        locH2O = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'H2O')
        locRetHDO = utilGeneral.WhereEqualIndices(retrieval_info.retrieval_info_obj.species, 'HDO')
        if (state_element_name == 'H2O') and (locHDO.size > 0) and (locRetHDO.size == 0):
            # get initial guess ratio...
            initialRatio = state_info.state_info_obj.initial['values'][locHDO[0], 0:len(result)] / state_info.state_info_obj.initial['values'][locH2O[0], 0:len(result)]

            # set HDO by initial ratio multiplied by retrieved H2O
            state_info.state_info_obj.current['values'][locHDO[0], 0:len(result)] = result * initialRatio
            if i_stateOneNext is not None and updateFlag is True:
                i_stateOneNext.values[locHDO[0], :] = state_info.state_info_obj.current['values'][locHDO[0], :]

        locH2O18 = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'H2O18')
        locH2O = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'H2O')
        locRetH2O18 = utilGeneral.WhereEqualIndices(retrieval_info.retrieval_info_obj.species, 'H2O18')
        if (state_element_name == 'H2O') and (locH2O18.size > 0) and (locRetH2O18.size == 0):
            # get initial guess ratio...
            initialRatio = state_info.state_info_obj.initial['values'][locH2O18[0], 0:len(result)] / state_info.state_info_obj.initial['values'][locH2O[0], 0:len(result)]
            # set HDO by initial ratio multiplied by retrieved H2O
            state_info.state_info_obj.current['values'][locH2O18[0], 0:len(result)] = result*initialRatio

        locH2O17 = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'H2O17')
        locH2O = utilGeneral.WhereEqualIndices(state_info.state_info_obj.species, 'H2O')
        locRetH2O17 = utilGeneral.WhereEqualIndices(retrieval_info.retrieval_info_obj.species, 'H2O17')
        if (state_element_name == 'H2O') and (locH2O17.size > 0) and (locRetH2O17.size == 0):
            # get initial guess ratio...
            initialRatio = state_info.state_info_obj.initial['values'][locH2O17[0], 0:len(result)] / state_info.state_info_obj.initial['values'][locH2O[0], 0:len(result)]
            # set HDO by initial ratio multiplied by retrieved H2O
            state_info.state_info_obj.current['values'][locH2O17[0], 0:len(result)] = result*initialRatio

    def state_element(self, name, step="current"):
        '''Return the state element with the given name.'''
        # We create the StateElement objects on first use
        if name not in self.current:
            (self.initialInitial[name], self.initial[name],
             self.current[name]) = self.state_element_handle_set.state_element_object(self, name)
        if(step == "initialInitial"):
            return self.initialInitial[name]
        elif(step == "initial"):
            return self.initial[name]
        elif(step == "current"):
            return self.current[name]
        else:
            raise RuntimeError("step must be initialInitial, initial, or current")
        
        
__all__ = ["StateElement", "StateElementHandle", "RetrievableStateElement",
           "StateElementHandleSet",
           "SoundingMetadata", "StateInfo"]
