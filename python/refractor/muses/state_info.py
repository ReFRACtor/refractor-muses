from __future__ import annotations # We can remove this when we upgrade to python 3.9
import abc
from .priority_handle_set import PriorityHandleSet
import refractor.muses.muses_py as mpy
import copy
import refractor.framework as rf
import numpy as np
import numbers

class SpeciesOrParametersState:
    '''Muses-py tends to call everything in its state "species",
    although in a few places things things are called
    "parameters". These should really be thought of as "things that go
    into a StateVector".

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
    def value(self):
        raise NotImplementedError

    
class SpeciesOrParametersHandle(object, metaclass=abc.ABCMeta):
    '''Return 3 SpeciesOrParametersState objects, for initialInitial, initial and
    current state. For many classes, this will just be a deepcopy of the same class,
    but at least for now the older muses-py code stores these in different places.
    We can perhaps change this interface in the future as we move out the old muses-py
    stuff.
    '''
    @abc.abstractmethod
    def species_object(self, state_info : "StateInfo",
                       species_name : str) -> \
            tuple[bool, tuple[SpeciesOrParametersState,
               SpeciesOrParametersState, SpeciesOrParametersState] | None]:
        raise NotImplementedError

class SpeciesOrParametersHandleSet(PriorityHandleSet):
    '''This maps a species name to the SpeciesOrParametersState object that handles
    it.'''
    def species_object(self, state_info : StateInfo, species_name : str) -> \
            tuple[SpeciesOrParametersState, SpeciesOrParametersState,
                  SpeciesOrParametersState]:
        return self.handle(state_info, species_name)

    def handle_h(self, h : SpeciesOrParametersHandle,
                 state_info : StateInfo, species_name : str)  -> \
            tuple[bool, tuple[SpeciesOrParametersState,
               SpeciesOrParametersState, SpeciesOrParametersState] | None]:
        return h.species_object(state_info, species_name)
    
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
        self.species_handle_set = copy.deepcopy(SpeciesOrParametersHandleSet.default_handle_set())
        self.initialInitial = {}
        self.initial = {}
        self.current = {}
        self.next_state = None
        
        # Odds an ends that are currently in the StateInfo. Doesn't exactly have
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
        
    @property
    def state_info_obj(self):
        return mpy.ObjectView(self.state_info_dict)

    def copy_current_initialInitial(self):
        self.state_info_dict["initialInitial"] = copy.deepcopy(self.state_info_dict["current"])

    def copy_current_initial(self):
        self.state_info_dict["initial"] = copy.deepcopy(self.state_info_dict["current"])

    def copy_state_one_next(self, state_one_next):
        self.state_info_dict["current"] = copy.deepcopy(state_one_next.__dict__)

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
    def species_on_levels(self):
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
        # The pressure is kind of like a SpeciesOnLevels, but it is a bit of a special
        # case. This is needed to interpret the rest of the data.
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
    
    def species_state(self, species_name, step="current"):
        '''Need to merge this with the one below, but for now leave as this.'''

        # We create the species objects on first use
        if species_name not in self.current:
            (self.initialInitial[species_name], self.initial[species_name],
             self.current[species_name]) = self.species_handle_set.species_object(self, species_name)
        if(step == "initialInitial"):
            return self.initialInitial[species_name]
        elif(step == "initial"):
            return self.initial[species_name]
        elif(step == "current"):
            return self.current[species_name]
        else:
            raise RuntimeError("step must be initialInitial, initial, or current")
        
        
__all__ = ["SpeciesOrParametersState", "SpeciesOrParametersHandle",
           "SpeciesOrParametersHandleSet",
           "SoundingMetadata", "StateInfo"]
