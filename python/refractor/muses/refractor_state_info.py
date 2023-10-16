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
    in the muses-py code. The RefractorStateInfo can be used to look these
    up.
    '''
    def __init__(self, name, value=None):
        self._name = name
        if(value is not None):
            # So we don't need special cases, always have a numpy array. A
            # single value is an array with one value.
            if(isinstance(value, numbers.Number)):
                self._value = np.array([value,])
            else:
                self._value = value.copy()
        else:
            self._value = np.array([0.0,])
        self._apriori_cov = np.array([[0.0,],])

    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self._value

class SpeciesOrParametersWithFrequencyState(SpeciesOrParametersState):
    '''Some of the species also have frequencies associated with them.
    We return these as Refractor SpectralDomain objects.

    TODO I'm pretty sure these are in nm, but this would be worth verifying.'''
    def __init__(self, name):
        super().__init__(name)
        self._sr = rf.SpectralDomain([0.0], rf.Unit("nm"))
        
    @property
    def spectral_range(self):
        return self._sr

    @property
    def wavelength(self):
        '''Short cut to return the spectral range in units of nm.'''
        return self._sr.convert_wave(rf.Unit("nm"))

class EmissivityState(SpeciesOrParametersWithFrequencyState):
    def __init__(self, state_info, step="current"):
        super().__init__("emissivity")
        if(step not in ("current", "initial", "initialInitial")):
            raise RuntimeError("Don't support anything other than the current, initial, or initialInitial step")
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0,state_info.state_info_dict["emisPars"]["num_frequencies"])
        self._value = state_info.state_info_dict[step]["emissivity"][r]
        self._sr = rf.SpectralDomain(state_info.state_info_dict["emisPars"]["frequency"][r], rf.Unit("nm"))
        # We may actually want constraint, we need to see how this maps
        self._apriori_cov = np.array([[0.0,],])
        # Couple other pieces of metadata that seem worth keeping
        self._camel_distance = state_info.state_info_dict["emisPars"]["camel_distance"]
        self._prior_source = state_info.state_info_dict["emisPars"]["emissivity_prior_source"]
        
    @property
    def camel_distance(self):
        # Not sure what this is, but seems worth keeping
        return self._camel_distance

    @property
    def prior_source(self):
        '''Source of prior.'''
        return self._prior_source

class CloudState(SpeciesOrParametersWithFrequencyState):
    def __init__(self, state_info, step="current"):
        super().__init__("emissivity")
        if(step not in ("current", "initial", "initialInitial")):
            raise RuntimeError("Don't support anything other than the current, initial, or initialInitial step")
        # Probably to support old IDL, the arrays are larger than the actual data.
        # We need to subset to get the actual data.
        r = range(0,state_info.state_info_dict["cloudPars"]["num_frequencies"])
        # Note cloud has 2 columns. I'm not sure why, or if this matters. But for
        # now just carry this through
        self._value = state_info.state_info_dict[step]["cloudEffExt"][:,r]
        self._sr = rf.SpectralDomain(state_info.state_info_dict["cloudPars"]["frequency"][r], rf.Unit("nm"))
        # We may actually want constraint, we need to see how this maps
        self._apriori_cov = np.array([[0.0,],])
        # Couple other pieces of metadata that seem worth keeping
        #self._camel_distance = state_info.state_info_dict["emisPars"]["camel_distance"]
        #self._prior_source = state_info.state_info_dict["emisPars"]["emissivity_prior_source"]
    
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
        self._surface_type = state_info.state_info_dict[step]['surfaceType'].upper()
        self._tai_time = state_info._tai_time
        self._sounding_id = state_info._sounding_id

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
    def tai_time(self):
        return self._tai_time

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
    RefractorStateInfo and makes in looks like we got it from a Level1bAirs file.
    We'll then eventually separate this out from RefractorStateInfo and put this
    over with the Observation.'''
    def __init__(self, rs):
        self.rs = rs

    def altitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["airs"]["satHeight"]), "km")

    def latitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["latitude"]), "deg")

    def longitude(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["longitude"]), "deg")

    def scan_angle(self, ind):
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["scanAng"]), "deg")

    def surface_altitude(self, ind):
        # Not clear if this is the best place for this, but for now shove here.
        # Framework didn't have this anywhere I don't think.
        return rf.DoubleWithUnit(float(self.rs.state_info_obj.current["tsa"]["surfaceAltitudeKm"]), "km")
    
# Not really clear how to handle this. But we'll start throwing something together.
# We'll want to separate these later on
class RefractorStateInfo:
    '''Like RefractorRetrievalInfo, this just wraps up the existing StateInfo
    class so we can figure out how it is used in code and get clear boundaries
    for the class.

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
    def __init__(self, state_info_dict, run_dir):
        self.state_info_dict = state_info_dict
        # Read some metadata that isn't already available
        tai_time = mpy.tes_file_get_preference(
            mpy.read_all_tes(f"{run_dir}/DateTime.asc")[1], "TAI_Time_of_ZPD")
        self._tai_time = float(tai_time)
        info_file = mpy.tes_file_get_struct(
            mpy.read_all_tes(f"{run_dir}/Measurement_ID.asc")[1])
        self._sounding_id = info_file['preferences']['key']
        
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

    def sounding_metadata(self, step="current"):
        return SoundingMetadata(self, step=step)
    
    def species_state(self, name, step="current"):
        '''Return the SpeciesOrParametersState for the give species/parameter name'''
        # TODO We will replace this with just collections of these objects, but for
        # now we map all this to the existing structure. Once the outside is fully
        # using this interface, we can clean up the insides here.
        if(name == "emissivity"):
            return EmissivityState(self, step=step)
        if(name == "cloudEffExt"):
            return CloudState(self, step=step)
        if(name in "PCLOUD", "PSUR"):
            return SpeciesOrParametersState(name, self.state_info_dict[step][name])
        raise KeyError(f"Don't recognize species name {name}")

    
        
        
