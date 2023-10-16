import refractor.muses.muses_py as mpy
import copy
import refractor.framework as rf

class SpeciesOrParameters:
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
    def __init__(self, name):
        self._name = name
        self._value = np.array([0.0,])
        self._apriori_cov = np.array([[0.0,],])

    @property
    def name(self):
        return self._name
    
    @property
    def value(self):
        return self._value

class SpeciesOrParametersWithFrequency(SpeciesOrParameters):
    '''Some of the species also have frequencies associated with them.
    We return these as Refractor SpectralRange objects.'''
    def __init__(self, name):
        super().__init__(name)
        self._sr = rf.SpectralRange([0.0], "nm")
        
    @property
    def spectral_range(self):
        return self._sr

    @property
    def frequency(self):
        '''Short cut to return the spectral range in units of blah'''
        return self._sr.data
    
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
    def __init__(self, state_info_dict):
        self.state_info_dict = state_info_dict

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
        
        
        
