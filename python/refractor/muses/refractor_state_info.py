import refractor.muses.muses_py as mpy
import copy

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
        
        
