from .strategy_table import StrategyTable
from .state_info import StateInfo
from .order_species import order_species
import refractor.muses.muses_py as mpy
import copy
import numpy as np
from scipy.linalg import block_diag

class ErrorAnalysis:
    '''This just groups together some of the error analysis stuff together,
    to put this together. Nothing more than a shuffling around of stuff already
    in muses-py
    '''
    def __init__(self, strategy_table: StrategyTable, state_info: StateInfo):
        self.initialize_error_initial(strategy_table, state_info)
        self.error_current = copy.deepcopy(self.error_initial)
        # Code seems to assume these are object view.
        self.error_initial = mpy.ObjectView(self.error_initial)
        self.error_current = mpy.ObjectView(self.error_current)

    def initialize_error_initial(self, strategy_table: StrategyTable,
                                 state_info: StateInfo):
        smeta = state_info.sounding_metadata()
        # List of state elements we need covariance from. This is all the elements
        # we will retrieve, plus any interferents that get added in. This list
        # is unique elements, sorted by the order_species sorting
        
        selem_list = order_species(set(strategy_table.retrieval_elements_all_step) |
                              set(strategy_table.error_analysis_interferents_all_step))
        # Note the odd seeming "capitalize" here. This is because get_prior_error
        # uses the map type to look up files, and rather than "linear" or "log" it
        # needs "Linear" or "Log"
        selem_map_type = [state_info.state_element(t).mapType.capitalize()
                          for t in selem_list]
        
        surfacetype = "OCEAN" if smeta.is_ocean else "LAND"

        num_selem_list = len(selem_list)
        num_pressures = len(state_info.pressure)

        row = 0
        rowFM = 0
        pressure_list = []
        species_list = []
        map_list = []


        # Make block diagonal covariance.
        matrix_list = []
        # AT_LINE 25 get_prior_error.pro 
        for species_name in selem_list:
            # Note the odd seeming "capitalize" here. This is because get_prior_error
            # uses the map type to look up files, and rather than "linear" or "log" it
            # needs "Linear" or "Log"
            maptype = state_info.state_element(species_name).mapType.capitalize()

            # MMS override covariance matrix director so we can add
            # Band 7 stuff
            i_directory = "../OSP/Strategy_Tables/tropomi_nir/Covariance/"
            (matrix, pressureSa) = mpy.get_prior_covariance(
                species_name, smeta.latitude.value, state_info.pressure, 
                surfacetype, state_info.nh3type,
                state_info.ch3ohtype, state_info.hcoohtype, maptype, i_directory)

            pressure_list.extend(pressureSa)
            species_list.extend([species_name] * matrix.shape[0])
            matrix_list.append(matrix)
            map_list.extend([maptype.lower()] * matrix.shape[0])
        
        initial = block_diag(*matrix_list)

        # Off diagonal blocks for covariance.
        for ispecie1 in range(num_selem_list):
            specie1 = selem_list[ispecie1]
            for ispecie2 in range(ispecie1+1, num_selem_list):
                specie2 = selem_list[ispecie2]

                # AT_LINE 51 get_prior_error.pro
                # Note that the parameter list in get_prior_cross_covariance() is not the same as in the IDL function get_prior_cross_covariance().

                (matrix, pressureSa) = mpy.get_prior_cross_covariance(specie1, specie2, smeta.latitude.value, state_info.pressure, surfacetype)
                if len(matrix.shape) > 1 and matrix[0, 0] >= -990:
                    n1 = []
                    for ii in range(len(species_list)):
                        if specie1 == species_list[ii]:
                            n1.append(ii)

                    n2 = []
                    for ii in range(len(species_list)):
                        if specie2 == species_list[ii]:
                            n2.append(ii)

                    n1 = np.asarray(n1)
                    n2 = np.asarray(n2)

                    array_2d_indices = np.ix_(n1, n2)
                    initial[array_2d_indices] = matrix[:, :]

                    array_2d_indices = np.ix_(n2, n1)
                    initial[array_2d_indices] = np.transpose(matrix)[:, :]

        self.error_initial = mpy.constraint_data(initial, pressure_list,
                                                 species_list, map_list)

        

    def error_analysis(self, rs: "RetrievalStrategy", results):
        '''Update results and error_current'''
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radianceNoise = {"radiance" : np.zeros_like(rs.radianceStep["radiance"]) }
        (results, self.error_current) = mpy.error_analysis_wrapper(
            rs.table_step,
            rs.strategy_table.analysis_directory,
            rs.radianceStep,
            radianceNoise,
            rs.retrievalInfo.retrieval_info_obj,
            rs.state_info.state_info_obj,
            self.error_initial,
            self.error_current,
            rs.windows,
            results
            )
        return results

        
