import refractor.muses.muses_py as mpy
import numpy as np
from scipy.linalg import block_diag
import copy
import os
import glob
import math
import logging
logger = logging.getLogger("py-retrieve")

class RetrievalInfo:
    '''Not sure if we'll keep this or not, but pull out RetrievalInfo stuff so
    we can figure out the interface and if we should replace this.

    A few functions seem sort of like member functions, we'll just make a list
    of these to sort out later but not try to get the full interface in place.

    update_state - I think this updates just doUpdateFM
    create_uip - This just reads values, mostly copying stuff over to the UIP
    systematic_jacobian - Pretty much just make the dummy retrieval_info_temp
    write_retrieval_input - Probably go away, since this is debug output and wouldn't
                            really apply if we change RetrievalInfo
    plot_results
    error_analysis_wrapper - Lots of stuff read here
    write_retrieval_summary
    write_products_one
    '''
    def __init__(self, error_analysis : "ErrorAnalysis",
                 strategy_table : "StrategyTable",
                 state_info : "StateInfo"):
        self.retrieval_dict = \
            self.init_data(error_analysis,
                           strategy_table,
                           state_info)
        self.retrieval_dict = self.retrieval_dict.__dict__

    @property
    def retrieval_info_obj(self):
        return mpy.ObjectView(self.retrieval_dict)

    @property
    def initialGuessList(self):
        '''This is the initial guess for the state vector (not the full state)'''
        return self.retrieval_dict["initialGuessList"]

    @property
    def constraintVector(self):
        '''This is the initial guess for the state vector (not the full state)'''
        return self.retrieval_dict["constraintVector"]

    def species_results(self, results, spcname, FM_Flag=True, INITIAL_Flag=False):
        return mpy.get_vector(results.resultsList, self.retrieval_info_obj, spcname,
                              FM_Flag, INITIAL_Flag)

    def species_initial(self, spcname, FM_Flag=True):
        return mpy.get_vector(self.initialGuessList, self.retrieval_info_obj,
                              spcname, FM_Flag, True)

    def species_constraint(self, spcname, FM_Flag=True):
        return mpy.get_vector(self.constraintVector, self.retrieval_info_obj,
                              spcname, FM_Flag, True)

    @property
    def initialGuessListFM(self):
        '''This is the initial guess for the FM state vector'''
        return self.retrieval_dict["initialGuessListFM"]
    
    @property
    def type(self):
        return self.retrieval_dict["type"]

    @property
    def apriori_cov(self):
        return self.retrieval_dict["Constraint"][0:self.n_totalParameters,0:self.n_totalParameters]

    @property
    def apriori(self):
        return self.retrieval_dict["constraintVector"][0:self.n_totalParameters]

    @property
    def species_names(self):
        return self.retrieval_dict["species"][0:self.retrieval_dict["n_species"]]

    @property
    def species_list(self):
        return np.array(self.retrieval_dict["speciesList"][0:self.n_totalParameters])

    @property
    def species_list_fm(self):
        return np.array(self.retrieval_dict["speciesListFM"][0:self.n_totalParametersFM])

    @property
    def pressure_list_fm(self):
        return self.retrieval_dict["pressureListFM"][0:self.n_totalParametersFM]

    @property
    def surface_type(self):
        return self.retrieval_dict["surfaceType"]

    @property
    def is_ocean(self):
        return self.surface_type == "OCEAN"
    
    # Synonyms used in the muses-py code.
    @property
    def speciesListFM(self):
        return self.species_list_fm

    @property
    def pressureListFM(self):
        return self.pressure_list_fm
    
    @property
    def n_species(self):
        return len(self.species_names)

    @property
    def minimumList(self):
        return self.retrieval_dict["minimumList"][0:self.n_totalParameters]

    @property
    def maximumList(self):
        return self.retrieval_dict["maximumList"][0:self.n_totalParameters]

    @property
    def maximumChangeList(self):
        return self.retrieval_dict["maximumChangeList"][0:self.n_totalParameters]
    
    @property
    def species_list_fm(self):
        return self.retrieval_dict["speciesListFM"][0:self.retrieval_dict["n_totalParametersFM"]]
    
    @property
    def n_totalParameters(self):
        # Might be a better place to get this, but start by getting from
        # initial guess
        return self.initialGuessList.shape[0]

    @property
    def n_totalParametersSys(self):
        return self.retrieval_dict["n_totalParametersSys"]

    @property
    def n_totalParametersFM(self):
        return self.retrieval_dict["n_totalParametersFM"]
    
    @property
    def n_speciesSys(self):
        return self.retrieval_dict["n_speciesSys"]
    
    @property
    def __doUpdateFM(self):
        return self.retrieval_dict["doUpdateFM"][0:self.n_totalParametersFM]

    @property
    def __initialGuessListFM(self):
        '''This is the initial guess for the FM state vector'''
        return self.retrieval_dict["initialGuessListFM"]

    @property
    def __species(self):
        # Not clear why these arrays are fixed size. Probably left over from IDL,
        # but go ahead and trim this
        return self.retrieval_dict["species"][0:self.retrieval_dict["n_species"]]

    @property
    def __n_species(self):
        return len(self.species)
        
    @property
    def __n_totalParametersFM(self):
        # Might be a better place to get this, but start by getting from
        # initial guess
        return self.initialGuessListFM.shape[0]

    def init_interferents(self, strategy_table, state_info, o_retrievalInfo,
                          error_analysis):
        '''Update the various "Sys" stuff in o_retrievalInfo to add in
        the error analysis interferents'''
        sys_tokens = strategy_table.error_analysis_interferents
        sys_tokens = mpy.flat_list(sys_tokens)
        if sys_tokens[0] in ('-',  ''):
            o_retrievalInfo.n_speciesSys = 0
            return
           
        sys_tokens = state_info.order_species(sys_tokens)
        o_retrievalInfo.n_speciesSys = len(sys_tokens)
        o_retrievalInfo.speciesSys.extend(sys_tokens)
        myspec = list(mpy.constraint_get_species(error_analysis.error_initial,
                                                 sys_tokens))
        o_retrievalInfo.n_totalParametersSys = len(myspec)
        for tk in sys_tokens:
            cnt = sum(t == tk for t in myspec)
            if cnt > 0:
                pstart = myspec.index(tk)
                o_retrievalInfo.parameterStartSys.append(pstart)
                o_retrievalInfo.parameterEndSys.append(pstart+cnt-1)
                o_retrievalInfo.speciesListSys.extend([tk] * cnt)
            else:
                o_retrievalInfo.parameterStartSys.append(-1)
                o_retrievalInfo.parameterEndSys.append(-1)

        
    def add_species(self, species_name, strategy_table, state_info,
                    o_retrievalInfo):
        selem = state_info.state_element(species_name)
        selem.update_initial_guess(strategy_table)
        
        row = o_retrievalInfo.n_totalParameters
        rowFM = o_retrievalInfo.n_totalParametersFM
        mm = len(selem.initialGuessList)
        nn = len(selem.initialGuessListFM)
        o_retrievalInfo.pressureList.extend(selem.pressureList)
        o_retrievalInfo.altitudeList.extend(selem.altitudeList)
        o_retrievalInfo.speciesList.extend([species_name] * mm)
        o_retrievalInfo.pressureListFM.append(selem.pressureListFM)
        o_retrievalInfo.altitudeListFM.append(selem.altitudeListFM)
        o_retrievalInfo.speciesListFM.extend([species_name] * nn)
        o_retrievalInfo.constraintVector.append(selem.constraintVector)
        o_retrievalInfo.initialGuessList.append(selem.initialGuessList)
        o_retrievalInfo.initialGuessListFM.append(selem.initialGuessListFM)
        o_retrievalInfo.constraintVectorListFM.append(selem.constraintVectorFM)
        o_retrievalInfo.minimumList.append(selem.minimum)
        o_retrievalInfo.maximumList.append(selem.maximum)
        o_retrievalInfo.maximumChangeList.append(selem.maximum_change)
        o_retrievalInfo.trueParameterList.append(selem.trueParameterList)
        o_retrievalInfo.trueParameterListFM.append(selem.trueParameterListFM)
        o_retrievalInfo.mapToState.append(selem.mapToState)
        o_retrievalInfo.mapToParameters.append(selem.mapToParameters)
        o_retrievalInfo.parameterStart.append(row)
        o_retrievalInfo.parameterEnd.append(row + mm - 1)
        o_retrievalInfo.n_parameters.append(mm)
        o_retrievalInfo.n_parametersFM.append(nn)
        o_retrievalInfo.mapTypeList.extend([selem.mapType] * mm)
        o_retrievalInfo.mapTypeListFM.extend([selem.mapType] * nn)
        o_retrievalInfo.mapType.append(selem.mapType)
        
        o_retrievalInfo.Constraint[row:row + mm, row:row + mm] = selem.constraintMatrix
        o_retrievalInfo.parameterStartFM.append(rowFM)
        o_retrievalInfo.parameterEndFM.append(rowFM + nn - 1)
        o_retrievalInfo.n_totalParameters = row + mm
        o_retrievalInfo.n_totalParametersFM = rowFM + nn

    def init_joint(self, o_retrievalInfo, strategy_table, state_info):
        '''This should get cleaned up somehow'''
        index_H2O = -1
        index_HDO = -1
        if 'H2O' in o_retrievalInfo.species:
            index_H2O = o_retrievalInfo.species.index('H2O')

        if 'HDO' in o_retrievalInfo.species:
            index_HDO = o_retrievalInfo.species.index('HDO')

        locs = [index_H2O, index_HDO]

        hdo_h2o_flag = 0
        if locs[0] >= 0 and locs[1] >= 0:
            # HDO and H2O both retrieved in this step
            # only allow PREMADE type?
            hdo_h2o_flag = 1
            names = ['H2O', 'HDO']
            species_dir = strategy_table.species_directory

            loop_count = 0
            for xx in range(2):
                for yy in range(2):
                    loop_count += 1
                    specie1 = names[xx]
                    specie2 = names[yy]

                    filename = species_dir + os.path.sep + specie1 + "_" + specie2 +'.asc'
                    files = glob.glob(filename)
                    if len(files) == 0:
                        # If cannot find file, look for one with the species names swapped.
                        filename = species_dir + os.path.sep + specie2 + "_" + specie1 +'.asc'

                    speciesInformationFilename = filename
                    (_, fileID) = mpy.read_all_tes_cache(filename, 'asc')

                    # AT_LINE 877 Get_Species_Information.pro
                    speciesInformationFile = mpy.tes_file_get_struct(fileID)

                    # get indices of location of where to place this matrix
                    # AT_LINE 880 Get_Species_Information.pro
                    loc11 = o_retrievalInfo.parameterStart[locs[xx]]
                    loc12 = o_retrievalInfo.parameterEnd[locs[xx]]   # Note the spelling of this key 'parameterEnd'
                    loc21 = o_retrievalInfo.parameterStart[locs[yy]]
                    loc22 = o_retrievalInfo.parameterEnd[locs[yy]]
                    loc11FM = o_retrievalInfo.parameterStartFM[locs[xx]]
                    loc12FM = o_retrievalInfo.parameterEndFM[locs[xx]] # Note the spelling of this key 'parameterEndFM'

                    # AT_LINE 887 Get_Species_Information.pro
                    mm = o_retrievalInfo.n_parameters[locs[0]]    # Note the spelling of this key 'n_parameters'
                    nn = o_retrievalInfo.n_parametersFM[locs[0]]  # Note the spelling of this key 'n_parametersFM'
                    mapType = 'log'
                    pressureListFM = o_retrievalInfo.pressureListFM[loc11FM:loc12FM]
                    pressureList = o_retrievalInfo.pressureListFM[loc11:loc12]

                    # We look for the tag 'constraintFilename' and it may not have the same case.
                    # So we will make everything lower case, find the index of that tag and use it to get to the value
                    # of 'constraintFilename' key in the tagNames.  Get the exact name of the tag using the index and then use
                    # that exact name to get to the value.
                    preferences = fileID['preferences']
                    tagNames = [x for x in preferences.keys()]  # Convert the dict keys into a regular list.
                    lowerTags = [x.lower() for x in tagNames]
                    ind_to_lower = -1
                    if 'constraintFilename'.lower() in lowerTags:
                        ind_to_lower = lowerTags.index('constraintFilename'.lower())
                        actual_tag = tagNames[ind_to_lower]

                    if ind_to_lower < 0:
                        raise RuntimeError(f"Name not found for constraintFilename. In file {speciesInformationFile}")

                    # AT_LINE 902 Get_Species_Information.pro 
                    filename = preferences[actual_tag]

                    # At this point, the file name may not actually exist since it may contain the '_87' in the name.
                    # ../OSP/Constraint/H2O_HDO/Constraint_Matrix_H2O_NADIR_LOG_90S_90N_87.asc
                    # The next function will remove it and will attempt to read it.

                    constraint_species = specie1 + "_" + specie2

                    constraintMatrix, pressurex = mpy.supplier_constraint_matrix_premade(
                        constraint_species, filename, mm,
                        i_nh3type = state_info.nh3type, 
                        i_ch3ohtype = state_info.ch3ohtype)
                    # PYTHON_NOTE: We add 1 to the end of the slice since Python does not include the slice end value.
                    o_retrievalInfo.Constraint[loc11:loc12+1, loc21:loc22+1] = constraintMatrix[:, :]
                    if loc11 != loc21:
                        o_retrievalInfo.Constraint[loc21:loc22+1, loc11:loc12+1] = np.transpose(constraintMatrix)[:, :]
                    species_name = 'HDOplusH2O' 
                # end for yy in range(0,1):
                # AT_LINE 921
            # end for xx in range(0,1):
            # AT_LINE 922
        # end if (locs[0] >= 0 and locs[1] >= 0):

    def init_data(self, error_analysis : "ErrorAnalysis",
                  strategy_table : "StrategyTable",
                  state_info : "StateInfo"):
        # This is a reworking of get_species_information in muses-py
        utilLevels = mpy.UtilLevels()


        o_retrievalInfo = None

        found_omi_species_flag = False
        found_tropomi_species_flag = False

        stateInfo = mpy.ObjectView(state_info.state_info_dict)


        # AT_LINE 13 Get_Species_Information.pro
        step = strategy_table.table_step

        # errors propagated from step to step - possibly used as covariances
        #                                       but perhaps also as propagated
        #                                       constraints.
        
        nn = len(state_info.pressure)

        # get retrieval parameters, including a list of retrieved species,
        # initial values for each parameter, true values for each parameter,
        # constraints for all parameters, maps for each parameter

        smeta = state_info.sounding_metadata()
        o_retrievalInfo = { 
            # Info by retrieval parameter
            'surfaceType' : 'OCEAN' if smeta.is_ocean else 'LAND',

            'speciesList' : [],
            'pressureList': [],
            'altitudeList': [],
            'mapTypeList' : [],

            'initialGuessList' : [],
            'constraintVector' : [],
            'trueParameterList': [],

            # optional allowed range and maximum stepsize during retrieval, set to -999 if not used
            'minimumList' : [],
            'maximumList' : [],
            'maximumChangeList' : [],

            'doUpdateFM'       : None,
            'speciesListFM'    : [],  
            'pressureListFM'     : [],
            'altitudeListFM'     : [],
            'initialGuessListFM' : [],
            'constraintVectorListFM' : [],
            'trueParameterListFM': [],

            'n_totalParametersFM': 0,

            'parameterStartFM': [],
            'parameterEndFM'  : [],
            'mapTypeListFM'   : [],

            # Info by species
            'n_speciesSys'     : 0,
            'speciesSys'       : [],
            'parameterStartSys': [],
            'parameterEndSys'  : [],
            'speciesListSys'   : [],

            'n_totalParametersSys': 0,
            'n_species'           : 0,
            'species'             : [],
            'parameterStart'      : [],
            'parameterEnd'        : [],

            'n_parametersFM' : [],
            'n_parameters'   : [],
            'mapType'        : [],
            'mapToState'     : [],
            'mapToParameters': [],

            # Constraint & SaTrue, & info for all parameters
            'n_totalParameters': 0,
            'Constraint'       : np.zeros(shape=(10*nn, 10*nn), dtype=np.float64),
            'type'             : None,
        }
        # o_retrievalInfo OBJECT_TYPE dict

        # AT_LINE 83 Get_Species_Information.pro
        o_retrievalInfo['type'] = strategy_table.retrieval_type

        o_retrievalInfo = mpy.ObjectView(o_retrievalInfo)  # Convert to object so we can use '.' to access member variables.


        # map types for all species
        # now in strategy table
       
        if strategy_table.retrieval_type.lower() in ('bt', 'forwardmodel'):
            pass
        else:
            o_retrievalInfo.species = state_info.order_species(strategy_table.retrieval_elements)
            o_retrievalInfo.n_species = len(o_retrievalInfo.species)

            for species_name in o_retrievalInfo.species:
                self.add_species(species_name, strategy_table, state_info,
                                 o_retrievalInfo)

            self.init_interferents(strategy_table, state_info, o_retrievalInfo,
                                   error_analysis)

        self.init_joint(o_retrievalInfo, strategy_table, state_info)
        
        # Convert to numpy arrays
        for key in ("initialGuessList", "constraintVector",
                    "trueParameterList", "trueParameterListFM",
                    "pressureListFM", "altitudeListFM",
                    "initialGuessListFM", "constraintVectorListFM",
                    "minimumList", "maximumList", "maximumChangeList"):
            if(len(o_retrievalInfo.__dict__[key]) > 0):
                o_retrievalInfo.__dict__[key] = np.concatenate(
                    [a.flatten() for a in o_retrievalInfo.__dict__[key]])
            else:
                o_retrievalInfo.__dict__[key] = np.zeros(0)
        # Few block diagonal matrixes
        for key in ("mapToState","mapToParameters"):
            o_retrievalInfo.__dict__[key] = block_diag(*o_retrievalInfo.__dict__[key])
        o_retrievalInfo.Constraint = \
            o_retrievalInfo.Constraint[0:o_retrievalInfo.n_totalParameters,
                                       0:o_retrievalInfo.n_totalParameters]
        o_retrievalInfo.doUpdateFM = np.zeros(o_retrievalInfo.n_totalParametersFM)
        o_retrievalInfo.speciesListSys = np.array(o_retrievalInfo.speciesListSys)
        # Not sure if these empty values are important or not, but for now
        # match what the existing muses-py code does.
        for k in ("speciesList","speciesListFM", "mapTypeList","speciesSys",
                  "speciesListSys", "mapTypeListFM"):
            if(len(o_retrievalInfo.__dict__[k]) == 0):
                o_retrievalInfo.__dict__[k] = ['',]
        for k in ('altitudeList','altitudeListFM','constraintVector',
                  'constraintVectorListFM', 'initialGuessList',
                  'initialGuessListFM','maximumChangeList','maximumList',
                  'minimumList','pressureList','pressureListFM',
                  'trueParameterList', 'trueParameterListFM'):
            if(len(o_retrievalInfo.__dict__[k]) == 0):
                o_retrievalInfo.__dict__[k] = np.array([0.0,])
        for k in ("doUpdateFM", "parameterStartSys", "parameterEndSys",):
            if(len(o_retrievalInfo.__dict__[k]) == 0):
                o_retrievalInfo.__dict__[k] = np.array([0,])
        for k in ('mapToParameters', 'mapToState', 'Constraint'):
            if(o_retrievalInfo.__dict__[k].shape[1] == 0):
                o_retrievalInfo.__dict__[k] = np.array([[0.0,]])

        # Check the constaint vector for sanity.
        if np.all(np.isfinite(o_retrievalInfo.constraintVector)) == False:
            raise RuntimeError(f"NaN's in constraint vector!! Constraint vector is: {o_retrievalInfo.constraintVector}. Check species {o_retrievalInfo.speciesList}")

        # Check the constaint matrix for sanity.
        if np.all(np.isfinite(o_retrievalInfo.Constraint)) == False:
            raise RuntimeError(f"NaN's in constraint matrix!! Constraint matrix is: {o_retrievalInfo.Constraint}. Check species {o_retrievalInfo.speciesList}")

        return o_retrievalInfo
        
    
__all__ = ["RetrievalInfo",]
