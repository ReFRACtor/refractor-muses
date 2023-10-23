import refractor.muses.muses_py as mpy
import numpy as np
import copy
import os
import glob
import math
import logging
logger = logging.getLogger("py-retrieve")

class RefractorRetrievalInfoOld:
    '''Original version of this, so we can debug/compare issues with the new one.
    '''
    def __init__(self, strategy_table : "StrategyTable",
                 state_info : "RefractorStateInfo"):
        (self.retrieval_dict, _, _) = \
            self.init_data(strategy_table.strategy_table_dict,
                           state_info.state_info_dict)
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
    def species_list_fm(self):
        return self.retrieval_dict["speciesListFM"][0:self.n_totalParametersFM]

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

    def init_data(self, strategy_table : "StrategyTable",
                  state_info : "RefractorStateInfo"):
        # This is a reworking of get_species_information in muses-py
        utilLevels = mpy.UtilLevels()


        o_retrievalInfo = None
        o_errorInitial = None
        o_errorCurrent = None

        found_omi_species_flag = False
        found_tropomi_species_flag = False

        stateInfo = mpy.ObjectView(state_info)
        i_table_struct = mpy.ObjectView(strategy_table)

        # true used for synthetic radiances only.  Here check if true has non-fill values.
        # the -999 (fill) messes up with log mapping, so only fill true if non-fill.
        if np.max(stateInfo.true['values']) > 0:
            have_true = True
        else:
            have_true = False

        if isinstance(i_table_struct.preferences, dict):
            preferences = mpy.ObjectView(i_table_struct.preferences)
        else:
            preferences = i_table_struct.preferences

        # AT_LINE 13 Get_Species_Information.pro
        step = i_table_struct.step
        # this should be the current state
        pressure = stateInfo.current['pressure'][:]
        surfacePressure = stateInfo.current['pressure'][0]
        num_pressures = len(pressure)

        # user specifies the number of forward model levels
        # AT_LINE 22 Get_Species_Information.pro
        if int(preferences.num_FMLevels) > num_pressures:
            i_table_struct.preferences['num_FMLevels'] = num_pressures
        else:
            pressure = pressure[0:int(preferences.num_FMLevels)]

        # Get it again to make sure we have the correct number.
        num_pressures = len(pressure)

        # errors propagated from step to step - possibly used as covariances
        #                                       but perhaps also as propagated
        #                                       constraints.

        # AT_LINE 32 Get_Species_Information.pro
        num_errorSpecies = len(i_table_struct.errorSpecies)
        nn = num_pressures

        # get retrieval parameters, including a list of retrieved species,
        # initial values for each parameter, true values for each parameter,
        # constraints for all parameters, maps for each parameter
        # 10/9/2019 Need to have different pressureList for every state

        # Info by retrieval parameter
        # AT_LINE 39 Get_Species_Information.pro
        o_retrievalInfo = { 
            # Info by retrieval parameter
            'surfaceType' : '',  

            'speciesList' : [''] * (10*nn+100),  
            'pressureList': [0.0 for x in range(10*nn+100)], 
            'altitudeList': [0.0 for x in range(10*nn+100)], # not needed in retrieval system; for archiving
            'mapTypeList' : [''] * (10*nn+100),  

            'initialGuessList' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'constraintVector' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'trueParameterList': np.zeros(shape=(10*nn+100), dtype=np.float64),

            # optional allowed range and maximum stepsize during retrieval, set to -999 if not used
            'minimumList' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'maximumList' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'maximumChangeList' : np.zeros(shape=(10*nn+100), dtype=np.float64),

            'doUpdateFM'       : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'speciesListFM'    : [''] * (10*nn+100),  
            'pressureListFM'     : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'altitudeListFM'     : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'constraintListFM' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'initialGuessListFM' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'constraintVectorListFM' : np.zeros(shape=(10*nn+100), dtype=np.float64),
            'trueParameterListFM': np.zeros(shape=(10*nn+100), dtype=np.float64),

            'n_totalParametersFM': 0,

            'parameterStartFM': [0  for x in range(18)],
            'parameterEndFM'  : [0  for x in range(18)],
            'mapTypeListFM'   : [''] * (10*nn+100),

            # Info by species
            'n_speciesSys'     : 0,
            'speciesSys'       : ["" for x in range(18)],
            'parameterStartSys': [0  for x in range(18)],
            'parameterEndSys'  : [0  for x in range(18)],
            'speciesListSys'   : np.asarray([''] * (10*nn+100), dtype='<U20'),

            'n_totalParametersSys': 0,
            'n_species'           : 0,
            'species'             : ["" for x in range(18)],
            'parameterStart'      : [0  for x in range(18)],
            'parameterEnd'        : [0  for x in range(18)],

            'n_parametersFM' : [0  for x in range(18)],
            'n_parameters'   : [0  for x in range(18)],
            'mapType'        : ["" for x in range(18)],
            'mapToState'     : np.zeros(shape=(nn*10+100, nn*10+100), dtype=np.float64),
            'mapToParameters': np.zeros(shape=(nn*10+100, nn*10+100), dtype=np.float64),

            # Constraint & SaTrue, & info for all parameters
            'n_totalParameters': 0,
            'Constraint'       : np.zeros(shape=(10*nn, 10*nn), dtype=np.float64),
            'type'             : '',
        }
        # o_retrievalInfo OBJECT_TYPE dict

        # AT_LINE 83 Get_Species_Information.pro
        o_retrievalInfo['type'] = mpy.table_get_entry(i_table_struct, step, "retrievalType")

        o_retrievalInfo = mpy.ObjectView(o_retrievalInfo)  # Convert to object so we can use '.' to access member variables.

        # update state based on output from previous steps
        # 1/2016 it seems like this doesn't belong here

        # AT_LINE 93 Get_Species_Information.pro
        row = 0
        rowFM = 0

        surfaceType = 'LAND'
        if stateInfo.surfaceType.upper() == 'OCEAN':
            surfaceType = 'OCEAN'
        o_retrievalInfo.surfaceType = surfaceType

        current = mpy.ObjectView(stateInfo.current) # Convert to object so we can use '.' to access member variables.

        # block diagonal part of covariance
        # AT_LINE 101 Get_Species_Information.pro
        latitude = current.latitude

        # map types for all species
        # now in strategy table

        if o_errorInitial is None:
            # get errorInitial, from the a priori covariances
            # requires the pressure grid to be set
            nh3type = stateInfo.nh3type
            hcoohtype = stateInfo.hcoohtype
            ch3ohtype = stateInfo.ch3ohtype

            # AT_LINE 109 Get_Species_Information.pro
            o_errorInitial = mpy.get_prior_error(i_table_struct.errorSpecies, i_table_struct.errorMaptype,
                                             stateInfo.current["pressure"], latitude, nh3type, hcoohtype,
                                             ch3ohtype, surfaceType)

        o_errorCurrent = copy.deepcopy(o_errorInitial)   # Make a separate copy of o_errorInitial to o_errorCurrent.
        # At this point, both o_errorInitial and o_errorCurrent are valid objects.

        # AT_LINE 117 Get_Species_Information.pro

        mpy.table_set_step(i_table_struct, step)

        # AT_LINE 121 Get_Species_Information.pro

        retrievalType = o_retrievalInfo.type.lower()

        if retrievalType == 'bt' or retrievalType == 'forwardmodel': # or retrievalType == 'fullfilter': EM NOTE - keep fullfilter in the same bracket as 'default'
            # no retrieval info
            # DEVELOPER_NOTE: Running for AIRS instrument test, the retrievalType is 'bt' so the code in the else part is not exercised.
            pass
        else:
            retrievalTypeStr = '_' + retrievalType
            if retrievalType == 'default':
                retrievalTypeStr = ''

            # order species stuff.  We also have to order everything that is associated with 
            # each species, e.g. mapType, initialGuessSource
            result = mpy.table_get_unpacked_entry(i_table_struct, step, "retrievalElements")

            retrievalElementsArray = mpy.order_species(result)

            num_species = len(retrievalElementsArray)

            # AT_LINE 134 Get_Species_Information.pro
            row = 0
            rowFM = 0

            # Needed upfront for created matrices.
            o_retrievalInfo.species[0:num_species] = retrievalElementsArray
            o_retrievalInfo.n_species = num_species

            # Store filenames for constraint matrices; after have all species
            # maps, go through and look for correlated matrices
            filenamesConstraint = ["" for x in range(num_species)]
            types = ["" for x in range(num_species)]

            # AT_LINE 146 Get_Species_Information.pro

            # BEGIN_MAIN_FOR_LOOP over species.
            for ii in range(num_species):
                # AT_LINE 148 Get_Species_Information.pro
                species_name = retrievalElementsArray[ii]

                # Open, read species file
                speciesInformationFilename = mpy.table_get_pref(i_table_struct, "speciesDirectory")  + os.path.sep + species_name + retrievalTypeStr + '.asc'

                files = glob.glob(speciesInformationFilename)
                if len(files) == 0:
                    # Look for alternate file.
                    speciesInformationFilename = mpy.table_get_pref(i_table_struct, "speciesDirectory")  + os.path.sep + species_name + '.asc'

                # AT_LINE 156 Get_Species_Information.pro
                (_, fileID) = mpy.read_all_tes_cache(speciesInformationFilename)
                speciesInformationFile = fileID['preferences']
                speciesInformationFile = mpy.ObjectView(speciesInformationFile) # It is now an object.

                # AT_LINE 157 Get_Species_Information.pro
                mapType = speciesInformationFile.mapType.lower()
                constraintType = speciesInformationFile.constraintType.lower()

                # spectral species
                # AT_LINE 161 Get_Species_Information.pro
                #self.m_debug_mode = True
                if 'OMI' in species_name and 'TROPOMI' not in species_name:
                    found_omi_species_flag = True
                    # CODE_NOT_TESTED_YET
                    # OMI parameters... all diagonal
                    # to change the names, update script_retrieval_setup_omi,
                    # get_species_information, order_species.pro,
                    # new_state_structures, update_state
                    # get_one_state, write_one_state, and table
                    # "OMI_CLOUDFRACTION","OMI_NRAD_WAV","OMI_OD_WAV","OMI_RING_SF","OMI_SURFALB","OMI_SURFALB_SLOPE"
                    # species filenames, covariance, constraint

                    tagnames = mpy.idl_tag_names(current)

                    omiInfo = mpy.ObjectView(current.omi)

                    # These are the potential key names: ['surface_albedo_uv1', 'surface_albedo_uv2', 'surface_albedo_slope_uv2', 'nradwav_uv1', 'nradwav_uv2', 'odwav_uv1', 'odwav_uv2', 'odwav_slope_uv1', 'odwav_slope_uv2', 'ring_sf_uv1', 'ring_sf_uv2', 'cloud_fraction', 'cloud_pressure', 'cloud_Surface_Albedo', 'xsecscaling', 'resscale_uv1', 'resscale_uv2', 'sza_uv1', 'raz_uv1', 'vza_uv1', 'sca_uv1', 'sza_uv2', 'raz_uv2', 'vza_uv2', 'sca_uv2', 'SPACECRAFTALTITUDE'])

                    # We don't need to look for the key because we know it exists as 'cloud_fraction' already.
                    # ind = []
                    # ind_names = [];  # List to hold the key name.
                    # for jj in range(len(tagnames)):
                    #    original_tag = tagnames[jj]
                    #    tagnames[jj] = tagnames[jj].replace('_','');  # Change cloud_fraction to cloudfraction so we can compare with 'OMICLOUDFRACTION'
                    #    # PYTHON_NOTE: Because the keys in Python are case sensitive, we change tagnames[jj] to uppercase.
                    #    # tagnames[jj].upper() is 'CLOUDFRACTION'
                    #    # species_name is 'OMICLOUDFRACTION'
                    #    # species_name[3:] is 'CLOUDFRACTION'
                    #    if (tagnames[jj].upper() == species_name[3:]):
                    #        ind.append(jj)
                    #        ind_names.append(original_tag);  # Note that if we had found cloudfraction, we save the original tag which is the real key.

                    # Because we are not certain the spelling of the keys in omi, we will perform a special search with get_omi_key() function.

                    actual_omi_key = mpy.get_omi_key(omiInfo, species_name)

                    # AT_LINE 177 Get_Species_Information.pro
                    # PYTHON_NOTE: To get access to a dictionary, we use a key instead of an index as IDL does.
                    # At this point, the value of actual_omi_key is the key we want to access the 'omi' dictionary.
                    initialGuessList = stateInfo.current['omi'][actual_omi_key]
                    constraintVector = stateInfo.constraint['omi'][actual_omi_key]
                    constraintVectorFM = stateInfo.constraint['omi'][actual_omi_key]

                    if have_true:
                        trueParameterList = stateInfo.true['omi'][actual_omi_key]

                    # It is also possible that these values are scalar, we convert them to an array of 1.
                    if np.isscalar(initialGuessList):
                        initialGuessList = np.asarray([initialGuessList])

                    if have_true:
                        if np.isscalar(trueParameterList):
                            trueParameterList = np.asarray([trueParameterList])

                    if np.isscalar(constraintVector):
                        constraintVector = np.asarray([constraintVector])
                        constraintVectorFM = np.asarray([constraintVectorFM])

                    initialGuessListFM = initialGuessList[:]
                    if have_true:
                        trueParameterListFM = trueParameterList[:]

                    # AT_LINE 184 Get_Species_Information.pro
                    nn = len(initialGuessList)
                    mm = len(initialGuessList)

                    if mm == 1:
                        mapToState = 1
                        mapToParameters = 1
                        num_retrievalParameters = 1

                        # it's difficult to get a nx1 array.  1xn is easy.
                        retrievalParameters = [0]
                        pressureList = [-2]
                        pressureListFM = [-2]
                        altitudeList = [-2]
                        altitudeListFM = [-2]

                        sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)  # Becareful that some values are of string type.
                        constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)  # Note the name change from .constraint to constraintMatrix
                    else:
                        # AT_LINE 202 Get_Species_Information.pro
                        mapToState = np.identity(mm)
                        mapToParameters = np.identity(mm)
                        num_retrievalParameters = mm

                        retrievalParameters = [ii for ii in range(mm)]   # INDGEN(mm)
                        pressureList = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        pressureListFM = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        altitudeList = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        altitudeListFM = [-2 for ii in range(mm)] # STRARR(mm)+'-2'

                        sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)
                        constraintMatrix = np.identity(mm)  # Note the name change from .constraint to constraintMatrix
                        indx = [ii for ii in range(mm)]
                        indx = np.asarray(indx)
                        constraintMatrix[indx, indx] = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)  # Note the name change from .constraint to constraintMatrix

                    # take log if it makes sense
                    if (speciesInformationFile.mapType) == 'LOG':
                        initialGuessListFM = np.log(initialGuessListFM)
                        constraintVector = np.log(constraintVector)
                        initialGuessList = np.log(initialGuessList)
                        if have_true:
                            trueParameterList = np.log(trueParameterList)
                            trueParameterListFM = np.log(trueParameterListFM)
                    # end if (speciesInformationFile.mapType) == 'LOG':
                elif 'TROPOMI' in species_name:
                    found_tropomi_species_flag = True
                    # EM NOTE - copied from the OMI section, since tropomi build is based on the omi build.
                    # OMI code above suggests 'not tested', but think this has just not been cleaned up yet.
                    # This section is necessary for TROPOMI since we are using similar fitting parameters,
                    # and structures to OMI.


                    tagnames = mpy.idl_tag_names(current)

                    tropomiInfo = mpy.ObjectView(current.tropomi)

                    actual_tropomi_key = mpy.get_tropomi_key(tropomiInfo, species_name)

                    # AT_LINE 177 Get_Species_Information.pro
                    # PYTHON_NOTE: To get access to a dictionary, we use a key instead of an index as IDL does.
                    # At this point, the value of actual_omi_key is the key we want to access the 'omi' dictionary.
                    initialGuessList = stateInfo.current['tropomi'][actual_tropomi_key]
                    if have_true:
                        trueParameterList = stateInfo.true['tropomi'][actual_tropomi_key]
                    constraintVector = stateInfo.constraint['tropomi'][actual_tropomi_key]
                    constraintVectorFM = stateInfo.constraint['tropomi'][actual_tropomi_key]

                    # It is also possible that these values are scalar, we convert them to an array of 1.
                    if np.isscalar(initialGuessList):
                        initialGuessList = np.asarray([initialGuessList])

                    if have_true:
                        if np.isscalar(trueParameterList):
                            trueParameterList = np.asarray([trueParameterList])

                    if np.isscalar(constraintVector):
                        constraintVector = np.asarray([constraintVector])

                    initialGuessListFM = initialGuessList[:]
                    if have_true:
                        trueParameterListFM = trueParameterList[:]

                    # AT_LINE 184 Get_Species_Information.pro
                    nn = len(initialGuessList)
                    mm = len(initialGuessList)

                    if mm == 1:
                        mapToState = 1
                        mapToParameters = 1
                        num_retrievalParameters = 1

                        # it's difficult to get a nx1 array.  1xn is easy.
                        retrievalParameters = [0]
                        pressureList = [-2]
                        pressureListFM = [-2]
                        altitudeList = [-2]
                        altitudeListFM = [-2]

                        sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)  # Becareful that some values are of string type.
                        constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)  # Note the name change from .constraint to constraintMatrix
                    else:
                        # AT_LINE 202 Get_Species_Information.pro
                        mapToState = np.identity(mm)
                        mapToParameters = np.identity(mm)
                        num_retrievalParameters = mm

                        retrievalParameters = [ii for ii in range(mm)]   # INDGEN(mm)
                        pressureList = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        pressureListFM = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        altitudeList = [-2 for ii in range(mm)] # STRARR(mm)+'-2'
                        altitudeListFM = [-2 for ii in range(mm)] # STRARR(mm)+'-2'

                        sSubaDiagonalValues = float(speciesInformationFile.sSubaDiagonalValues)
                        constraintMatrix = np.identity(mm)  # Note the name change from .constraint to constraintMatrix
                        indx = [ii for ii in range(mm)]
                        indx = np.asarray(indx)
                        constraintMatrix[indx, indx] = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)  # Note the name change from .constraint to constraintMatrix

                    # take log if it makes sense
                    if (speciesInformationFile.mapType) == 'LOG':
                        initialGuessListFM = np.log(initialGuessListFM)
                        constraintVector = np.log(constraintVector)
                        initialGuessList = np.log(initialGuessList)
                        if have_true:
                            trueParameterListFM = np.log(trueParameterListFM)
                            trueParameterList = np.log(trueParameterList)
                    # end if (speciesInformationFile.mapType) == 'LOG':
                # IDL AT_LINE 240 Get_Species_Information:

                elif species_name == 'NIRALBBRDFPL' or species_name == 'NIRALBLAMBPL':
                    if species_name == 'NIRALBBRDFPL':
                        mult = 1
                        if stateInfo.current['nir']['albtype'] == 1:
                            mult = 1/0.07
                    else:
                        mult = 1
                        if stateInfo.current['nir']['albtype'] == 2:
                            mult = 0.07

                    if stateInfo.current['nir']['albtype'] == 3:
                        raise RuntimeError("Mismatch in albedo type")

                    # get all full-state grid quantities
                    initialGuessListFM = stateInfo.current['nir']['albpl']*mult
                    nn = len(initialGuessListFM)
                    initialGuessListFM = initialGuessListFM.reshape(nn)
                    if have_true:
                        trueParameterListFM = stateInfo.true['nir']['albpl'].reshape(nn)*mult
                    constraintVectorFM = stateInfo.constraint['nir']['albpl'].reshape(nn)*mult
                    pressureListFM = stateInfo.current['nir']['albplwave']
                    altitudeListFM = stateInfo.current['nir']['albplwave']
                    mm = int(speciesInformationFile.num_retrieval_levels)

                    # albedo
                    if (mapType == 'linearpca') or (mapType == 'logpca'):

                        # this is state vector of the form:
                        # current_fs = apriori_fs + mapToState @ current for linearpca
                        # or 
                        # log(current_fs) = log(apriori_fs) + mapToState @ log(current) for logpca
                        # Doing current_fs = mapToState @ current does not work
                        # because the maps do not have a good span of the state, e.g. the stratosphere does not have sensitivity.
                        # when I tried this I got Tatm = [300, ..., 62, 37, -20, -27]
                        # so it must be apriori + offset


                        mapsFilename = speciesInformationFile.mapsFilename

                        # retrieval "levels"
                        retrievalParameters = np.array(range(mm))+1

                        # implemented for OCO-2, but if used for other satellite
                        # need to change # of full state levels to read correct file
                        # for oco-2:  maps_TATM_Linear_20_3.nc, where 20 is # of full state pressures
                        (mapDict, _, _) = cdf_read_dict(mapsFilename)
                        mapToState = np.transpose(mapDict['to_state'])
                        mapToParameters = np.transpose(mapDict['to_pars'])

                        pressureList = np.zeros(mm,dtype=np.float)-999
                        altitudeList = np.zeros(mm,dtype=np.float)-999

                        filename = speciesInformationFile.constraintFilename
                        (constraintStruct, constraintPressure) = mpy.constraint_read(filename)
                        constraintMatrix = mpy.constraint_get(constraintStruct)
                        constraintPressure = mpy.constraint_get_pressures(constraintStruct)

                        # since the "true" is relative to the a priori
                        # the "true state" is set to e.g. 0.8 if the a priori
                        # is off by -0.8K
                        if mapType == 'linearpca': 
                            constraintVector = np.zeros(mm,dtype = np.float32) + 0
                            initialGuessList = np.transpose(mapToParameters) @ (trueParameterListFM - constraintVectorFM)
                            initialGuessListFM = constraintVectorFM + np.transpose(mapToState) @ initialGuessList
                            if have_true:
                                trueParameterList = np.transpose(mapToParameters) @ (trueParameterListFM - constraintVectorFM)
                        else:
                            constraintVector = np.zeros(mm,dtype = np.float32) + 0
                            initialGuessList = np.transpose(mapToParameters) @ (np.log(trueParameterListFM) - np.log(constraintVectorFM))
                            initialGuessListFM = np.exp(np.log(constraintVectorFM) + mapToState @ initialGuessList)
                            if have_true:
                                trueParameterList = np.transpose(mapToParameters) @ (np.log(trueParameterListFM) - np.log(constraintVectorFM))
                    # end part of elif (mapType == 'linearpca') or (mapType == 'logpca'):

                    else:

                        if speciesInformationFile.constraintType == 'Diagonal':
                            values = (speciesInformationFile.sSubaDiagonalValues).split(',')
                            for kk in range(0,len(values)):
                                values[kk] = float(values[kk])
                            constraintMatrix = np.identity(mm)
                            for kk in range(0,len(values)):
                                constraintMatrix[kk,kk] = 1/values[kk]
                        elif speciesInformationFile.constraintType == 'Full':
                            (covariance, _) = mpy.constraint_read(speciesInformationFile.constraintFilename)
                            constraintMatrix = np.invert(covariance['data'])
                            pressureList = covariance['pressure']
                            altitudeList = pressureList
                        elif speciesInformationFile.constraintType == 'PREMADE':
                            filename = speciesInformationFile.constraintFilename

                            constraintMatrix, pressurex = mpy.supplier_constraint_matrix_premade(
                                species_name, 
                                filename, 
                                mm, 
                                i_nh3type = stateInfo.nh3type, 
                                i_ch3ohtype = stateInfo.ch3ohtype)

                            pressureList = pressurex
                            altitudeList = pressurex
                        else:
                            raise RuntimeError(f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}")

                        #get maps from constraintMatrix
                        if mm == nn:
                            mapToState = np.identity(mm)
                            mapToParameters = np.identity(mm)

                            retrievalParameters = range(mm)
                            pressureList = pressureListFM
                            altitudeList = altitudeListFM
                            constraintVector = constraintVectorFM
                            initialGuessList = initialGuessListFM
                            if have_true:
                                trueParameterList = trueParameterListFM
                        else:
                            # ensure each band edge matches, match to pressureListFM, then create maps
                            ind1 = np.where(pressureList < 1.0)[0]
                            ind2 = np.where(pressureListFM < 1.0)[0]
                            if len(ind2) > 0:
                                pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                                pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                            ind1 = np.where((pressureList > 1.0) * (pressureList < 2.0))[0]
                            ind2 = np.where((pressureListFM > 1.0) * (pressureListFM < 2.0))[0]
                            if len(ind2) > 0:
                                pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                                pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                            ind1 = np.where((pressureList > 2.0) * (pressureList < 2.2))[0]
                            ind2 = np.where((pressureListFM > 2.0) * (pressureListFM < 2.2))[0]
                            if len(ind2) > 0:
                                pressureList[min(ind1)] = np.min(pressureListFM[ind2])
                                pressureList[max(ind1)] = np.max(pressureListFM[ind2])

                        # change pressurelist to best matching pressurelistFM
                        # this is needed for mapping
                        inds = []
                        for iq in range(len(pressureList)):
                            xx = np.min(abs(pressureList[iq] - pressureListFM))
                            ind = np.where(abs(pressureList[iq] - pressureListFM) == xx)[0][0]
                            pressureList[iq] = pressureListFM[ind]
                            inds.append(ind)

                        inds = np.array(inds)+1
                        maps = mpy.make_maps(pressureListFM, inds, i_linearFlag = True)

                        constraintVector = maps['toPars'].transpose() @ constraintVectorFM
                        initialGuessList = maps['toPars'].transpose() @ initialGuessListFM
                        if have_true:
                            trueParameterList = maps['toPars'].transpose() @ trueParameterListFM

                        mapToParameters = maps['toPars']
                        mapToState = maps['toState']

                elif 'NIRAERX' in species_name:

                    # aerosol.  Match type
                    types = (speciesInformationFile.types).split(',')
                    myinds = []
                    for ii in range(mm):
                        ind = np.where(np.array(stateInfo.current['nir']['aertype'][ii]) == types)[0]
                        myinds.append(ind[0])

                    #mykeys = ['sSubaDiagonalValues','minimum','maximum','maximumChange']
                    value = (speciesInformationFile.sSubaDiagonalValues).split(',')
                    if len(value) > 1:
                        speciesInformationFile.sSubaDiagonalValues = np.array(value)[myinds]
                    value = (speciesInformationFile.minimum).split(',')
                    if len(value) > 1:
                        speciesInformationFile.minimum = np.array(value)[myinds]
                    value = (speciesInformationFile.maximum).split(',')
                    if len(value) > 1:
                        speciesInformationFile.maximum = np.array(value)[myinds]
                    value = (speciesInformationFile.maximumChange).split(',')
                    if len(value) > 1:
                        speciesInformationFile.maximumChange = np.array(value)[myinds]

                    # make key
                    mykey = species_name[3:]
                    mykey = mykey.lower()

                    npar = len(stateInfo.current['nir'][mykey][:])
                    initialGuessList = stateInfo.current['nir'][mykey]
                    constraintVector = stateInfo.constraint['nir'][mykey]
                    constraintVectorFM = stateInfo.constraint['nir'][mykey]
                    initialGuessListFM = copy.deepcopy(initialGuessList)
                    if have_true:
                        trueParameterListFM = copy.deepcopy(trueParameterList)
                        trueParameterList = stateInfo.true['nir'][mykey]

                    nn = len(initialGuessListFM)
                    mm = len(initialGuessList)
                    num_retrievalParameters = mm

                    # aerosol.  Match type
                    types = (speciesInformationFile.types).split(',')
                    myinds = []
                    for ii in range(mm):
                        ind = np.where(np.array(stateInfo.current['nir']['aertype'][ii]) == types)[0]
                        myinds.append(ind[0])

                    val = np.array((speciesInformationFile.sSubaDiagonalValues).split(','))
                    constraintMatrix = np.identity(mm)
                    for jj in range(mm):
                        constraintMatrix[jj,jj] = 1/np.float(val[myinds[jj]])/np.float(val[myinds[jj]])

                    mapToState = np.identity(mm)
                    mapToParameters = np.identity(mm)

                elif 'NIR' in species_name:
                    # other NIR parameters... all diagonal

                    # match up fields in stateOne.nir to parameter names
                    #mylist1 = list(stateInfo.current['nir'].keys())
                    #mylist2 = mylist1.copy()
                    #for jj in range(len(mylist2)):
                    #    mylist2[jj] = mylist2[jj].replace('_','')
                    #    mylist2[jj] = 'NIR' + mylist2[jj].upper()

                    # make key
                    mykey = species_name[3:]
                    mykey = mykey.lower()


                    if 'NIRAER' in species_name:
                        f = file_class(speciesInformationFilename)
                        # even if these are #'s, expressed as strings.
                        types = f.get_preference("types").split(',')
                        sSubaDiagonalValues = f.get_preference("sSubaDiagonalValues").split(',')
                        minimum = f.get_preference("minimum").split(',')
                        maximum = f.get_preference("maximum").split(',')
                        maximumChange = f.get_preference("maximumChange").split(',')

                        # aerosol.  Select down to exact types used.
                        # each type can have a different constraint (or max/min/maxchange)
                        #types = (speciesInformationFile.types).split(',')
                        myinds = []
                        for ik in range(stateInfo.current['nir']['naer']):
                            if (str(stateInfo.current['nir']['aertype'][ik].dtype))[1] == 'S':
                                ind = np.where(stateInfo.current['nir']['aertype'][ik].decode() == np.array(types))[0]
                            else:
                                ind = np.where(stateInfo.current['nir']['aertype'][ik] == np.array(types))[0]

                            if len(ind) == 0:
                                pass

                            myinds.append(ind[0])

                        # select values corresponding to aerosol types used and
                        # place into speciesInformationFile
                        myinds = np.array(myinds)
                        speciesInformationFile.sSubaDiagonalValues = ','.join(np.array(sSubaDiagonalValues)[myinds])
                        speciesInformationFile.minimum = ','.join(np.array(minimum)[myinds])
                        speciesInformationFile.maximum = ','.join(np.array(maximum)[myinds])
                        speciesInformationFile.maximumChange = ','.join(np.array(maximumChange)[myinds])
                        speciesInformationFile.types = ','.join(np.array(types)[myinds])


                    # for NIRALBLAMB, the polynomial order is set by the # listed in speciesfile, sSubaDiagonalValues
                    if species_name == 'NIRALBLAMB' or species_name == 'NIRALBBRDF' or species_name == 'NIRALBCM':
                        # check retrieval versus state type 
                        mult = 1
                        if species_name == 'NIRALBLAMB':
                            if stateInfo.current['nir']['albtype'] == 2:
                                mult = 0.07
                            if stateInfo.current['nir']['albtype'] == 3:
                                raise RuntimeError("Mismatch in albedo type")

                        if species_name == 'NIRALBBRDF':
                            if stateInfo.current['nir']['albtype'] == 1:
                                mult = 1/0.07
                            if stateInfo.current['nir']['albtype'] == 3:
                                raise RuntimeError("Mismatch in albedo type")


                        npoly = np.int(len((speciesInformationFile.sSubaDiagonalValues).split(','))/3)

                        # get initial maps.  Maps will be updated when ReFRACtor is run
                        nfs = len(stateInfo.current['nir']['albpl'])
                        filename = '../OSP/OCO2/map'+str(nfs)+'x'+str(np.int(npoly*3))+'.nc'
                        (_, mapToParameters, _) = nc_read_variable(filename, 'topars')
                        (_, mapToState, _) = nc_read_variable(filename, 'tostate')

                        initialGuessListFM = stateInfo.current['nir']['albpl'] * mult
                        initialGuessList = mapToParameters @ initialGuessListFM
                        if have_true:
                            trueParameterListFM = stateInfo.true['nir']['albpl'] * mult
                            trueParameterList = mapToParameters @ trueParameterListFM
                        constraintVector = mapToParameters @ (stateInfo.constraint['nir']['albpl'] * mult)
                        constraintVectorFM = mapToState @  mapToParameters @ (stateInfo.constraint['nir']['albpl'] * mult)

                        mapToParameters = mapToParameters.T
                        mapToState = mapToState.T

                    elif species_name == 'NIRDISP':
                        npoly = int(len((speciesInformationFile.sSubaDiagonalValues).split(','))/3)
                        # get only the first npoly entries
                        initialGuessList = stateInfo.current['nir'][mykey][:,0:npoly].reshape(npoly*3)
                        constraintVector = (stateInfo.constraint['nir'][mykey][:,0:npoly]).reshape(npoly*3)
                        constraintVectorFM = (stateInfo.constraint['nir'][mykey][:,0:npoly]).reshape(npoly*3)
                        initialGuessListFM = initialGuessList.copy()
                        if have_true:
                            trueParameterList = (stateInfo.true['nir'][mykey][:,0:npoly]).reshape(npoly*3)
                            trueParameterListFM = trueParameterList.copy()
                    elif species_name == 'NIREOF':
                        npar = len(stateInfo.current['nir'][mykey][:,0])
                        nband = len(stateInfo.current['nir'][mykey][0,:])
                        initialGuessList = stateInfo.current['nir'][mykey].reshape(npar*nband)
                        constraintVector = stateInfo.constraint['nir'][mykey].reshape(npar*nband)
                        constraintVectorFM = stateInfo.constraint['nir'][mykey].reshape(npar*nband)
                        initialGuessListFM = copy.deepcopy(initialGuessList)
                        if have_true:
                            trueParameterList = stateInfo.true['nir'][mykey].reshape(npar*nband)
                            trueParameterListFM = copy.deepcopy(trueParameterList)
                    # elif species_name == 'NIRCLOUD3D':
                    #     npar = len(stateInfo.current['nir'][mylist1[indnir]][:,0])
                    #     nband = len(stateInfo.current['nir'][mylist1[indnir]][0,:])
                    #     initialGuessList = stateInfo.current['nir'][mylist1[indnir]].reshape(npar*nband)
                    #     trueParameterList = stateInfo.true['nir'][mylist1[indnir]].reshape(npar*nband)
                    #     constraintVector = stateInfo.constraint['nir'][mylist1[indnir]].reshape(npar*nband)
                    #     initialGuessListFM = deepcopy(initialGuessList)
                    #     trueParameterListFM = deepcopy(trueParameterList)
                    elif species_name == 'NIRWIND':
                        npar = 1
                        initialGuessList = [stateInfo.current['nir']['wind']]
                        constraintVector = [stateInfo.constraint['nir']['wind']]
                        constraintVectorFM = [stateInfo.constraint['nir']['wind']]
                        initialGuessListFM = [copy.deepcopy(initialGuessList)]
                        if have_true:
                            trueParameterListFM = [stateInfo.true['nir']['wind']]
                            trueParameterList = [stateInfo.true['nir']['wind']]
                    else:
                        initialGuessList = stateInfo.current['nir'][mykey]
                        constraintVector = stateInfo.constraint['nir'][mykey]
                        constraintVectorFM = stateInfo.constraint['nir'][mykey]
                        initialGuessListFM = copy.deepcopy(initialGuessList)
                        if have_true:
                            trueParameterList = stateInfo.true['nir'][mykey]
                            trueParameterListFM = copy.deepcopy(trueParameterList)

                    nn = len(initialGuessListFM)
                    mm = len(initialGuessList)
                    num_retrievalParameters = mm

                    if mm == 1:
                        mapToState = 1
                        mapToParameters = 1

                        # it's difficult to get a nx1 array.  1xn is easy.
                        retrievalParameters = [0]
                        pressureList = [-2]
                        pressureListFM = [-2]
                        altitudeList = [-2]
                        altitudeListFM = [-2]

                        val = (speciesInformationFile.sSubaDiagonalValues).split(',')
                        for jj in range(len(val)):
                            val[jj] = np.double(val[jj])
                        sSubaDiagonalValues = val
                        constraintMatrix = [1/val[0]/val[0]]

                    elif mm == nn:
                        mapToState = np.identity(mm)
                        mapToParameters = np.identity(mm)

                        retrievalParameters = range(mm)
                        pressureList = np.zeros(mm)-2
                        pressureListFM = np.zeros(mm)-2
                        altitudeList = np.zeros(mm)-2
                        altitudeListFM = np.zeros(mm)-2

                        if speciesInformationFile.constraintType == 'Diagonal':
                            val = (speciesInformationFile.sSubaDiagonalValues).split(',')
                            for jj in range(len(val)):
                                val[jj] = np.double(val[jj])
                            sSubaDiagonalValues = val
                            constraintMatrix = np.identity(mm)
                            for jj in range(len(val)):
                                constraintMatrix[jj,jj] = 1/sSubaDiagonalValues[jj]/sSubaDiagonalValues[jj]
                        elif speciesInformationFile.constraintType == 'Full':
                            (covariance, _) = mpy.constraint_read(speciesInformationFile.sSubaFilename)
                            constraintMatrix = np.linalg.inv(covariance['data'])

                            pressureList = covariance['pressure']
                            altitudeList = covariance['pressure']
                            pressureListFM = covariance['pressure']
                            altitudeListFM = covariance['pressure']
                        elif speciesInformationFile.constraintType == 'PREMADE':
                            filename = speciesInformationFile.constraintFilename                 

                            constraintMatrix, pressurex = mpy.supplier_constraint_matrix_premade(
                                species_name,
                                filename,
                                mm, 
                                i_nh3type = stateInfo.nh3type, 
                                i_ch3ohtype = stateInfo.ch3ohtype)

                            pressureList = pressurex.copy()
                            pressureListFM = pressurex.copy()
                            #pressurelistFM = stateInfo.current.nir.albplwave
                            altitudeList = pressurex.copy()
                            altitudeListFM = pressurex.copy()
                            #altitudeListFM = stateInfo.current.nir.albplwave                    

                        else:
                            raise RuntimeError(f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}")
                    else:
                        # already made maps, above

                        retrievalParameters = range(mm)
                        pressureList = np.zeros(mm)-2
                        pressureListFM = np.zeros(nn)-2
                        altitudeList = np.zeros(mm)-2
                        altitudeListFM = np.zeros(nn)-2

                    if species_name == 'NIRALBBRDF' or species_name == 'NIRALBLAMB' or species_name == 'NIRALBCM':
                        pressureListFM = (stateInfo.current['nir']['albplwave']).reshape(len(stateInfo.current['nir']['albplwave']))


                        if speciesInformationFile.constraintType == 'Diagonal':
                            val = (speciesInformationFile.sSubaDiagonalValues).split(',')
                            for jj in range(len(val)):
                                val[jj] = np.double(val[jj])
                            sSubaDiagonalValues = val
                            constraintMatrix = np.identity(mm)
                            for jj in range(mm):
                                constraintMatrix[jj,jj] = 1/sSubaDiagonalValues[jj]/sSubaDiagonalValues[jj]
                        elif speciesInformationFile.constraintType == 'Full':
                            file = mpy.constraint_read(speciesInformationFile.sSubaFilename)
                            constraintMatrix = np.invert(file['data'])
                        else:
                            raise RuntimeError(f"Unknown type for {speciesInformationFile.filename} constraintType is {speciesInformationFile.constraintType}")


                    # change to log if specified
                    if (speciesInformationFile.mapType).upper() == 'LOG':
                        initialGuessListFM = np.log(initialGuessListFM)
                        constraintVector = np.log(constraintVector)
                        initialGuessList = np.log(initialGuessList)
                        if have_true:
                            trueParameterList = np.log(trueParameterList)
                            trueParameterListFM = np.log(trueParameterListFM)

                    if 'ALB' in species_name:
                        ind = (np.where(initialGuessListFM < -990))[0]
                        if len(ind) > 0:
                            raise RuntimeError("Error -999 in albedo. Possibly mismatch between state file inputs and strategy table for ALBLAMB vs. ALBBRDF")


                elif (species_name == 'EMIS') or (species_name == 'CLOUDEXT') or \
                     (species_name == 'CALSCALE') or (species_name == 'CALOFFSET'):

                    # IDL AT_LINE 243 Get_Species_Information:
                    microwindows = mpy.table_new_mw_from_step(i_table_struct, step)

                    # Select non-UV windows
                    ind = []
                    for ff in range(len(microwindows)):
                        if 'UV' not in microwindows[ff]['filter']:
                            ind.append(ff)

                    temp_microwindows = microwindows
                    microwindows = []
                    for ff in range(len(ind)):
                        microwindows.append(temp_microwindows[ind[ff]])

                    # Get species specific things, e.g. get frequency grid
                    # for EMIS from EMIS pars
                    # AT_LINE 250 Get_Species_Information.pro
                    if species_name == 'EMIS':
                        frequencyIn = stateInfo.emisPars['frequency'][0:int(stateInfo.emisPars['num_frequencies'])]
                        # AT_LINE 252 Get_Species_Information.pro
                        stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn)

                        # AT_LINE 254 Get_Species_Information.pro
                        nn = len(stepFMSelect)
                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]
                        mm = len(ind)
                        freq = stateInfo.emisPars['frequency'][0:nn]
                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]

                        # AT_LINE 258 Get_Species_Information.pro
                        freqRet = freq[ind]
                        freqRet = np.asarray(freqRet)

                        # in this, all frequencies that are between other
                        # frequencies are mapped.  Then take out frequencies
                        # more than 20 from the retrieved on both sides

                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]
                        ind = ind + 1

                        linearFlag = True
                        averageFlag = False
                        maps = mpy.make_maps(stateInfo.emisPars['frequency'][0:nn], ind, linearFlag, averageFlag)
                        mapToState = maps['toState'] # SPECIES_NAME 'EMIS'
                        mapToParameters = maps['toPars'] # mapToParameters.shape (121, 2)

                        # if an EMIS frequency was not retrieved but is in a
                        # gap larger than 50 cm-1, then remove it
                        # All frequencies bracketed by retrieved frequencies
                        # are AUTOMATICALLY interpolated by mapping, above.
                        # AT_LINE 274 Get_Species_Information.pro
                        num_rows_cleared = 0
                        for kk in range(0, nn):
                            ind = np.where(freqRet < freq[kk])
                            if len(ind[0]) > 0:
                                ind = ind[0]
                                ind1 = np.amax(ind)
                                ind1_arr = []
                                ind1_arr.append(ind1)
                                ind1 = np.asarray(ind1_arr) # Convert a list of 1 element into an array of 1 element.
                            else:
                                ind1 = [] # An empty list since there are none that fit the criteria: freqRet < freq[kk]

                            ind = np.where(freqRet > freq[kk])
                            if len(ind[0]) > 0:
                                ind = ind[0]
                                ind = np.asarray(ind)
                                ind2 = np.amin(ind)
                                ind2_arr = []
                                ind2_arr.append(ind2)
                                ind2 = np.asarray(ind2_arr) # Convert a list of 1 element into an array of 1 element.
                            else:
                                ind2 = [] # An empty list since there are none that fit the criteria: freqRet > freq[kk]

                            ind = np.where(np.absolute(freqRet - freq[kk]) < 0.001)
                            ind = ind[0]

                            # AT_LINE 278 Get_Species_Information.pro
                            if len(ind) == 0 and len(ind1) > 0 and len(ind2) > 0:
                                # calculate the frequency difference between the
                                # given frequency point and the closest points with larger and smaller
                                # frequency
                                diff1 = freq[kk] - freqRet[ind1[0]]  # We only want 1 value from freqRet
                                diff2 = freqRet[ind2[0]] - freq[kk]  # We only want 1 value from freqRet 
                                if diff1 + diff2 > 50:
                                    # zero out interpolation if gap larger
                                    # than 50 cm-1
                                    mapToState[:, kk] = 0
                                    num_rows_cleared = num_rows_cleared + 1
                        # end for kk in range(0, nn):

                        # Set pars for retrieval
                        # AT_LINE 294 Get_Species_Information.pro
                        pressureListFM = stateInfo.emisPars['frequency'][0:nn]
                        altitudeListFM = stateInfo.emisPars['frequency'][0:nn] / 100. # just for spacing
                        constraintVectorFM = stateInfo.constraint['emissivity'][0:nn]
                        initialGuessListFM = stateInfo.current['emissivity'][0:nn]
                        if have_true:
                            trueParameterListFM = stateInfo.true['emissivity'][0:nn]
                    # end part of if (species_name == 'EMIS'):
                    # AT_LINE 300 Get_Species_Information.pro

                    # clouds complicated by capability of having two true
                    # clouds and bracket mode for IGR step
                    # AT_LINE 305 Get_Species_Information.pro
                    if species_name == 'CLOUDEXT':
                        # get IGR frequency mode
                        # AT_LINE 308 Get_Species_Information.pro
                        filename = mpy.table_get_pref(i_table_struct, "CloudParameterFilename")
                        if not os.path.isfile(filename):
                            raise RuntimeError(f"File not found:  {filename}")

                        (_, fileID) = mpy.read_all_tes_cache(filename)
                        freqMode = mpy.tes_file_get_preference(fileID, "CLOUDEXT_IGR_Min_Freq_Spacing")
                        freqMode = (freqMode.split())[0] # In case the preference contains multiple tokens, we only wish to get the first one.

                        # get which freqs are used in this step... consider
                        # step type
                        # AT_LINE 318 Get_Species_Information.pro
                        frequencyIn = stateInfo.cloudPars['frequency'][0:int(stateInfo.cloudPars['num_frequencies'])]
                        stepType = mpy.table_get_entry(i_table_struct, step, "retrievalType")
                        stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn, stepType, freqMode)

                        nn = len(stepFMSelect)
                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]
                        mm = len(ind)

                        averageFlag = False
                        if freqMode.lower() == 'one':
                            averageFlag = True
                            mm = 1

                        ind = np.where(stepFMSelect != 0)[0]
                        ind = ind + 1
                        linearFlag = True

                        # Note: lines 333 and 334 in IDL are unnecessary since mapToState and mapToParameters gets assigned to maps.toState and maps.toPars.
                        # AT_LINE 336 Get_Species_Information.pro
                        maps = mpy.make_maps(stateInfo.cloudPars['frequency'][0:nn], ind, linearFlag, averageFlag)
                        maps = mpy.ObjectView(maps)
                        mapToState = maps.toState 
                        mapToParameters = maps.toPars  

                        # AT_LINE 340 Get_Species_Information.pro
                        pressureListFM = stateInfo.cloudPars['frequency'][0:nn]                   
                        altitudeListFM = stateInfo.cloudPars['frequency'][0:nn] / 100.0           
                        constraintVectorFM = np.log(stateInfo.constraint['cloudEffExt'][0, 0:nn]) 
                        initialGuessListFM = np.log(stateInfo.current['cloudEffExt'][0, 0:nn]) 
                        if have_true:
                            trueParameterListFM = np.log(stateInfo.true['cloudEffExt'][0, 0:nn]) 

                            # get true values for 2 clouds.
                            # try to combine the two clouds into 1, since we'll have
                            # only 1 cloud Jacobian, for the linear estimate
                            # AT_LINE 351 Get_Species_Information.pro
                            if int(stateInfo.true['num_clouds']) == 2:
                                # take larger cloud; then try to fold smaller cloud in
                                c1 = stateInfo.true['cloudEffExt'][0, 0:nn]
                                c2 = stateInfo.true['cloudEffExt'][1, 0:nn]
                                trueParameterListFM = np.log(c1 + c2)
                        # end part of if (species_name == 'CLOUDEXT'):

                    # AT_LINE 360 Get_Species_Information.pro
                    if species_name == 'CALSCALE':
                        # get which freqs are used in this step

                        # AT_LINE 363 Get_Species_Information.pro
                        frequencyIn = stateInfo.calibrationPars['frequency'][0:int(stateInfo.calibrationPars['num_frequencies'])]
                        stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn, stepType, freqMode)

                        # AT_LINE 365 Get_Species_Information.pro
                        nn = len(stepFMSelect)
                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]
                        mm = len(ind)
                        mapToParameters = np.zeros(shape=(nn, mm), dtype=np.float64)
                        mapToState = np.zeros(shape=(mm, nn), dtype=np.float64)

                        # For CALSCALE species, do NOT allow interpolation.  Just have 1:1
                        # mapping.  E.g. if retrieve in 2B1 and 2A1 
                        # filters, results should not go into the 1B2 filter. 
                        # ideally interpolation allowed within filters only
                        count = 0
                        for ik in range(0, nn):
                            if stepFMSelect[ik] == 1:
                                mapToState[count, ik] = 1
                                mapToParameters[ik, count] = 1
                                count = count + 1

                        pressureListFM = stateInfo.calibrationPars['frequency'][0:nn]
                        altitudeListFM = stateInfo.calibrationPars['frequency'][0:nn] / 100.0
                        constraintVectorFM = stateInfo.constraint['calibrationScale'][0:nn]
                        initialGuessListFM = stateInfo.current['calibrationScale'][0:nn]
                        if have_true:
                            trueParameterListFM = stateInfo.true['calibrationScale'][0:nn]
                        assert False
                    # end part of if (species_name == 'CALSCALE'):

                    # AT_LINE 391 Get_Species_Information.pro
                    if species_name == 'CALOFFSET':
                        # get which freqs are used in this step
                        frequencyIn = stateInfo.calibrationPars['frequency'][0:stateInfo.calibrationPars['num_frequencies']]
                        stepFMSelect = mpy.mw_frequency_needed(microwindows, frequencyIn)

                        # AT_LINE 397 Get_Species_Information.pro
                        nn = len(stepFMSelect)
                        ind = np.where(stepFMSelect != 0)
                        ind = ind[0]
                        mm = len(ind)

                        # AT_LINE 400 Get_Species_Information.pro
                        mapToParameters = np.zeros(shape=(nn, mm), dtype=np.float64)
                        mapToState = np.zeros(shape=(mm, nn), dtype=np.float64)

                        # For CALOFFSET species, do NOT allow interpolation.  Just have 1:1
                        # mapping 

                        # AT_LINE 407 Get_Species_Information.pro
                        count = 0
                        for ik in range(0, nn):
                            if stepFMSelect[ik] == 1:
                                mapToState[count, ik] = 1
                                mapToParameters[ik, count] = 1
                                count = count + 1

                        pressureListFM = stateInfo.calibrationPars['frequency'][0:nn]
                        altitudeListFM = stateInfo.calibrationPars['frequency'][0:nn] / 100.0
                        constraintVectorFM = stateInfo.constraint['calibrationOffset'][0:nn]
                        initialGuessListFM = stateInfo.current['calibrationOffset'][0:nn]
                        if have_true:
                            trueParameterListFM = stateInfo.true['calibrationOffset'][0:nn]
                        assert False
                    # end part of if (species_name == 'CALOFFSET'):

                    # AT_LINE 425 Get_Species_Information.pro

                    # now map FM to retrieval grid.  If log parameter, the
                    # FM is already in log

                    ind = np.where(stepFMSelect != 0)
                    ind = ind[0]
                    mm = len(ind)
                    m = np.asarray(mapToParameters)

                    if mm > 1:
                        altitudeList = np.matmul(altitudeListFM, m)
                        pressureList = np.matmul(pressureListFM, m)
                        constraintVector = np.matmul(constraintVectorFM, m)
                        initialGuessList = np.matmul(initialGuessListFM, m)
                        if have_true:
                            trueParameterList = np.matmul(trueParameterListFM, m)
                    else:
                        altitudeList = np.sum(m * altitudeListFM)
                        pressureList = np.sum(m * pressureListFM)
                        constraintVector = np.sum(m * constraintVectorFM)
                        initialGuessList = np.sum(m * initialGuessListFM)
                        if have_true:
                            trueParameterList = np.sum(m * trueParameterListFM)

                    # AT_LINE 443 Get_Species_Information.pro
                    # Now get constraint matrix.
                    if constraintType.lower() == 'tikhonov':
                        constraintMatrix = np.zeros(shape=(mm, mm), dtype=np.float64) # DBLARR(mm,mm)
                        myError = (1 / (0.1))**2
                        constraintMatrix[0, 0] = myError
                        constraintMatrix[mm-1, mm-1] = myError
                        for ll in range(0, mm-2):
                            if pressureList[ll+1] - pressureList[ll] < 30:
                                constraintMatrix[ll, ll] = constraintMatrix[ll, ll] + myError
                                constraintMatrix[ll+1, ll] = constraintMatrix[ll+1, ll] - myError
                                constraintMatrix[ll, ll+1] = constraintMatrix[ll, ll+1] - myError
                                constraintMatrix[ll+1, ll+1] = constraintMatrix[ll+1, ll+1] + myError
                            else:
                                constraintMatrix[ll, ll] = constraintMatrix[ll, ll] + myError
                                constraintMatrix[ll+1, ll+1] = constraintMatrix[ll+1, ll+1] + myError
                    elif constraintType.lower() == 'full':
                        sSubaFilename = speciesInformationFile.sSubaFilename
                        constraintMatrix = mpy.supplier_constraint_matrix_ssuba(constraintVector,
                                                                            species_name,
                                                                            mapType,
                                                                            mapToParameters,
                                                                            pressureListFM,
                                                                            pressureList,
                                                                            sSubaFilename)
                    elif constraintType.lower == 'premade':
                        raise RuntimeError("premade type not implemented for spectral type check Supplier_Constraint_Matrix for previous implementation")

                        # problem with pre-made pre-inverted
                        # constraint... when select some frequency it
                        # seems to be ill-conditioned when inverted to
                        # covariance... 
                    else:
                        raise RuntimeError(f"Unknown constraint type for {species_name} {constraintType}")

                    # done with spectral species 

                # end part of elif (species_name == 'EMIS')     or (species_name == 'CLOUDEXT')
                #                  (species_name == 'CALSCALE') or (species_name == 'CALOFFSET'):

                # AT_LINE 489 Get_Species_Information.pro
                elif species_name == 'PTGANG':

                    # AT_LINE 491 Get_Species_Information.pro
                    nn = 1
                    mm = 1

                    mapToState = 1
                    mapToParameters = 1

                    # it's difficult to get a nx1 array.  1xn is easy.
                    retrievalParameters = [0]
                    altitudeList = [-2]
                    altitudeListFM = [-2]
                    pressureList = [-2]
                    pressureListFM = [-2]
                    num_retrievalParameters = 1

                    # set pars for retrieval
                    constraintVector = stateInfo.constraint['tes']['boresightNadirRadians']
                    constraintVectorFM = stateInfo.constraint['tes']['boresightNadirRadians']
                    initialGuessList = stateInfo.current['tes']['boresightNadirRadians']
                    initialGuessListFM = stateInfo.current['tes']['boresightNadirRadians']
                    if have_true:
                        trueParameterList = stateInfo.true['tes']['boresightNadirRadians']
                        trueParameterListFM = stateInfo.true['tes']['boresightNadirRadians']

                    raise RuntimeError(f"Need more coding from developer. species_name {species_name}")
                    # Constraint = INVERT(Constraint_Get(errorInitial, species))

                # end part of elif (species_name == 'PTGANG'):

                # AT_LINE 515 Get_Species_Information.pro
                elif species_name == 'PCLOUD':
                    nn = 1
                    mm = 1

                    mapToState = 1
                    mapToParameters = 1

                    retrievalParameters = [0]
                    altitudeList = [-2]
                    altitudeListFM = [-2]
                    pressureList = [-2]
                    pressureListFM = [-2]
                    num_retrievalParameters = 1

                    # set pars for retrieval
                    constraintVector = math.log(stateInfo.constraint['PCLOUD'][0])
                    constraintVectorFM = math.log(stateInfo.constraint['PCLOUD'][0])
                    initialGuessList = math.log(stateInfo.current['PCLOUD'][0])
                    initialGuessListFM = math.log(stateInfo.current['PCLOUD'][0])
                    if have_true:
                        trueParameterList = math.log(stateInfo.true['PCLOUD'][0])
                        trueParameterListFM = math.log(stateInfo.true['PCLOUD'][0])

                    # get constraint
                    stepType = mpy.table_get_entry(i_table_struct, step, "retrievalType")
                    tag_names = mpy.idl_tag_names(speciesInformationFile)
                    if 'sSubaDiagonalValues' not in tag_names:
                        raise RuntimeError(f"Preference 'sSubaDiagonalValues' NOT found in file {speciesInformationFile.filename}")

                    # AT_LINE 546 Get_Species_Information.pro
                    constraintMatrix = np.float64(speciesInformationFile.sSubaDiagonalValues)
                    constraintMatrix = 1 / constraintMatrix / constraintMatrix

                    if have_true:

                        # pick thickest cloud for cloud height
                        if stateInfo.true['num_clouds'] == 2:
                            # take larger cloud; then try to fold smaller cloud in
                            c1 = stateInfo.true['cloudEffExt'][0, :]
                            c2 = stateInfo.true['cloudEffExt'][1, :]

                            if np.sum(c1) > np.sum(c2):
                                trueParameterList = math.log(stateInfo.true['PCLOUD'][1])
                # end part of elif (species_name == 'PCLOUD'):
                # AT_LINE 558 Get_Species_Information.pro
                elif species_name == 'RESSCALE':
                    nn = 2
                    mm = 2

                    mapToState = np.identity(2)
                    mapToParameters = np.identity(2)

                    retrievalParameters = [0]
                    altitudeList = [2]
                    altitudeListFM = [2]
                    pressureList = [2]
                    pressureListFM = [2]
                    num_retrievalParameters = 2

                    # set pars for retrieval
                    constraintVector = [1, 1]
                    constraintVectorFM = [1, 1]
                    initialGuessList = [0, 0]
                    initialGuessListFM = [1, 1]
                    if have_true:
                        trueParameterList = [1, 1]
                        trueParameterListFM = [1, 1]

                    sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)
                    constraintMatrix = 1 / sSubaDiagonalValues / sSubaDiagonalValues
                # end part of elif (species_name == 'RESSCALE'):
                elif species_name == 'TSUR':
                    # AT_LINE 583 Get_Species_Information.pro
                    nn = 1
                    mm = 1

                    mapToState = 1
                    mapToParameters = 1

                    # it's difficult to get a nx1 array.  1xn is easy.
                    # AT_LINE 592 Get_Species_Information.pro
                    retrievalParameters = [0]
                    altitudeList = [-2]
                    altitudeListFM = [-2]
                    pressureList = [-2]
                    pressureListFM = [-2]
                    num_retrievalParameters = 1

                    # set pars for retrieval
                    # AT_LINE 600 Get_Species_Information.pro
                    constraintVectorFM = stateInfo.constraint['TSUR']
                    constraintVector = stateInfo.constraint['TSUR']
                    initialGuessList = stateInfo.current['TSUR']
                    initialGuessListFM = stateInfo.current['TSUR']
                    if have_true:
                        trueParameterList = stateInfo.true['TSUR']
                        trueParameterListFM = stateInfo.true['TSUR']

                    tag_names = mpy.idl_tag_names(speciesInformationFile)
                    upperTags = [x.upper() for x in tag_names]
                    step_name = mpy.table_get_entry(i_table_struct, step, "stepName")
                    full_step_label = ('sSubaDiagonalValues-'  + step_name).upper()
                    if full_step_label in upperTags:
                        sSubaDiagonalValues = np.asarray(speciesInformationFile[upperTags.index(full_step_label)])
                        logger.warning(f"using step-dependent constraint for TSUR, value: {1 / (sSubaDiagonalValues * sSubaDiagonalValues)}")
                    else:
                        sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)

                    constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)
                # end part of elif (species_name == 'TSUR'):
                elif species_name == 'PSUR':
                    # AT_LINE 583 Get_Species_Information.pro
                    nn = 1
                    mm = 1

                    mapToState = 1
                    mapToParameters = 1

                    # it's difficult to get a nx1 array.  1xn is easy.
                    # AT_LINE 592 Get_Species_Information.pro
                    retrievalParameters = [0]
                    altitudeList = [-2]
                    altitudeListFM = [-2]
                    pressureList = [-2]
                    pressureListFM = [-2]
                    num_retrievalParameters = 1

                    # set pars for retrieval
                    # AT_LINE 696 Get_Species_Information.pro
                    constraintVector = stateInfo.constraint['pressure'][0]
                    constraintVectorFM = stateInfo.constraint['pressure'][0]
                    initialGuessList = stateInfo.current['pressure'][0]
                    initialGuessListFM = stateInfo.current['pressure'][0]
                    if have_true:
                        trueParameterList = stateInfo.true['pressure'][0]
                        trueParameterListFM = stateInfo.true['pressure'][0]

                    tag_names = mpy.idl_tag_names(speciesInformationFile)
                    upperTags = [x.upper() for x in tag_names]
                    step_name = mpy.table_get_entry(i_table_struct, step, "stepName")
                    full_step_label = ('sSubaDiagonalValues-'  + step_name).upper()
                    if full_step_label in upperTags:
                        sSubaDiagonalValues = np.asarray(speciesInformationFile[upperTags.index(full_step_label)])
                        logger.warning(f"using step-dependent constraint for PSUR, value: {1 / (sSubaDiagonalValues * sSubaDiagonalValues)}")
                    else:
                        sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)

                    constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)
                # end part of elif (species_name == 'PSUR'):
                elif (mapType == 'linearscale') or (mapType == 'logscale'):
                    # AT_LINE 718 Get_Species_Information.pro
                    if species_name not in stateInfo.species:
                        raise RuntimeError("Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}")
                    retrievalParameters = 5 # level 5

                    num_retrievalParameters = 1
                    mm = num_retrievalParameters
                    nn = stateInfo.num_pressures                   

                    pressureList = pressure[retrievalParameters-1]
                    pressureListFM = pressure
                    altitudeList = stateInfo.current['heightKm'][retrievalParameters-1]
                    altitudeListFM = stateInfo.current['heightKm']                

                    # map isn't used but size is useful
                    #maps = {'toPars':np.zeros(shape=(nn,mm), dtype=np.float64),
                    #    'toState':np.zeros(shape=(mm,nn), dtype=np.float64)}
                    #maps = make_maps(stateInfo.current['pressure'], retrievalParameters)
                    #maps = ObjectView(maps)
                    mapToParameters = np.zeros(shape=(nn,mm), dtype=np.float64) + 1/20 # not sure about this
                    mapToState = np.zeros(shape=(mm,nn), dtype=np.float64) + 1 # only for mapping Jacobian                

                    # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
                    ind = np.where(species_name == np.asarray(stateInfo.species))[0]

                    initialGuessFM = stateInfo.current['values'][ind, :]   # Keep the array as 2 dimensions so we can multiply them later.
                    currentGuessFM = stateInfo.current['values'][ind, :]
                    constraintVectorFM = stateInfo.constraint['values'][ind, :]
                    constraintMatrix = stateInfo.constraint['values'][ind, :]
                    if have_true:
                        trueStateFM = stateInfo.true['values'][ind, :]

                    sSubaDiagonalValues = np.float64(speciesInformationFile.sSubaDiagonalValues)
                    constraintMatrix = 1 / (sSubaDiagonalValues * sSubaDiagonalValues)

                    # AT_LINE 752 Get_Species_Information.pro

                    # since the "true" is relative to the initial guess
                    # the "true state" is set to e.g. 0.8 if the initial guess
                    # is off by -0.8K
                    if mapType == 'linearscale': 
                        constraintVector = 0
                        initialGuessList = np.mean(initialGuessFM - constraintVectorFM)
                        initialGuessListFM = constraintVectorFM + initialGuessList
                        if have_true:
                            trueParameterList = np.mean(trueStateFM - initialGuessFM)
                            trueParameterListFM = np.copy(trueStateFM)
                    else:
                        constraintVector = 1
                        initialGuessList = np.mean(constraintVectorFM / initialGuessFM)
                        initialGuessListFM = constraintVectorFM * initialGuessList
                        if have_true:
                            trueParameterList = np.mean(trueStateFM / initialGuessFM)
                            trueParameterListFM = np.copy(trueStateFM)                
                # end part of elif (mapType == 'linearscale') or (mapType == 'logscale'):

                elif (mapType == 'linearpca') or (mapType == 'logpca'):

                    # this is state vector of the form:
                    # current = apriori + mapToState @ currentGuess for linearpca
                    # or 
                    # log(current) = log(apriori) + mapToState @ log(currentGuess) for logpca
                    # Doing current = mapToState @ currentGuess does not work
                    # because the maps do not have a good span of the state, e.g. the stratosphere does not have sensitivity.
                    # when I tried this I got Tatm = [300, ..., 62, 37, -20, -27]
                    # so it must be aprior + offset

                    if species_name not in stateInfo.species:
                        raise RuntimeError(f"Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}")

                    mapsFilename = speciesInformationFile.mapsFilename.replace('64_',str(stateInfo.num_pressures)+'_')

                    # retrieval "levels"
                    levels_tokens = speciesInformationFile.retrievalLevels.split(',')
                    int_levels_arr = [int(x) for x in levels_tokens]
                    retrievalParameters = int_levels_arr

                    num_retrievalParameters = len(retrievalParameters)
                    mm = num_retrievalParameters
                    nn = stateInfo.num_pressures

                    pressureListFM = pressure
                    altitudeListFM = stateInfo.current['heightKm']

                    # implemented for OCO-2, but if used for other satellite
                    # need to change # of full state levels to read correct file
                    # for oco-2:  maps_TATM_Linear_20_3.nc, where 20 is # of full state pressures
                    (mapDict, _, _) = cdf_read_dict(mapsFilename)
                    mapToState = np.transpose(mapDict['to_state'])
                    mapToParameters = np.transpose(mapDict['to_pars'])

                    altitudeList = np.transpose(mapToParameters) @ altitudeListFM
                    pressureList = np.transpose(mapToParameters) @ pressure # nonsense values, e.g. [1595,  1833, 594]
                    pressureList[:] = -999
                    altitudeList[:] = -999

                    filename = speciesInformationFile.constraintFilename
                    (constraintStruct, constraintPressure) = mpy.constraint_read(filename)
                    constraintMatrix = mpy.constraint_get(constraintStruct)
                    constraintPressure = mpy.constraint_get_pressures(constraintStruct)

                    # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
                    ind = np.where(species_name == np.asarray(stateInfo.species))[0]

                    initialGuessFM = stateInfo.current['values'][ind, :].reshape(nn)   # Keep the array as 2 dimensions so we can multiply them later.
                    currentGuessFM = stateInfo.current['values'][ind, :].reshape(nn)
                    constraintVectorFM = stateInfo.constraint['values'][ind, :].reshape(nn)
                    if have_true:
                        trueStateFM = stateInfo.true['values'][ind, :].reshape(nn)


                    # since the "true" is relative to the a priori
                    # the "true state" is set to e.g. 0.8 if the a priori
                    # is off by -0.8K
                    # atmospheric parameters
                    if mapType == 'linearpca': 
                        constraintVector = np.zeros(mm,dtype = np.float32) + 0
                        initialGuessList = np.transpose(mapToParameters) @ (initialGuessFM - constraintVectorFM)
                        initialGuessListFM = np.copy(constraintVectorFM) + np.transpose(mapToState) @ initialGuessList

                        if have_true:
                            trueParameterList = np.transpose(mapToParameters) @ (trueState - constraintVectorFM)
                            trueParameterListFM = np.copy(trueState)
                    else:
                        constraintVector = np.zeros(mm,dtype = np.float32) + 0
                        initialGuessList = np.transpose(mapToParameters) @ (np.log(initialGuessFM) - np.log(constraintVectorFM))
                        initialGuessListFM = np.copy(constraintVectorFM) + np.transpose(np.mapToState) @ initialGuessList
                        if have_true:
                            trueParameterList = np.transpose(mapToParameters) @ (np.log(trueState) - np.log(constraintVectorFM))
                            trueParameterListFM = np.copy(trueState)                
                # end part of elif (mapType == 'linearpca') or (mapType == 'logpca'):

                else:
                    # AT_LINE 629 Get_Species_Information.pro
                    # line parameter, e.g. H2O, CO2, O3, TATM, ...

                    if species_name not in stateInfo.species:
                        raise RuntimeError(f"Species not found in stateInfo.  This usually means your spectral windows do not include this species OR the L2_Setup does not list this species. Looking for species: {species_name}")

                    # maps
                    if (mapType == 'linear') or (mapType == 'log'):

                        # We read in the retrieval levels and modify for
                        # current pressure grid
                        # Because the value of speciesInformationFile.retrievalLevels is a long string of:
                        #    '1,2,3,4,5,6,7,8,10,12,14,16,18,21,24,27,30,33,36,39,42,45,48,51,53,54,55,58,60,62,64,66'
                        # We need to split it up.
                        levels_tokens = speciesInformationFile.retrievalLevels.split(',')
                        int_levels_arr = [int(x) for x in levels_tokens]
                        levels0 = np.asarray(int_levels_arr)
    #                    if len(levels0) == len(i_table_struct.pressureFM) and len(levels0) == len(stateInfo.current['pressure']):
    #                        # for sigma pressure grid of OCO-2
    #                        i_table_struct.pressureFM = stateInfo.current['pressure'].copy()
                        retrievalParameters = mpy.supplier_retrieval_levels_tes(levels0, i_table_struct.pressureFM, stateInfo.current['pressure'])

                        # PYTHON_NOTE: It is possible that some values in i_levels may index passed the size of pressure.
                        # The size of pressure may be 63 and one indices may be 64.
                        any_values_greater_than_size = (retrievalParameters > pressure.size).any()
                        if any_values_greater_than_size:
                            o_cleaned_retrievalParameters = utilLevels.RemoveIndicesTooBig(retrievalParameters, pressure, function_name)
                            # Reassign retrievalParameters to o_cleaned_retrievalParameters so it will contain indices that are within size of pressure.
                            retrievalParameters = o_cleaned_retrievalParameters

                        # AT_LINE 636 Get_Species_Information.pro
                        num_retrievalParameters = len(retrievalParameters)
                        mm = num_retrievalParameters
                        nn = stateInfo.num_pressures

                        pressureList = pressure[retrievalParameters-1]
                        pressureListFM = pressure
                        altitudeList = stateInfo.current['heightKm'][retrievalParameters-1]
                        altitudeListFM = stateInfo.current['heightKm']

                        maps = mpy.make_maps(stateInfo.current['pressure'], retrievalParameters)
                        maps = mpy.ObjectView(maps)
                        mapToParameters = maps.toPars
                        mapToState = maps.toState
                    else:
                        # AT_LINE 699 Get_Species_Information.pro
                        raise RuntimeError('Only linear/log/pca maps implemented')

                    # AT_LINE 652 Get_Species_Information.pro

                    # PYTHON_NOTE: Because the value of stateInfo.species is a list, we have to convert to numpy array.
                    ind = np.where(species_name == np.asarray(stateInfo.species))[0]

                    initialGuessFM = stateInfo.current['values'][ind, :]   # Keep the array as 2 dimensions so we can multiply them later.
                    currentGuessFM = stateInfo.current['values'][ind, :]
                    constraintVectorFM = stateInfo.constraint['values'][ind, :]
                    if have_true:
                        trueStateFM = stateInfo.true['values'][ind, :]

                    # allows two NH3 steps which start at different initial
                    # values.  Put '2' for steptype in table for 2nd NH3 step (need to
                    # duplicate microwindows, quality flag, and species files)
                    if o_retrievalInfo.type == '2' and species_name == 'NH3':
                        logger.info("TWO STEP NH3.  SECOND STEP GUESS 2/3 ORIGINAL")
                        currentGuess = initialGuess * 2/3.

                    # AT_LINE 668 Get_Species_Information.pro
                    if mapType.lower() == 'log':
                        if have_true:
                            if mpy.table_get_pref(i_table_struct, "mapTrueFullStateVector") == 'yes':
                                trueStateFM = np.exp(np.matmul(np.matmul(mapToState, mapToParameters), np.log(trueStateFM)))

                        if mpy.table_get_pref(i_table_struct, "mapInitialGuess") == 'yes':
                            initialGuessFM = np.exp(np.matmul(np.log(initialGuessFM), np.matmul(mapToParameters, mapToState)))
                            currentGuessFM = np.exp(np.matmul(np.log(currentGuessFM), np.matmul(mapToParameters, mapToState)))
                    else:
                        if have_true:
                            if mpy.table_get_pref(i_table_struct, "mapTrueFullStateVector") == 'yes':
                                trueStateFM = np.matmul(np.matmul(mapToState, mapToParameters), trueStateFM)

                        if mpy.table_get_pref(i_table_struct, "mapInitialGuess") == 'yes':
                            initialGuess = np.matmul(initialGuessFM, np.matmul(mapToParameters, mapToState), initialGuessFM) # TRICKY_LOGIC
                            currentGuess = np.matmul(currentGuessFM, np.matmul(mapToParameters, mapToState))              # TRICKY_LOGIC

                    if mapType.lower() == 'log':
                        constraintVector = np.matmul(np.log(constraintVectorFM), mapToParameters)
                    else:
                        constraintVector = np.matmul(constraintVectorFM, mapToParameters)

                        # AT_LINE 693 src_ms-2018-12-10/Get_Species_Information.pro
                        if len(constraintVectorFM.shape) >= 2 and constraintVectorFM.shape[0] == 1:
                            # Re-shape back to 1-D array: from (1,30) to (30,)
                            constraintVector = np.reshape(constraintVector, (constraintVector.shape[1]))

                        if np.amin(constraintVector) < 0 and np.amax(constraintVector) > 0:
                            # fix issue with mapping going to negative #s
                            logger.info(f"Fix negative mapping: constraintVector: species_name: {species_name}")

                            ind1 = np.where(constraintVector < 0)[0]
                            ind2 = np.where(constraintVector > 0)[0]
                            constraintVector[ind1] = np.amin(constraintVector[ind2])
                        # end if np.amin(constraintCector) < 0 and np.amax(constraintVector) > 0:

                    # AT_LINE 693 Get_Species_Information.pro

                    initialGuessFM = currentGuessFM
                    errorMessages = 'no'

                    # for jointly retrieved... don't populate
                    # constraint check H2O-HDO, if so, get off diagonal also
                    locs = [(np.where(np.asarray(o_retrievalInfo.species) == 'H2O'))[0], (np.where(np.asarray(o_retrievalInfo.species) == 'HDO'))[0]]
                    num_retrievalPressures = len(retrievalParameters)

                    # AT_LINE 702 Get_Species_Information.pro
                    if locs[0] >= 0 and locs[1] >= 0 and (species_name == 'H2O' or species_name == 'HDO'):
                        constraintMatrix = np.zeros(shape=(num_retrievalPressures, num_retrievalPressures), dtype=np.float64)
                    else:
                        # AT_LINE 708 Get_Species_Information.pro
                        if constraintType == 'premade':
                            filename = speciesInformationFile.constraintFilename
                            if filename[0] == '':
                                raise RuntimeError(f"Name not found for PREMADE constraint: {filename}")

                            nh3type = stateInfo.nh3type
                            ch3ohtype = stateInfo.ch3ohtype

                            constraintMatrix, pressurex = mpy.supplier_constraint_matrix_premade(species_name,
                                            filename,
                                            num_retrievalPressures, 
                                            i_nh3type = stateInfo.nh3type, 
                                            i_ch3ohtype = stateInfo.ch3ohtype)
                        # AT_LINE 727 Get_Species_Information.pro
                        elif constraintType == 'covariance':
                            filename = speciesInformationFile.sSubaFilename
                            if filename[0] == '':
                                raise RuntimeError(f"Name not found for Covariance constraint In file {speciesInformationFile.filename}")

                            nh3type = stateInfo.nh3type
                            ch3ohtype = stateInfo.ch3ohtype
                            constraintMatrix = mpy.supplier_constraint_matrix_ssuba(species_name,
                                                                                mapType,
                                                                                mapToParameters,
                                                                                pressureListFM,
                                                                                pressureList,
                                                                                filename,
                                                                                i_nh3type = stateInfo.nh3type, 
                                                                                i_ch3ohtype = stateInfo.ch3ohtype)

                        # AT_LINE 747 Get_Species_Information.pro
                        elif species_name == 'O3' and constraintType == 'McPeters':
                            raise RuntimeError('Constraint type McPeters and O3 species not implemented yet')
                        else:
                            raise RuntimeError(f"Constraint type not implemented: {constraintType}")
                        # end else portion of if constraintType == 'premade':
                    # end else portion of if locs[0] >= 0 and locs[1] >= 0 and (species_name == 'H2O' or species_name == 'HDO'):

                    # AT_LINE 771 Get_Species_Information.pro
                    if constraintType == 'Scale':
                        if mapType == 'linear':
                            initialGuessList = np.mean(initialGuessListFM - constraintVectorFM)
                            initialGuessListFM = constraintVectorFM + initialGuessList
                            if have_true:
                                trueParameterList = np.mean(trueStateFM - constraintVectorFM)
                                trueParameterListFM = constraintVectorFM + trueParameterList
                        else:
                            initialGuessList = np.mean(initialGuessListFM / constraintVectorFM)
                            initialGuessListFM = constraintVectorFM * initialGuessList
                            if have_true:
                                trueParameterList = np.mean(trueStateFM / constraintVectorFM)
                                trueParameterListFM = constraintVectorFM * trueParameterList
                    else:    
                        if mapType == 'log':
                            initialGuessList = np.matmul(np.log(initialGuessFM), mapToParameters)
                            initialGuessListFM = np.log(initialGuessFM)
                            if have_true:
                                trueParameterList = np.matmul(np.log(trueState), mapToParameters)
                                trueParameterListFM = np.log(trueStateFM)
                        elif mapType == 'linear':
                            initialGuessList = np.matmul(initialGuessFM, mapToParameters)
                            initialGuessListFM = np.copy(initialGuessFM)
                            if have_true:
                                trueParameterList = np.matmul(trueStateFM, mapToParameters)
                                trueParameterListFM = np.copy(trueStateFM)

                            if len(initialGuessList.shape) >= 2 and initialGuessList.shape[0] == 1:
                                # Re-shape back to 1-D array: from (1,30) to (30,)
                                initialGuessList = np.reshape(initialGuessList, (initialGuessList.shape[1]))

                            if have_true:
                                if len(trueParameterList.shape) >= 2 and trueParameterList.shape[0] == 1:
                                    # Re-shape back to 1-D array: from (1,30) to (30,)
                                    trueParameterList = np.reshape(trueParameterList, (trueParameterList.shape[1]))

                            # AT_LINE 789 src_ms-2018-12-10/Get_Species_Information.pro Get_Species_Information

                            if np.amin(initialGuessList) < 0 and np.max(initialGuessList) > 0:
                                logger.info(f"Fix negative mapping: initialGuessList: species_name: {species_name}")
                                # fix issue with mapping going to negative #s
                                ind1 = np.where(initialGuessList < 0)[0]
                                ind2 = np.where(initialGuessList > 0)[0]
                                initialGuessList[ind1] = np.amin(initialGuessList[ind2])
                            # end if np.amin(initialGuessList) < 0 and np.max(initialGuessList) > 0:

                            if have_true:
                                if np.amin(trueParameterList) < 0 and np.amax(trueParameterList) > 0:
                                    logger.info(f"Fix negative mapping: initialGuessList: species_name: {species_name}")
                                    # fix issue with mapping going to negative #s
                                    ind1 = np.where(trueParameterList < 0)[0]
                                    ind2 = np.where(trueParameterList > 0)[0]
                                    trueParameterList[ind1] = np.amin(trueParameterList[ind2])
                                # end if np.amin(trueParameterList) < 0 and np.amax(trueParameterList) > 0:
                        else:
                            raise RuntimeError(f"mapType not handled: {mapType}")
                    # end else part of if constraintType == 'Scale':

                    loc = (np.where(np.asarray(stateInfo.species) == species_name))[0]
                    if loc.size == 0:
                        raise RuntimeError(f"FM species not found {species_name}. Are you running the step this species is in?")

                    # AT_LINE 789 Get_Species_Information.pro
                    stateInfo.initial['values'][loc, :] = initialGuessFM[:]
                    stateInfo.current['values'][loc, :] = currentGuessFM[:]
                    if have_true:
                        stateInfo.true['values'][loc, :] = trueStateFM[:]

                # end else part from AT_LINE 629 Get_Species_Information.pro
                # end else part of if 'OMI' in species_name:

                # AT_LINE 799 Get_Species_Information.pro

                # Convert any scalar values to array so we can use the [:] index syntax.
                if np.isscalar(altitudeList):
                    altitudeList = np.asanyarray([altitudeList])
                else:
                    altitudeList = np.asanyarray(altitudeList)

                if np.isscalar(altitudeListFM):
                    altitudeListFM = np.asanyarray([altitudeListFM])
                else:
                    altitudeListFM = np.asanyarray(altitudeListFM)

                if np.isscalar(pressureList):
                    pressureList = np.asanyarray([pressureList])
                else:
                    pressureList = np.asanyarray(pressureList)

                if np.isscalar(pressureListFM):
                    pressureListFM = np.asanyarray([pressureListFM])
                else:
                    pressureListFM = np.asanyarray(pressureListFM)

                if np.isscalar(constraintVector):
                    constraintVector = np.asanyarray([constraintVector])
                else:
                    constraintVector = np.asanyarray(constraintVector)

                if np.isscalar(initialGuessList):
                    initialGuessList = np.asanyarray([initialGuessList])
                else:
                    initialGuessList = np.asanyarray(initialGuessList)

                if np.isscalar(initialGuessListFM):
                    initialGuessListFM = np.asanyarray([initialGuessListFM])
                else:
                    initialGuessListFM = np.asanyarray(initialGuessListFM)

                if np.isscalar(constraintVectorFM):
                    constraintVectorFM = np.asanyarray([constraintVectorFM])
                else:
                    constraintVectorFM = np.asanyarray(constraintVectorFM)


                if have_true:
                    if np.isscalar(trueParameterList):
                        trueParameterList = np.asanyarray([trueParameterList])
                    else:
                        trueParameterList = np.asanyarray(trueParameterList)

                    if np.isscalar(trueParameterListFM):
                        trueParameterListFM = np.asanyarray([trueParameterListFM])
                    else:
                        trueParameterListFM = np.asanyarray(trueParameterListFM)


                # check minimum, maximum, maximumChange
                # if parameters missing, set to -999 (do not check min, max, max change)
                # allow single value or value for each retrieval parameter
                # Note below code does not account for differing # of pressure values

                minimum = np.zeros(shape=(mm), dtype=np.float64)-999
                try:
                    ff = (speciesInformationFile.minimum).split(',')
                    if len(ff) == 1:
                        minimum = minimum*0 + np.float(ff[0])
                    else:
                        for ix in range(len(ff)):
                            minimum[ix] = minimum[ix]*0 + np.float(ff[ix])
                except:
                    pass

                maximum = np.zeros(shape=(mm), dtype=np.float64)-999
                try:
                    ff = (speciesInformationFile.maximum).split(',')
                    if len(ff) == 1:
                        maximum = maximum*0 + np.float(ff[0])
                    else:
                        for ix in range(len(ff)):
                            maximum[ix] = maximum[ix]*0 + np.float(ff[ix])
                except:
                    pass

                maximum_change = np.zeros(shape=(mm), dtype=np.float64)-999
                try:
                    ff = (speciesInformationFile.maximumChange).split(',')
                    if len(ff) == 1:
                        maximum_change = maximum_change*0 + np.float(ff[0])
                    else:
                        for ix in range(len(ff)):
                            maximum_change[ix] = maximum_change[ix]*0 + np.float(ff[ix])
                except:
                    pass

                o_retrievalInfo.pressureList[row:row + mm] = pressureList[:]
                o_retrievalInfo.altitudeList[row:row + mm] = altitudeList[:]
                o_retrievalInfo.speciesList[row:row + mm] = [species_name] * mm

                o_retrievalInfo.pressureListFM[rowFM:rowFM + nn] = pressureListFM[:]
                o_retrievalInfo.altitudeListFM[rowFM:rowFM + nn] = altitudeListFM[:]
                o_retrievalInfo.speciesListFM[rowFM:rowFM + nn] = [species_name] * nn

                # AT_LINE 807 Get_Species_Information.pro
                o_retrievalInfo.constraintVector[row:row + mm] = constraintVector[:]
                o_retrievalInfo.initialGuessList[row:row + mm] = initialGuessList[:]
                o_retrievalInfo.initialGuessListFM[rowFM:rowFM + nn] = initialGuessListFM[:]
                o_retrievalInfo.constraintVectorListFM[rowFM:rowFM + nn] = constraintVectorFM[:]

                if have_true:
                    o_retrievalInfo.trueParameterList[row:row + mm] = trueParameterList[:]
                    o_retrievalInfo.trueParameterListFM[rowFM:rowFM + nn] = trueParameterListFM[:]

                o_retrievalInfo.minimumList[row:row + mm] = minimum[:]
                o_retrievalInfo.maximumList[row:row + mm] = maximum[:]
                o_retrievalInfo.maximumChangeList[row:row + mm] = maximum_change[:]


                o_retrievalInfo.mapToState[row:row + mm, rowFM:rowFM + nn] = mapToState
                o_retrievalInfo.mapToParameters[rowFM:rowFM + nn, row:row + mm] = mapToParameters

                o_retrievalInfo.species[ii] = species_name
                o_retrievalInfo.parameterStart[ii] = row
                o_retrievalInfo.parameterEnd[ii] = row + mm - 1
                o_retrievalInfo.n_parameters[ii] = mm
                o_retrievalInfo.n_parametersFM[ii] = nn

                o_retrievalInfo.mapTypeList[row:row + mm] = [mapType] * mm
                o_retrievalInfo.mapTypeListFM[rowFM:rowFM + nn] = [mapType] * nn

                o_retrievalInfo.mapType[ii] = mapType

                # AT_LINE 826 Get_Species_Information.pro
                o_retrievalInfo.Constraint[row:row + mm, row:row + mm] = constraintMatrix

                o_retrievalInfo.n_totalParameters = row + mm

                o_retrievalInfo.parameterStartFM[ii] = rowFM

                rowFM = rowFM + nn
                o_retrievalInfo.parameterEndFM[ii] = rowFM - 1

                o_retrievalInfo.n_totalParametersFM = rowFM

                row = row + mm
            # end for ii in range(num_species) at line 146 Get_Species_Information.pro
            # END_MAIN_FOR_LOOP over species.

            # 02/03/2020: Finally found the missing Python code which may be the cause of CLOUDEXT species not matching in the resultsList retrieval.
            # AT_LINE 861 Get_Species_Information.py

            sys_tokens = mpy.table_get_unpacked_entry(i_table_struct, step, "errorAnalysisInterferents")

            # For some reason, the returned value of sys_tokens is a list of list so we just desire a regular list.
            sys_tokens = mpy.flat_list(sys_tokens)

            # added 6/2023 SSK.  Inside constraint_get_species, it has code:  i_constraintStruct.species, so
            # o_errorInitial needs to be an object.  So convert it here.
            if isinstance(o_errorInitial, dict):
                o_errorInitial = mpy.ObjectView(o_errorInitial)


            if sys_tokens[0] != '-' and sys_tokens[0] != '':
                sys_tokens = mpy.order_species(sys_tokens)
                o_retrievalInfo.n_speciesSys = len(sys_tokens)
                o_retrievalInfo.speciesSys[0:o_retrievalInfo.n_speciesSys] = sys_tokens[:]
                o_retrievalInfo.n_totalParametersSys = len(mpy.constraint_get_species(o_errorInitial, sys_tokens))
            else:
                o_retrievalInfo.n_speciesSys = 0
            # end else portion of if (sys_tokens[0] != '-' and sys_tokens[0] != ''):

            # AT_LINE 872 Get_Species_Information.py
            if sys_tokens[0] != '-' and sys_tokens[0] != '':
                myspec = mpy.constraint_get_species(o_errorInitial, sys_tokens)
                for ll in range(0, len(sys_tokens)):
                    my_ind = np.where(np.asarray(myspec) == sys_tokens[ll])[0]
                    # PYTHON_NOTE: If the len of my_ind, that means we don't have any matching species names.
                    if len(my_ind) > 0:
                        o_retrievalInfo.parameterStartSys[ll] = np.amin(my_ind)
                        o_retrievalInfo.parameterEndSys[ll] = np.amax(my_ind)
                        o_retrievalInfo.speciesListSys[my_ind] = o_retrievalInfo.speciesSys[ll]
                    else:
                        o_retrievalInfo.parameterStartSys[ll] = -1
                        o_retrievalInfo.parameterEndSys[ll] = -1
            # end for ll = 0,len(sys_tokens):

        # end else portion of if retrievalType == 'bt' or retrievalType == 'forwardmodel' or retrievalType == 'fullfilter':

        # AT_LINE 862 Get_Species_Information.py
        # get joint constraint for H2O-HDO.  Simplify this code

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
            species_dir = mpy.table_get_pref(i_table_struct, "speciesDirectory")

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
                    mapToParameters = o_retrievalInfo.mapToParameters[loc11FM:loc12FM, loc11:loc12] # Note the spelling of this key 'mapToParameters'
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
                    nh3type = stateInfo.nh3type
                    ch3ohtype = stateInfo.ch3ohtype

                    constraintMatrix, pressurex = mpy.supplier_constraint_matrix_premade(constraint_species,
                                                                        filename,
                                                                        mm,
                                                                        i_nh3type = stateInfo.nh3type, 
                                                                        i_ch3ohtype = stateInfo.ch3ohtype)
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
        # AT_LINE 924 Get_Species_Information.py

        # AT_LINE 926 Get_Species_Information.py

        # Check the constaint vector for sanity.
        if np.all(np.isfinite(o_retrievalInfo.constraintVector) == False):
            raise RuntimeError(f"NaN's in constraint vector!! Constraint vector is: {o_retrievalInfo.constraintVector}. Check species {o_retrievalInfo.speciesList}")

        # Check the constaint matrix for sanity.
        if np.all(np.isfinite(o_retrievalInfo.Constraint)) == False:
            raise RuntimeError(f"NaN's in constraint matrix!! Constraint matrix is: {o_retrievalInfo.Constraint}. Check species {o_retrievalInfo.speciesList}")

        # get to right sizes
        # resave to a structure that is "just right"
        # except for maps, which are different sizes.
        # could put into a super map but leave for now.
        # AT_LINE 950 Get_Species_Information.py
        nnn = o_retrievalInfo.n_totalParametersFM
        mmm = o_retrievalInfo.n_totalParameters

        sss = o_retrievalInfo.n_totalParametersSys
        if sss == 0:
            sss = 1

        nsys = o_retrievalInfo.n_speciesSys
        if nsys == 0:
            nsys = 1

        r = o_retrievalInfo

        if mmm == 0:
            mmm = 1

        if nnn == 0:
            nnn = 1

        # AT_LINE 960 Get_Species_Information.py
        o_retrievalInfo = { 
            # Info by retrieval parameter
            'surfaceType' : r.surfaceType,                    
            'speciesList' : r.speciesList[0:mmm],             
            'pressureList': r.pressureList[0:mmm],            
            'altitudeList': r.altitudeList[0:mmm],            
            'mapTypeList' : r.mapTypeList[0:mmm],             

            'minimumList' : r.minimumList[0:mmm],             
            'maximumList' : r.maximumList[0:mmm],             
            'maximumChangeList' : r.maximumChangeList[0:mmm],             

            'initialGuessList' :  r.initialGuessList[0:mmm],  
            'constraintVector' : r.constraintVector[0:mmm],   
            'trueParameterList': r.trueParameterList[0:mmm],  
            'doUpdateFM'       : r.doUpdateFM[0:nnn],         
            'speciesListFM'    : r.speciesListFM[0:nnn],      

            'pressureListFM'     : r.pressureListFM[0:nnn],      
            'altitudeListFM'     : r.altitudeListFM[0:nnn],      
            'initialGuessListFM' : r.initialGuessListFM[0:nnn],  
            'constraintVectorListFM' : r.constraintVectorListFM[0:nnn],  
            'trueParameterListFM': r.trueParameterListFM[0:nnn], 
            'n_totalParametersFM': r.n_totalParametersFM,        

            'parameterStartFM': r.parameterStartFM[0:r.n_species],  
            'parameterEndFM'  : r.parameterEndFM[0:r.n_species],    
            'mapTypeListFM'   : r.mapTypeListFM[0:nnn],             

            # Info by species
            'n_speciesSys'     : r.n_speciesSys,                 
            'speciesSys'       : r.speciesSys[0:nsys],           
            'parameterStartSys': r.parameterStartSys[0:nsys],    
            'parameterEndSys'  : r.parameterEndSys[0:nsys],      
            'speciesListSys'   : r.speciesListSys[0:sss],        

            'n_totalParametersSys': r.n_totalParametersSys,          
            'n_species'           : r.n_species,                     
            'species'             : r.species[0:r.n_species],        
            'parameterStart'      : r.parameterStart[0:r.n_species], 
            'parameterEnd'        : r.parameterEnd[0:r.n_species],   

            'n_parametersFM' : r.n_parametersFM[0:r.n_species],      
            'n_parameters'   : r.n_parameters[0:r.n_species],        
            'mapType'        : r.mapType[0:r.n_species],             
            'mapToState'     : r.mapToState[0:mmm, 0:nnn],           
            'mapToParameters': r.mapToParameters[0:nnn, 0:mmm],       

            # Constraint & SaTrue, & info for all parameters
            'n_totalParameters': r.n_totalParameters,        
            'Constraint'       : r.Constraint[0:mmm, 0:mmm],  
            'type'             : r.type                     
        }


        # Convert any dictionaries to objects.
        if isinstance(o_retrievalInfo, dict):
            o_retrievalInfo = mpy.ObjectView(o_retrievalInfo)

        if isinstance(o_errorInitial, dict):
            o_errorInitial = mpy.ObjectView(o_errorInitial)

        if isinstance(o_errorCurrent, dict):
            o_errorCurrent = mpy.ObjectView(o_errorCurrent)


        return (o_retrievalInfo, o_errorInitial, o_errorCurrent)
        
    
