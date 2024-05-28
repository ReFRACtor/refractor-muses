from .current_state import CurrentStateDict
from .observation_handle import mpy_radiance_from_observation_list
import refractor.framework as rf
from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
import numpy as np
import datetime
import pytz

logger = logging.getLogger("py-retrieve")

class RetrievalOutput:
    '''Observer of RetrievalStrategy, common behavior for Products files.'''
    def notify_add(self, retrieval_strategy):
        self.retrieval_strategy = retrieval_strategy

    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy_step = retrieval_strategy_step

    @property
    def strategy_table(self):
        return self.retrieval_strategy.strategy_table

    @property
    def step_directory(self):
        return self.strategy_table.step_directory

    @property
    def input_directory(self):
        return self.strategy_table.input_directory

    @property
    def analysis_directory(self):
        return self.strategy_table.analysis_directory

    @property
    def elanor_directory(self):
        return self.strategy_table.elanor_directory
    
    @property
    def windows(self):
        return self.retrieval_strategy.strategy_table.microwindows()

    @property
    def errorCurrent(self):
        return self.retrieval_strategy.error_analysis.error_current
    
    @property
    def special_tag(self):
        if self.retrieval_strategy.retrieval_type != 'default':
            return f"-{self.retrieval_strategy.retrieval_type}"
        return ""

    @property
    def species_tag(self):
        res = self.retrieval_strategy.step_name
        res = res.rstrip(', ')
        if 'EMIS' in res and res.index('EMIS') > 0:
            res = res.replace('EMIS', '')
        if res.endswith(',_OMI'):
            res = res.replace(',_OMI', '_OMI')  #  Change "H2O,O3,_OMI" to "H2O,O3_OMI"
        res = res.rstrip(', ')
        return res

    @property
    def quality_name(self):
        return self.retrieval_strategy.quality_name
    
    @property
    def table_step(self):
        return self.retrieval_strategy.table_step

    @property
    def number_table_step(self):
        return self.retrieval_strategy.number_table_step

    @property
    def results(self):
        return self.retrieval_strategy_step.results

    @property
    def state_info(self):
        return self.retrieval_strategy.state_info

    @property
    def radiance_full(self):
        olist = [self.retrieval_strategy.observation_handle_set.observation(iname, None, None,None)
                 for iname in self.retrieval_strategy.instrument_name_all]
        return mpy_radiance_from_observation_list(olist, full_band=True)

    @property
    def obs_list(self):
        return self.retrieval_strategy_step.cfunc.obs_list
    
    @property
    def radiance_step(self):
        return mpy.ObjectView(self.retrieval_strategy.radiance_step)

    @property
    def retrievalInfo(self):
        return self.retrieval_strategy.retrieval_info
    
    @property
    def instruments(self):
        return self.retrieval_strategy.instruments


class CdfWriteTes:
    '''We want to be able to extend the variables written out, so we have our own
    lightly modified version of cdf_write_tes. This basically just makes some of
    the fixed sized arrays things that can be modified.

    Since we want to do this by run we wrap this in a class, which is little more than
    cdf_write_tes function plus some arrays that can be extended.
    '''
    def __init__(self):
        pass

    def write(self, dataOut, filenameOut, tracer_species="species",
              retrieval_pressures=None,
              write_met=False, version=None, liteVersion=None, runtimeAttributes=None,
              state_element_out=None):
        '''We pass in state_element_out for StateElement not otherwise handled. This
        separates out the species that were in muses-py vs stuff we may have added.
        Perhaps we'll get all the StateElements handled the same way at some point,
        muses-py is really overly complicated. On the other hand, this is just output
        code which tends to get convoluted to create specific output, so perhaps
        this isn't so much an issue.'''
        if runtimeAttributes is None:
            runtimeAttributes = {}
        if(state_element_out is None):
            state_element_out = {}
        dims = {}
        tagnames = list(dataOut.keys())
        
        if 'FILENAME' in dataOut:
            del dataOut['FILENAME']

        if 'NCEP_TEMPERATURESURFACE' in dataOut:
            del dataOut['NCEP_TEMPERATURESURFACE']

        if 'NCEP_TEMPERATURE' in dataOut:
            del dataOut['NCEP_TEMPERATURE']

        # correct naming mistakes
        # Loop through all keys and look for '_TROPOSPHERI'
        for key_name, v in dataOut.items():
            if '_TROPOSPHERI' in key_name:
                new_key = key_name.split('_')[0]
                del dataOut[key_name]
                dataOut[new_key] = v
            elif ('OZONETROPOSPHERI' in key_name or
                  'OZONEUPPER' in key_name or
                  'OZONELOWER' in key_name):
                new_key = key_name[5:]  # Extract from 'UPPER" on from 'OZONE'
                del dataOut[key_name]
                dataOut[new_key] = v
            elif 'OZONEDOFS' in key_name:
                new_key = key_name[5:]  # Extract from 'DOFS' of 'OZONEDOFS'
                del  dataOut[key_name]
                dataOut[new_key] = v

        tracer_species = tracer_species.lstrip().rstrip()

        if 'STEPNAME' in dataOut:
            del dataOut['STEPNAME']

        # TODO Probably bad that this is hardcoded. Should perhaps pass this
        # in instead
        grid_pressure_FM = [
            -999, 
            1.21153e+03, 1.10070e+03, 1.0e+03,
            908.514, 
            825.402, 
            749.893, 
            681.291, 618.966,
            562.342, 510.898, 
            464.160, 421.698,
            383.117, 348.069, 316.227,
            287.298, 261.016, 237.137, 215.444,
            195.735, 177.829, 161.561, 146.779, 133.352, 121.152, 110.069, 100.000,
            90.8518000, 
            82.5406000, 
            74.9896000, 
            68.1295000, 61.8963000,
            56.2339000, 51.0896000, 
            46.4158000, 42.1696000, 
            38.3119000, 34.8071000, 31.6229000, 
            28.7299000, 26.1017000, 23.7136000, 21.5443000, 
            19.5734000, 17.7828000, 16.156, 14.678, 13.3352000, 12.1153000, 11.007,
            10.000, 9.0851400, 8.25402, 6.8129100, 5.10898, 4.64160, 3.16227, 2.6101600, 2.15443, 1.6156000, 1.33352, 1.,
            0.681292, 0.3831180, 0.215443, 0.1
        ]
        if len(dataOut['PRESSURE']) == 20:
            grid_pressure_FM = np.array([1.00000,0.947364,0.894734,0.842105,0.789472,0.736844,0.684210,0.631580,0.578948,0.526316,0.473684, 0.421053, 0.368420, 0.315789, 0.263158,0.210526,     0.157895,     0.105263,    0.0526316,  0.000100000])


        # AT_LINE 161 TOOLS/cdf_write_tes.pro
        if len(dataOut['PRESSURE']) != len(grid_pressure_FM):
            if retrieval_pressures is None:
                raise RuntimeError("You must include retrieval_pressures, which is the grid_pressure")
            grid_pressure = retrieval_pressures
            dims['dim_pressure'] = len(grid_pressure)
            dims['dim_pressure_fm'] = len(grid_pressure_FM)
        else:
            dims['dim_pressure'] = len(grid_pressure_FM)


        grid_rtvmr_levels = []
        grid_rtvmr_map = []
        if 'RTVMR' in dataOut:
            #my_lists.extend(['GRID_RTVMR_MAP', 'GRID_RTVMR_LEVELS'])
            grid_rtvmr_levels = [0, 1]
            dims['dim_rtvmr_levels'] = len(grid_rtvmr_levels)

            grid_rtvmr_map = [0, 1, 2, 3, 4]
            dims['dim_rtvmr_map'] = len(grid_rtvmr_map)

        if 'NIR_AEROD' in dataOut:
            dims['dim_nir_aerosol'] = len(dataOut['NIR_AEROD'])



        # AT_LINE 145 TOOLS/cdf_write_tes.pro
        grid_iters = []
        grid_iterlist = []
        if 'LMRESULTS_RESNORM' in dataOut:
            grid_iters = np.arange(0, len(dataOut['lmresults_resnorm'.upper()]))
            grid_iterlist = np.arange(0, len(dataOut['lmresults_iterlist'.upper()][0, :]))
            grid_iterlist2 = np.arange(0, len(dataOut['lmresults_iterlist'.upper()][:, 0]))

        if len(grid_iters) > 0:
            dims['dim_iters'] = len(grid_iters)
            dims['dim_iterlist'] = len(grid_iterlist)

        # AT_LINE 157 TOOLS/cdf_write_tes.pro


        # AT_LINE 170 TOOLS/cdf_write_tes.pro
        if 'CT_CO2' in dataOut:
            raise RuntimeError("Not implemented yet")

        # AT_LINE 181 TOOLS/cdf_write_tes.pro
        if 'NCEP_TEMPERATURE' in dataOut:
            grid_ncep = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 10]
            dims['dim_ncep'] = len(grid_ncep)
        # end if 'NCEP_TEMPERATURE' in dataOut:

        # AT_LINE 186 TOOLS/cdf_write_tes.pro
        if 'EMISSIVITY' in dataOut:
            GRID_EMISSIVITY = np.asarray([600, 650, 675, 700, 725, 750, 770, 780, 790, 800, 810, 830, 850, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1105, 1110, 1115, 1120, 1125, 1130, 1135, 1140, 1145, 1150, 1155, 1160, 1165, 1170, 1175, 1180, 1185, 1190, 1195, 1200, 1205, 1210, 1215, 1220, 1225, 1230, 1235, 1240, 1245, 1250, 1255, 1260, 1265, 1270, 1275, 1280, 1285, 1290, 1295, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1830, 1850, 1870, 1900, 1930, 1950, 1960, 1980, 2000, 2030, 2050, 2060, 2100, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280, 2300, 2320, 2340, 2360, 2380, 2400, 2460, 2500, 2540, 2560, 2600, 2700, 2800, 2900, 3000, 3100])
            dims['dim_emissivity'] = len(GRID_EMISSIVITY)
        # end if 'EMISSIVITY' in dataOut:

        # AT_LINE 191 TOOLS/cdf_write_tes.pro
        if len(dataOut['CLOUDEFFECTIVEOPTICALDEPTH']) == 25:
            GRID_CLOUD = np.asarray([600, 650, 700, 750, 800, 850, 900, 950, 975, 1000, 1025, 1050, 1075, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1900, 2000, 2100, 2200, 2250])
        else:
            GRID_CLOUD = np.asarray([600, 650, 700, 750, 800, 850, 900, 950, 975, 1000, 1025, 1050, 1075, 1100, 1140, 1200, 1250, 1300, 1350, 1400, 1900, 2000, 2040, 2060, 2080, 2100, 2200, 2250])
        # end if len(dataOut['CLOUDEFFECTIVEOPTICALDEPTH']) == 25:
        dims['dim_cloud'] = len(GRID_CLOUD)

        #### AT_LINE 199 TOOLS/cdf_write_tes.pro
        if 'MAP' in list(dataOut.keys()):
            if len(dataOut['pressure'.upper()]) != len(dataOut['map'.upper()][:, 0]):
                # for composite files (e.g. HDO-H2O) also stack of pressures
                grid_pressure_composite = np.concatenate((retrieval_pressures, retrieval_pressures), axis=0)
                dims['dim_pressure_composite'] = len(grid_pressure_composite)

        # AT_LINE 208 TOOLS/cdf_write_tes.pro
        # add and define RTVMR field names
        if 'RTVMR' in dataOut:
            grid_rtvmr_levels = [0, 1]
            grid_rtvmr_map = [0, 1, 2, 3, 4]
        # end if 'RTVMR' in dataOut:

        if 'LMRESULTS_RESNORM' in dataOut:
            # INDGEN(N_ELEMENTS(data[0].lmresults_resnorm))
            grid_iters = np.arange(0, len(dataOut['lmresults_resnorm'.upper()]))
            # INDGEN(N_ELEMENTS(data[0].lmresults_iterlist[0,*]))
            grid_iterlist = np.arange(0, len(dataOut['lmresults_iterlist'.upper()][0, :]))
        # end if 'LMRESULTS_RESNORM' in dataOut:

        #my_lists.extend(['GRID_COLUMN', 'GRID_FILTER'])
        GRID_COLUMN = [0, 1, 2, 3, 4]
        dims['dim_column'] = len(GRID_COLUMN)

        GRID_FILTER = np.arange(len(dataOut['FILTER_INDEX']))
        dims['dim_filter'] = len(GRID_FILTER)


        dataOut['GRID_PRESSURE_FM'] = grid_pressure_FM

        # form new time called YYYYMMDD.fractionofday
        ymd = mpy.tai(dataOut['TIME'])
        yyear = ymd['year']
        ymonth = ymd['month']
        yday = ymd['day']
        yyyymmdd = np.float64(yyear*10000.0 + ymonth*100.0 + yday*1.0)

        dataOut['YYYYMMDD'] = yyyymmdd

        ut_hour = ymd['digitalHour']

        dataOut['UT_HOUR'] = ut_hour

        # cant handle things on the freq grid (expects all arrays to be on presure grid)
        # cant handle string variable type
        notEqualList = [
            'CALFREQ',
            'CALSCALE',
            'CALOFFSET',
            'STEPNAMES',
            'UTCTIME',
            'VERTICALCOORDINATE',
            'SCAN_RESOLUTION',
            'CLOUDFREQUENCY',
            'NCEP_PRESSURE'
        ]

        if 'OMI_NRADWAV' in dataOut:
            grid_2 = [0, 1]
            dims['size2'] = 2

            grid_3 = [0, 1, 2]
            dims['size3'] = 3

        (dataNew, dims) = mpy.cdf_var_add_strings(dataOut, dims)
        dataOut = dataNew 

        #===============================
        # netcdf file global variable
        #===============================
        
        global_attributes = {
            'Name': 'TES Level 2 Product',
            'Version': version,
            'Lite Version': liteVersion,
            'Copyright': 'Jet Propulsion Laboratory, California Institute for Technology, Pasadena, CA',
            'Content': 'Global observations',
            'Source': 'TES HDF-EOS files available at http://eosweb.larc.nasa.gov/PRODOCS/tes/products/level2.html',
            'Contact': 'Susan Kulawik <susan.kulawik@nasa.gov> on TES NetCDF data'
        }
        # It is convenient in testing to have a fixed creation_date, just so we can
        # compare files created at different times with h5diff and have them compare as
        # identical
        if("MUSES_FAKE_CREATION_DATE" in os.environ):
            history_entry = 'created '+ os.environ["MUSES_FAKE_CREATION_DATE"]
        else:
            history_entry = 'created '+ datetime.datetime.now(tz=pytz.utc).strftime("%Y%m%dT%H%M%SZ")

        # Create .met file and retrieve the output
        if write_met:
            raise RuntimeError("Function Met_Write() has not been implemented.  Must exit for now.")
        else:
            global_attr = {
            }

        global_attr['fileversion'] = 2
        global_attr['history'] = history_entry

        # PYTHON_NOTE: Below are an exhaustive definition of attributes of every possible variable.  Because this is Python,
        # we make sure the part before the _attr is uppercase because later on, we will use it to access the attributes defined
        # here.
        SPECIES_attr = {
            'Longname':tracer_species+' volume mixing ratio',
            'Units':'volume mixing ratio relative to dry air',
            'FillValue':-999.00,
            'MisingValue':-999.00
        }

        mpy.cdf_var_attributes['SPECIES_attr'] = SPECIES_attr        

        SPECIES_FM_attr = {
            'Longname':tracer_species+' volume mixing ratio',
            'Units':'volume mixing ratio relative to dry air on fm grid',
            'FillValue':-999.00,
            'MisingValue':-999.00
        }

        mpy.cdf_var_attributes['SPECIES_FM_attr'] = SPECIES_FM_attr        

        #===============================
        # Link attributes to their respective variables
        #===============================
        # use ordering in the array names
        # if that variable exists, then add its attribute to the list

        # AT_LINE 758 TOOLS/cdf_write_tes.pro
        dict_of_variables_and_their_attributes = {}

        groupvarnames = mpy.cdf_var_names()

        utilList = mpy.UtilList()
        names = utilList.GetColumnFromList(groupvarnames, 1)
        names = utilList.GetUniqueValues(names)
        names = [x.upper() for x in names]

        for ii in range(0, len(names)):
            tag_name = names[ii]
            if tag_name in dataOut:
                attr_name = names[ii] + '_attr'  
                if attr_name in mpy.cdf_var_attributes:
                    dict_of_variables_and_their_attributes[tag_name] = mpy.cdf_var_attributes[attr_name]
                else:    
                    logger.info('Using generic attribute spec for '+attr_name+'.') 
                    # Create a generic_attr depend on name of the variable.
                    # A string variable has 'GRID_STRING' or 'Grid_String' in it.
                    if 'STRING' in attr_name or 'String' in attr_name:  
                        generic_attr = {'Longname': 'string', 'Units': '', 'FillValue': -999, 'MissingValue': -999} 
                    else:
                        generic_attr = {'Longname': '', 'Units': '', 'FillValue': -999, 'MissingValue': -999}

                    dict_of_variables_and_their_attributes[tag_name] = generic_attr

                # If there are extra attributes whose values are defined at runtime, add them in now
                dict_of_variables_and_their_attributes[tag_name].update(runtimeAttributes.get(tag_name, dict()))
        # for ii in range(0,len(names)):

        #===============================
        # Link data to their respective variables
        #===============================
        # use tagnames to put data into newdata
        dataNew_keys = list(dataNew.keys())
        newdata = {}
        for ii in range(0, len(names)):
            tag_name = names[ii]
            if tag_name in dataOut:
                if tag_name not in dataNew_keys:
                    raise RuntimeError("tag_name not in dataNew_keys")
                newdata[tag_name] = dataNew[tag_name]

        structIn = newdata

        # To ensure we have matching values in both list: structIn and dict_of_variables_and_their_attributes, we extract
        # from dict_of_variables_and_their_attributes based on key.
        structUnits = []
        generic_attr = {'Longname':'', 'Units':'', 'Fillvalue':-999.0, 'Missingvalue':-999.0}
        structKeys = list(structIn.keys())
        for ii in range(len(structKeys)):
            tag_name = structKeys[ii]
            if tag_name in dict_of_variables_and_their_attributes:
                attributes = dict_of_variables_and_their_attributes[tag_name]
                structUnits.append(attributes)
            else:
                structUnits.append(deepcopy(generic_attr))

        # PYTHON_NOTE: We create a dictionary of each possible variables in their exact cases for when they are written to NetCDF file.
        # The variable name will have an uppercase and lowercase mixed.
        # If a name is not found in exact_cased_variable_names, it will be written as is.
        exact_cased_variable_names = mpy.cdf_var_map()

        for ii in range(len(structKeys)):
            tag_name = structKeys[ii]
            variable_data = structIn[tag_name]
            if isinstance(variable_data, list):
                structIn[tag_name] = np.asarray(variable_data)

        # StateElements we haven't already gotten. These are
        # StateElement that weren't originally in muses-py.
        
        for selem in state_element_out:
            structIn[selem.name] = selem.value
            # For simplicity, value is always a numpy array. If it is size 1,
            # we want to pull this out so the data is written in netcdf as
            # a scalar rather than a array of size 1
            if(len(selem.value.shape) == 1 and selem.value.shape[0] == 1):
                structIn[selem.name] = selem.value[0]
            structUnits.append(selem.net_cdf_struct_units())
            exact_cased_variable_names[selem.name] = selem.net_cdf_variable_name()
            groupvarnames.append([selem.net_cdf_group_name(),
                                  selem.net_cdf_variable_name()])

        #===============================
        # write data (Use the service of cdf_write() function.)
        #===============================

        # Note that for this call to cdf_write:
        #   1.  We pass in dims which is a list containing all dimension names and their sizes.
        #   2.  We also pass in a dictionary containing the exact variable names.
        mpy.cdf_write(
            structIn, 
            filenameOut, 
            structUnits, 
            dims=dims, 
            lowercase=False,
            exact_cased_variable_names=exact_cased_variable_names, 
            groupvarnames=groupvarnames,
            global_attr=global_attr
        )

    def write_lite(self, stepNumber, filenameIn, qualityFilename, instrument,
                   liteDirectory, data1In, data2=None, species_name='', step=0,
                   times_species_retrieved=0, state_element_out=None):
        '''This is a lightly edited version of make_lite_casper_script_retrieval,
        mainly we want this to call our cdf_write_tes so we can add new species in.

        We pass in state_element_out for StateElement not otherwise handled. This
        separates out the species that were in muses-py vs stuff we may have added.
        Perhaps we'll get all the StateElements handled the same way at some point,
        muses-py is really overly complicated. On the other hand, this is just output
        code which tends to get convoluted to create specific output, so perhaps
        this isn't so much an issue.
        '''
        data1 = copy.deepcopy(data1In)

        version = 'v006'
        version = 'CASPER'

        versionLite = 'v08'
        pressuresMax = 1040

        starttai = mpy.tai({'year': 2003, 'month': 1, 'day': 1, 'hour': 0, 'minute': 0,
                            'second': 0}, True)
        endtai = mpy.tai({'year': 2103, 'month': 1, 'day': 1, 'hour': 0, 'minute': 0,
                            'second': 0}, True)

        if not filenameIn.endswith(".nc"):
            filename = os.path.basename(filenameIn)
            nc_pos = filename.index('.')

            optional_separator = ''
            if not (os.path.dirname(filenameIn)).endswith(os.path.sep):
                optional_separator = os.path.sep

            filenameOut = os.path.dirname(filenameIn) + optional_separator + 'Lite_' + filename[0:nc_pos+1] + 'nc'
        else:
            filenameOut = filenameIn

        runs = filenameIn
        directory = None
        useData = True
        dataAnc = copy.deepcopy(data1)
        (data, data2, pressuresMax) = mpy.make_one_lite(
            species_name, runs, starttai, endtai, instrument,
            directory, pressuresMax,
            qualityFilename, liteDirectory,
            version, versionLite,
            useData, data1, data2, dataAnc,
            step
        )
        tracer = species_name
        retrieval_pressures = pressuresMax
        version = version
        liteVersion = versionLite
        write_met = False
        self.write(data, filenameOut, tracer, retrieval_pressures, write_met, version, liteVersion, state_element_out=state_element_out)
        return data2
        
__all__ = ["RetrievalOutput", "CdfWriteTes"] 
