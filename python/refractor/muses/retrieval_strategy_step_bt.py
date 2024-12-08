import refractor.muses.muses_py as mpy
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_strategy_step import (RetrievalStrategyStep,
                                      RetrievalStrategyStepSet)
import refractor.framework as rf
import numpy as np
import os
import copy
from loguru import logger

class RetrievalStrategyStepBT(RetrievalStrategyStep):
    '''Brightness Temperature strategy step. This handles steps with the retrieval
    type "BT". This then selects one of the following BT_IG_Refine steps to
    execute.'''
    def __init__(self):
        super().__init__()
        self.notify_update_target(None)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.BTstruct = [{'diff':0.0, 'obs':0.0, 'fit':0.0} for i in range(100)]
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "bt":
            return (False,  None)
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        jacobian_speciesNames = ['H2O']
        jacobian_specieslist = ['H2O']
        jacobianOut = None
        mytiming = None
        logger.info("Running run_forward_model ...")
        self.cfunc = rs.create_cost_function(
            fix_apriori_size=True, jacobian_speciesIn=jacobian_speciesNames)
        radiance_fm = self.cfunc.max_a_posteriori.model
        freq_fm = np.concatenate([fm.spectral_domain_all().data
                                  for fm in self.cfunc.max_a_posteriori.forward_model])
        # Put into structure expected by modify_from_bt
        radiance_res = {"radiance" : radiance_fm,
                        "frequency" : freq_fm }
        self.modify_from_bt(
            rs._strategy_executor.strategy._stable,
            rs.step_number,
            radiance_res,
            rs._state_info,
            self.BTstruct)
        logger.info(f"Step: {rs.step_number},  Total Steps (after modify_from_bt): {rs.number_retrieval_step}")
        rs.state_info.next_state_dict = copy.deepcopy(rs.state_info.state_info_dict["current"])
        return (True, None)

    def modify_from_bt(self, strategy_table, step, radianceModel, state_info,
                       BTStruct):
        stateInfo = state_info.state_info_dict
    
        # If the dimension of radianceModel['radiance'] is 2, we check to see if we can flatten it.
        if (len(radianceModel['radiance'].shape) == 2) and (radianceModel['radiance'].shape[0] == 1): # As in the case of (1,14)
            radianceModel['radiance'] = radianceModel['radiance'].flatten()
    
        table_struct = strategy_table.strategy_table_obj

        radianceStep = mpy_radiance_from_observation_list(
            self.cfunc.obs_list, include_bad_sample=False)
        
        if isinstance(radianceModel, dict):
            radianceModel = mpy.ObjectView(radianceModel)
    
        filename = mpy.table_get_step_filename(table_struct, "observation")
    
        # issue with negative radiances, so take mean
        frequency = radianceStep["frequency"]
        radianceBtObs = mpy.bt(np.mean(frequency), np.mean(radianceStep["radiance"]))
        radianceBtFit = mpy.bt(np.mean(frequency), np.mean(radianceModel.radiance))
    
        BTStruct[step]['diff'] = radianceBtFit[0] - radianceBtObs[0]
        BTStruct[step]['obs'] = radianceBtObs[0]
        BTStruct[step]['fit'] = radianceBtFit[0]
    
        bt_diff = BTStruct[step]['diff']
    
        # If next step is NOT BT, evaluate what to do with "cloud" 
        doo = False
        if step == table_struct.numRows:
            # only BT characterization for limb not nadir
            pass
        else:
            nextStepRetrievalType = mpy.table_get_entry(table_struct, step+1, "retrievalType")
            if nextStepRetrievalType != 'BT': 
                doo = True

        if not doo:
            return
        
        filename = mpy.table_get_pref(table_struct, "CloudParameterFilename")
        (_, fileID) = mpy.read_all_tes_cache(filename, 'asc')

        # The values returned from tes_file_get_column() are list of strings, we need to convert each to float and then to an array.
        BTLow = [float(value) for ii, value in enumerate(mpy.tes_file_get_column(fileID, 0))]
        BTLow = np.asarray(BTLow)
        
        BTHigh = [float(value) for ii, value in enumerate(mpy.tes_file_get_column(fileID, 1))]
        BTHigh = np.asarray(BTHigh)
        
        tsurIG = [float(value) for ii, value in enumerate(mpy.tes_file_get_column(fileID, "TSUR_IG"))]
        tsurIG = np.asarray(tsurIG)
        
        cloudIG = [float(value) for ii, value in enumerate(mpy.tes_file_get_column(fileID, "CLOUDEXT_IG"))]
        cloudIG = np.asarray(cloudIG)

        cloudPrefs = fileID['preferences']

        x = mpy.tes_file_get_preference(fileID, 'note')
        if x is None or x == '':
            raise RuntimeError(f"Please update CloudParameterFilename: {filename} To be for fit-obs not obs-fit")

        row = np.where((bt_diff >= BTLow) & (bt_diff <= BTHigh))[0]
        if row.size == 0:
            raise RuntimeError(f"No entry in file, {filename} For BT difference of {bt_diff}")

        # AT_LINE 97 modify_from_bt.pro
        available = ""
        col = mpy.tes_file_get_column(fileID, "SPECIES_IGR")
        col = np.asarray(col)
        expected = col[row]
    
        # AT_LINE 101 modify_from_bt.pro

        # for IGR and TSUR modification for TSUR, must be daytime land
        if (stateInfo['current']['tsa']['dayFlag'] == 0 or \
            stateInfo['surfaceType'].upper() == 'OCEAN' or \
            stateInfo['surfaceType'].upper() == 'FRESH') and ('TSUR' in expected):
            logger.info('Must be land, daytime for TSUR IGR')
            expected = '-'
            tsurIG[row] = 0


        # AT_LINE 112 modify_from_bt.pro
        found = False
        istep = step + 1 # Start with next step.
        my_type = mpy.table_get_entry(table_struct, istep, "retrievalType")

        while (istep < table_struct.numRows+1 and
               my_type.lower() == 'bt_ig_refine'):
            my_list = mpy.table_get_entry(table_struct, istep, "retrievalElements")
            if my_list != expected:
                available = available + my_list + "   " 

                mpy.table_delete_row(table_struct, istep)

                istep = istep - 1
            else:
                found = True

            istep = istep + 1
            if istep < table_struct.numRows + 1:
                my_type = mpy.table_get_entry(table_struct, istep, "retrievalType")
        # end while (istep < table.numRows+1 and my_type.lower() == 'bt_ig_refine'):

        if not found and expected != '-':
            raise RuntimeError(f'''Specified IG refinement not found (MUST be retrievalType BT_IG_Refine AND species listed in correct order).
   Expected retrieved species: {expected}
   Available from table:       {available}''')

        output_directory = table_struct.outputDirectory
        output_filename = output_directory + os.path.sep + 'Table-final.asc'

        # AT_LINE 141 modify_from_bt.pro
        o_write_status = mpy.table_write(table_struct, output_filename)
        (o_read_status, table_struct) = mpy.table_read(output_filename)

        if cloudIG[row] > 0:
            stateInfo['initial']['cloudEffExt'][:] = cloudIG[row]
            stateInfo['current']['cloudEffExt'][:] = cloudIG[row]
            stateInfo['constraint']['cloudEffExt'][:] = cloudIG[row]

        if tsurIG[row] != 0:
            # use difference in observed - fit to change TSUR.  Note, we
            # assume weak clouds. 
            stateInfo['initial']['TSUR'] = stateInfo['initial']['TSUR'] + BTStruct[step]['obs'] - BTStruct[step]['fit']
            stateInfo['current']['TSUR'] = stateInfo['initial']['TSUR'] 
            stateInfo['constraint']['TSUR'] = stateInfo['initial']['TSUR'] 
    

        # NOTE: This will assign tableStrategy.__dict__ to table_struct internally
        mpy.table_set_step(table_struct, step)

RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepBT())
    
__all__ = ["RetrievalStrategyStepBT",]
    
