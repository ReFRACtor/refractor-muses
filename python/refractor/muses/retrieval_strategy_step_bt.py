import refractor.muses.muses_py as mpy
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_strategy_step import (RetrievalStrategyStep,
                                      RetrievalStrategyStepSet)
from .tes_file import TesFile
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
        self.modify_from_bt(
            rs.retrieval_config,
            rs._strategy_executor.strategy._stable,
            rs.step_number,
            rs._state_info,
            self.BTstruct)
        logger.info(f"Step: {rs.step_number},  Total Steps (after modify_from_bt): {rs.number_retrieval_step}")
        rs.state_info.next_state_dict = copy.deepcopy(rs.state_info.state_info_dict["current"])
        return (True, None)

    def modify_from_bt(self, retrieval_config, strategy_table, step, state_info,
                       BTStruct):
        stateInfo = state_info.state_info_dict
    
        table_struct = strategy_table.strategy_table_obj

        # Note from py-retrieve: issue with negative radiances, so take mean
        #
        # I'm not actually sure that is true, we filter out bad samples. But
        # regardless, use the mean like py-retrieve does.
        frequency = np.concatenate([fm.spectral_domain_all().data
                           for fm in self.cfunc.max_a_posteriori.forward_model])
        radiance_bt_obs = mpy.bt(np.mean(frequency),
                                 np.mean(self.cfunc.max_a_posteriori.measurement))
        radiance_bt_fit = mpy.bt(np.mean(frequency),
                                 np.mean(self.cfunc.max_a_posteriori.model))
    
        BTStruct[step]['diff'] = radiance_bt_fit[0] - radiance_bt_obs[0]
        BTStruct[step]['obs'] = radiance_bt_obs[0]
        BTStruct[step]['fit'] = radiance_bt_fit[0]
    
        bt_diff = BTStruct[step]['diff']
    
        # If next step is NOT BT, evaluate what to do with "cloud". Otherwise,
        # we are done.
        if (step == table_struct.numRows or
            mpy.table_get_entry(table_struct, step+1, "retrievalType") == 'BT'):
            return

        cfile = TesFile(retrieval_config["CloudParameterFilename"])
        BTLow = np.array(cfile.table["BT_low"])
        BTHigh = np.array(cfile.table["BT_high"])
        tsurIG = np.array(cfile.table["TSUR_IG"])
        cloudIG = np.array(cfile.table["CLOUDEXT_IG"])
        row = np.where((bt_diff >= BTLow) & (bt_diff <= BTHigh))[0]
        if row.size == 0:
            raise RuntimeError(f"No entry in file, {filename} For BT difference of {bt_diff}")

        available = ""
        col = np.array(cfile.table["SPECIES_IGR"])
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
    
