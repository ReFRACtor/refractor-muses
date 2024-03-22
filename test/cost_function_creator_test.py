from test_support import *
from refractor.muses import CostFunctionCreator

@require_muses_py
def test_fm_obs_creator_tropomi(joint_tropomi_uip_step_12, joint_tropomi_obs_step_12):
    rf_uip = joint_tropomi_uip_step_12
    creator = CostFunctionCreator()
    
    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = creator.cost_function_from_uip(rf_uip, joint_tropomi_obs_step_12, None)

@require_muses_py
def test_fm_obs_creator_omi(joint_omi_uip_step_8, joint_omi_obs_step_8):
    rf_uip = joint_omi_uip_step_8
    creator = CostFunctionCreator()
    
    # We test using this else where. Here, just make sure we can
    # call the creation function.
    cf = creator.cost_function_from_uip(rf_uip, joint_omi_obs_step_8, None)
    
    
