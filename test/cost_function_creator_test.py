from test_support import *
from refractor.muses import CostFunctionCreator

@require_muses_py
def test_fm_obs_creator_tropomi(joint_tropomi_uip_step_12):
    rf_uip = joint_tropomi_uip_step_12
    creator = CostFunctionCreator()
    
    # We test using this else where. Here, just make sure we can
    # call the creation function.
    (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint, basis_matrix) = creator.fm_and_fake_obs(rf_uip)

@require_muses_py
def test_fm_obs_creator_omi(joint_omi_uip_step_8):
    rf_uip = joint_omi_uip_step_8
    creator = CostFunctionCreator()
    
    # We test using this else where. Here, just make sure we can
    # call the creation function.
    (fm_list, obs_list, sv, sv_apriori, sv_sqrt_constraint, basis_matrix) = creator.fm_and_fake_obs(rf_uip)
    
    
