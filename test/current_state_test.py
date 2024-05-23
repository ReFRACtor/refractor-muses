from test_support import *
from refractor.muses import (CurrentStateDict, CurrentStateUip, 
                             RetrievalStrategyStepRetrieve,
                             RetrievalStrategy, MusesRunDir,
                             CurrentStateStateInfo)
import numpy as np

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "starting retrieval steps"):
            raise StopIteration()
        
def test_current_state_dict():
    d = {"TROPOMISOLARSHIFTBAND3" : 1.0,
         "TROPOMIRADIANCESHIFTBAND3" : 2.0,
         "TROPOMIRADSQUEEZEBAND3" : 3.0
         }
    cs = CurrentStateDict(d, ["TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3"])
    coeff, mp = cs.object_state(["TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3",
                                 "TROPOMIRADSQUEEZEBAND3"])
    npt.assert_allclose(coeff, [1.0,2.0,3.0])
    npt.assert_allclose(mp.retrieval_indexes, [0,1])
    cs = CurrentStateDict(d, ["TROPOMISOLARSHIFTBAND3", "TROPOMIRADSQUEEZEBAND3"])
    coeff, mp = cs.object_state(["TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3",
                                 "TROPOMIRADSQUEEZEBAND3"])
    npt.assert_allclose(coeff, [1.0,2.0,3.0])
    npt.assert_allclose(mp.retrieval_indexes, [0,2])

def test_current_state_uip(joint_tropomi_uip_step_12):
    rf_uip = joint_tropomi_uip_step_12
    cs = CurrentStateUip(rf_uip)
    print(cs.fm_sv_loc)
    print(cs.fm_state_vector_size)
    coeff, mp = cs.object_state(["TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3",
                                 "TROPOMIRADSQUEEZEBAND3"])
    print(coeff)
    print(mp)
    print(mp.retrieval_indexes)
                    
def test_current_state_state_info(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    # TODO - We should have a constructor for StateInfo. Don't currently,
    # so we just run RetrievalStrategy to the beginning and stop
    try:
        with all_output_disabled():
            #r = MusesRunDir(joint_omi_test_in_dir,
            r = MusesRunDir(joint_tropomi_test_in_dir,
                            osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    # Sets up retrieval_info
    for i in range(rs.number_table_step):
        rs.table_step = i
        with rs.chdir_run_dir():
            rs.retrievalInfo.stepNumber = rs.table_step
            rs.retrievalInfo.stepName = rs.step_name
            rs.get_initial_guess()
            rs.create_windows(all_step=False)

            # Create UIP one, so we can compare
            rstep = RetrievalStrategyStepRetrieve()
            rf_uip = rstep.uip_func(rs, do_systematic=False, jacobian_speciesIn=None)
            csuip = CurrentStateUip(rf_uip)
    
        cs = CurrentStateStateInfo(rs.state_info, rs.retrievalInfo)
        assert cs.fm_state_vector_size == csuip.fm_state_vector_size
        assert cs.fm_sv_loc == csuip.fm_sv_loc
    # TODO
    # We need to add in the full_state_value testing, but we don't have that in place
    # yet for CurrentStateStateInfo
    
    
