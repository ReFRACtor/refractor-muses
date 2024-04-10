from test_support import *
from refractor.muses import CurrentStateDict, CurrentStateUip
import numpy as np

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
    cs.retrieval_element = ["TROPOMISOLARSHIFTBAND3", "TROPOMIRADSQUEEZEBAND3"]
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
                    
