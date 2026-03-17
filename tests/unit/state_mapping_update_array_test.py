from __future__ import annotations
from refractor.muses import (
    StateMappingUpdateArray
)
import refractor.framework as rf  # type: ignore
import numpy as np
import numpy.testing as npt
import pytest

# We don't know exactly what we want here. Set test aside until we can come back
# to this
@pytest.mark.skip
def test_state_mapping_update_array():
    smap = StateMappingUpdateArray(np.array([True, False, False, True]))
    assert smap.name == "update array"
    full_state = rf.ArrayAd_double_1(4, 2)
    full_state[0] = rf.AutoDerivativeDouble(1, 0, 2)
    full_state[1] = rf.AutoDerivativeDouble(2)
    full_state[2] = rf.AutoDerivativeDouble(3)
    full_state[3] = rf.AutoDerivativeDouble(4, 1, 2)
    npt.assert_allclose(smap.retrieval_state(full_state).value, [1,4])
    npt.assert_allclose(smap.retrieval_state(full_state).jacobian, [[1,0],[0,1]])
    update_coeff = rf.ArrayAd_double_1(2, 2)
    update_coeff[0] = rf.AutoDerivativeDouble(11, 0, 2)
    update_coeff[1] = rf.AutoDerivativeDouble(14, 1, 2)
    npt.assert_allclose(smap.mapped_state(update_coeff).value, [11,2,3,14])
    npt.assert_allclose(smap.mapped_state(update_coeff).jacobian, [[1,0],[0,0],[0,0],[0,1]])
    npt.assert_allclose(smap.retrieval_state(smap.mapped_state(update_coeff)).value, [11,14])
    npt.assert_allclose(smap.retrieval_state(smap.mapped_state(update_coeff)).jacobian, [[1,0],[0,1]])
    
          
    

