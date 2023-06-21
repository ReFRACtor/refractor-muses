from test_support import *
from refractor.muses import ConstantDict
import numpy as np

def test_constant_dict():
    underlying_dict = { 'k1' : 'v1', 'k2' : { 'k2_k1' : 'v2'}, 'k3' : np.array([1,2,3]) }
    cdict = ConstantDict(underlying_dict)
    # Check that we can't change things.
    with pytest.raises(ValueError):
        cdict['k1'] = 5
    with pytest.raises(ValueError):
        cdict['k2']['k2_k1']=5
    with pytest.raises(ValueError):
        cdict['k3'][2] = 5
    # Mark as writable, we should be able to change things now
    assert cdict['k1'] != 5
    assert cdict['k2']['k2_k1'] != 5
    assert cdict['k3'][2] != 5
    assert underlying_dict['k1'] != 5
    assert underlying_dict['k2']['k2_k1'] != 5
    assert underlying_dict['k3'][2] != 5
    cdict.writable = True
    cdict['k1'] = 5
    cdict['k2']['k2_k1'] = 5
    cdict['k3'][2] = 5
    assert cdict['k1'] == 5
    assert cdict['k2']['k2_k1'] == 5
    assert cdict['k3'][2] == 5
    assert underlying_dict['k1'] == 5
    assert underlying_dict['k2']['k2_k1'] == 5
    assert underlying_dict['k3'][2] == 5
