import numpy as np
import numpy.testing as npt
from test_support import *

# Think this will go away, we'll leave in place for now. Need to merge in
# Josh's stuff
@require_muses_py
def test_forward_model(tropomi_uip_band7_step_1, clean_up_replacement_function):
    print("Have a environment for starting to put together a NIR forward model")
