from refractor.muses import (MusesAirsObservationNew, MusesRunDir)
from test_support import *

def test_muses_airs_observation(osp_dir):
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    obs = MusesAirsObservationNew(fname, xtrack, atrack, channel_list, osp_dir=osp_dir)
    
