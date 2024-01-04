import os
import re


from refractor import framework as rf
from refractor.framework import creator, refractor_config

from .retrieval_run_config import retrieval_run_config

@refractor_config
def run_config(**kwargs):

    config_def = retrieval_run_config(**kwargs)

    config_def['radiative_transfer'] = {
        'creator': creator.rt.PCARt,
        'primary_absorber': "O3",
        'bin_method': rf.PCABinning.UVVSWIR_V4,
        'num_bins': 11,
        'num_eofs': 4,
        'num_streams': 8,
        'num_mom': 16,
        'use_thermal_emission': False,
        'use_solar_sources': True,
    }

    return config_def
