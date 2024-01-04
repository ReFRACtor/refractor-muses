import os
import numpy as np

from refractor.framework import refractor_config, read_shelve

from .retrieval_base_config import retrieval_base_config_definition


@refractor_config
def simulation_example_config(**kwargs):
    """Simulate radiances using the retrieval initial guess set up. Used for self consistency retrieval tests."""

    # test data are within omi dir in refractor_test_data repo at same level as omi
    if("REFRACTOR_TEST_DATA" in os.environ):
        test_in_dir = f"{os.environ['REFRACTOR_TEST_DATA']}/omi/in/"
    else:
        test_in_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../../../../../refractor_test_data/omi/in")

    l1b_filename = os.path.join(test_in_dir, "OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4")
    irr_filename = os.path.join(test_in_dir, "OMI-Aura_L1-OML1BIRR_2016m0414t0337-o62486_v003-2016m0621t174011.he4")
    atmosphere_filename = os.path.join(test_in_dir, "atmosphere.bin.gz")

    along_track_index = 786
    across_track_indexes = [26, 52]  # UV1, UV2

    config_def = retrieval_base_config_definition(l1b_filename, along_track_index, across_track_indexes, irradiance_file=irr_filename, **kwargs)

    # Disable bad sample mask to conver full window range
    del config_def['spec_win']['bad_sample_mask']

    # Set windows to full range
    config_def['spec_win']['micro_windows'] = config_def['spec_win']['full_ranges']

    config_def['atmosphere'] = read_shelve(atmosphere_filename)

    # Remove the need for retrieval values
    config_def['order'].remove('retrieval')
    del config_def['retrieval']

    return config_def
