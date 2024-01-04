import numpy as np

from refractor.framework import refractor_config

from .retrieval_run_config import retrieval_run_config

@refractor_config
def simulation_config(**kwargs):
    """Simulate radiances using the retrieval initial guess set up. Used for self consistency retrieval tests."""

    config_def = retrieval_run_config(**kwargs)

    # Disable bad sample mask to conver full window range
    del config_def['spec_win']['bad_sample_mask']

    # Set windows to full range
    config_def['spec_win']['micro_windows'] = config_def['spec_win']['full_ranges']

    return config_def
