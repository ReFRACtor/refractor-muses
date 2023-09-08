# These are the arguments passed to cli function that loads this code.
from py_retrieve.cli import params
from refractor.muses import RefractorMusesIntegration
from refractor.tropomi import TropomiInstrumentHandle
from refractor.omi import OmiInstrumentHandle

# Configuration to use our own TROPOMI and OMI forward models.

r = RefractorMusesIntegration()
r.instrument_handle_set.add_handle(TropomiInstrumentHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
r.instrument_handle_set.add_handle(OmiInstrumentHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)

# Replace run_retrieval and run_forward_model with RefractorMusesIntegration
r.register_with_muses_py()
