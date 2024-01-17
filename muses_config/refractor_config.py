# These are the arguments passed to cli function that loads this code.
from py_retrieve.cli import params
from refractor.muses import RetrievalStrategy
from refractor.tropomi import TropomiInstrumentHandle
from refractor.omi import OmiInstrumentHandle

# Configuration to use our own TROPOMI and OMI forward models.

rs = RetrievalStrategy(None)
rs.instrument_handle_set.add_handle(TropomiInstrumentHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
rs.instrument_handle_set.add_handle(OmiInstrumentHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)

# Replace top level script_retrieval_ms with RetrievalStrategy.
rs.register_with_muses_py()
