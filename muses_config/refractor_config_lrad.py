# These are the arguments passed to cli function that loads this code.
from refractor.muses import RetrievalStrategy, RetrievalStrategyCaptureObserver
from refractor.tropomi import TropomiForwardModelHandle
from refractor.omi import OmiForwardModelHandle
import refractor.framework as rf

# Configuration to use our own TROPOMI and OMI forward models.

# Turn on logging for refractor. This is independent of python logger, although we
# could probably integrate this together if it is important
rf.Logger.set_implementation(rf.FpLogger())

rs = RetrievalStrategy(None)
rs.forward_model_handle_set.add_handle(TropomiForwardModelHandle(use_pca=False,
                                       use_lrad=True, lrad_second_order=True),
                                       priority_order=100)
rs.forward_model_handle_set.add_handle(OmiForwardModelHandle(use_pca=True,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
if False:
    # If desired, capture each step so we can rerun this for debugging
    rscap = RetrievalStrategyCaptureObserver("retrieval_step",
                                             "starting run_step")
    rs.add_observer(rscap)

# Replace top level script_retrieval_ms with RetrievalStrategy.
rs.register_with_muses_py()
