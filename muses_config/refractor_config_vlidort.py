# These are the arguments passed to cli function that loads this code.
from refractor.muses import RetrievalStrategy, RetrievalStrategyCaptureObserver
import refractor.framework as rf

# Configuration to use our wrapper around the py-retrieve vlidort code. Should be
# close to running py-retrieve without refractor, but tests our RetrievalStrategy
# class

# Turn on logging for refractor. This is independent of python logger, although we
# could probably integrate this together if it is important
rf.Logger.set_implementation(rf.FpLogger())

# Note that the VLIDORT code depends on MusesCrisObservation writing a pickle
# file that it then reads elsewhere. This is a bad design - we really are using
# the file as a hidden variable. But to support the old py-retrieve code, we need
# to do this.
rs = RetrievalStrategy(None, write_tropomi_radiance_pickle=True)
if False:
    # If desired, capture each step so we can rerun this for debugging
    rscap = RetrievalStrategyCaptureObserver("retrieval_step",
                                             "starting run_step")
    rs.add_observer(rscap)

# Replace top level script_retrieval_ms with RetrievalStrategy.
rs.register_with_muses_py()
