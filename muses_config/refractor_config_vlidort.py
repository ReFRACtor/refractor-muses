# These are the arguments passed to cli function that loads this code.
from py_retrieve.cli import params
from refractor.muses import RetrievalStrategy
from refractor.tropomi import TropomiForwardModelHandle
from refractor.omi import OmiForwardModelHandle
import refractor.framework as rf

# Configuration to use our wrapper around the py-retrieve vlidort code. Should be
# close to running py-retrieve without refractor, but tests our RetrievalStrategy
# class

# Turn on logging for refractor. This is independent of python logger, although we
# could probably integrate this together if it is important
rf.Logger.set_implementation(rf.FpLogger())

rs = RetrievalStrategy(None)

# Replace top level script_retrieval_ms with RetrievalStrategy.
rs.register_with_muses_py()
