from .refractor_capture_directory import muses_py_call
import logging
import refractor.muses.muses_py as mpy
import os

logger = logging.getLogger("py-retrieve")

class RetrievalStrategy:
    '''This is an attempt to make the muses-py script_retrieval_ms more like our
    JointRetrieval stuff (pretty dated, but
    https://github.jpl.nasa.gov/refractor/joint_retrieval)

    This is a replacement for script_retrieval_ms, that tries to do a few things:

    1. Simplifies the core code, the script_retrieval_ms is really pretty long and
        is a sequence of "do one thing, then another, then aother). We do this by:
    2. Moving output out of this class, and having separate classes handle this. We
       use the standard ReFRACtor approach of having observers. This tend to give a
       much cleaner interface with clear seperation.
    3. Adopt a extensively, configurable way to handle the initial guess (similiar
       to the OCO-2 InitialGuess structure)
    4. Handle species information as a separate class, which allows us to easily
       extend the list of jacobian parameters (e.g, add EOFs). The existing code
       uses long lists of hardcoded values, this attempts to be a more adaptable.

    This has a number of advantages, for example having InitialGuess separated out
    allows us to do unit testing in ways that don't require updating the OSP
    directories with new covariance stuff, for example.
    '''
    # TODO  Add handling of writeOutput, writePlots, debug. I think we can probably
    # do that by just adding Observers
    def __init__(self, filename, vlidort_cli=None):
        logger.info(f"Strategy table filename {filename}")
        self.filename = os.path.abspath(filename)
        self.run_dir = os.path.dirname(self.filename)
        self.vlidort_cli = vlidort_cli
        # May want to rework this, but for now uses muses-py functions
        curdir = os.getcwd()
        try:
            os.chdir(self.run_dir)
            _, self.strategy_table = mpy.table_read(self.filename)
            self.strategy_table = self.strategy_table.__dict__
        finally:
            os.chdir(curdir)

    def retrieval_ms(self):
        '''This is script_retrieval_ms in muses-py'''
        # Wrapper around calling mpy. We can perhaps pull some this out, but
        # for now we'll do that.
        with muses_py_call(self.run_dir,
                           vlidort_cli=self.vlidort_cli):
            self.retrieval_ms_body()

    def retrieval_ms_body(self):
        mpy.script_retrieval_ms(self.filename)
