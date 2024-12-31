import refractor.muses.muses_py as mpy
from refractor.muses import (
    register_replacement_function_in_block,
    RefractorCaptureDirectory,
    muses_py_call,
)
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
import pickle

if mpy.have_muses_py:

    class _FakeParamsExecption(Exception):
        def __init__(self, params):
            self.params = params

    class _CaptureParams(mpy.ReplaceFunctionObject):
        def __init__(self, func_count=1):
            self.func_count = func_count

        def should_replace_function(self, func_name, params):
            self.func_count -= 1
            print(f"hi there, in replace. func_count = {self.func_count}")
            if self.func_count <= 0:
                return True
            return False

        def replace_function(self, func_name, params):
            raise _FakeParamsExecption(params)


@contextmanager
def _all_output_disabled():
    """Suppress stdout, stderr, and logging"""
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()) as sout:
            with redirect_stderr(io.StringIO()) as serr:
                yield
    finally:
        logging.disable(previous_level)


class MusesForwardModelStep:
    """This class is used to capture the arguments to a muses-py
    run_forward_model, and to then call that function. This is little more than
    the argument list plus a bit of support code.

    We are interested in this because in some way the run_forward_model
    is used by script_retrieval_ms to calculate the systematic jacobian.
    I'm not sure what is going on here, or the exact logic, but we need to
    replace this like we are the RefractorResidualFmJacobian."""

    def __init__(self, params=None):
        self.params = params
        self.capture_directory = RefractorCaptureDirectory()
        self.run_path = None

    @property
    def run_forward_model_path(self):
        """The path we run run_forward_model in."""
        return self.run_path + "/" + self.capture_directory.runbase

    def run_forward_model(
        self, vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli"
    ):
        """Run the retrieval step with the saved parameters"""
        with muses_py_call(self.run_forward_model_path, vlidort_cli=vlidort_cli):
            return mpy.run_forward_model(**self.params)

    @classmethod
    def create_from_table(
        cls,
        strategy_table,
        step=1,
        capture_directory=False,
        save_pickle_file=None,
        vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
        suppress_noisy_output=True,
    ):
        """This grabs the arguments passed to run_forward_model and stores them
        to allow calling this later."""
        # TODO Note there is some duplication with create_from_table we
        # have in RefractorUip. We could possible extract this out
        # somehow into a base class. But right now we only have a
        # few lasses, so this probably isn't worth it. So we are currently
        # just duplicating the code.
        with muses_py_call(os.path.dirname(strategy_table), vlidort_cli=vlidort_cli):
            try:
                with register_replacement_function_in_block(
                    "run_forward_model", _CaptureParams(func_count=step)
                ):
                    # This is pretty noisy, so suppress printing. We can revisit
                    # this if needed, but I think this is a good idea
                    if suppress_noisy_output:
                        with _all_output_disabled() as f:
                            mpy.script_retrieval_ms(os.path.basename(strategy_table))
                    else:
                        mpy.script_retrieval_ms(os.path.basename(strategy_table))
            except _FakeParamsExecption as e:
                res = cls(params=e.params)
        if capture_directory:
            # Not needed, run_forward_model creates this itself
            vlidort_input = None
            res.capture_directory.save_directory(
                os.path.dirname(strategy_table), vlidort_input
            )
        if save_pickle_file is not None:
            pickle.dump(res, open(save_pickle_file, "wb"))
        return res

    @classmethod
    def load_forward_model_step(
        cls,
        save_pickle_file,
        path=".",
        change_to_dir=False,
        osp_dir=None,
        gmao_dir=None,
    ):
        """This is the pair to create_from_table, it loads a MusesRetrievalStep
        from a pickle file, extracts the saved directory, and optionally
        changes to that directory."""
        res = pickle.load(open(save_pickle_file, "rb"))
        res.run_path = os.path.abspath(path)
        res.capture_directory.extract_directory(
            path=path, change_to_dir=change_to_dir, osp_dir=osp_dir, gmao_dir=gmao_dir
        )
        return res


__all__ = [
    "MusesForwardModelStep",
]
