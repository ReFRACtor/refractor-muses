import refractor.muses.muses_py as mpy
from refractor.muses import (
    register_replacement_function_in_block,
    RefractorCaptureDirectory,
    muses_py_call,
    osswrapper,
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


class MusesResidualFmJacobian:
    """This class is used to capture the arguments to a py-retrieve
    residual_fm_jacobian step, and to then call that step. This is little
    more than the argument list plus a bit of support code."""

    def __init__(self, params=None):
        self.params = params
        self.capture_directory = RefractorCaptureDirectory()

    @property
    def run_dir(self):
        """The path we run residual_fm_jacobian in."""
        return self.capture_directory.rundir

    def residual_fm_jacobian(
        self,
        vlidort_nstokes=2,
        vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
    ):
        """Run the retrieval step with the saved parameters"""
        with muses_py_call(
            self.run_dir, vlidort_cli=vlidort_cli, vlidort_nstokes=vlidort_nstokes
        ):
            with osswrapper(self.params["uip"]):
                return mpy.residual_fm_jacobian(**self.params)

    @classmethod
    def create_from_retrieval_step(
        cls,
        rstep,
        iteration=1,
        capture_directory=False,
        save_pickle_file=None,
        vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
        suppress_noisy_output=True,
    ):
        """This grabs the arguments passed to residual_fm_jacobian and stores
        them to allow calling this later."""
        try:
            with register_replacement_function_in_block(
                "residual_fm_jacobian", _CaptureParams(func_count=iteration)
            ):
                # This is pretty noisy, so suppress printing. We can revisit
                # this if needed, but I think this is a good idea
                if suppress_noisy_output:
                    with _all_output_disabled() as f:
                        rstep.run_retrieval(vlidort_cli=vlidort_cli)
                else:
                    rstep.run_retrieval(vlidort_cli=vlidort_cli)
        except _FakeParamsExecption as e:
            res = cls(params=e.params)
        if capture_directory:
            # Not needed, residual_fm_jacobian creates this itself
            vlidort_input = None
            if "uip_OMI" in res.params["uip"].__dict__:
                vlidort_input = res.params["uip"].__dict__["uip_OMI"]["vlidort_input"]
            if "uip_TROPOMI" in res.params["uip"].__dict__:
                vlidort_input = res.params["uip"].__dict__["uip_TROPOMI"][
                    "vlidort_input"
                ]
            res.capture_directory.save_directory(
                rstep.run_retrieval_path, vlidort_input
            )
        if save_pickle_file is not None:
            pickle.dump(res, open(save_pickle_file, "wb"))
        return res

    @classmethod
    def load_residual_fm_jacobian(
        cls,
        save_pickle_file,
        path=".",
        change_to_dir=False,
        osp_dir=None,
        gmao_dir=None,
    ):
        """This is the pair to create_from_table, it loads a
        MusesResidualFmJacobian from a pickle file, extracts the
        saved directory, and optionally changes to that directory."""
        res = pickle.load(open(save_pickle_file, "rb"))
        res.capture_directory.extract_directory(
            path=path, change_to_dir=change_to_dir, osp_dir=osp_dir, gmao_dir=gmao_dir
        )
        return res
