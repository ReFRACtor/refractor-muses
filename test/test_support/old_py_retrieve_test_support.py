# This is test support for comparing against the old py-retrieve. We pull this
# out of test_support.py just to make it clear that this is only used for the
# comparison with the old py-retrieve code.
#
# The py-retrieve code is complicated to match and to test against. This was
# very useful during initial development to make sure the ReFRACtor code was
# working and generated similar results to py-retrieve.
#
# At this point, the comparisons here are mostly done. In the future, changes
# to ReFRACtor will be compared against the existing ReFRACtor results, we
# will compare against py-retrieve less and less frequently. At some point,
# the cost of maintaining this old code won't be worth it - so these tests
# will likely disappear over time.
#
# But for now, we keep this old functionality to support investigating any
# old issue that pop up.

import os
import pytest
from refractor.muses import RefractorUip
import refractor.muses.muses_py as mpy
from refractor.old_py_retrieve_wrapper import (
    MusesResidualFmJacobian,
    MusesRetrievalStep,
)
import os
from .test_support import (
    joint_tropomi_test_in_dir,
    joint_omi_test_in_dir,
    omi_test_in_dir,
    tropomi_test_in_dir,
)

# Marker for tests against py-retrieve code. See comments at top
# of this file
old_py_retrieve_test = pytest.mark.old_py_retrieve_test


@pytest.fixture(scope="function")
def clean_up_replacement_function():
    """Remove any replacement functions that have been added when the
    test ends"""
    if not mpy.have_muses_py:
        raise pytest.skip("test requires muses_py")
    try:
        yield
    finally:
        mpy.unregister_replacement_function_all()
        osswrapper.register_with_muses_py()


def load_uip(dir_in, step_number=1, osp_dir=None, gmao_dir=None):
    return RefractorUip.load_uip(
        f"{dir_in}/uip_step_{step_number}.pkl",
        change_to_dir=True,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )


@pytest.fixture(scope="function")
def omi_uip_step_2(isolated_dir, osp_dir, gmao_dir):
    """Return a RefractorUip for strategy step 2, and also unpack all the
    support files into a directory"""
    return load_uip(omi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir)


@pytest.fixture(scope="function")
def tropomi_uip_step_2(isolated_dir, osp_dir, gmao_dir):
    """Return a RefractorUip for strategy step 2, and also unpack all the
    support files into a directory"""
    return load_uip(
        tropomi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir
    )


@pytest.fixture(scope="function")
def clean_up_replacement_function():
    """Remove any replacement functions that have been added when the
    test ends"""
    if not mpy.have_muses_py:
        raise pytest.skip("test requires muses_py")
    try:
        yield
    finally:
        mpy.unregister_replacement_function_all()
        osswrapper.register_with_muses_py()


def _muses_residual_fm_jac(
    dir_in,
    step_number=1,
    iteration=1,
    osp_dir=None,
    gmao_dir=None,
    path=".",
    change_to_dir=True,
):
    """This reads parameters that can be use to call the py-retrieve function
    residual_fm_jac. See muses_capture in refractor-muses for collecting this.
    """
    return MusesResidualFmJacobian.load_residual_fm_jacobian(
        f"{dir_in}/residual_fm_jac_{step_number}_{iteration}.pkl",
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
        change_to_dir=change_to_dir,
    )


def joint_omi_residual_fm_jac(path="refractor"):
    """This returns the old MusesResidualFmJacobian. This is used for
    backwards testing against py-retrieve code.

    Directory is created given the path, and we change into that
    directory."""
    step_number = 8
    iteration = 2
    osp_dir = os.environ.get("MUSES_OSP_PATH", None)
    gmao_dir = os.environ.get("MUSES_GMAO_PATH", None)
    return _muses_residual_fm_jac(
        joint_omi_test_in_dir,
        step_number=step_number,
        iteration=iteration,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
    )


def joint_tropomi_residual_fm_jac(path="refractor"):
    """This returns the old MusesResidualFmJacobian. This is used for
    backwards testing against py-retrieve code.

    Directory is created given the path, and we change into that
    directory."""

    step_number = 12
    iteration = 2
    osp_dir = os.environ.get("MUSES_OSP_PATH", None)
    gmao_dir = os.environ.get("MUSES_GMAO_PATH", None)
    return _muses_residual_fm_jac(
        joint_tropomi_test_in_dir,
        step_number=step_number,
        iteration=iteration,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
    )


@pytest.fixture(scope="function")
def joint_omi_uip_step_8(isolated_dir, osp_dir, gmao_dir):
    """Return a RefractorUip for strategy step 1, and also unpack all the
    support files into a directory"""
    return load_uip(
        joint_omi_test_in_dir, step_number=8, osp_dir=osp_dir, gmao_dir=gmao_dir
    )


@pytest.fixture(scope="function")
def joint_tropomi_uip_step_12(isolated_dir, osp_dir, gmao_dir):
    """Return a RefractorUip for strategy step 1, and also unpack all the
    support files into a directory"""
    return load_uip(
        joint_tropomi_test_in_dir, step_number=12, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
