# Functions very much like a fixture. These return the the old MusesResidualFmJacobian
from refractor.old_py_retrieve_wrapper import (
    MusesResidualFmJacobian,
)


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
        dir_in / f"residual_fm_jac_{step_number}_{iteration}.pkl",
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
        change_to_dir=change_to_dir,
    )


def joint_omi_residual_fm_jac(
    osp_dir, gmao_dir, joint_omi_test_in_dir, path="refractor"
):
    """This returns the old MusesResidualFmJacobian. This is used for
    backwards testing against py-retrieve code.

    Directory is created given the path, and we change into that
    directory."""
    step_number = 8
    iteration = 2
    return _muses_residual_fm_jac(
        joint_omi_test_in_dir,
        step_number=step_number,
        iteration=iteration,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
    )


def joint_tropomi_residual_fm_jac(
    osp_dir, gmao_dir, joint_tropomi_test_in_dir, path="refractor"
):
    """This returns the old MusesResidualFmJacobian. This is used for
    backwards testing against py-retrieve code.

    Directory is created given the path, and we change into that
    directory."""

    step_number = 12
    iteration = 2
    return _muses_residual_fm_jac(
        joint_tropomi_test_in_dir,
        step_number=step_number,
        iteration=iteration,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        path=path,
    )
