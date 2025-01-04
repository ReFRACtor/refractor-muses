from refractor.old_py_retrieve_wrapper import (
    MusesResidualFmJacobian,
    MusesRetrievalStep,
    RefractorTropOrOmiFmPyRetrieve,
)
from refractor.muses import (
    MusesRunDir,
    RefractorUip,
)
import pytest
import os
import refractor.muses.muses_py as mpy


def load_muses_retrieval_step(
    dir_in, step_number=1, osp_dir=None, gmao_dir=None, change_to_dir=True
):
    """This reads parameters that can be use to call the py-retrieve function
    run_retrieval. See muses_capture in refractor-muses for collecting this.
    """
    return MusesRetrievalStep.load_retrieval_step(
        dir_in / f"run_retrieval_step_{step_number}.pkl",
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        change_to_dir=change_to_dir,
    )


# ---------------------------------------------------------------
# This is used to test out the old residual_fm_jacobian function.
# This is the old py-retrieve function. We don't actually use this
# anymore, but it is useful to make sure the old function works in case
# we need to use this in the future to track down some problem.
#
# We can probably eventually remove this - at some point it may be more
# work to maintain this old compatibility function than it is worth.


@pytest.mark.parametrize(
    "step_number",
    [
        12,
    ],
)
@pytest.mark.parametrize(
    "iteration",
    [
        2,
    ],
)
@pytest.mark.capture_test
def test_capture_joint_tropomi_residual_fm_jac(
    isolated_dir,
    step_number,
    iteration,
    osp_dir,
    gmao_dir,
    joint_tropomi_test_in_dir,
    vlidort_cli,
):
    rstep = load_muses_retrieval_step(
        joint_tropomi_test_in_dir,
        step_number=step_number,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    MusesResidualFmJacobian.create_from_retrieval_step(
        rstep,
        iteration=iteration,
        capture_directory=True,
        save_pickle_file=joint_tropomi_test_in_dir
        / f"residual_fm_jac_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


# ---------------------------------------------------------------
# This is used to test out the old residual_fm_jacobian function.
# This is the old py-retrieve function. We don't actually use this
# anymore, but it is useful to make sure the old function works in case
# we need to use this in the future to track down some problem.
#
# We can probably eventually remove this - at some point it may be more
# work to maintain this old compatibility function than it is worth.
@pytest.mark.parametrize(
    "step_number",
    [
        8,
    ],
)
@pytest.mark.parametrize(
    "iteration",
    [
        2,
    ],
)
@pytest.mark.capture_test
def test_capture_joint_omi_residual_fm_jac(
    isolated_dir,
    step_number,
    iteration,
    osp_dir,
    joint_omi_test_in_dir,
    gmao_dir,
    vlidort_cli,
):
    rstep = load_muses_retrieval_step(
        joint_omi_test_in_dir,
        step_number=step_number,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    MusesResidualFmJacobian.create_from_retrieval_step(
        rstep,
        iteration=iteration,
        capture_directory=True,
        save_pickle_file=joint_omi_test_in_dir
        / f"residual_fm_jac_{step_number}_{iteration}.pkl",
        suppress_noisy_output=False,
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@pytest.mark.capture_test
def test_capture_tropomi_refractor_fm(
    isolated_dir,
    step_number,
    iteration,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    tropomi_test_in_dir,
):
    rstep = load_muses_retrieval_step(
        tropomi_test_in_dir, step_number=step_number, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(
        rstep,
        iteration,
        tropomi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize(
    "step_number",
    [
        12,
    ],
)
@pytest.mark.parametrize("iteration", [1, 2, 3])
@pytest.mark.capture_test
def test_capture_joint_tropomi_refractor_fm(
    isolated_dir,
    step_number,
    iteration,
    osp_dir,
    gmao_dir,
    joint_tropomi_test_in_dir,
    vlidort_cli,
):
    # Note this is the TROPOMI part only, we save stuff after CrIS has been run
    rstep = load_muses_retrieval_step(
        joint_tropomi_test_in_dir,
        step_number=step_number,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(
        rstep,
        iteration,
        joint_tropomi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize(
    "iteration",
    [
        2,
    ],
)
@pytest.mark.capture_test
def test_capture_omi_refractor_fm(
    isolated_dir,
    step_number,
    iteration,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    omi_test_in_dir,
):
    rstep = load_muses_retrieval_step(
        omi_test_in_dir, step_number=step_number, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(
        rstep,
        iteration,
        omi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


# ---------------------------------------------------------------
# Capture the old UIP and retrieval step. This is just used for
# backwards testing against py-retrieve. We don't use these in
# refractor any more
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@pytest.mark.capture_initial_test
def test_capture_initial_tropomi(
    isolated_dir,
    step_number,
    do_uip,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    tropomi_test_in_dir,
):
    capoutdir = tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = r.run_dir / "Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            str(fname),
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )


# ---------------------------------------------------------------
# Capture the old UIP and retrieval step. This is just used for
# backwards testing against py-retrieve. We don't use these in
# refractor any more
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@pytest.mark.capture_initial_test
def test_capture_initial_omi(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, omi_test_in_dir, vlidort_cli
):
    capoutdir = omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = r.run_dir / "Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )


# ---------------------------------------------------------------
# Capture the old UIP and retrieval step. This is just used for
# backwards testing against py-retrieve. We don't use these in
# refractor any more
@pytest.mark.parametrize(
    "step_number",
    [
        8,
    ],
)
@pytest.mark.parametrize("do_uip", [True, False])
@pytest.mark.capture_initial_test
def test_capture_initial_joint_omi(
    isolated_dir,
    step_number,
    do_uip,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    joint_omi_test_in_dir,
):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = r.run_dir / "Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )


# ---------------------------------------------------------------
# Capture the old UIP and retrieval step. This is just used for
# backwards testing against py-retrieve. We don't use these in
# refractor any more
@pytest.mark.parametrize(
    "step_number",
    [
        12,
    ],
)
@pytest.mark.parametrize("do_uip", [True, False])
@pytest.mark.capture_initial_test
def test_capture_initial_joint_tropomi(
    isolated_dir,
    step_number,
    joint_tropomi_test_in_dir,
    do_uip,
    osp_dir,
    gmao_dir,
    vlidort_cli,
):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = r.run_dir / "Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )


# This doesn't work. This isn't overly important, this data was a kludge to
# get something initial working. This should get replaced with Josh's test cases.
# Leave in place in case we need to come back to this, but for now this doesn't work.
@pytest.mark.skip
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@pytest.mark.capture_initial_test
def test_capture_initial_tropomi_band7(
    isolated_dir,
    step_number,
    do_uip,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    tropomi_band7_test_in_dir,
):
    capoutdir = tropomi_band7_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = r.run_dir / "Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=capoutdir / f"run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
