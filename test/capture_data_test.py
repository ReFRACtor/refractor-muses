from test_support import *
from refractor.old_py_retrieve_wrapper import RefractorTropOrOmiFmPyRetrieve
from refractor.muses import (
    MusesRunDir,
    RetrievalStrategy,
    RetrievalStepCaptureObserver,
    StateInfoCaptureObserver,
)
from refractor.tropomi import TropomiForwardModelHandle, TropomiSwirForwardModelHandle
from refractor.omi import OmiForwardModelHandle

# This contains all the capture tests. Note that there is no requirement at
# all that this be in only one file, but we just collect everything here so
# it is easier to know where the capture tests are.

# Note it is perfectly fine to run the capture steps in parallel (i.e.,
# pytest with -n 10 or whatever). This generates them faster, and
# everything is done in its own directory so this is clean.

# This assumes the data files are already in refractor_test_data. If not,
# you can either manually copy them there, or uses
# MusesRunDir.save_run_directory

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
@capture_test
def test_capture_joint_tropomi_residual_fm_jac(
    isolated_dir, step_number, iteration, osp_dir, gmao_dir, vlidort_cli
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
        save_pickle_file=f"{joint_tropomi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl",
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
@capture_test
def test_capture_joint_omi_residual_fm_jac(
    isolated_dir, step_number, iteration, osp_dir, gmao_dir, vlidort_cli
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
        save_pickle_file=f"{joint_omi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl",
        suppress_noisy_output=False,
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
def test_capture_tropomi_refractor_fm(
    isolated_dir, step_number, iteration, osp_dir, gmao_dir, vlidort_cli
):
    rstep = load_muses_retrieval_step(
        tropomi_test_in_dir, step_number=step_number, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(
        rstep,
        iteration,
        f"{tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize(
    "step_number",
    [
        12,
    ],
)
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
def test_capture_joint_tropomi_refractor_fm(
    isolated_dir, step_number, iteration, osp_dir, gmao_dir, vlidort_cli
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
        f"{joint_tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize(
    "iteration",
    [
        2,
    ],
)
@capture_test
def test_capture_omi_refractor_fm(
    isolated_dir, step_number, iteration, osp_dir, gmao_dir, vlidort_cli
):
    rstep = load_muses_retrieval_step(
        omi_test_in_dir, step_number=step_number, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(
        rstep,
        iteration,
        f"{omi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli,
    )


@capture_test
def test_capture_tropomi_cris_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli
):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    ihandle = TropomiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{joint_tropomi_test_in_dir}/state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        f"{joint_tropomi_test_in_dir}/retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()


@capture_test
def test_capture_airs_omi_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli
):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    ihandle = OmiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{joint_omi_test_in_dir}/state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        f"{joint_omi_test_in_dir}/retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()


@capture_test
def test_capture_airs_irk(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    r = MusesRunDir(airs_irk_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{airs_irk_test_in_dir}/state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        f"{airs_irk_test_in_dir}/retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()


# ---------------------------------------------------------------
# Capture the old UIP and retrieval step. This is just used for
# backwards testing against py-retrieve. We don't use these in
# refractor any more
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
def test_capture_initial_tropomi(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, vlidort_cli
):
    capoutdir = tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
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
        2,
    ],
)
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
def test_capture_initial_omi(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, vlidort_cli
):
    capoutdir = omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
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
@capture_initial_test
def test_capture_initial_joint_omi(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, vlidort_cli
):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
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
@capture_initial_test
def test_capture_initial_joint_tropomi(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, vlidort_cli
):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )


@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
def test_capture_initial_tropomi_band7(
    isolated_dir, step_number, do_uip, osp_dir, gmao_dir, vlidort_cli
):
    capoutdir = tropomi_band7_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if do_uip:
        RefractorUip.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
    else:
        MusesRetrievalStep.create_from_table(
            fname,
            step=step_number,
            capture_directory=True,
            save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
            suppress_noisy_output=False,
            vlidort_cli=vlidort_cli,
        )
