from test_support import *
from refractor.old_py_retrieve_wrapper import (RefractorMusesIntegration, MusesForwardModelStep,
                                               RefractorTropOrOmiFmPyRetrieve)
from refractor.muses import (MusesRunDir, RetrievalStrategy,
                             RetrievalStrategyCaptureObserver,
                             RetrievalStepResultCaptureObserver,
                             StateInfoCaptureObserver)
from refractor.tropomi import TropomiForwardModelHandle

# This contains all the capture tests. Note that there is no requirement at
# all that this be in only one file, but we just collect everything here so
# it is easier to know where are the capture tests are.

# Note it is perfectly fine to run the capture steps in parallel (i.e.,
# pytest with -n 10 or whatever). This generates them faster, and
# everything is done in its own directory so this is clean.

@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
@require_muses_py
def test_capture_tropomi_residual_fm_jac(isolated_dir, step_number, iteration,
                                         osp_dir, gmao_dir,
                                         vlidort_cli):
    rstep = load_muses_retrieval_step(tropomi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    MusesResidualFmJacobian.create_from_retrieval_step(
        rstep, iteration=iteration, capture_directory=True,
        save_pickle_file=f"{tropomi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl", suppress_noisy_output=False, vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [12,])
@pytest.mark.parametrize("iteration", [1,2,3])
@capture_test
@require_muses_py
def test_capture_joint_tropomi_residual_fm_jac(isolated_dir, step_number, iteration,
                                       osp_dir, gmao_dir,vlidort_cli):
    rstep = load_muses_retrieval_step(joint_tropomi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    MusesResidualFmJacobian.create_from_retrieval_step(
        rstep, iteration=iteration, capture_directory=True,
        save_pickle_file=f"{joint_tropomi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl",vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
@require_muses_py
def test_capture_omi_residual_fm_jac(isolated_dir, step_number, iteration,
                                 osp_dir, gmao_dir, vlidort_cli):
    rstep = load_muses_retrieval_step(omi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    MusesResidualFmJacobian.create_from_retrieval_step\
        (rstep, iteration=iteration, capture_directory=True,
         save_pickle_file=f"{omi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl", suppress_noisy_output=False, vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [8,])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
@require_muses_py
def test_capture_joint_omi_residual_fm_jac(isolated_dir, step_number, iteration,
                                 osp_dir, gmao_dir, vlidort_cli):
    rstep = load_muses_retrieval_step(joint_omi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    MusesResidualFmJacobian.create_from_retrieval_step\
        (rstep, iteration=iteration, capture_directory=True,
         save_pickle_file=f"{joint_omi_test_in_dir}/residual_fm_jac_{step_number}_{iteration}.pkl", suppress_noisy_output=False, vlidort_cli=vlidort_cli)
    
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
@require_muses_py
def test_capture_tropomi_refractor_fm(isolated_dir, step_number, iteration,
                              osp_dir, gmao_dir, vlidort_cli):
    rstep = load_muses_retrieval_step(tropomi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(rstep, iteration,
        f"{tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
        vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [12,])
@pytest.mark.parametrize("iteration", [1,2,3])
@capture_test
@require_muses_py
def test_capture_joint_tropomi_refractor_fm(isolated_dir, step_number, iteration,
                                    osp_dir, gmao_dir, vlidort_cli):
    # Note this is the TROPOMI part only, we save stuff after CrIS has been run
    rstep = load_muses_retrieval_step(joint_tropomi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(rstep,
              iteration,
              f"{joint_tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
              vlidort_cli=vlidort_cli)
    
@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("iteration", [1, 2, 3])
@capture_test
@require_muses_py
def test_capture_omi_refractor_fm(isolated_dir, step_number, iteration,
                              osp_dir, gmao_dir, vlidort_cli):
    rstep = load_muses_retrieval_step(omi_test_in_dir,
                                 step_number=step_number,osp_dir=osp_dir,
                                 gmao_dir=gmao_dir)
    RefractorTropOrOmiFmPyRetrieve.uip_from_muses_retrieval_step(rstep,
              iteration,
              f"{omi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl",
              vlidort_cli=vlidort_cli)

@capture_test
@require_muses_py
def test_capture_tropomi_cris_retrieval_strategy(isolated_dir, osp_dir, gmao_dir,
                                                 vlidort_cli,
                                                 clean_up_replacement_function):
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    ihandle = TropomiForwardModelHandle(use_pca=True, use_lrad=False,
                                        lrad_second_order=False)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{joint_tropomi_test_in_dir}/state_info_step",
        "starting run_step")
    rs.add_observer(rscap)
    rscap2 = RetrievalStepResultCaptureObserver(
        f"{joint_tropomi_test_in_dir}/retrieval_result_step",
        "retrieval step")
    rs.add_observer(rscap2)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()

@capture_test
@require_muses_py
def test_capture_airs_irk(isolated_dir, osp_dir, gmao_dir,
                          vlidort_cli,
                          clean_up_replacement_function):
    r = MusesRunDir(airs_irk_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{airs_irk_test_in_dir}/state_info_step",
        "starting run_step")
    rs.add_observer(rscap)
    rscap2 = RetrievalStrategyCaptureObserver(
        f"{airs_irk_test_in_dir}/retrieval_irk",
        "IRK step")
    rs.add_observer(rscap2)
    rscap3 = RetrievalStrategyCaptureObserver(
        f"{airs_irk_test_in_dir}/retrieval_ready_output",
        "retrieval step")
    rs.add_observer(rscap3)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()
    
# These next set of captures duplicates what muses-capture program
# does. You don't need to run these if you already ran
# muses-capture. But it can be useful to run these here to regenerate
# data, both because it uses our stashed version meaning we don't need
# access to the full MUSES input directory, and because these can run
# in parallel.

@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
@require_muses_py
def test_capture_initial_tropomi(isolated_dir, step_number, do_uip, osp_dir,
                                 gmao_dir, vlidort_cli):
    capoutdir = tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if(do_uip):
        RefractorUip.create_from_table(fname, step=step_number,
                                       capture_directory=True,
                                       save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
                                       suppress_noisy_output=False,
                                       vlidort_cli=vlidort_cli)
    else:
        MusesRetrievalStep.create_from_table(fname, step=step_number,
                              capture_directory=True,
                              save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
                              suppress_noisy_output=False,
                              vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
@require_muses_py
def test_capture_initial_omi(isolated_dir, step_number, do_uip, osp_dir, gmao_dir,
                                 vlidort_cli):
    capoutdir = omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if(do_uip):
        RefractorUip.create_from_table(fname, step=step_number,
                                       capture_directory=True,
                                       save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
                                       suppress_noisy_output=False,
                                       vlidort_cli=vlidort_cli)
    else:
        MusesRetrievalStep.create_from_table(fname, step=step_number,
                              capture_directory=True,
                              save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
                              suppress_noisy_output=False,
                              vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [8,])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
@require_muses_py
def test_capture_initial_joint_omi(isolated_dir, step_number, do_uip, osp_dir, gmao_dir,
                                 vlidort_cli):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_omi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if(do_uip):
        RefractorUip.create_from_table(fname, step=step_number,
                                       capture_directory=True,
                                       save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
                                       suppress_noisy_output=False,
                                       vlidort_cli=vlidort_cli)
    else:
        MusesRetrievalStep.create_from_table(fname, step=step_number,
                              capture_directory=True,
                              save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
                              suppress_noisy_output=False,
                              vlidort_cli=vlidort_cli)
        
@pytest.mark.parametrize("step_number", [12,])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
@require_muses_py
def test_capture_initial_joint_tropomi(isolated_dir, step_number, do_uip, osp_dir, gmao_dir,
                                        vlidort_cli):
    os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
    capoutdir = joint_tropomi_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if(do_uip):
        RefractorUip.create_from_table(fname, step=step_number,
                                       capture_directory=True,
                                       save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
                                       suppress_noisy_output=False,
                                       vlidort_cli=vlidort_cli)
    else:
        MusesRetrievalStep.create_from_table(fname, step=step_number,
                              capture_directory=True,
                              save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
                              suppress_noisy_output=False,
                              vlidort_cli=vlidort_cli)

@pytest.mark.parametrize("step_number", [1, 2])
@pytest.mark.parametrize("do_uip", [True, False])
@capture_initial_test
@require_muses_py
def test_capture_initial_tropomi_band7(isolated_dir, step_number, do_uip, osp_dir, gmao_dir,
                                 vlidort_cli):
    capoutdir = tropomi_band7_test_in_dir
    r = MusesRunDir(capoutdir, osp_dir, gmao_dir)
    fname = f"{r.run_dir}/Table.asc"
    if(do_uip):
        RefractorUip.create_from_table(fname, step=step_number,
                                       capture_directory=True,
                                       save_pickle_file=f"{capoutdir}/uip_step_{step_number}.pkl",
                                       suppress_noisy_output=False,
                                       vlidort_cli=vlidort_cli)
    else:
        MusesRetrievalStep.create_from_table(fname, step=step_number,
                              capture_directory=True,
                              save_pickle_file=f"{capoutdir}/run_retrieval_step_{step_number}.pkl",
                              suppress_noisy_output=False,
                              vlidort_cli=vlidort_cli)
        
