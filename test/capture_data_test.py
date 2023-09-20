from test_support import *
from refractor.muses import *

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
    MusesResidualFmJacobian.create_from_retrieval_step\
        (rstep, iteration=iteration, capture_directory=True,
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
    MusesResidualFmJacobian.create_from_retrieval_step\
        (rstep, iteration=iteration, capture_directory=True,
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
    
@capture_test
@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_capture_joint_tropomi_run_forward_model(isolated_dir, osp_dir, gmao_dir,
                                         call_num, vlidort_cli):
    '''muses-py calls run_forward_model to calculate the systematic jacobian.
    It only does this for some steps, and for the cris-tropomi (but not
    tropomi run. Capture this so we have test data to work with.'''
     # This is the last call to run_forward_model in the retrieval
    rdir = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    fname = f"{rdir.run_dir}/Table.asc"
    MusesForwardModelStep.create_from_table(fname, step=call_num,
                                            capture_directory=True,
                                            save_pickle_file=f"{joint_tropomi_test_in_dir}/run_forward_model_call_{call_num}.pkl",
                                            vlidort_cli=vlidort_cli,
                                            suppress_noisy_output=False)

@capture_test
@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_capture_joint_omi_run_forward_model(isolated_dir, osp_dir, gmao_dir,
                                         call_num, vlidort_cli):
    '''muses-py calls run_forward_model to calculate the systematic jacobian.
    It only does this for some steps, and for the airs-omi (but not
    tropomi run. Capture this so we have test data to work with.'''
     # This is the last call to run_forward_model in the retrieval
    rdir = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    fname = f"{rdir.run_dir}/Table.asc"
    MusesForwardModelStep.create_from_table(fname, step=call_num,
                                            capture_directory=True,
                                            save_pickle_file=f"{joint_omi_test_in_dir}/run_forward_model_call_{call_num}.pkl",
                                            vlidort_cli=vlidort_cli,
                                            suppress_noisy_output=False)
    
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
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    rmi.register_with_muses_py()
    r = MusesRunDir(joint_tropomi_test_in_dir,
                    osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc")
    rs.clear_observers()
    obs = RetrievalStrategyCaptureObserver(f"{joint_tropomi_test_in_dir}/retrieval_strategy_retrieval_step", "retrieval step")
    rs.add_observer(obs)
    rs.retrieval_ms()
    
    
