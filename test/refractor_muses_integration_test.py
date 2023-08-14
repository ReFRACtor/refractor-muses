from test_support import *
from refractor.muses import (FmObsCreator, CostFunction,
                             MusesForwardmodelStep,
                             RefractorMusesIntegration)
import refractor.muses.muses_py as mpy

@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_run_forward_model_joint_tropomi(call_num,
                                         isolated_dir, osp_dir, gmao_dir,
                                         vlidort_cli):
    pfile = f"{joint_tropomi_test_in_dir}/run_forward_model_call_{call_num}.pkl"
    curdir = os.path.curdir
    rrefractor = MusesForwardmodelStep.load_forward_model_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    (uip, o_radianceOut, o_jacobianOut) = rmi.run_forward_model(**rrefractor.params)
        
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesForwardmodelStep.load_forward_model_step(pfile,
                       osp_dir=osp_dir, gmao_dir=gmao_dir, path="muses_py",
                       change_to_dir=True)
    (uip2, o_radianceOut2, o_jacobianOut2) = rmuses_py.run_forward_model(vlidort_cli=vlidort_cli)
        
    for k in o_radianceOut.keys():
        #print(k)
        if(isinstance(o_radianceOut[k], np.ndarray) and
           np.can_cast(o_radianceOut[k], np.float64)):
            npt.assert_allclose(o_radianceOut[k], o_radianceOut2[k])
        elif(isinstance(o_radianceOut[k], np.ndarray)):
            assert np.all(o_radianceOut[k] == o_radianceOut2[k])
        else:
            assert o_radianceOut[k] == o_radianceOut2[k]

    for k in o_jacobianOut.keys():
        #print(k)
        if(isinstance(o_jacobianOut[k], np.ndarray) and
           np.can_cast(o_jacobianOut[k], np.float64)):
           npt.assert_allclose(o_jacobianOut[k], o_jacobianOut2[k])
        elif(isinstance(o_jacobianOut[k], np.ndarray)):
            assert np.all(o_jacobianOut[k] == o_jacobianOut2[k])
        else:
            assert o_jacobianOut[k] == o_jacobianOut2[k]

@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_run_forward_model_joint_omi(call_num,
                                         isolated_dir, osp_dir, gmao_dir,
                                         vlidort_cli):
    pfile = f"{joint_omi_test_in_dir}/run_forward_model_call_{call_num}.pkl"
    curdir = os.path.curdir
    rrefractor = MusesForwardmodelStep.load_forward_model_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    (uip, o_radianceOut, o_jacobianOut) = rmi.run_forward_model(**rrefractor.params)
        
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesForwardmodelStep.load_forward_model_step(pfile,
                       osp_dir=osp_dir, gmao_dir=gmao_dir, path="muses_py",
                       change_to_dir=True)
    (uip2, o_radianceOut2, o_jacobianOut2) = rmuses_py.run_forward_model(vlidort_cli=vlidort_cli)
        
    for k in o_radianceOut.keys():
        #print(k)
        if(isinstance(o_radianceOut[k], np.ndarray) and
           np.can_cast(o_radianceOut[k], np.float64)):
            npt.assert_allclose(o_radianceOut[k], o_radianceOut2[k])
        elif(isinstance(o_radianceOut[k], np.ndarray)):
            assert np.all(o_radianceOut[k] == o_radianceOut2[k])
        else:
            assert o_radianceOut[k] == o_radianceOut2[k]

    for k in o_jacobianOut.keys():
        #print(k)
        if(isinstance(o_jacobianOut[k], np.ndarray) and
           np.can_cast(o_jacobianOut[k], np.float64)):
           npt.assert_allclose(o_jacobianOut[k], o_jacobianOut2[k])
        elif(isinstance(o_jacobianOut[k], np.ndarray)):
            assert np.all(o_jacobianOut[k] == o_jacobianOut2[k])
        else:
            assert o_jacobianOut[k] == o_jacobianOut2[k]
            
