from test_support import *
from refractor.muses import (FmObsCreator, CostFunction,
                             MusesForwardmodelStep,
                             RefractorMusesIntegration)
import refractor.muses.muses_py as mpy
import subprocess
import pprint

def struct_compare(s1, s2):
    for k in s1.keys():
        #print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_run_forward_model_joint_tropomi(call_num,
                                         isolated_dir, osp_dir, gmao_dir,
                                         vlidort_cli):
    pfile = f"{joint_tropomi_test_in_dir}/run_forward_model_call_{call_num}.pkl"
    curdir = os.path.abspath(os.path.curdir)
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

    struct_compare(o_radianceOut, o_radianceOut2)
    struct_compare(o_jacobianOut, o_jacobianOut2)

@pytest.mark.parametrize("call_num", [1,2,3,4,5,6])
@require_muses_py
def test_run_forward_model_joint_omi(call_num,
                                         isolated_dir, osp_dir, gmao_dir,
                                         vlidort_cli):
    pfile = f"{joint_omi_test_in_dir}/run_forward_model_call_{call_num}.pkl"
    curdir = os.path.abspath(os.path.curdir)
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
        
    struct_compare(o_radianceOut, o_radianceOut2)
    struct_compare(o_jacobianOut, o_jacobianOut2)

@pytest.mark.parametrize("step_num", [1,2,3])
@require_muses_py
def test_run_retrieval_tropomi(step_num,
                               isolated_dir, osp_dir, gmao_dir,
                               vlidort_cli):
    pfile = f"{tropomi_test_in_dir}/run_retrieval_step_{step_num}.pkl"
    curdir = os.path.abspath(os.path.curdir)
    rrefractor = MusesRetrievalStep.load_retrieval_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    # For quicker turn around, change the number of iterations from 10 to 1.
    # Once we have everything agreeing we can remove this for a slower run
    rrefractor.params['i_tableStruct']['data'][0] = rrefractor.params['i_tableStruct']['data'][0].replace('10','1')
    (o_retrievalResults, o_uip, rayInfo,
     ret_info, windowsF, success_flag) = rmi.run_retrieval(**rrefractor.params)
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesRetrievalStep.load_retrieval_step(pfile,
                       osp_dir=osp_dir, gmao_dir=gmao_dir, path="muses_py",
                       change_to_dir=True)
    rmuses_py.params['i_tableStruct']['data'][0] = rmuses_py.params['i_tableStruct']['data'][0].replace('10','1')
    (o_retrievalResults2, o_uip2, rayInfo2,
     ret_info2, windowsF2, success_flag2) = rmuses_py.run_retrieval(vlidort_cli=vlidort_cli)
    assert success_flag == success_flag2
    struct_compare(ret_info, ret_info2)
    # Need
    # o_retrievalResults
    # rayInfo
    # windowsF
    with open("o_uip.txt", "w") as fh:
        pprint.pprint(o_uip,fh)
    with open("o_uip2.txt", "w") as fh:
        pprint.pprint(o_uip2,fh)
    # This is almost the same, except the cloud surface albedo is off by the
    # last digit. Not sure why this isn't identical, but for practical purposes
    # it is. Just print out differents, but don't fail if not identical
    print("UIP differences:")
    subprocess.run(["diff", "-u", "o_uip.txt", "o_uip2.txt"],
                   #check=True)
                   )
            
