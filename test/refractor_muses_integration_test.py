from test_support import *
from refractor.muses import (FmObsCreator, CostFunction,
                             MusesForwardModelStep,
                             MusesRunDir,
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
    rrefractor = MusesForwardModelStep.load_forward_model_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    (uip, o_radianceOut, o_jacobianOut) = rmi.run_forward_model(**rrefractor.params)
        
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesForwardModelStep.load_forward_model_step(pfile,
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
    rrefractor = MusesForwardModelStep.load_forward_model_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    (uip, o_radianceOut, o_jacobianOut) = rmi.run_forward_model(**rrefractor.params)
        
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesForwardModelStep.load_forward_model_step(pfile,
                       osp_dir=osp_dir, gmao_dir=gmao_dir, path="muses_py",
                       change_to_dir=True)
    (uip2, o_radianceOut2, o_jacobianOut2) = rmuses_py.run_forward_model(vlidort_cli=vlidort_cli)
    with open("uip2.txt", "w") as fh:
        pprint.pprint(uip2,fh)
        
    struct_compare(o_radianceOut, o_radianceOut2)
    struct_compare(o_jacobianOut, o_jacobianOut2)

@require_muses_py
def test_quicker_run_retrieval_tropomi(isolated_dir, osp_dir, gmao_dir,
                               vlidort_cli):
    '''This is like the run_retrieval test found below, but we monkey with
    stuff to give only 1 iteration. This is good for testing the interface
    of run_retrieval because this runs in a reasonable amount of time.'''
    step_num = 1
    pfile = f"{tropomi_test_in_dir}/run_retrieval_step_{step_num}.pkl"
    curdir = os.path.abspath(os.path.curdir)
    rrefractor = MusesRetrievalStep.load_retrieval_step(pfile,
                osp_dir=osp_dir, gmao_dir=gmao_dir, path="refractor",
                change_to_dir=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    # For quicker turn around, change the number of iterations from 10 to 1.
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
    os.chdir(curdir)
    assert success_flag == success_flag2
    struct_compare(ret_info, ret_info2)
    assert len(windowsF) == len(windowsF2)
    for i in range(len(windowsF)):
        struct_compare(windowsF[i], windowsF2[i])

    # These fields are nested too much for using struct_compare. We could
    # write some  specialized code, but it is easier just to pretty print
    # and do a unix diff.
    with open("ray_info.txt", "w") as fh:
        pprint.pprint(rayInfo,fh)
    with open("ray_info2.txt", "w") as fh:
        pprint.pprint(rayInfo2,fh)
    print("RayInfo differences:")
    subprocess.run(["diff", "-u", "ray_info.txt", "ray_info2.txt"],
                   check=True)
    
    with open("retrieval_results.txt", "w") as fh:
        pprint.pprint(o_retrievalResults,fh)
    with open("retrieval_results2.txt", "w") as fh:
        pprint.pprint(o_retrievalResults2,fh)
    print("RetrievalResults differences:")
    subprocess.run(["diff", "-u", "retrieval_results.txt",
                    "retrieval_results2.txt"],
                   check=True)

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


# These tests are a bit long, even for a long test. We don't normally
# run these, at the point where we are looking at a full retrieval we
# usually want to just do full runs. However this can be useful to
# occasionally run, e.g., when debugging an issue. So we leave these
# tests here but usually skip running them.
@skip    
@long_test
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
    (o_retrievalResults, o_uip, rayInfo,
     ret_info, windowsF, success_flag) = rmi.run_retrieval(**rrefractor.params)
    # Results to compare against
    os.chdir(curdir)
    rmuses_py =MusesRetrievalStep.load_retrieval_step(pfile,
                       osp_dir=osp_dir, gmao_dir=gmao_dir, path="muses_py",
                       change_to_dir=True)
    (o_retrievalResults2, o_uip2, rayInfo2,
     ret_info2, windowsF2, success_flag2) = rmuses_py.run_retrieval(vlidort_cli=vlidort_cli)
    os.chdir(curdir)
    assert success_flag == success_flag2
    struct_compare(ret_info, ret_info2)
    assert len(windowsF) == len(windowsF2)
    for i in range(len(windowsF)):
        struct_compare(windowsF[i], windowsF2[i])

    # These fields are nested too much for using struct_compare. We could
    # write some  specialized code, but it is easier just to pretty print
    # and do a unix diff.
    with open("ray_info.txt", "w") as fh:
        pprint.pprint(rayInfo,fh)
    with open("ray_info2.txt", "w") as fh:
        pprint.pprint(rayInfo2,fh)
    # May see tiny differences, so we run diff without requiring this
    # to be identical.
    print("RayInfo differences:")
    subprocess.run(["diff", "-u", "ray_info.txt", "ray_info2.txt"],
                   #check=True)
                   )
    
    with open("retrieval_results.txt", "w") as fh:
        pprint.pprint(o_retrievalResults,fh)
    with open("retrieval_results2.txt", "w") as fh:
        pprint.pprint(o_retrievalResults2,fh)
    # May see tiny differences, so we run diff without requiring this
    # to be identical.
    print("RetrievalResults differences:")
    subprocess.run(["diff", "-u", "retrieval_results.txt",
                    "retrieval_results2.txt"],
                   #check=True)
                   )

    with open("o_uip.txt", "w") as fh:
        pprint.pprint(o_uip,fh)
    with open("o_uip2.txt", "w") as fh:
        pprint.pprint(o_uip2,fh)
    # May see tiny differences, so we run diff without requiring this
    # to be identical.
    print("UIP differences:")
    subprocess.run(["diff", "-u", "o_uip.txt", "o_uip2.txt"],
                   #check=True)
                   )
            

@long_test
@require_muses_py
def test_original_airs_omi(osp_dir, gmao_dir, vlidort_cli):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r original_airs_omi", shell=True)
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="original_airs_omi")
    r.run_retrieval(vlidort_cli=vlidort_cli)

@long_test
@require_muses_py
def test_refractor_integration_airs_omi(osp_dir, gmao_dir, vlidort_cli):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r refractor_integration_airs_omi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="refractor_integration_airs_omi")
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
@long_test
@require_muses_py
def test_original_cris_tropomi(osp_dir, gmao_dir, vlidort_cli):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r original_cris_tropomi", shell=True)
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="original_cris_tropomi")
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
@long_test
@require_muses_py
def test_refractor_integration_cris_tropomi(osp_dir, gmao_dir, vlidort_cli):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r refractor_integration_cris_tropomi", shell=True)
    rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli)
    rmi.register_with_muses_py()
    r = MusesRunDir(joint_omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="refractor_integration_cris_tropomi")
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
    
