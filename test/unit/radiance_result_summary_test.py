import refractor.muses.muses_py as mpy  # type: ignore
from refractor.muses import mpy_radiance_from_observation_list, FilterIdentifier, FilterResultSummary, RadianceResultSummary
from pytest import approx

def test_radiance_result_summary(joint_tropomi_step_12_output):
    rs, rstp, _ = joint_tropomi_step_12_output
    ret_res = mpy.ObjectView(rstp.slv.retrieval_results())
    obs_list = rstp.cfunc.obs_list
    rstep = mpy.ObjectView(mpy_radiance_from_observation_list(obs_list, include_bad_sample=True))
    fsummary = FilterResultSummary(rstep)
    fslice = fsummary.filter_slice[0]
    rsum = RadianceResultSummary(rstep.radiance[fslice],
                                 ret_res.radiance["radiance"][fslice],
                                 rstep.NESR[fslice])
    assert rsum.radiance_residual_mean == approx(-0.016962118397373542)
    
