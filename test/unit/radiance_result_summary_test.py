import refractor.muses.muses_py as mpy  # type: ignore
from refractor.muses import mpy_radiance_from_observation_list, FilterResultSummary, RadianceResultSummary
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
                                 ret_res.radianceIterations[0, 0, :][fslice],
                                 rstep.NESR[fslice])
    assert rsum.radiance_residual_mean == approx(-0.016962118397373542)
    assert rsum.radiance_residual_rms == approx(4.097508548116728)
    assert rsum.radiance_residual_mean_initial == approx(-10.533994365730303)
    assert rsum.radiance_residual_rms_initial == approx(17.98897105856759)
    assert rsum.radiance_snr == approx(623.9468620299166)
    assert rsum.radiance_residual_rms_relative_continuum == approx(0.000526006841888145)
    assert rsum.radiance_continuum == approx(0.08342921788652663)
    assert rsum.residual_slope == approx(0.03338386427243988)
    assert rsum.residual_quadratic == approx(-1.5004715029448592)

