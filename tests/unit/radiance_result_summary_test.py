from refractor.muses import (
    mpy_radiance_from_observation_list,
    FilterResultSummary,
    RadianceResultSummary,
    AttrDictAdapter,
)
from pytest import approx


def test_radiance_result_summary(joint_tropomi_step_12_output):
    rs, rstp, _ = joint_tropomi_step_12_output
    ret_res = rstp.slv.retrieval_results()
    obs_list = rstp.cfunc.obs_list
    rstep = AttrDictAdapter(
        mpy_radiance_from_observation_list(obs_list, include_bad_sample=True)
    )
    fsummary = FilterResultSummary(rstep)
    fslice = fsummary.filter_slice[0]
    rsum = RadianceResultSummary(
        rstep.radiance[fslice],
        ret_res.radiance["radiance"][fslice],
        ret_res.radianceIterations[0, 0, :][fslice],
        rstep.NESR[fslice],
    )
    if False:
        print(rsum.radiance_residual_mean)
        print(rsum.radiance_residual_rms)
        print(rsum.radiance_residual_mean_initial)
        print(rsum.radiance_residual_rms_initial)
        print(rsum.radiance_snr)
        print(rsum.radiance_residual_rms_relative_continuum)
        print(rsum.radiance_continuum)
        print(rsum.residual_slope)
        print(rsum.residual_quadratic)

    assert rsum.radiance_residual_mean == approx(-0.018559420372403317)
    assert rsum.radiance_residual_rms == approx(4.099550280515104)
    assert rsum.radiance_residual_mean_initial == approx(-10.401067796728588)
    assert rsum.radiance_residual_rms_initial == approx(18.302985458656355)
    assert rsum.radiance_snr == approx(623.9484593318916)
    assert rsum.radiance_residual_rms_relative_continuum == approx(0.000526028739914585)
    assert rsum.radiance_continuum == approx(0.08342918922614802)
    assert rsum.residual_slope == approx(0.036068230091460075)
    assert rsum.residual_quadratic == approx(-1.508706609387551)
