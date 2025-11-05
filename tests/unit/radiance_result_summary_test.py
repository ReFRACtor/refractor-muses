from refractor.muses import (
    mpy_radiance_from_observation_list,
    FilterResultSummary,
    RadianceResultSummary,
    AttrDictAdapter
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

    assert rsum.radiance_residual_mean == approx(-0.01894818332062393)
    assert rsum.radiance_residual_rms == approx(4.097515323702615)
    assert rsum.radiance_residual_mean_initial == approx(-10.533994208879513)
    assert rsum.radiance_residual_rms_initial == approx(17.98897124186496)
    assert rsum.radiance_snr == approx(623.9488480948398)
    assert rsum.radiance_residual_rms_relative_continuum == approx(
        0.0005259849508727864
    )
    assert rsum.radiance_continuum == approx(0.08342909399663756)
    assert rsum.residual_slope == approx(0.03695308972199365)
    assert rsum.residual_quadratic == approx(-1.5049600230580278)
