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
        print(f"""
    assert rsum.radiance_residual_mean == approx({rsum.radiance_residual_mean})
    assert rsum.radiance_residual_rms == approx({rsum.radiance_residual_rms})
    assert rsum.radiance_residual_mean_initial == approx({rsum.radiance_residual_mean_initial})
    assert rsum.radiance_residual_rms_initial == approx({rsum.radiance_residual_rms_initial})
    assert rsum.radiance_snr == approx({rsum.radiance_snr})
    assert rsum.radiance_residual_rms_relative_continuum == approx({rsum.radiance_residual_rms_relative_continuum})
    assert rsum.radiance_continuum == approx({rsum.radiance_continuum})
    assert rsum.residual_slope == approx({rsum.residual_slope})
    assert rsum.residual_quadratic == approx({rsum.residual_quadratic})
        """)
    assert rsum.radiance_residual_mean == approx(-0.016827086759100926)
    assert rsum.radiance_residual_rms == approx(4.098160029918225)
    assert rsum.radiance_residual_mean_initial == approx(-10.57412927730087)
    assert rsum.radiance_residual_rms_initial == approx(17.912617700297393)
    assert rsum.radiance_snr == approx(623.9467269982783)
    assert rsum.radiance_residual_rms_relative_continuum == approx(
        0.0005526478397821612
    )
    assert rsum.radiance_continuum == approx(0.08342167978847297)
    assert rsum.residual_slope == approx(0.03845152071341276)
    assert rsum.residual_quadratic == approx(-1.4554959003247652)
