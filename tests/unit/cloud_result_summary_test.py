from refractor.muses import CloudResultSummary, ErrorAnalysis
from pytest import approx
import numpy.testing as npt


def test_cloud_result_summary(joint_tropomi_step_12_output):
    rs, rstep, _ = joint_tropomi_step_12_output
    current_state = rs.current_state
    result_list = rstep.results.resultsList
    e = ErrorAnalysis(rs.current_state, rs.current_strategy_step, rstep.results)
    csum = CloudResultSummary(current_state, result_list, e)
    if False:
        print(f"""
    assert csum.cloudODAve == approx({csum.cloudODAve})
    assert csum.cloudODVar == approx({csum.cloudODVar})
    assert csum.cloudODAveError == approx({csum.cloudODAveError})
    assert csum.emisDev == approx({csum.emisDev})
    assert csum.emissionLayer == approx({csum.emissionLayer})
    assert csum.ozoneCcurve == approx({csum.ozoneCcurve})
    assert csum.ozone_slope_QA == approx({csum.ozone_slope_QA})
    npt.assert_allclose(
        csum.deviation_QA, {csum.deviation_QA.tolist()}
    )
    npt.assert_allclose(csum.num_deviations_QA, {csum.num_deviations_QA.tolist()})
    npt.assert_allclose(csum.DeviationBad_QA, {csum.DeviationBad_QA.tolist()})
""")
    assert csum.cloudODAve == approx(0.5291746065634516)
    assert csum.cloudODVar == approx(0.8773752354232953)
    assert csum.cloudODAveError == approx(0.0)
    assert csum.emisDev == approx(0.0012045889451099967)
    assert csum.emissionLayer == approx(-6.993888393789803)
    assert csum.ozoneCcurve == approx(1)
    assert csum.ozone_slope_QA == approx(7.2938369516522945)
    npt.assert_allclose(
        csum.deviation_QA, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    npt.assert_allclose(csum.num_deviations_QA, [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    npt.assert_allclose(csum.DeviationBad_QA, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
