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
        print(csum.cloudODAve)
        print(csum.cloudODVar)
        print(csum.cloudODAveError)
        print(csum.emisDev)
        print(csum.emissionLayer)
        print(csum.ozoneCcurve)
        print(csum.ozone_slope_QA)
        print(csum.deviation_QA)
        print(csum.num_deviations_QA)
        print(csum.DeviationBad_QA)
    assert csum.cloudODAve == approx(0.5293772644232951)
    assert csum.cloudODVar == approx(0.7167961581629615)
    assert csum.cloudODAveError == approx(0.0)
    assert csum.emisDev == approx(0.0012042654037386358)
    assert csum.emissionLayer == approx(-6.990282365802727)
    assert csum.ozoneCcurve == approx(1.0)
    assert csum.ozone_slope_QA == approx(7.187930140233559)
    npt.assert_allclose(
        csum.deviation_QA, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    npt.assert_allclose(csum.num_deviations_QA, [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    npt.assert_allclose(csum.DeviationBad_QA, [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
