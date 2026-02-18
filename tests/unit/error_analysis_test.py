from refractor.muses import ErrorAnalysis
from fixtures.retrieval_step_fixture import set_up_run_to_location
from pytest import approx
import numpy.testing as npt
import numpy as np


def test_error_analysis(isolated_dir, joint_tropomi_test_in_dir, ifile_hlp):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "retrieval step",
        ifile_hlp,
    )
    e = ErrorAnalysis(rs.current_state, rs.current_strategy_step, rstep.results)
    if False:
        with np.printoptions(precision=12):
            print(f"""
    assert e.radianceResidualRMSSys == approx(0)
    assert e.KDotDL == approx({repr(e.KDotDL)})
    assert e.LDotDL == approx({repr(e.LDotDL)})
    assert e.maxKDotDLSys == approx(0)
    assert e.KDotDL_species == ["TROPOMICLOUDFRACTION"]
    assert e.Sb is None
    npt.assert_allclose(e.Sa, {repr(e.Sa)} )
    npt.assert_allclose(e.Sa_ret, {repr(e.Sa_ret)} )
    npt.assert_allclose(e.KtSyK, {repr(e.KtSyK)} )
    npt.assert_allclose(e.KtSyKFM, {repr(e.KtSyKFM)} )
    npt.assert_allclose(e.Sx, {repr(e.Sx)} )
    npt.assert_allclose(e.Sx_smooth, {repr(e.Sx_smooth)} )
    npt.assert_allclose(e.Sx_ret_smooth, {repr(e.Sx_ret_smooth)} )
    npt.assert_allclose(e.Sx_smooth_self, {repr(e.Sx_smooth_self)} )
    npt.assert_allclose(e.Sx_ret_crossState, {repr(e.Sx_ret_crossState)} )
    npt.assert_allclose(e.Sx_rand, {repr(e.Sx_rand)} )
    npt.assert_allclose(e.Sx_ret_rand, {repr(e.Sx_ret_rand)} )
    npt.assert_allclose(e.Sx_ret_mapping, {repr(e.Sx_ret_mapping)} )
    npt.assert_allclose(e.Sx_sys, {repr(e.Sx_sys)} )
    npt.assert_allclose(e.Sx_ret_sys, {repr(e.Sx_ret_sys)})
    npt.assert_allclose(e.A, {repr(e.A)})
    npt.assert_allclose(e.A_ret, {repr(e.A_ret)})
    npt.assert_allclose(e.GMatrix, {repr(e.GMatrix)})
    npt.assert_allclose(e.GMatrixFM, {repr(e.GMatrixFM)})
    npt.assert_allclose(e.deviationVsErrorSpecies, {repr(e.deviationVsErrorSpecies)})
    npt.assert_allclose(e.deviationVsRetrievalCovarianceSpecies, {repr(e.deviationVsRetrievalCovarianceSpecies)} )
    npt.assert_allclose(e.deviationVsAprioriCovarianceSpecies, {repr(e.deviationVsAprioriCovarianceSpecies)} )
    npt.assert_allclose(e.degreesOfFreedomForSignal, {repr(e.degreesOfFreedomForSignal)} )
    npt.assert_allclose(e.degreesOfFreedomNoise, {repr(e.degreesOfFreedomNoise)} )
    npt.assert_allclose(e.KDotDL_list, {repr(e.KDotDL_list)} )
    npt.assert_allclose(e.KDotDL_byspecies, {repr(e.KDotDL_byspecies)} )
    npt.assert_allclose(e.KDotDL_byfilter, {repr(e.KDotDL_byfilter)} )
    npt.assert_allclose(e.errorFM, {repr(e.errorFM)} )
    npt.assert_allclose(e.precision, {repr(e.precision)} )
    npt.assert_allclose(e.GdL, {repr(e.GdL)})
    npt.assert_allclose(e.ch4_evs, {repr(e.ch4_evs)})
            """)

    assert e.radianceResidualRMSSys == approx(0)
    assert e.KDotDL == approx(np.float64(0.0006153856967129692))
    assert e.LDotDL == approx(np.float64(0.002344153194986892))
    assert e.maxKDotDLSys == approx(0)
    assert e.KDotDL_species == ["TROPOMICLOUDFRACTION"]
    assert e.Sb is None
    npt.assert_allclose(e.Sa, np.array([[0.01]]))
    npt.assert_allclose(e.Sa_ret, np.array([[0.01]]))
    npt.assert_allclose(e.KtSyK, np.array([[2779734.98080042]]))
    npt.assert_allclose(e.KtSyKFM, np.array([[2779734.98080042]]))
    npt.assert_allclose(e.Sx, np.array([[3.5974551e-07]]))
    npt.assert_allclose(e.Sx_smooth, np.array([[2.07067504e-14]]))
    npt.assert_allclose(e.Sx_ret_smooth, np.array([[2.07067504e-14]]))
    npt.assert_allclose(e.Sx_smooth_self, np.array([[2.07067504e-14]]))
    npt.assert_allclose(e.Sx_ret_crossState, np.array([[0.0]]))
    npt.assert_allclose(e.Sx_rand, np.array([[3.5974549e-07]]))
    npt.assert_allclose(e.Sx_ret_rand, np.array([[3.5974549e-07]]))
    npt.assert_allclose(e.Sx_ret_mapping, np.array([[0.0]]))
    npt.assert_allclose(e.Sx_sys, np.array([[0.0]]))
    npt.assert_allclose(e.Sx_ret_sys, np.array([[0.0]]))
    npt.assert_allclose(e.A, np.array([[0.99999856]]))
    npt.assert_allclose(e.A_ret, np.array([[0.99999856]]))
    npt.assert_allclose(
        e.GMatrix,
        np.array([[1.7254163], [1.73855986], [1.74505535], [1.73818519], [1.73501069]]),
    )
    npt.assert_allclose(
        e.GMatrixFM,
        np.array([[1.7254163], [1.73855986], [1.74505535], [1.73818519], [1.73501069]]),
    )
    npt.assert_allclose(e.deviationVsErrorSpecies, np.array([0.0]))
    npt.assert_allclose(e.deviationVsRetrievalCovarianceSpecies, np.array([0.0]))
    npt.assert_allclose(e.deviationVsAprioriCovarianceSpecies, np.array([0.0]))
    npt.assert_allclose(e.degreesOfFreedomForSignal, np.array([0.99999856]))
    npt.assert_allclose(e.degreesOfFreedomNoise, np.array([1.43898403e-06]))
    npt.assert_allclose(e.KDotDL_list, [0.0006153856967129692])
    npt.assert_allclose(e.KDotDL_byspecies, [0.000615385696712969])
    npt.assert_allclose(e.KDotDL_byfilter, np.array([0.0, 0.0]))
    npt.assert_allclose(e.errorFM, np.array([0.000599787888]))
    npt.assert_allclose(e.precision, np.array([0.000599787871]))
    npt.assert_allclose(
        e.GdL,
        np.array(
            [
                [
                    6.599045497853e-05,
                    1.728327994577e-04,
                    1.760011519570e-05,
                    -2.888365708178e-04,
                    3.288030543579e-05,
                ],
                [
                    6.649314481721e-05,
                    1.741493730759e-04,
                    1.773418608627e-05,
                    -2.910368164326e-04,
                    3.313077492324e-05,
                ],
                [
                    6.674157219153e-05,
                    1.748000186667e-04,
                    1.780044340193e-05,
                    -2.921241693069e-04,
                    3.325455597535e-05,
                ],
                [
                    6.647881511095e-05,
                    1.741118427505e-04,
                    1.773036425354e-05,
                    -2.909740961010e-04,
                    3.312363502522e-05,
                ],
                [
                    6.635740293936e-05,
                    1.737938572856e-04,
                    1.769798277948e-05,
                    -2.904426817426e-04,
                    3.306314037813e-05,
                ],
            ]
        ),
    )
    npt.assert_allclose(e.ch4_evs, [])
