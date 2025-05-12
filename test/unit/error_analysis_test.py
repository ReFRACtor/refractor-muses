from refractor.muses import ErrorAnalysis
from fixtures.retrieval_step_fixture import set_up_run_to_location
from pytest import approx
import numpy.testing as npt


def test_error_analysis(
    isolated_dir, joint_tropomi_test_in_dir, osp_dir, gmao_dir, vlidort_cli
):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "retrieval step",
        osp_dir,
        gmao_dir,
        vlidort_cli,
    )
    e = ErrorAnalysis(rs.current_state, rs.current_strategy_step, rstep.results)
    if False:
        print(e.radianceResidualRMSSys)
        print(e.KDotDL)
        print(e.LDotDL)
        print(e.maxKDotDLSys)
        print(e.KDotDL_species)
        print(repr(e.Sb))
        print(repr(e.Sa))
        print(repr(e.Sa_ret))
        print(repr(e.KtSyK))
        print(repr(e.KtSyKFM))
        print(repr(e.Sx))
        print(repr(e.Sx_smooth))
        print(repr(e.Sx_ret_smooth))
        print(repr(e.Sx_smooth_self))
        print(repr(e.Sx_ret_crossState))
        print(repr(e.Sx_rand))
        print(repr(e.Sx_ret_rand))
        print(repr(e.Sx_ret_mapping))
        print(repr(e.Sx_sys))
        print(repr(e.Sx_ret_sys))
        print(repr(e.A))
        print(repr(e.A_ret))
        print(repr(e.GMatrix))
        print(repr(e.GMatrixFM))
        print(repr(e.deviationVsErrorSpecies))
        print(repr(e.deviationVsRetrievalCovarianceSpecies))
        print(repr(e.deviationVsAprioriCovarianceSpecies))
        print(repr(e.degreesOfFreedomForSignal))
        print(repr(e.degreesOfFreedomNoise))
        print(repr(e.KDotDL_list))
        print(repr(e.KDotDL_byspecies))
        print(repr(e.KDotDL_byfilter))
        print(repr(e.errorFM))
        print(repr(e.precision))
        print(repr(e.GdL))
        print(repr(e.ch4_evs))

    assert e.radianceResidualRMSSys == approx(0)
    assert e.KDotDL == approx(0.000615366226801187)
    assert e.LDotDL == approx(0.002344087795348228)
    assert e.maxKDotDLSys == approx(0)
    assert e.KDotDL_species == ["TROPOMICLOUDFRACTION"]
    assert e.Sb is None
    npt.assert_allclose(e.Sa, [[0.01]])
    npt.assert_allclose(e.Sa_ret, [[0.01]])
    npt.assert_allclose(e.KtSyK, [[2779760.7342367]])
    npt.assert_allclose(e.KtSyKFM, [[2779760.7342367]])
    npt.assert_allclose(e.Sx, [[3.59742178e-07]])
    npt.assert_allclose(e.Sx_smooth, [[2.07063667e-14]])
    npt.assert_allclose(e.Sx_ret_smooth, [[2.07063667e-14]])
    npt.assert_allclose(e.Sx_smooth_self, [[2.07063667e-14]])
    npt.assert_allclose(e.Sx_ret_crossState, [[0.0]])
    npt.assert_allclose(e.Sx_rand, [[3.59742157e-07]])
    npt.assert_allclose(e.Sx_ret_rand, [[3.59742157e-07]])
    npt.assert_allclose(e.Sx_ret_mapping, [[0.0]])
    npt.assert_allclose(e.Sx_sys, [[0.0]])
    npt.assert_allclose(e.Sx_ret_sys, [[0.0]])
    npt.assert_allclose(e.A, [[0.99999856]])
    npt.assert_allclose(e.A_ret, [[0.99999856]])
    npt.assert_allclose(
        e.GMatrix,
        [[1.72540835], [1.73855184], [1.74504728], [1.73817712], [1.73500259]],
    )
    npt.assert_allclose(
        e.GMatrixFM,
        [[1.72540835], [1.73855184], [1.74504728], [1.73817712], [1.73500259]],
    )
    npt.assert_allclose(e.deviationVsErrorSpecies, [0.0])
    npt.assert_allclose(e.deviationVsRetrievalCovarianceSpecies, [0.0])
    npt.assert_allclose(e.deviationVsAprioriCovarianceSpecies, [0.0])
    npt.assert_allclose(e.degreesOfFreedomForSignal, [0.99999856])
    npt.assert_allclose(e.degreesOfFreedomNoise, [1.4389707e-06])
    npt.assert_allclose(e.KDotDL_list, [0.000615366226801187])
    npt.assert_allclose(e.KDotDL_byspecies, [0.000615366226801187])
    npt.assert_allclose(e.KDotDL_byfilter, [0.0, 0.0])
    npt.assert_allclose(e.errorFM, [0.0005997851094818544])
    npt.assert_allclose(e.precision, [0.0005997851094818544])
    npt.assert_allclose(
        e.GdL,
        [
            [
                6.59867250e-05,
                1.72831279e-04,
                1.76009321e-05,
                -2.88833102e-04,
                3.28812309e-05,
            ],
            [
                6.64893862e-05,
                1.74147841e-04,
                1.77350091e-05,
                -2.91033321e-04,
                3.31317073e-05,
            ],
            [
                6.67377988e-05,
                1.74798479e-04,
                1.78012693e-05,
                -2.92120657e-04,
                3.32554915e-05,
            ],
            [
                6.64750557e-05,
                1.74110306e-04,
                1.77311867e-05,
                -2.90970594e-04,
                3.31245664e-05,
            ],
            [
                6.63536483e-05,
                1.73792318e-04,
                1.76988032e-05,
                -2.90439177e-04,
                3.30640690e-05,
            ],
        ],
    )
    npt.assert_allclose(e.ch4_evs, [])
