from refractor.muses import ErrorAnalysis
from fixtures.retrieval_step_fixture import set_up_run_to_location
from pytest import approx
import numpy.testing as npt


def test_error_analysis(isolated_dir, joint_tropomi_test_in_dir, ifile_hlp):
    rs, rstep, _ = set_up_run_to_location(
        joint_tropomi_test_in_dir,
        10,
        "retrieval step",
        ifile_hlp,
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
    assert e.KDotDL == approx(0.0006150135677765839)
    assert e.LDotDL == approx(0.002343842720102936)
    assert e.maxKDotDLSys == approx(0)
    assert e.KDotDL_species == ["TROPOMICLOUDFRACTION"]
    assert e.Sb is None
    npt.assert_allclose(e.Sa, [[0.01]])
    npt.assert_allclose(e.Sa_ret, [[0.01]])
    npt.assert_allclose(e.KtSyK, [[2779730.446896]])
    npt.assert_allclose(e.KtSyKFM, [[2779730.446896]])
    npt.assert_allclose(e.Sx, [[3.59746097e-07]])
    npt.assert_allclose(e.Sx_smooth, [[2.07068179e-14]])
    npt.assert_allclose(e.Sx_ret_smooth, [[2.07068179e-14]])
    npt.assert_allclose(e.Sx_smooth_self, [[2.07068179e-14]])
    npt.assert_allclose(e.Sx_ret_crossState, [[0.0]])
    npt.assert_allclose(e.Sx_rand, [[3.59746077e-07]])
    npt.assert_allclose(e.Sx_ret_rand, [[3.59746077e-07]])
    npt.assert_allclose(e.Sx_ret_mapping, [[0.0]])
    npt.assert_allclose(e.Sx_sys, [[0.0]])
    npt.assert_allclose(e.Sx_ret_sys, [[0.0]])
    npt.assert_allclose(e.A, [[0.99999856]])
    npt.assert_allclose(e.A_ret, [[0.99999856]])
    npt.assert_allclose(
        e.GMatrix, [[1.72541806], [1.73856162], [1.7450569], [1.73818634], [1.73501157]]
    )
    npt.assert_allclose(
        e.GMatrixFM,
        [[1.72541806], [1.73856162], [1.7450569], [1.73818634], [1.73501157]],
    )
    npt.assert_allclose(e.deviationVsErrorSpecies, [0.0])
    npt.assert_allclose(e.deviationVsRetrievalCovarianceSpecies, [0.0])
    npt.assert_allclose(e.deviationVsAprioriCovarianceSpecies, [0.0])
    npt.assert_allclose(e.degreesOfFreedomForSignal, [0.99999856])
    npt.assert_allclose(e.degreesOfFreedomNoise, [1.43898638e-06])
    npt.assert_allclose(e.KDotDL_list, [0.0006150135677765839])
    npt.assert_allclose(e.KDotDL_byspecies, [0.0006150135677765839])
    npt.assert_allclose(e.KDotDL_byfilter, [0.0, 0.0])
    npt.assert_allclose(e.errorFM, [0.0005997883770280405])
    npt.assert_allclose(e.precision, [0.0005997883770280405])
    npt.assert_allclose(
        e.GdL,
        [
            [
                6.60240745e-05,
                1.72867895e-04,
                1.76143404e-05,
                -2.88861463e-04,
                3.28222523e-05,
            ],
            [
                6.65270201e-05,
                1.74184736e-04,
                1.77485196e-05,
                -2.91061897e-04,
                3.30722794e-05,
            ],
            [
                6.67755656e-05,
                1.74835491e-04,
                1.78148282e-05,
                -2.92149307e-04,
                3.31958378e-05,
            ],
            [
                6.65126600e-05,
                1.74147137e-04,
                1.77446885e-05,
                -2.90999070e-04,
                3.30651406e-05,
            ],
            [
                6.63911755e-05,
                1.73829060e-04,
                1.77122781e-05,
                -2.90467565e-04,
                3.30047476e-05,
            ],
        ],
    )
    npt.assert_allclose(e.ch4_evs, [])
