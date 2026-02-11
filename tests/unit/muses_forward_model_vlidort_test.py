from refractor.muses import (
    InstrumentIdentifier,
)
from refractor.omi import OmiFmObjectCreator
from refractor.tropomi import TropomiFmObjectCreator
from refractor.muses_py_fm import (
    MusesTropomiForwardModel,
    MusesOmiForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import numpy.testing as npt


@require_muses_py_fm
def test_muses_tropomi_forward_model_vlidort(joint_tropomi_step_12):
    # Note, we need the run dir for these tests because match_py_retrieve puts files in
    # run directory
    rs, rstep, _ = joint_tropomi_step_12
    obs_tropomi = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
    )
    obs_tropomi.spectral_window.include_bad_sample = True
    ocreator = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.retrieval_config,
        obs_tropomi,
        use_vlidort=True,
        match_py_retrieve=True,
    )
    fm = ocreator.forward_model
    # Set up jacobians
    ocreator.fm_sv.update_state(ocreator.fm_sv.state, ocreator.fm_sv.state_covariance)

    s = fm.radiance_all()
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesTropomiForwardModel(
        rs.current_state,
        obs_tropomi,
        rs.retrieval_config,
    )
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian.copy()
    # MusesTropomiForwardModel includes the observation jacobian, which we include
    # separately (in the cost function, not the forward model). So zero out
    # so we can compare. Note that we get different jacobians for the observation
    # also, py-retrieve used finite differences while we have analytic jacobians.
    # Results in small differences.
    jaccmp[:,-3:] = 0
    # MusesTropomiForwardModel incorrectly does not scale the surface parameters by
    # the clear fraction. Fix so we can compare
    jaccmp[:,-6:-3] *= 1 - ocreator.cloud_fraction.cloud_fraction.value    
    assert rad.shape == radcmp.shape
    assert jac.shape == jaccmp.shape
    npt.assert_allclose(rad, radcmp)
    # MusesTropomiForwardModel incorrectly does not handle the raman scattering
    # jacobian correctly. The jaccmp should be scaled by the raman scattering, but
    # isn't. We can't easily correct for this here, so we just compare with a loose
    # tolerance.  The jacobians are still very similar, just scaled differently
    npt.assert_allclose(jac, jaccmp, rtol=1e-2)

@require_muses_py_fm
def test_muses_omi_forward_model_vlidort(joint_omi_step_8):
    # Note, we need the run dir for these tests because match_py_retrieve puts files in
    # run directory
    rs, rstep, _ = joint_omi_step_8
    obs_omi = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
    )
    obs_omi.spectral_window.include_bad_sample = True
    ocreator = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.retrieval_config,
        obs_omi,
        use_vlidort=True,
        match_py_retrieve=True,
    )
    fm = ocreator.forward_model
    # Set up jacobians
    ocreator.fm_sv.update_state(ocreator.fm_sv.state, ocreator.fm_sv.state_covariance)

    s = fm.radiance_all()
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian

    fmcmp = MusesOmiForwardModel(rs.current_state, obs_omi, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian.copy()
    # MusesOmiForwardModel includes the observation jacobian, which we include
    # separately (in the cost function, not the forward model). So zero out
    # so we can compare. Note that we get different jacobians for the observation
    # also, py-retrieve used finite differences while we have analytic jacobians.
    # Results in small differences.
    jaccmp[:,-4:] = 0
    # MusesOmiForwardModel incorrectly does not scale the surface parameters by
    # the clear fraction. Fix so we can compare
    jaccmp[:,-7:-4] *= 1 - ocreator.cloud_fraction.cloud_fraction.value    
    assert rad.shape == radcmp.shape
    assert jac.shape == jaccmp.shape
    # Slightly larger difference. We use the same code for the raman scattering, but
    # the data passed doesn't get truncated like it does with the old ring external
    # program. So we end up with slightly different value
    npt.assert_allclose(rad, radcmp, rtol=2e-6)
    # MusesTropomiForwardModel incorrectly does not handle the raman scattering
    # jacobian correctly. The jaccmp should be scaled by the raman scattering, but
    # isn't. We can't easily correct for this here, so we just compare with a loose
    # tolerance.  The jacobians are still very similar, just scaled differently
    npt.assert_allclose(jac, jaccmp, atol=6e-3)
