from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
)
from refractor.omi import StateElementOmiCloudFraction, StateElementOmiSurfaceAlbedo
import numpy as np
import numpy.testing as npt


def test_omi_cloud_fraction_state_element(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    selem = StateElementOmiCloudFraction.create(
        StateElementIdentifier("OMICLOUDFRACTION"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    vexpect = np.array([0.32976987957954407])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_omi_surface_albedo_state_element(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    selem = StateElementOmiSurfaceAlbedo.create(
        StateElementIdentifier("OMISURFACEALBEDOUV1"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    vexpect = np.array([0.052])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)
