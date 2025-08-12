from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
)
from refractor.tropomi import (
    StateElementTropomiCloudFraction,
    StateElementTropomiCloudPressure,
    StateElementTropomiSurfaceAlbedo,
)
import numpy as np
import numpy.testing as npt


def test_tropomi_cloud_fraction_state_element(cris_tropomi_old_shandle):
    _, _, rconfig, strat, obs_hset, smeta = cris_tropomi_old_shandle
    selem = StateElementTropomiCloudFraction.create(
        StateElementIdentifier("TROPOMICLOUDFRACTION"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
    )
    vexpect = np.array([0.219607874751091])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_tropomi_cloud_pressure_state_element(cris_tropomi_old_shandle):
    _, _, rconfig, strat, obs_hset, smeta = cris_tropomi_old_shandle
    selem = StateElementTropomiCloudPressure.create(
        StateElementIdentifier("TROPOMICLOUDPRESSURE"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
    )
    vexpect = np.array([642.80609375])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_tropomi_surface_alebdo_band7_state_element(tropomi_swir_old_shandle):
    _, _, rconfig, strat, obs_hset, smeta = tropomi_swir_old_shandle
    selem = StateElementTropomiSurfaceAlbedo.create(
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND7"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
        band=7,
        cov_is_constraint=True,
    )
    vexpect = np.array(
        [
            0.061612524825922714,
        ]
    )
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_tropomi_surface_albedo_state_element(cris_tropomi_old_shandle):
    _, _, rconfig, strat, obs_hset, smeta = cris_tropomi_old_shandle
    selem = StateElementTropomiSurfaceAlbedo.create(
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        retrieval_config=rconfig,
        strategy=strat,
        observation_handle_set=obs_hset,
        sounding_metadata=smeta,
        band=3,
    )
    vexpect = np.array([0.04])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)
