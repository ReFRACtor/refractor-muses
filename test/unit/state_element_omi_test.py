from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
)
from refractor.omi import StateElementOmiCloudFraction, StateElementOmiSurfaceAlbedo
import numpy as np
import numpy.testing as npt


def test_omi_cloud_fraction_state_element(osp_dir, joint_omi_obs_step_8):
    _, omi_obs = joint_omi_obs_step_8
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the omi_obs cloud fraction.
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOmiCloudFraction(
        StateElementIdentifier("OMICLOUDFRACTION"),
        omi_obs,
        latitude,
        "LAND",
        species_directory,
        covariance_directory,
    )
    vexpect = np.array([0.32976987957954407])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_omi_surface_state_element(osp_dir, joint_omi_obs_step_8):
    _, omi_obs = joint_omi_obs_step_8
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the omi_obs monthly minimum surface reflectance
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOmiSurfaceAlbedo(
        StateElementIdentifier("OMISURFACEALBEDOUV1"),
        omi_obs,
        latitude,
        "LAND",
        species_directory,
        covariance_directory,
    )
    vexpect = np.array([0.052])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)
