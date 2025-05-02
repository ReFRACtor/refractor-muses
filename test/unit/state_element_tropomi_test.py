from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
)
from refractor.tropomi import StateElementTropomiCloudFraction, StateElementTropomiCloudPressure
import numpy as np
import numpy.testing as npt


def test_tropomi_cloud_fraction_state_element(osp_dir, joint_tropomi_obs_step_12):
    _, tropomi_obs = joint_tropomi_obs_step_12
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the tropomi_obs cloud fraction.
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-CrIS-TROPOMI-v7" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementTropomiCloudFraction(
        StateElementIdentifier("TROPOMICLOUDFRACTION"),
        tropomi_obs,
        latitude,
        species_directory,
        covariance_directory,
    )
    vexpect = np.array(
        [
            0.219607874751091
        ]
    )
    npt.assert_allclose(selem.value, vexpect)
    npt.assert_allclose(selem.step_initial_value, vexpect)
    npt.assert_allclose(selem.retrieval_initial_value, vexpect)

def test_tropomi_cloud_pressure_state_element(osp_dir, joint_tropomi_obs_step_12):
    _, tropomi_obs = joint_tropomi_obs_step_12
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the tropomi_obs cloud fraction.
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-CrIS-TROPOMI-v7" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementTropomiCloudPressure(
        StateElementIdentifier("TROPOMICLOUDPRESSURE"),
        tropomi_obs,
    )
    vexpect = np.array(
        [
            642.80609375
        ]
    )
    npt.assert_allclose(selem.value, vexpect)
    npt.assert_allclose(selem.step_initial_value, vexpect)
    npt.assert_allclose(selem.retrieval_initial_value, vexpect)
    
