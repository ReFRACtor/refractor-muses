from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    RetrievalConfiguration,
    InstrumentIdentifier,
    FilterIdentifier,
    MeasurementIdFile,
    MusesTropomiObservation,
)
from refractor.tropomi import (
    StateElementTropomiCloudFraction,
    StateElementTropomiCloudPressure,
    StateElementTropomiSurfaceAlbedo,
)
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
    vexpect = np.array([0.219607874751091])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_tropomi_cloud_pressure_state_element(osp_dir, joint_tropomi_obs_step_12):
    _, tropomi_obs = joint_tropomi_obs_step_12
    selem = StateElementTropomiCloudPressure(
        StateElementIdentifier("TROPOMICLOUDPRESSURE"),
        tropomi_obs,
    )
    vexpect = np.array([642.80609375])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)


def test_tropomi_surface_alebdo_band7_state_element(tropomi_swir, josh_osp_dir):
    r = tropomi_swir
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=josh_osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("TROPOMI"): [FilterIdentifier("BAND7")],
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    tropomi_obs = MusesTropomiObservation.create_from_id(
        measurement_id,
        None,
        None,
        None,
        None,
        osp_dir=josh_osp_dir,
    )
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the tropomi_obs cloud fraction.
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        josh_osp_dir
        / "Strategy_Tables"
        / "laughner"
        / "OSP-CrIS-TROPOMI-swir-co-dev"
        / "Species-66"
    )
    covariance_directory = josh_osp_dir / "Covariance" / "Covariance"
    selem = StateElementTropomiSurfaceAlbedo(
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND7"),
        tropomi_obs,
        latitude,
        species_directory,
        covariance_directory,
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


def test_tropomi_surface_albedo_state_element(osp_dir, joint_tropomi_obs_step_12):
    _, tropomi_obs = joint_tropomi_obs_step_12
    # State element is just a StateElementOspFile (already tested), but with
    # the apriori/initial guess coming from the tropomi_obs cloud fraction.
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-CrIS-TROPOMI-v7" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementTropomiSurfaceAlbedo(
        StateElementIdentifier("TROPOMISURFACEALBEDOBAND3"),
        tropomi_obs,
        latitude,
        species_directory,
        covariance_directory,
        band=3,
    )
    vexpect = np.array([0.04])
    npt.assert_allclose(selem.value_fm, vexpect)
    npt.assert_allclose(selem.step_initial_fm, vexpect)
    npt.assert_allclose(selem.retrieval_initial_fm, vexpect)
