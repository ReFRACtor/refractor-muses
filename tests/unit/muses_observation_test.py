from refractor.muses import (
    MusesRunDir,
    MusesTropomiObservation,
    SimulatedObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    CurrentStateDict,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
)
import copy
import os
import numpy.testing as npt
import pytest


def test_measurement_id(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    flist = {
        InstrumentIdentifier("OMI"): [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        InstrumentIdentifier("AIRS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    mid = MeasurementIdFile(r.run_dir / "Measurement_ID.asc", rconfig, flist)
    assert mid.filter_list_dict == flist
    assert float(mid["OMI_Longitude"]) == pytest.approx(-154.7512664794922)
    assert int(mid["OMI_XTrack_UV1_Index"]) == 10
    assert (
        os.path.basename(mid["OMI_Cloud_filename"])
        == "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    assert mid["omi_calibrationFilename"] == str(
        osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    )


def test_simulated_obs(isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    # Determined by looking a the full run
    filter_list_dict = {
        InstrumentIdentifier("TROPOMI"): [FilterIdentifier("BAND3")],
        InstrumentIdentifier("CRIS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    # This is the microwindows file for step 12, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    cs = CurrentStateDict(
        {
            "TROPOMISOLARSHIFTBAND3": 0.1,
            "TROPOMIRADIANCESHIFTBAND3": 0.2,
            "TROPOMIRADSQUEEZEBAND3": 0.3,
        },
        [
            "TROPOMISOLARSHIFTBAND3",
        ],
    )
    obs = MusesTropomiObservation.create_from_id(
        measurement_id,
        None,
        cs,
        swin_dict[InstrumentIdentifier("TROPOMI")],
        None,
        osp_dir=osp_dir,
        write_tropomi_radiance_pickle=True,
    )
    rad = [
        copy.copy(obs.radiance(0).spectral_range.data),
    ]
    rad[0] *= 0.75
    obssim = SimulatedObservation(obs, rad)
    npt.assert_allclose(obssim.spectral_domain(0).data, obs.spectral_domain(0).data)
    npt.assert_allclose(
        obssim.radiance(0).spectral_range.data,
        obs.radiance(0).spectral_range.data * 0.75,
    )
