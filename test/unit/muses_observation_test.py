from refractor.muses import (
    MusesRunDir,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesTropomiObservation,
    MusesOmiObservation,
    MusesTesObservation,
    SimulatedObservation,
    FileFilterMetadata,
    MeasurementIdFile,
    RetrievalConfiguration,
    CurrentStateDict,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
)
import refractor.framework as rf
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


def test_create_muses_airs_observation(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_omi_test_in_dir
):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    # Determined by looking a the full run
    filter_list_dict = {
        InstrumentIdentifier("OMI"): [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        InstrumentIdentifier("AIRS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs = MusesAirsObservation.create_from_id(
        measurement_id,
        None,
        None,
        swin_dict[InstrumentIdentifier("AIRS")],
        None,
        osp_dir=osp_dir,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


def test_muses_tes_observation(isolated_dir, osp_dir, gmao_dir, tes_test_in_dir):
    channel_list = [
        FilterIdentifier("2B1"),
        FilterIdentifier("1B2"),
        FilterIdentifier("2A1"),
        FilterIdentifier("1A1"),
    ]
    l1b_index = [54, 54, 54, 54]
    l1b_avgflag = 0
    run = 2147
    sequence = 388
    scan = 2
    fname = (
        tes_test_in_dir.parent / "TES-Aura_L1B-Nadir_FP2B_r0000002147-o00978_F04_07.h5"
    )
    obs = MusesTesObservation.create_from_filename(
        fname,
        l1b_index,
        l1b_avgflag,
        run,
        sequence,
        scan,
        channel_list,
        osp_dir=osp_dir,
    )
    print(obs)


def test_create_muses_tes_observation(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, tes_test_in_dir
):
    # This depends on files in Susan's OSP. Run if these are available, but
    # don't fail if they aren't
    if not (osp_dir / "Strategy_Tables/ssund").exists():
        pytest.skip("Don't have support files in osp_dir for TES data.")
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(tes_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    # Determined by looking a the full run
    filter_list_dict = {
        InstrumentIdentifier("TES"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ]
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    # This is the microwindows file for step 0, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ssund/OSP-R14/MWDefinitions/Windows_Nadir_EMIS_CLOUDEXT_PAN_PREP.asc"
    )
    fmeta = FileFilterMetadata(
        osp_dir
        / "Strategy_Tables/ssund/Defaults/Default_Spectral_Windows_Definition_File_Filters.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile, filter_metadata=fmeta)
    obs = MusesTesObservation.create_from_id(
        measurement_id,
        None,
        None,
        swin_dict[InstrumentIdentifier("TES")],
        None,
        osp_dir=osp_dir,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


def test_create_muses_tropomi_observation(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_tropomi_test_in_dir
):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
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
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


def test_create_muses_cris_observation(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_tropomi_test_in_dir
):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
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
    obs = MusesCrisObservation.create_from_id(
        measurement_id,
        None,
        None,
        swin_dict[InstrumentIdentifier("CRIS")],
        None,
        osp_dir=osp_dir,
        write_tropomi_radiance_pickle=True,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


def test_create_muses_omi_observation(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_omi_test_in_dir
):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    # Determined by looking a the full run
    filter_list_dict = {
        InstrumentIdentifier("OMI"): [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        InstrumentIdentifier("AIRS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    cs = CurrentStateDict(
        {
            "OMINRADWAVUV1": 0.1,
            "OMINRADWAVUV2": 0.11,
            "OMIODWAVUV1": 0.2,
            "OMIODWAVUV2": 0.21,
            "OMIODWAVSLOPEUV1": 0.3,
            "OMIODWAVSLOPEUV2": 0.31,
        },
        [
            "OMINRADWAVUV1",
        ],
    )
    obs = MusesOmiObservation.create_from_id(
        measurement_id,
        None,
        cs,
        swin_dict[InstrumentIdentifier("OMI")],
        None,
        osp_dir=osp_dir,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


def test_omi_bad_sample(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    )
    cld_filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    obs = MusesOmiObservation.create_from_filename(
        filename,
        xtrack_uv1,
        xtrack_uv2,
        atrack,
        utc_time,
        calibration_filename,
        [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        cld_filename=cld_filename,
        osp_dir=osp_dir,
    )
    sv = rf.StateVector()
    sv.add_observer(obs)
    x2 = [0.01, 0.02, 0.03, 0.04, 0.001, 0.002]
    sv.update_state(x2)
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict[InstrumentIdentifier("OMI")]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(1).data)
    print(obs.spectral_domain(1).sample_index)
    # Check handling of data with bad samples. Should get set to -999
    print(obs.radiance(1).spectral_range.data)


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
