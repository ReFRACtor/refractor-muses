from refractor.muses import (
    MusesRunDir,
    MusesTropomiObservation,
    MusesOmiObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    CurrentStateDict,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
    osp_setup,
)
import refractor.framework as rf
from fixtures.require_check import require_muses_py
from fixtures.compare_run import compare_muses_py_dict


def test_create_muses_tropomi_observation(
    isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir
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


def test_create_muses_omi_observation(
    isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir
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


@require_muses_py
def test_tropomi_steps(isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir):
    import refractor.muses.muses_py as mpy

    filename_dict = {
        "CLOUD": str(
            joint_tropomi_test_in_dir.parent
            / "S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
        ),
        "BAND3": str(
            joint_tropomi_test_in_dir.parent
            / "S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc"
        ),
        "IRR_BAND_1to6": str(
            joint_tropomi_test_in_dir.parent
            / "S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
        ),
    }
    xtrack_dict = {"CLOUD": "226", "BAND3": "226", "IRR_BAND_1to6": "226"}
    atrack_dict = {"CLOUD": "2995", "BAND3": "2995"}
    windows = [{"instrument": "TROPOMI", "filter": "BAND3"}]
    utc_time = "2019-08-07T06:24:33.584090Z"
    erad = mpy.combine_tropomi_erad(filename_dict, xtrack_dict, atrack_dict, windows)
    erad2 = MusesTropomiObservation.combine_tropomi_erad(
        filename_dict, xtrack_dict, atrack_dict, windows
    )
    compare_muses_py_dict(erad2, erad, "erad")

    with osp_setup(osp_dir):
        o_tropomi = mpy.read_tropomi(
            filename_dict, xtrack_dict, atrack_dict, utc_time, windows
        )
        for i in range(len(o_tropomi["Earth_Radiance"]["ObservationTable"]["ATRACK"])):
            surfaceAltitude = mpy.read_tropomi_surface_altitude(
                o_tropomi["Earth_Radiance"]["ObservationTable"]["Latitude"][i],
                o_tropomi["Earth_Radiance"]["ObservationTable"]["Longitude"][i],
            )
            o_tropomi["Earth_Radiance"]["ObservationTable"]["TerrainHeight"][i] = (
                surfaceAltitude
            )

    o_tropomi2 = MusesTropomiObservation.read_tropomi(
        filename_dict, xtrack_dict, atrack_dict, utc_time, windows, osp_dir=osp_dir
    )
    compare_muses_py_dict(o_tropomi2, o_tropomi, "read_tropomi")


@require_muses_py
def test_omi_steps(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    import refractor.muses.muses_py as mpy

    filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    )
    cld_filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    xtrack_uv2 = 20
    atrack = 1139
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"

    with osp_setup(osp_dir):
        o_omi = mpy.read_omi(
            str(filename),
            xtrack_uv2,
            atrack,
            utc_time,
            str(calibration_filename),
            cldFilename=str(cld_filename),
        )
    o_omi2 = MusesOmiObservation.read_omi(
        filename,
        xtrack_uv2,
        atrack,
        utc_time,
        calibration_filename,
        cld_filename,
        osp_dir,
    )

    compare_muses_py_dict(o_omi2, o_omi, "read_omi")
