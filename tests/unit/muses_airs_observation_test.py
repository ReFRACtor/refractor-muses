from refractor.muses import (
    MusesRunDir,
    MusesAirsObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
    osp_setup,
)
from fixtures.require_check import require_muses_py
from fixtures.compare_run import compare_muses_py_dict


def test_create_muses_airs_observation(
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
    swin_dict = MusesSpectralWindow.create_dict_from_file(
        mwfile, rconfig.input_file_helper
    )
    obs = MusesAirsObservation.create_from_id(
        measurement_id,
        None,
        None,
        swin_dict[InstrumentIdentifier("AIRS")],
        None,
        rconfig.input_file_helper,
        osp_dir=osp_dir,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


@require_muses_py
def test_airs_steps(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    import refractor.muses_py as mpy

    filename = (
        joint_omi_test_in_dir.parent
        / "AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    )
    xtrack = 29
    atrack = 49
    windows = [
        {"filter": "2B1"},
        {"filter": "1B2"},
        {"filter": "2A1"},
        {"filter": "1A1"},
    ]

    i_fileid = {}
    i_fileid["preferences"] = {
        "AIRS_filename": str(filename),
        "AIRS_XTrack_Index": xtrack,
        "AIRS_ATrack_Index": atrack,
    }
    with osp_setup(osp_dir):
        o_airs = mpy.read_airs(i_fileid, windows)
    o_airs2 = MusesAirsObservation.read_airs(filename, xtrack, atrack, osp_dir)
    compare_muses_py_dict(o_airs2, o_airs, "read_airs")
