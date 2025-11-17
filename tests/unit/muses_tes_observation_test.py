from refractor.muses import (
    MusesRunDir,
    MusesTesObservation,
    FileFilterMetadata,
    MeasurementIdFile,
    RetrievalConfiguration,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
    osp_setup,
)
import pytest
from fixtures.require_check import require_muses_py
from fixtures.compare_run import compare_muses_py_dict


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


def test_create_muses_tes_observation(isolated_dir, osp_dir, gmao_dir, tes_test_in_dir):
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


@require_muses_py
def test_tes_steps(isolated_dir, osp_dir, gmao_dir, tes_test_in_dir):
    import refractor.muses.muses_py as mpy

    filename = (
        tes_test_in_dir.parent / "TES-Aura_L1B-Nadir_FP2B_r0000002147-o00978_F04_07.h5"
    )
    l1b_index = [54, 54, 54, 54]
    l1b_avgflag = 0
    windows = [
        {"filter": "2B1"},
        {"filter": "1B2"},
        {"filter": "2A1"},
        {"filter": "1A1"},
    ]

    i_fileid = {}
    i_fileid["preferences"] = {
        "TES_filename_L1B": str(filename),
        "TES_filename_L1B_Index": l1b_index,
        "TES_L1B_Average_Flag": l1b_avgflag,
    }
    with osp_setup(osp_dir):
        o_tes = mpy.read_tes_l1b(i_fileid, windows)
    o_tes2 = MusesTesObservation.read_tes(
        filename, l1b_index, l1b_avgflag, windows, osp_dir
    )

    compare_muses_py_dict(o_tes2, o_tes, "read_tes")
