from refractor.muses import (
    MusesRunDir,
    MusesCrisObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
    osp_setup,
)
from fixtures.require_check import require_muses_py
from fixtures.compare_run import compare_muses_py_dict


def test_create_muses_cris_observation(
    isolated_dir, ifile_hlp, joint_tropomi_test_in_dir
):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_tropomi_test_in_dir, ifile_hlp)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", ifile_hlp=ifile_hlp
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
        ifile_hlp.osp_dir
        / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(
        mwfile, rconfig.input_file_helper
    )
    obs = MusesCrisObservation.create_from_id(
        measurement_id,
        None,
        None,
        swin_dict[InstrumentIdentifier("CRIS")],
        None,
        rconfig.input_file_helper,
    )
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    print(obs.surface_altitude)


@require_muses_py
def test_cris_steps(isolated_dir, ifile_hlp, joint_tropomi_test_in_dir):
    import refractor.muses_py as mpy

    filename = (
        joint_tropomi_test_in_dir.parent
        / "nasa_fsr_SNDR.SNPP.CRIS.20190807T0624.m06.g065.L1B.std.v02_22.G.190905161252.nc"
    )
    xtrack = 8
    atrack = 4
    pixel_index = 5

    i_fileid = {
        "CRIS_filename": str(filename),
        "CRIS_XTrack_Index": xtrack,
        "CRIS_ATrack_Index": atrack,
        "CRIS_Pixel_Index": pixel_index,
    }
    with osp_setup(ifile_hlp):
        o_cris = mpy.read_nasa_cris_fsr(i_fileid)
    o_cris2 = MusesCrisObservation.read_cris(
        filename, xtrack, atrack, pixel_index, ifile_hlp
    )
    compare_muses_py_dict(o_cris2, o_cris, "read_cris")
