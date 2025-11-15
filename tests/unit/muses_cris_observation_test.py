from refractor.muses import (
    MusesRunDir,
    MusesCrisObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    MusesSpectralWindow,
    InstrumentIdentifier,
    FilterIdentifier,
)


def test_create_muses_cris_observation(
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
