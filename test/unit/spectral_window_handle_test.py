from refractor.muses import (
    MusesRunDir,
    RetrievalConfiguration,
    SpectralWindowHandleSet,
    CurrentStrategyStepDict,
    MeasurementIdFile,
    InstrumentIdentifier,
    FilterIdentifier,
    StrategyStepIdentifier,
)


def test_muses_py_spectral_window_handle(
    osp_dir, isolated_dir, gmao_dir, joint_omi_test_in_dir
):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        f"{r.run_dir}/Table.asc", osp_dir=osp_dir
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
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig, flist)
    swin_handle_set = SpectralWindowHandleSet.default_handle_set()
    swin_handle_set.notify_update_target(mid)
    # For step 8
    current_strategy_step = CurrentStrategyStepDict(
        {
            "retrieval_elements": [
                "H2O",
                "O3",
                "TSUR",
                "CLOUDEXT",
                "PCLOUD",
                "OMICLOUDFRACTION",
                "OMISURFACEALBEDOUV1",
                "OMISURFACEALBEDOUV2",
                "OMISURFACEALBEDOSLOPEUV2",
                "OMINRADWAVUV1",
                "OMINRADWAVUV2",
                "OMIODWAVUV1",
                "OMIODWAVUV2",
            ],
            "strategy_step": StrategyStepIdentifier(8, "H2O,O3_OMI"),
            "max_num_iterations": "15",
            "retrieval_type": "joint",
        },
        mid,
    )
    swin_dict = swin_handle_set.spectral_window_dict(current_strategy_step)
    print(swin_dict)


def test_muses_py_spectral_window_handle_empty_band(
    osp_dir, isolated_dir, gmao_dir, joint_omi_test_in_dir
):
    """Test step 3, which has an empty OMI band, to make sure it is handled correctly"""
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        f"{r.run_dir}/Table.asc", osp_dir=osp_dir
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
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig, flist)
    swin_handle_set = SpectralWindowHandleSet.default_handle_set()
    swin_handle_set.notify_update_target(mid)
    # For step 3
    current_strategy_step = CurrentStrategyStepDict(
        {
            "retrieval_elements": ["OMICLOUDFRACTION"],
            "strategy_step": StrategyStepIdentifier(3, "OMICLOUDFRACTION"),
            "max_num_iterations": "10",
            "retrieval_type": "OMICLOUD_IG_Refine",
        },
        mid,
    )
    swin_dict = swin_handle_set.spectral_window_dict(current_strategy_step)
    print(swin_dict)
