from refractor.muses import (
    MusesRunDir,
    RetrievalConfiguration,
    SpectralWindowHandleSet,
    CurrentStrategyStepOEImp,
    MeasurementIdFile,
    InstrumentIdentifier,
    FilterIdentifier,
    StrategyStepIdentifier,
    RetrievalType,
    StateElementIdentifier,
    CreatorDict,
)


def test_muses_py_spectral_window_handle(
    ifile_hlp, isolated_dir, gmao_dir, joint_omi_test_in_dir
):
    r = MusesRunDir(joint_omi_test_in_dir, ifile_hlp)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        f"{r.run_dir}/Table.asc", ifile_hlp=ifile_hlp
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
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig)
    cdict = CreatorDict()
    strategy_context = cdict.strategy_context
    strategy_context.update_strategy_context(
        measurement_id=mid,
        retrieval_config=rconfig,
        filter_list_dict=flist,
        creator_dict=cdict,
    )
    swin_handle_set = SpectralWindowHandleSet.default_handle_set_with_context(
        strategy_context
    )
    # For step 8
    current_strategy_step = CurrentStrategyStepOEImp(
        strategy_context,
        swin_handle_set,
        RetrievalType("joint"),
        [
            StateElementIdentifier("H2O"),
            StateElementIdentifier("O3"),
            StateElementIdentifier("TSUR"),
            StateElementIdentifier("CLOUDEXT"),
            StateElementIdentifier("PCLOUD"),
            StateElementIdentifier("OMICLOUDFRACTION"),
            StateElementIdentifier("OMISURFACEALBEDOUV1"),
            StateElementIdentifier("OMISURFACEALBEDOUV2"),
            StateElementIdentifier("OMISURFACEALBEDOSLOPEUV2"),
            StateElementIdentifier("OMINRADWAVUV1"),
            StateElementIdentifier("OMINRADWAVUV2"),
            StateElementIdentifier("OMIODWAVUV1"),
            StateElementIdentifier("OMIODWAVUV2"),
        ],
        StrategyStepIdentifier(8, "H2O,O3_OMI"),
        {},
        [],
        [],
        [],
        None,
    )

    swin_dict = swin_handle_set.spectral_window_dict(
        current_strategy_step, strategy_context.filter_list_dict
    )
    print(swin_dict)


def test_muses_py_spectral_window_handle_empty_band(
    ifile_hlp, isolated_dir, gmao_dir, joint_omi_test_in_dir
):
    """Test step 3, which has an empty OMI band, to make sure it is handled correctly"""
    r = MusesRunDir(joint_omi_test_in_dir, ifile_hlp)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        f"{r.run_dir}/Table.asc", ifile_hlp=ifile_hlp
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
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig)
    cdict = CreatorDict()
    strategy_context = cdict.strategy_context
    strategy_context.update_strategy_context(
        measurement_id=mid,
        retrieval_config=rconfig,
        filter_list_dict=flist,
        creator_dict=cdict,
    )
    swin_handle_set = SpectralWindowHandleSet.default_handle_set_with_context(
        strategy_context
    )
    # For step 3
    current_strategy_step = CurrentStrategyStepOEImp(
        strategy_context,
        swin_handle_set,
        RetrievalType("OMICLOUD_IG_Refine"),
        [StateElementIdentifier("OMICLOUDFRACTION")],
        StrategyStepIdentifier(3, "OMICLOUDFRACTION"),
        {},
        [],
        [],
        [],
        None,
    )
    swin_dict = swin_handle_set.spectral_window_dict(
        current_strategy_step, strategy_context.filter_list_dict
    )
    print(swin_dict)
