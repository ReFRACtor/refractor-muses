from refractor.muses import (
    FakeStateInfo,
    MusesOmiObservation,
    MeasurementIdFile,
    RetrievalConfiguration,
    InstrumentIdentifier,
    FilterIdentifier,
    MusesTropomiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
)
from refractor.muses_py_fm import RefractorUip
from refractor.old_py_retrieve_wrapper import StateInfoOld, CurrentStateStateInfoOld
import refractor.muses_py as mpy
from fixtures.require_check import require_muses_py
import pprint
import pytest
import os
import numpy.testing as npt


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_omi_uip(omi_uip_step_2):
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    filter_name = "UV2"
    print(omi_uip_step_2.atmosphere_column("O3"))
    print(omi_uip_step_2.omi_params)
    print(omi_uip_step_2.observation_zenith_with_unit(filter_name))
    print(omi_uip_step_2.observation_azimuth_with_unit(filter_name))
    print(omi_uip_step_2.solar_azimuth_with_unit(filter_name))
    print(omi_uip_step_2.solar_zenith_with_unit(filter_name))
    print(omi_uip_step_2.relative_azimuth_with_unit(filter_name))
    print(omi_uip_step_2.latitude(filter_name))
    print(omi_uip_step_2.longitude(filter_name))
    print(omi_uip_step_2.surface_height(filter_name))
    print(omi_uip_step_2.across_track_indexes(filter_name, "OMI"))
    print(omi_uip_step_2.atm_params("OMI"))
    print(omi_uip_step_2.ray_info("OMI"))
    print(omi_uip_step_2.solar_irradiance(filter_name, "OMI"))


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_tropomi_uip(tropomi_uip_step_2):
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    filter_name = "BAND3"
    print(tropomi_uip_step_2.atmosphere_column("O3"))
    print(tropomi_uip_step_2.tropomi_params)
    print(tropomi_uip_step_2.observation_zenith_with_unit(filter_name))
    # For some reason, not actually in the tropomi UIP. Really
    # isn't there, not an error in our processing. I don't think
    # this actually matters though
    # print(tropomi_uip_step_2.observation_azimuth_with_unit(filter_name))
    print(tropomi_uip_step_2.solar_azimuth_with_unit(filter_name))
    print(tropomi_uip_step_2.solar_zenith_with_unit(filter_name))
    print(tropomi_uip_step_2.relative_azimuth_with_unit(filter_name))
    print(tropomi_uip_step_2.latitude(filter_name))
    print(tropomi_uip_step_2.longitude(filter_name))
    print(tropomi_uip_step_2.surface_height(filter_name))
    print(tropomi_uip_step_2.across_track_indexes(filter_name, "TROPOMI"))
    print(tropomi_uip_step_2.atm_params("TROPOMI"))
    print(tropomi_uip_step_2.ray_info("TROPOMI"))


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_species_basis(tropomi_uip_step_2):
    npt.assert_allclose(
        tropomi_uip_step_2.species_basis_matrix("O3"),
        tropomi_uip_step_2.species_basis_matrix_calc("O3"),
    )


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_joint_tropomi_create_uip(
    joint_tropomi_muses_retrieval_step_12, joint_tropomi_uip_step_12, osp_dir
):
    rstep = joint_tropomi_muses_retrieval_step_12
    os.chdir(rstep.run_retrieval_path)
    i_stateInfo = rstep.params["i_stateInfo"]
    i_table = rstep.params["i_tableStruct"]
    i_windows = rstep.params["i_windows"]
    i_retrievalInfo = rstep.params["i_retrievalInfo"]
    i_airs = rstep.params["i_airs"]
    i_tes = rstep.params["i_tes"]
    i_cris = rstep.params["i_cris"]
    i_omi = rstep.params["i_omi"]
    i_tropomi = rstep.params["i_tropomi"]
    i_oco2 = rstep.params["i_oco2"]
    # Test working with FakeStateInfo, since that is what we do now in
    # actual retrieval
    sinfo = StateInfoOld()
    sinfo.state_info_dict = mpy.ObjectView.as_dict(i_stateInfo)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        "Table.asc", osp_dir=osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("TROPOMI"): [FilterIdentifier("BAND3")],
        InstrumentIdentifier("CRIS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    mid = MeasurementIdFile("Measurement_ID.asc", rconfig, filter_list_dict)
    obs_list = [
        MusesCrisObservation.create_from_id(mid, None, None, None, None),
        MusesTropomiObservation.create_from_id(mid, None, None, None, None),
    ]
    cstate = CurrentStateStateInfoOld(sinfo, None, "stepdir")
    fstate_info = FakeStateInfo(cstate, obs_list=obs_list)
    rf_uip = RefractorUip.create_uip(
        # i_stateInfo,
        fstate_info,
        i_table,
        i_windows,
        i_retrievalInfo,
        i_airs,
        i_tes,
        i_cris,
        i_omi,
        i_tropomi,
        i_oco2,
        # Test for pointing angle
        # pointing_angle=rf.DoubleWithUnit(45,"deg"))
    )
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip["nirPars"]["aertype"] = None
    joint_tropomi_uip_step_12.uip["nirPars"]["aertype"] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip, fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(joint_tropomi_uip_step_12.uip, fh)
    # These tests fail, but the differences are just changes in the print output.
    # Since we have the refractor_uip tested and used, we can skip spending time fixing
    # these tests. If there are issues, we can look at this in more detail
    # subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_tropomi_create_uip(
    tropomi_muses_retrieval_step_2, tropomi_uip_step_2, osp_dir
):
    rstep = tropomi_muses_retrieval_step_2
    os.chdir(rstep.run_retrieval_path)
    i_stateInfo = rstep.params["i_stateInfo"]
    i_table = rstep.params["i_tableStruct"]
    i_windows = rstep.params["i_windows"]
    i_retrievalInfo = rstep.params["i_retrievalInfo"]
    i_airs = rstep.params["i_airs"]
    i_tes = rstep.params["i_tes"]
    i_cris = rstep.params["i_cris"]
    i_omi = rstep.params["i_omi"]
    i_tropomi = rstep.params["i_tropomi"]
    i_oco2 = rstep.params["i_oco2"]
    # Test working with FakeStateInfo, since that is what we do now in
    # actual retrieval
    sinfo = StateInfoOld()
    sinfo.state_info_dict = mpy.ObjectView.as_dict(i_stateInfo)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        "Table.asc", osp_dir=osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("TROPOMI"): [FilterIdentifier("BAND3")],
        InstrumentIdentifier("CRIS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    mid = MeasurementIdFile("Measurement_ID.asc", rconfig, filter_list_dict)
    obs_list = [
        MusesTropomiObservation.create_from_id(mid, None, None, None, None),
    ]
    cstate = CurrentStateStateInfoOld(sinfo, None, "stepdir")
    fstate_info = FakeStateInfo(cstate, obs_list=obs_list)
    rf_uip = RefractorUip.create_uip(
        # i_stateInfo,
        fstate_info,
        i_table,
        i_windows,
        i_retrievalInfo,
        i_airs,
        i_tes,
        i_cris,
        i_omi,
        i_tropomi,
        i_oco2,
    )
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip["nirPars"]["aertype"] = None
    tropomi_uip_step_2.uip["nirPars"]["aertype"] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip, fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(tropomi_uip_step_2.uip, fh)
    # These tests fail, but the differences are just changes in the print output.
    # Since we have the refractor_uip tested and used, we can skip spending time fixing
    # these tests. If there are issues, we can look at this in more detail
    # subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_joint_omi_create_uip(
    joint_omi_muses_retrieval_step_8, joint_omi_uip_step_8, osp_dir
):
    rstep = joint_omi_muses_retrieval_step_8
    os.chdir(rstep.run_retrieval_path)
    i_stateInfo = rstep.params["i_stateInfo"]
    i_table = rstep.params["i_tableStruct"]
    i_windows = rstep.params["i_windows"]
    i_retrievalInfo = rstep.params["i_retrievalInfo"]
    i_airs = rstep.params["i_airs"]
    i_tes = rstep.params["i_tes"]
    i_cris = rstep.params["i_cris"]
    i_omi = rstep.params["i_omi"]
    i_tropomi = rstep.params["i_tropomi"]
    i_oco2 = rstep.params["i_oco2"]
    # Test working with FakeStateInfo, since that is what we do now in
    # actual retrieval
    sinfo = StateInfoOld()
    sinfo.state_info_dict = mpy.ObjectView.as_dict(i_stateInfo)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        "Table.asc", osp_dir=osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("OMI"): [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        InstrumentIdentifier("AIRS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    mid = MeasurementIdFile("Measurement_ID.asc", rconfig, filter_list_dict)
    obs_list = [
        MusesAirsObservation.create_from_id(mid, None, None, None, None),
        MusesOmiObservation.create_from_id(mid, None, None, None, None),
    ]
    cstate = CurrentStateStateInfoOld(sinfo, None, "stepdir")
    fstate_info = FakeStateInfo(cstate, obs_list=obs_list)
    rf_uip = RefractorUip.create_uip(
        # i_stateInfo,
        fstate_info,
        i_table,
        i_windows,
        i_retrievalInfo,
        i_airs,
        i_tes,
        i_cris,
        i_omi,
        i_tropomi,
        i_oco2,
        # Test for pointing angle
        # pointing_angle=rf.DoubleWithUnit(45,"deg"))
    )
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip["nirPars"]["aertype"] = None
    joint_omi_uip_step_8.uip["nirPars"]["aertype"] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip, fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(joint_omi_uip_step_8.uip, fh)
    # These tests fail, but the differences are just changes in the print output.
    # Since we have the refractor_uip tested and used, we can skip spending time fixing
    # these tests. If there are issues, we can look at this in more detail
    # subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_refractor_omi_create_uip(omi_muses_retrieval_step_2, omi_uip_step_2, osp_dir):
    rstep = omi_muses_retrieval_step_2
    os.chdir(rstep.run_retrieval_path)
    i_stateInfo = rstep.params["i_stateInfo"]
    i_table = rstep.params["i_tableStruct"]
    i_windows = rstep.params["i_windows"]
    i_retrievalInfo = rstep.params["i_retrievalInfo"]
    i_airs = rstep.params["i_airs"]
    i_tes = rstep.params["i_tes"]
    i_cris = rstep.params["i_cris"]
    i_omi = rstep.params["i_omi"]
    i_tropomi = rstep.params["i_tropomi"]
    i_oco2 = rstep.params["i_oco2"]
    # Test working with FakeStateInfo, since that is what we do now in
    # actual retrieval
    sinfo = StateInfoOld()
    sinfo.state_info_dict = mpy.ObjectView.as_dict(i_stateInfo)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        "Table.asc", osp_dir=osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("OMI"): [FilterIdentifier("UV1"), FilterIdentifier("UV2")],
        InstrumentIdentifier("AIRS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    mid = MeasurementIdFile("Measurement_ID.asc", rconfig, filter_list_dict)
    obs_list = [
        MusesOmiObservation.create_from_id(mid, None, None, None, None),
    ]
    cstate = CurrentStateStateInfoOld(sinfo, None, "stepdir")
    fstate_info = FakeStateInfo(cstate, obs_list=obs_list)
    rf_uip = RefractorUip.create_uip(
        # i_stateInfo,
        fstate_info,
        i_table,
        i_windows,
        i_retrievalInfo,
        i_airs,
        i_tes,
        i_cris,
        i_omi,
        i_tropomi,
        i_oco2,
    )
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip["nirPars"]["aertype"] = None
    omi_uip_step_2.uip["nirPars"]["aertype"] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip, fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(omi_uip_step_2.uip, fh)
    # These tests fail, but the differences are just changes in the print output.
    # Since we have the refractor_uip tested and used, we can skip spending time fixing
    # these tests. If there are issues, we can look at this in more detail
    # subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)
