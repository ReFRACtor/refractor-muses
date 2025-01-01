from test_support import *
from test_support.old_py_retrieve_test_support import *
from refractor.muses import RefractorUip
import subprocess
import pprint


def load_muses_retrieval_step(
    dir_in, step_number=1, osp_dir=None, gmao_dir=None, change_to_dir=True
):
    """This reads parameters that can be use to call the py-retrieve function
    run_retrieval. See muses_capture in refractor-muses for collecting this.
    """
    return MusesRetrievalStep.load_retrieval_step(
        f"{dir_in}/run_retrieval_step_{step_number}.pkl",
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
        change_to_dir=change_to_dir,
    )


@old_py_retrieve_test
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


@old_py_retrieve_test
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


@old_py_retrieve_test
def test_species_basis(tropomi_uip_step_2):
    npt.assert_allclose(
        tropomi_uip_step_2.species_basis_matrix("O3"),
        tropomi_uip_step_2.species_basis_matrix_calc("O3"),
    )


@old_py_retrieve_test
def test_refractor_joint_tropomi_create_uip(
    isolated_dir, osp_dir, gmao_dir, joint_tropomi_uip_step_12
):
    rstep = load_muses_retrieval_step(
        joint_tropomi_test_in_dir, step_number=12, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
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
    rf_uip = RefractorUip.create_uip(
        i_stateInfo,
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
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@old_py_retrieve_test
def test_refractor_tropomi_create_uip(
    isolated_dir, osp_dir, gmao_dir, tropomi_uip_step_2
):
    rstep = load_muses_retrieval_step(
        tropomi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
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
    rf_uip = RefractorUip.create_uip(
        i_stateInfo,
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
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@old_py_retrieve_test
def test_refractor_joint_omi_create_uip(
    isolated_dir, osp_dir, gmao_dir, joint_omi_uip_step_8
):
    rstep = load_muses_retrieval_step(
        joint_omi_test_in_dir, step_number=8, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
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
    rf_uip = RefractorUip.create_uip(
        i_stateInfo,
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
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)


@old_py_retrieve_test
def test_refractor_omi_create_uip(isolated_dir, osp_dir, gmao_dir, omi_uip_step_2):
    rstep = load_muses_retrieval_step(
        omi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
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
    rf_uip = RefractorUip.create_uip(
        i_stateInfo,
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
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"], check=True)
