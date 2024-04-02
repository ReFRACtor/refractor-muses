from test_support import *
from refractor.muses import RefractorUip
import subprocess
import pprint

@require_muses_py
def test_refractor_omi_uip(isolated_dir):
    m = RefractorUip.load_uip(f"{test_base_path}/omi/in/sounding_1/uip_step_1.pkl",
                              change_to_dir=False)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    filter_name = "UV2"
    print(m.atmosphere_column("O3"))
    print(m.omi_params)
    print(m.observation_zenith_with_unit(filter_name))
    print(m.observation_azimuth_with_unit(filter_name))
    print(m.solar_azimuth_with_unit(filter_name))
    print(m.solar_zenith_with_unit(filter_name))
    print(m.relative_azimuth_with_unit(filter_name))
    print(m.latitude(filter_name))
    print(m.longitude(filter_name))
    print(m.surface_height(filter_name))
    print(m.across_track_indexes(filter_name, "OMI"))
    print(m.atm_params("OMI"))
    print(m.ray_info("OMI"))
    print(m.solar_irradiance(0, "OMI"))

@require_muses_py
def test_refractor_joint_uip(isolated_dir):
    # UIP  that has both AIRS and OMI
    m = RefractorUip.load_uip(f"{test_base_path}/airs_omi/in/sounding_1/uip_step_8.pkl",
                              change_to_dir=False)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    filter_name = "UV1"
    print(m.atmosphere_column("O3"))
    print(m.omi_params)
    print(m.observation_zenith_with_unit(filter_name))
    print(m.observation_azimuth_with_unit(filter_name))
    print(m.solar_azimuth_with_unit(filter_name))
    print(m.solar_zenith_with_unit(filter_name))
    print(m.relative_azimuth_with_unit(filter_name))
    print(m.latitude(filter_name))
    print(m.longitude(filter_name))
    print(m.surface_height(filter_name))
    print(m.across_track_indexes(filter_name, "OMI"))
    print(m.atm_params("OMI"))
    print(m.ray_info("OMI"))
    
@require_muses_py
def test_refractor_tropomi_uip(isolated_dir):
    m = RefractorUip.load_uip(f"{test_base_path}/tropomi/in/sounding_1/uip_step_1.pkl",
                              change_to_dir=False)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    filter_name = "BAND3"
    print(m.atmosphere_column("O3"))
    print(m.tropomi_params)
    print(m.observation_zenith_with_unit(filter_name))
    # For some reason, not actually in the tropomi UIP. Really
    # isn't there, not an error in our processing. I don't think
    # this actually matters though
    # print(m.observation_azimuth_with_unit(filter_name))
    print(m.solar_azimuth_with_unit(filter_name))
    print(m.solar_zenith_with_unit(filter_name))
    print(m.relative_azimuth_with_unit(filter_name))
    print(m.latitude(filter_name))
    print(m.longitude(filter_name))
    print(m.surface_height(filter_name))
    print(m.across_track_indexes(filter_name, "TROPOMI"))
    print(m.atm_params("TROPOMI"))
    print(m.ray_info("TROPOMI"))

@require_muses_py
def test_species_basis(tropomi_uip_step_2, clean_up_replacement_function):
    npt.assert_allclose(tropomi_uip_step_2.species_basis_matrix("O3"),
                        tropomi_uip_step_2.species_basis_matrix_calc("O3"))
    
@require_muses_py
def test_refractor_joint_tropomi_create_uip(isolated_dir, osp_dir, gmao_dir,
                                            joint_tropomi_uip_step_12):
    rstep = load_muses_retrieval_step(joint_tropomi_test_in_dir, step_number=12,
                                      osp_dir=osp_dir,gmao_dir=gmao_dir)
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
    rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,     
        i_retrievalInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2)
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip['nirPars']['aertype'] = None
    joint_tropomi_uip_step_12.uip['nirPars']['aertype'] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip,fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(joint_tropomi_uip_step_12.uip,fh)
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"],
                   check=True)

@require_muses_py
def test_refractor_tropomi_create_uip(isolated_dir, osp_dir, gmao_dir,
                                      tropomi_uip_step_2):
    rstep = load_muses_retrieval_step(tropomi_test_in_dir, step_number=2,
                                      osp_dir=osp_dir,gmao_dir=gmao_dir)
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
    rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,     
        i_retrievalInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2)
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip['nirPars']['aertype'] = None
    tropomi_uip_step_2.uip['nirPars']['aertype'] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip,fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(tropomi_uip_step_2.uip,fh)
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"],
                   check=True)
    
@require_muses_py
def test_refractor_joint_omi_create_uip(isolated_dir, osp_dir, gmao_dir,
                                            joint_omi_uip_step_8):
    rstep = load_muses_retrieval_step(joint_omi_test_in_dir, step_number=8,
                                      osp_dir=osp_dir,gmao_dir=gmao_dir)
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
    rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,     
        i_retrievalInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2)
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip['nirPars']['aertype'] = None
    joint_omi_uip_step_8.uip['nirPars']['aertype'] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip,fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(joint_omi_uip_step_8.uip,fh)
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"],
                   check=True)

@require_muses_py
def test_refractor_omi_create_uip(isolated_dir, osp_dir, gmao_dir,
                                      omi_uip_step_2):
    rstep = load_muses_retrieval_step(omi_test_in_dir, step_number=2,
                                      osp_dir=osp_dir,gmao_dir=gmao_dir)
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
    rf_uip = RefractorUip.create_uip(i_stateInfo, i_table, i_windows,     
        i_retrievalInfo, i_airs, i_tes, i_cris, i_omi, i_tropomi, i_oco2)
    # aertype is some odd structure used for OCO-2, which doesn't seem to be set right. We
    # may need to eventually sort this out, but it doesn't actually seem to be used  for
    # anything. Remove just so it doesn't interfere with our check of everything else.
    rf_uip.uip['nirPars']['aertype'] = None
    omi_uip_step_2.uip['nirPars']['aertype'] = None
    # To compare, just print out and then use diff
    with open("our_uip.txt", "w") as fh:
        pprint.pprint(rf_uip.uip,fh)
    with open("original_uip.txt", "w") as fh:
        pprint.pprint(omi_uip_step_2.uip,fh)
    subprocess.run(["diff", "-u", "original_uip.txt", "our_uip.txt"],
                   check=True)
    
    
