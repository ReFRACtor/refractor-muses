from test_support import *
import numpy as np
import pandas as pd
import numpy.testing as npt
import os
import refractor.muses.muses_py as mpy
from refractor.old_py_retrieve_wrapper import (RefractorTropOmiFm, RefractorTropOmiFmMusesPy,
                                               RefractorTropOrOmiFmPyRetrieve)
from refractor.tropomi import TropomiFmObjectCreator
import refractor.framework as rf

#============================================================================
# This set of classes replace the lower level call to tropomi_fm in
# muses-py. This was used when initially comparing ReFRACtor and muses-py.
# This has been replaced with RefractorMusesIntegration which is higher
# in the call chain and has a cleaner interface.
# We'll leave these classes here for now, since it can be useful to do
# lower level comparisons. But these should largely be considered deprecated
#============================================================================

@pytest.mark.parametrize("step_number", [1, 2])
@require_muses_py
def test_refractor_fm_muses_py(isolated_dir, step_number, osp_dir, gmao_dir,
                               vlidort_cli):
    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration=2
    pfile = tropomi_test_in_dir + f"/refractor_fm_{step_number}_{iteration}.pkl"
    r = RefractorTropOmiFmMusesPy()
    (o_jacobian, o_radiance,
     o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)
    # Compare with py_retrieve. Should be identical, since we are doing the
    # same thing here and just going through different plumbing
    r2 = RefractorTropOrOmiFmPyRetrieve(func_name="tropomi_fm")
    (o_jacobian2, o_radiance2,
     o_measured_radiance_tropomi2, o_success_flag2) = r2.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="py_retrieve/", vlidort_cli=vlidort_cli)
    assert o_success_flag == 1
    assert o_success_flag2 == 1
    # The logic in pack_tropomi_jacobian over counts the size of
    # atmosphere jacobians by 1 for each species. This is harmless,
    # it gives an extra row of zeros that then gets trimmed before leaving
    # fm_wrapper. But we need to trim this to compare in our unit test.
    assert np.abs(o_jacobian - o_jacobian2[:o_jacobian.shape[0],:]).max() < 1e-15
    assert np.abs(o_radiance - o_radiance2).max() < 1e-15
    assert np.abs(o_measured_radiance_tropomi["measured_radiance_field"] -
                  o_measured_radiance_tropomi2["measured_radiance_field"]).max() < 1e-15
    assert np.abs(o_measured_radiance_tropomi["measured_nesr"] -
                  o_measured_radiance_tropomi2["measured_nesr"]).max() < 1e-15

@pytest.mark.parametrize("step_number", [12,])
@require_muses_py
def test_refractor_joint_fm_muses_py(isolated_dir, step_number, osp_dir,
                                     gmao_dir, vlidort_cli):
    # Note this is the TROPOMI part only, we save stuff after CrIS has been run

    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration=2
    pfile = f"{joint_tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl"
    r = RefractorTropOmiFmMusesPy()
    (o_jacobian, o_radiance,
     o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)
    # Compare with py_retrieve. Should be identical, since we are doing the
    # same thing here and just going through different plumbing
    r2 = RefractorTropOrOmiFmPyRetrieve(func_name="tropomi_fm")
    (o_jacobian2, o_radiance2,
     o_measured_radiance_tropomi2, o_success_flag2) = r2.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="py_retrieve/", vlidort_cli=vlidort_cli)
    assert o_success_flag == 1
    assert o_success_flag2 == 1
    # RefractorTropOmiFmMusesPy does both CRIS and TROPOMI. We could probably pull out
    # just the TROPOMI part in the class, but for now we don't. But the data is in the
    # CRIS then TROPOMI order, so we can just pull out the end.
    # The logic in pack_tropomi_jacobian over counts the size of
    # atmosphere jacobians by 1 for each species. This is harmless,
    # it gives an extra row of zeros that then gets trimmed before leaving
    # fm_wrapper. But we need to trim this to compare in our unit test.
    assert np.abs(o_radiance[-o_radiance2.shape[0]:] - o_radiance2).max() < 1e-15
    assert np.abs(o_measured_radiance_tropomi["measured_radiance_field"] -
                  o_measured_radiance_tropomi2["measured_radiance_field"]).max() < 1e-15
    assert np.abs(o_measured_radiance_tropomi["measured_nesr"] -
                  o_measured_radiance_tropomi2["measured_nesr"]).max() < 1e-15
    # Subset the full jacobian of the joint CRIS and TROPOMI to just the
    # TROPOMI wavelengths
    o_jacobian = o_jacobian[:,-o_radiance2.shape[0]:]
    # Trim the extra row on the jacobian returned by tropomi_fm
    o_jacobian2 = o_jacobian2[:-1,:]
    assert np.abs(o_jacobian - o_jacobian2).max() < 1e-15
    
    
@pytest.mark.parametrize("step_number", [1, 2])
@require_muses_py
def test_refractor_fm_refractor(isolated_dir, step_number, osp_dir, gmao_dir,
                                vlidort_cli, tropomi_obs_step_1):
    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration=2
    # Get much better agreement with nstokes=1 for vlidort.
    #vlidort_nstokes=2
    vlidort_nstokes=1
    pfile = tropomi_test_in_dir + f"/refractor_fm_{step_number}_{iteration}.pkl"
    # Do a lidort run, just to leave PCA out of our checks
    r = RefractorTropOmiFm(tropomi_obs_step_1, use_pca=False, use_lrad=False,
                           lrad_second_order=False)
    (o_jacobian, o_radiance,
     o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_ref/", vlidort_cli=vlidort_cli)
    # Compare with py-retrieve run
    r2 = RefractorTropOmiFmMusesPy(py_retrieve_vlidort_nstokes=vlidort_nstokes)
    (o_jacobian2, o_radiance2,
     o_measured_radiance_tropomi2, o_success_flag2) = r2.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)
    if True:
        # Part of jacobian that comes from the measured_radiance
        print(r2.rf_uip.state_vector_params("TROPOMI"))
        print(o_jacobian[-5:-3, :])
        print(o_jacobian2[-5:-3, :])
    print("******************************************")
    print(o_radiance)
    print(o_radiance2)
    print("   diff:")
    print(o_radiance2-o_radiance)
    print("   diff %:")
    print((o_radiance2-o_radiance) / o_radiance2 * 100.0)
    # We should have the differences be better than 0.1%, based on testing.
    # Just check to make sure this doesn't suddenly jump if we break something
    assert np.max(np.abs((o_radiance2-o_radiance) / o_radiance2 * 100.0)) < 0.15

@pytest.mark.parametrize("step_number", [12,])
@require_muses_py
def test_refractor_joint_fm_refractor(isolated_dir, step_number, osp_dir,
                                      gmao_dir, vlidort_cli,
                                      joint_tropomi_obs_step_12):
    # Note this is the TROPOMI part only, we save stuff after CrIS has been run

    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration=2
    # Get much better agreement with nstokes=1 for vlidort.
    #vlidort_nstokes=2
    vlidort_nstokes=1
    pfile = f"{joint_tropomi_test_in_dir}/refractor_fm_{step_number}_{iteration}.pkl"
    # Do a lidort run, just to leave PCA out of our checks
    obs_cris, obs_tropomi = joint_tropomi_obs_step_12 
    r = RefractorTropOmiFm(obs_tropomi, use_pca=False, use_lrad=False,
                           lrad_second_order=False)
    (o_jacobian, o_radiance,
     o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_ref/", vlidort_cli=vlidort_cli)
    # Compare with py-retrieve run
    r2 = RefractorTropOmiFmMusesPy(py_retrieve_vlidort_nstokes=vlidort_nstokes)
    (o_jacobian2, o_radiance2,
     o_measured_radiance_tropomi2, o_success_flag2) = r2.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)
    # RefractorTropOmiFmMusesPy does both CRIS and TROPOMI. We could probably pull out
    # just the TROPOMI part in the class, but for now we don't. But the data is in the
    # CRIS then TROPOMI order, so we can just pull out the end.
    # The logic in pack_tropomi_jacobian over counts the size of
    # atmosphere jacobians by 1 for each species. This is harmless,
    # it gives an extra row of zeros that then gets trimmed before leaving
    # fm_wrapper. But we need to trim this to compare in our unit test.
    o_radiance2 = o_radiance2[-o_radiance.shape[0]:]
    print(o_radiance)
    print(o_radiance2)
    print("   diff:")
    print(o_radiance2-o_radiance)
    print("   diff %:")
    print((o_radiance2-o_radiance) / o_radiance2 * 100.0)
    # We should have the differences be better than 0.1%, based on testing.
    # Just check to make sure this doesn't suddenly jump if we break something
    assert np.max(np.abs((o_radiance2-o_radiance) / o_radiance2 * 100.0)) < 0.1

    # Optionally generate plots. Note this needs pandas and seaborn which isn't strictly
    # a requirement of ReFRACtor, although it is generally available
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        sns.set_theme()
        sd = r.obj_creator.forward_model.spectral_domain(0).data
        d = pd.DataFrame({"Wavelength (nm)" : sd, "ReFRACtor Radiance" : o_radiance,
                          "py-retrieve Radiance" : o_radiance2})
        sns.relplot(data=pd.melt(d, ["Wavelength (nm)"], value_name="Radiance",
                                 var_name="Type"),
                    x="Wavelength (nm)", y="Radiance", hue='Type',
                    kind="line").set(title=f"CrIS-TROPOMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot1.png", dpi=300)
        d2 = pd.DataFrame({"Wavelength (nm)" : sd,
                           "Difference Radiance" : o_radiance-o_radiance2})
        sns.relplot(data=d2,
                    x="Wavelength (nm)", y="Difference Radiance",
                    kind="line").set(title=f"CrIS-TROPOMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot2.png", dpi=300)
    

@require_muses_py
def test_refractor_detailed_fm_refractor(isolated_dir, osp_dir, gmao_dir,
                                         vlidort_cli, tropomi_obs_step_2):
    '''Look at each piece in detail, so make sure we are agreeing'''
    step_number = 2
    iteration=2
    # Get much better agreement with nstokes=1 for vlidort.
    #vlidort_nstokes=2
    vlidort_nstokes=1
    pfile = tropomi_test_in_dir + f"/refractor_fm_{step_number}_{iteration}.pkl"
    # Do a lidort run, just to leave PCA out of our checks
    r = RefractorTropOmiFm(tropomi_obs_step_2, use_pca=False, use_lrad=False,
                           lrad_second_order=False)
    (o_jacobian, o_radiance,
     o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_ref/", vlidort_cli=vlidort_cli)
    # Compare with py-retrieve run
    r2 = RefractorTropOmiFmMusesPy(py_retrieve_debug=True, py_retrieve_vlidort_nstokes=vlidort_nstokes)
    (o_jacobian2, o_radiance2,
     o_measured_radiance_tropomi2, o_success_flag2) = r2.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)

    print("Comparing Raman ring")
    for do_cloud in (False, True):
        npt.assert_allclose(r.raman_ring_spectrum(do_cloud).spectral_domain.data,
                            r2.raman_ring_spectrum(do_cloud).spectral_domain.data)
        npt.assert_allclose(r.raman_ring_spectrum(do_cloud).spectral_range.data,
                            r2.raman_ring_spectrum(do_cloud).spectral_range.data,
                            atol=2e-6)    

    print("Comparing surface albedo")
    for do_cloud in (False, True):
        npt.assert_allclose(r.surface_albedo(do_cloud).spectral_domain.data,
                            r2.surface_albedo(do_cloud).spectral_domain.data)
        npt.assert_allclose(r.surface_albedo(do_cloud).spectral_range.data,
                            r2.surface_albedo(do_cloud).spectral_range.data,
                            atol=1e-6)

    print("Comparing geometry")
    for do_cloud in (False, True):
        npt.assert_allclose(r.geometry(do_cloud), r2.geometry(do_cloud),
                            atol=1e-3)
    
    print("Comparing pressure levels")
    for do_cloud in (False, True):
        npt.assert_allclose(r.pressure_grid(do_cloud).value,
                            r2.pressure_grid(do_cloud).convert("Pa").value)


    print("Comparing temperature levels")
    for do_cloud in (False, True):
        npt.assert_allclose(r.temperature_grid(do_cloud).value,
                            r2.temperature_grid(do_cloud).value)

    print("Comparing altitude levels")
    for do_cloud in (False, True):
        npt.assert_allclose(r.altitude_grid(do_cloud).value,
                            r2.altitude_grid(do_cloud).value)
        
    print("Comparing gas_number_density")
    for do_cloud in (False, True):
        npt.assert_allclose(r.gas_number_density(do_cloud).value,
                            r2.gas_number_density(do_cloud).value)

    print("Comparing taur")
    for do_cloud in (False, True):
        sd, taur = r.taur(do_cloud)
        sd2, taur2 = r2.taur(do_cloud)
        npt.assert_allclose(sd.data, sd2.data)
        npt.assert_allclose(taur, taur2, rtol=2e-4)

    print("Comparing taug")
    for do_cloud in (False, True):
        sd, taug = r.taug(do_cloud)
        sd2, taug2 = r2.taug(do_cloud)
        npt.assert_allclose(sd.data, sd2.data)
        npt.assert_allclose(taug, taug2, atol=1e-7)
    
    print("Comparing taut")
    for do_cloud in (False, True):
        sd, taut = r.taut(do_cloud)
        sd2, taut2 = r2.taut(do_cloud)
        npt.assert_allclose(sd.data, sd2.data)
        npt.assert_allclose(taut, taut2, rtol=2e-4)

    print("Compare rt output. Note that this *is* different, but if all the input agree then the difference is just LIDORT vs VLIDORT")
    print(f"Number vlidort stokes: {vlidort_nstokes}")
    for do_cloud in (False, True):
        spec = r.rt_radiance(do_cloud)
        spec2 = r2.rt_radiance(do_cloud)
        npt.assert_allclose(spec.spectral_domain.data, spec2.spectral_domain.data)
        print(f"   py-retrieve for {'cloud' if do_cloud else 'clear'}:")
        print(spec2.spectral_range.data)
        print(f"   refractor for {'cloud' if do_cloud else 'clear'}:")
        print(spec.spectral_range.data)
        print( "   diff:")
        print(spec2.spectral_range.data - spec.spectral_range.data)
        print("   diff %:")
        print((spec2.spectral_range.data - spec.spectral_range.data) / spec2.spectral_range.data * 100.0)

    print("Compare final radiance output. Note that this *is* different, but if all the input agree then the difference is just LIDORT vs VLIDORT")
    print(f"Number vlidort stokes: {vlidort_nstokes}")
    print("   py-retrieve:")
    print(o_radiance2)
    print("   refractor:")
    print(o_radiance)
    print("   diff:")
    print(o_radiance2-o_radiance)
    print("   diff %:")
    print((o_radiance2-o_radiance) / o_radiance2 * 100.0)
    # Optionally generate plots. Note this needs pandas and seaborn which isn't strictly
    # a requirement of ReFRACtor, although it is generally available
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        sns.set_theme()
        sd = r.obj_creator.forward_model.spectral_domain(0).data
        d = pd.DataFrame({"Wavelength (nm)" : sd, "ReFRACtor Radiance" : o_radiance,
                          "py-retrieve Radiance" : o_radiance2})
        sns.relplot(data=pd.melt(d, ["Wavelength (nm)"], value_name="Radiance",
                                 var_name="Type"),
                    x="Wavelength (nm)", y="Radiance", hue='Type',
                    kind="line").set(title=f"TROPOMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot1.png", dpi=300)
        d2 = pd.DataFrame({"Wavelength (nm)" : sd,
                           "Difference Radiance" : o_radiance-o_radiance2})
        sns.relplot(data=d2,
                    x="Wavelength (nm)", y="Difference Radiance",
                    kind="line").set(title=f"TROPOMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot2.png", dpi=300)


# We don't normally run these. The FD take a long time to run for all
# the rows. But this is useful to leave around for an initial look at this
# - once we have things verified we can just check that the jacobian
# doesn't change a lot.
@skip    
@pytest.mark.parametrize("index", list(range(34)))
@pytest.mark.parametrize("do_refractor", [True, False])
def test_jac_fd(isolated_dir, osp_dir, gmao_dir, index, do_refractor,
                vlidort_cli):
    '''Look at each piece in detail, so make sure we are agreeing'''
    
    # Just pick a step and iteration. We could loop over these, but
    # I think looking at one point should be good for testing the jacobians
    step_number = 2
    iteration=2
    pfile = tropomi_test_in_dir + f"/refractor_fm_{step_number}_{iteration}.pkl"
    # can check ReFRACtor or py-retrieve
    if do_refractor:
        r = RefractorTropOmiFm(use_pca=False, use_lrad=False,
                               lrad_second_order=False)
        (o_jacobian, o_radiance,
         o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_ref/", vlidort_cli=vlidort_cli)
    else:
        r = RefractorTropOmiFmMusesPy()
        (o_jacobian, o_radiance,
         o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)

    # Front of jacobian is the O3 VMR. Note this is the retrieval vector,
    # not the state vector. We have:
    # 0-24 - O3 levels
    # 25 - Cloud Fraction
    # 26 - TROPOMISURFACEALBEDOBAND3
    # 27 - TROPOMISURFACEALBEDOSLOPEBAND3
    # 28 - TROPOMISURFACEALBEDOSLOPEORDER2BAND3
    # 29 - TROPOMISOLARSHIFTBAND3
    # 30 - TROPOMIRADIANCESHIFTBAND3
    # 31 - TROPOMIRINGSFBAND3
    # 32 - TROPOMITEMPSHIFTBAND3
    # 33 - TROPOMICLOUDSURFACEALBEDO
    nm = ["O3 VMR"] * 25
    nm.extend(["Cloud Fraction", "TROPOMISURFACEALBEDOBAND3", "TROPOMISURFACEALBEDOSLOPEBAND3", "TROPOMISURFACEALBEDOSLOPEORDER2BAND3", "TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3", "TROPOMIRINGSFBAND3", "TROPOMITEMPSHIFTBAND3", "TROPOMICLOUDSURFACEALBEDO"])

    # Delta's are empirical, we don't really have a great way of coming
    # up with the "right" step size.
    delta = [0.01,]*34
    delta[25] = 0.01
    delta[26] = 1e-4
    delta[27] = 1e-4
    delta[28] = 1e-4
    delta[29] = 1e-4
    delta[30] = 1e-4
    delta[31] = 1e-4
    #delta[32] = 
    delta[33] = 1e-4
    jfd, jcalc = r.fd_jac(index, delta[index])
    print(f"Finite difference jacobian for index {index} ({nm[index]}) using delta {delta[index]}:")
    print("  Jac finite difference: ", jfd)
    print("  Jac calculated:        ", jcalc)
    # This will often be a distribution, e.g. a few larger points and lots of
    # smaller ones. Use pandas describe as an simple way to capture this.
    print(f"Finite difference jacobian for index {index} ({nm[index]}) using delta {delta[index]}:")
    print(pd.DataFrame(np.abs(jfd-jcalc)).describe())
    # Optionally generate plots. Note this needs pandas and seaborn which isn't strictly
    # a requirement of ReFRACtor, although it is generally available
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        sns.set_theme()
        sd = TropomiFmObjectCreator(r.rf_uip).forward_model.spectral_domain(0).data
        d = pd.DataFrame({"Wavelength (nm)" : sd, "Jac FD" : jfd,
                          "Jac calc" : jcalc})
        sns.relplot(data=pd.melt(d, ["Wavelength (nm)"], value_name="Jac",
                                 var_name="Type"),
                    x="Wavelength (nm)", y="Jac", hue='Type',
                    kind="line").set(title=f"Index {index} ({nm[index]}) using delta {delta[index]}")
        plt.tight_layout()
        plt.savefig("plot1.png", dpi=300)
        d2 = pd.DataFrame({"Wavelength (nm)" : sd,
                           "Difference Jac" : jfd-jcalc})
        sns.relplot(data=d2,
                    x="Wavelength (nm)", y="Difference Jac",
                    kind="line").set(title=f"Index {index} ({nm[index]}) using delta {delta[index]}")
        plt.tight_layout()
        plt.savefig("plot2.png", dpi=300)

# We don't normally run these. The FD take a long time to run for all
# the rows. But this is useful to leave around for an initial look at this
# - once we have things verified we can just check that the jacobian
# doesn't change a lot.

# Also this one doesn't currently work. We haven't figured out the logic for separating
# out the tropomi part vs. the joint. We don't have a strong reason to work that out,
# so just skip for now
@skip    
@pytest.mark.parametrize("index", list(range(32)))
@pytest.mark.parametrize("do_refractor", [True, False])
def test_jac_joint_fd(isolated_dir, osp_dir, gmao_dir, index, do_refractor,
                      vlidort_cli):
    '''Look at each piece in detail, so make sure we are agreeing'''
    
    # Just pick a step and iteration. We could loop over these, but
    # I think looking at one point should be good for testing the jacobians
    step_number = 12
    iteration=2
    pfile = joint_test_in_dir + f"sounding_1/refractor_fm_{step_number}_{iteration}.pkl"
    # can check ReFRACtor or py-retrieve
    if do_refractor:
        r = RefractorTropOmiFm(use_pca=False, use_lrad=False,
                               lrad_second_order=False)
        (o_jacobian, o_radiance,
         o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_ref/", vlidort_cli=vlidort_cli)
    else:
        r = RefractorTropOmiFmMusesPy()
        (o_jacobian, o_radiance,
         o_measured_radiance_tropomi, o_success_flag) = r.run_pickle_file(pfile,osp_dir=osp_dir,gmao_dir=gmao_dir,path="fm_muses_py/", vlidort_cli=vlidort_cli)

    # Front of jacobian is the O3 VMR. Note this is the retrieval vector,
    # not the state vector. We have:
    # 0-24 - O3 levels
    # 25 - Cloud Fraction
    # 26 - TROPOMISOLARSHIFTBAND3
    # 27 - TROPOMIRADIANCESHIFTBAND3
    # 28 - TROPOMIRADSQUEEZEBAND3
    # 29 - resscale_O0_BAND3
    # 30 - resscale_O1_BAND3
    # 31 - resscale_O2_BAND3
    nm = ["O3 VMR"] * 25
    nm.extend(["Cloud Fraction", "TROPOMISOLARSHIFTBAND3", "TROPOMIRADIANCESHIFTBAND3", "TROPOMIRADSQUEEZEBAND3", "resscale_O0_BAND3", "resscale_O1_BAND3", "resscale_O2_BAND3"])

    # Delta's are empirical, we don't really have a great way of coming
    # up with the "right" step size.
    delta = [0.01,]*34
    delta[25] = 0.01
    delta[26] = 1e-4
    delta[27] = 1e-4
    delta[28] = 1e-4
    delta[29] = 1e-4
    delta[30] = 1e-4
    delta[31] = 1e-4
    jfd, jcalc = r.fd_jac(index, delta[index])
    print(f"Finite difference jacobian for index {index} ({nm[index]}) using delta {delta[index]}:")
    print("  Jac finite difference: ", jfd)
    print("  Jac calculated:        ", jcalc)
    # This will often be a distribution, e.g. a few larger points and lots of
    # smaller ones. Use pandas describe as an simple way to capture this.
    print(f"Finite difference jacobian for index {index} ({nm[index]}) using delta {delta[index]}:")
    print(pd.DataFrame(np.abs(jfd-jcalc)).describe())
    # Optionally generate plots. Note this needs pandas and seaborn which isn't strictly
    # a requirement of ReFRACtor, although it is generally available
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib
        sns.set_theme()
        sd = RefractorObjectCreator(r.rf_uip).forward_model.spectral_domain(0).data
        d = pd.DataFrame({"Wavelength (nm)" : sd, "Jac FD" : jfd,
                          "Jac calc" : jcalc})
        sns.relplot(data=pd.melt(d, ["Wavelength (nm)"], value_name="Jac",
                                 var_name="Type"),
                    x="Wavelength (nm)", y="Jac", hue='Type',
                    kind="line").set(title=f"Index {index} ({nm[index]}) using delta {delta[index]}")
        plt.tight_layout()
        plt.savefig("plot1.png", dpi=300)
        d2 = pd.DataFrame({"Wavelength (nm)" : sd,
                           "Difference Jac" : jfd-jcalc})
        sns.relplot(data=d2,
                    x="Wavelength (nm)", y="Difference Jac",
                    kind="line").set(title=f"Index {index} ({nm[index]}) using delta {delta[index]}")
        plt.tight_layout()
        plt.savefig("plot2.png", dpi=300)
        



