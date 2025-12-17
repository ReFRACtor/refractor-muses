import numpy as np
import pandas as pd
import numpy.testing as npt
from refractor.muses import RetrievalConfiguration, MeasurementIdFile
from refractor.old_py_retrieve_wrapper import (
    RefractorOmiFmMusesPy,
    RefractorOmiFm,
    RefractorTropOrOmiFmPyRetrieve,
)
from fixtures.require_check import require_muses_py, require_muses_py_fm
import pytest

# ============================================================================
# This set of classes replace the lower level call to omi_fm in
# muses-py. This was used when initially comparing ReFRACtor and muses-py.
#
# This is no longer used, ReFRACtor has completely replaced the foward
# model.
#
# We'll leave these classes here for now, since it can be useful to do
# lower level comparisons. But these should be considered deprecated
# ============================================================================


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
@require_muses_py_fm
@pytest.mark.parametrize("step_number", [1, 2])
def test_refractor_fm_muses_py(
    isolated_dir, step_number, osp_dir, gmao_dir, omi_test_in_dir
):
    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration = 2
    pfile = omi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl"
    r = RefractorOmiFmMusesPy()
    (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag) = (
        r.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="fm_muses_py/",
        )
    )
    # Compare with py_retrieve. Should be identical, since we are doing the
    # same thing here and just going through different plumbing
    r2 = RefractorTropOrOmiFmPyRetrieve(func_name="omi_fm")
    (o_jacobian2, o_radiance2, o_measured_radiance_omi2, o_success_flag2) = (
        r2.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="py_retrieve/",
        )
    )
    assert o_success_flag == 1
    assert o_success_flag2 == 1
    # The logic in pack_omi_jacobian over counts the size of
    # atmosphere jacobians by 1 for each species. This is harmless,
    # it gives an extra row of zeros that then gets trimmed before leaving
    # fm_wrapper. But we need to trim this to compare in our unit test.
    assert np.abs(o_jacobian - o_jacobian2[: o_jacobian.shape[0], :]).max() < 1e-15
    assert np.abs(o_radiance - o_radiance2).max() < 1e-15
    assert (
        np.abs(
            o_measured_radiance_omi["measured_radiance_field"]
            - o_measured_radiance_omi2["measured_radiance_field"]
        ).max()
        < 1e-15
    )
    assert (
        np.abs(
            o_measured_radiance_omi["measured_nesr"]
            - o_measured_radiance_omi2["measured_nesr"]
        ).max()
        < 1e-15
    )


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
@require_muses_py_fm
@pytest.mark.parametrize("step_number", [1, 2])
def test_refractor_fm_refractor(
    isolated_dir,
    step_number,
    osp_dir,
    gmao_dir,
    omi_obs_step_1,
    omi_obs_step_2,
    omi_test_in_dir,
):
    # Just pick an iteration to use. Not sure that we care about looping
    # here.
    iteration = 2
    # Get much better agreement with nstokes=1 for vlidort.
    # vlidort_nstokes=2
    vlidort_nstokes = 1
    pfile = omi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl"
    # Do a lidort run, just to leave PCA out of our checks
    if step_number == 1:
        obs = omi_obs_step_1
    elif step_number == 2:
        obs = omi_obs_step_2
    obs.spectral_window.include_bad_sample = True
    rconf = RetrievalConfiguration.create_from_strategy_file(
        omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    flist = {"OMI": ["UV1", "UV2"]}
    mid = MeasurementIdFile(omi_test_in_dir / "Measurement_ID.asc", rconf, flist)
    r = RefractorOmiFm(
        obs, mid, rconf, use_pca=False, use_lrad=False, lrad_second_order=False
    )
    (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag) = (
        r.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="fm_muses_ref/",
        )
    )
    # Compare with py-retrieve run
    r2 = RefractorOmiFmMusesPy(py_retrieve_vlidort_nstokes=vlidort_nstokes)
    (o_jacobian2, o_radiance2, o_measured_radiance_omi2, o_success_flag2) = (
        r2.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="fm_muses_py/",
        )
    )
    print("******************************************")
    print(o_radiance)
    print(o_radiance2)
    print("   diff:")
    print(o_radiance2 - o_radiance)
    print("   diff %:")
    print((o_radiance2 - o_radiance) / o_radiance2 * 100.0)
    # We should have the differences be better than 0.1%, based on testing.
    # Just check to make sure this doesn't suddenly jump if we break something
    assert np.max(np.abs((o_radiance2 - o_radiance) / o_radiance2 * 100.0)) < 0.1


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
@require_muses_py_fm
def test_refractor_detailed_fm_refractor(
    isolated_dir, osp_dir, gmao_dir, omi_obs_step_2, omi_test_in_dir
):
    """Look at each piece in detail, so make sure we are agreeing"""
    step_number = 2
    iteration = 2
    # Get much better agreement with nstokes=1 for vlidort.
    # vlidort_nstokes=2
    vlidort_nstokes = 1
    pfile = omi_test_in_dir / f"refractor_fm_{step_number}_{iteration}.pkl"
    # Do a lidort run, just to leave PCA out of our checks
    omi_obs_step_2.spectral_window.include_bad_sample = True
    rconf = RetrievalConfiguration.create_from_strategy_file(
        omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    flist = {"OMI": ["UV1", "UV2"]}
    mid = MeasurementIdFile(omi_test_in_dir / "Measurement_ID.asc", rconf, flist)
    r = RefractorOmiFm(
        omi_obs_step_2,
        mid,
        rconf,
        use_pca=False,
        use_lrad=False,
        lrad_second_order=False,
    )
    (o_jacobian, o_radiance, o_measured_radiance_omi, o_success_flag) = (
        r.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="fm_muses_ref/",
        )
    )
    # Compare with py-retrieve run
    r2 = RefractorOmiFmMusesPy(
        py_retrieve_debug=True, py_retrieve_vlidort_nstokes=vlidort_nstokes
    )
    (o_jacobian2, o_radiance2, o_measured_radiance_omi2, o_success_flag2) = (
        r2.run_pickle_file(
            pfile,
            osp_dir=osp_dir,
            gmao_dir=gmao_dir,
            path="fm_muses_py/",
        )
    )

    print("Comparing Raman ring")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.raman_ring_spectrum(do_cloud).spectral_domain.data,
            r2.raman_ring_spectrum(do_cloud).spectral_domain.data,
        )
        npt.assert_allclose(
            r.raman_ring_spectrum(do_cloud).spectral_range.data,
            r2.raman_ring_spectrum(do_cloud).spectral_range.data,
            atol=2e-5,
        )

    print("Comparing surface albedo")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.surface_albedo(do_cloud).spectral_domain.data,
            r2.surface_albedo(do_cloud).spectral_domain.data,
        )
        npt.assert_allclose(
            r.surface_albedo(do_cloud).spectral_range.data,
            r2.surface_albedo(do_cloud).spectral_range.data,
            atol=1e-6,
        )

    print("Comparing geometry")
    for do_cloud in (False, True):
        npt.assert_allclose(r.geometry(do_cloud), r2.geometry(do_cloud), atol=1e-3)

    print("Comparing pressure levels")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.pressure_grid(do_cloud).value,
            r2.pressure_grid(do_cloud).convert("Pa").value,
        )

    print("Comparing temperature levels")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.temperature_grid(do_cloud).value, r2.temperature_grid(do_cloud).value
        )

    print("Comparing altitude levels")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.altitude_grid(do_cloud).value, r2.altitude_grid(do_cloud).value
        )

    print("Comparing gas_number_density")
    for do_cloud in (False, True):
        npt.assert_allclose(
            r.gas_number_density(do_cloud).value, r2.gas_number_density(do_cloud).value
        )

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

    print(
        "Compare rt output. Note that this *is* different, but if all the input agree then the difference is just LIDORT vs VLIDORT"
    )
    print(f"Number vlidort stokes: {vlidort_nstokes}")
    for do_cloud in (False, True):
        spec = r.rt_radiance(do_cloud)
        spec2 = r2.rt_radiance(do_cloud)
        npt.assert_allclose(spec.spectral_domain.data, spec2.spectral_domain.data)
        print(f"   py-retrieve for {'cloud' if do_cloud else 'clear'}:")
        print(spec2.spectral_range.data)
        print(f"   refractor for {'cloud' if do_cloud else 'clear'}:")
        print(spec.spectral_range.data)
        print("   diff:")
        print(spec2.spectral_range.data - spec.spectral_range.data)
        print("   diff %:")
        print(
            (spec2.spectral_range.data - spec.spectral_range.data)
            / spec2.spectral_range.data
            * 100.0
        )

    print(
        "Compare final radiance output. Note that this *is* different, but if all the input agree then the difference is just LIDORT vs VLIDORT"
    )
    print(f"Number vlidort stokes: {vlidort_nstokes}")
    print("   py-retrieve:")
    print(o_radiance2)
    print("   refractor:")
    print(o_radiance)
    print("   diff:")
    print(o_radiance2 - o_radiance)
    print("   diff %:")
    print((o_radiance2 - o_radiance) / o_radiance2 * 100.0)
    # Optionally generate plots. Note this needs pandas and seaborn which isn't strictly
    # a requirement of ReFRACtor, although it is generally available
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme()
        sd = np.hstack(
            [
                r.obj_creator.forward_model.spectral_domain(i).data
                for i in range(r.obj_creator.forward_model.num_channels)
            ]
        )
        d = pd.DataFrame(
            {
                "Wavelength (nm)": sd,
                "ReFRACtor Radiance": o_radiance,
                "py-retrieve Radiance": o_radiance2,
            }
        )
        sns.relplot(
            data=pd.melt(
                d, ["Wavelength (nm)"], value_name="Radiance", var_name="Type"
            ),
            x="Wavelength (nm)",
            y="Radiance",
            hue="Type",
            kind="line",
        ).set(title=f"OMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot1.png", dpi=300)
        d2 = pd.DataFrame(
            {"Wavelength (nm)": sd, "Difference Radiance": o_radiance - o_radiance2}
        )
        sns.relplot(
            data=d2, x="Wavelength (nm)", y="Difference Radiance", kind="line"
        ).set(title=f"OMI Strategy step {step_number}, iteration {iteration}")
        plt.tight_layout()
        plt.savefig("plot2.png", dpi=300)


# Could do FD jacobian like we do with refractor_trop_omi_fm_test.py, if
# we run into any issues
