from __future__ import annotations
from . import muses_py  # type: ignore
from functools import partial
from typing import Any

# We have a number of places where we use muses-py. There are a few places where we
# will always want muses-py - for example running the old muses-py forward models. For
# most other places, we want to eventually move this functionality over to refractor-muses.
#
# This module provides exactly what we are calling in muses-py, to make it clear what
# dependencies we still have. Code should import this, rather than muses-py directly.
#
# Note that this does not apply at all to the old_py_retrieve_wrapper package. The whole
# point of that package is to test our code against the old py-retrieve code, so of course
# it calls muses-py all over. The entire submodule depends on having muses-py available.

# TODO - Clean up the calls where we don't want the dependency on muses-py

have_muses_py = muses_py.have_muses_py


def muses_py_wrapper(funcname: str, *args: Any, **kwargs: Any) -> Any:
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return getattr(muses_py, funcname)(*args, **kwargs)


# Synonym, just to make it clear where we have dependencies we intend to keep
muses_py_wrapper_keep = muses_py_wrapper


def muses_py_util_wrapper(funcname: str, *args: Any, **kwargs: Any) -> Any:
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    t = muses_py.UtilList()
    return getattr(t, funcname)(*args, **kwargs)


# Minor function, we need to replace this
mpy_WhereEqualIndices = partial(muses_py_util_wrapper, "WhereEqualIndices")

# Used in cloud_result_summary, we need to replace
mpy_get_one_map = partial(muses_py_wrapper, "get_one_map")
mpy_ccurve_jessica = partial(muses_py_wrapper, "ccurve_jessica")
mpy_quality_deviation = partial(muses_py_wrapper, "quality_deviation")
mpy_compute_cloud_factor = partial(muses_py_wrapper, "compute_cloud_factor")

# Used in column_result_summary
mpy_column = partial(muses_py_wrapper, "column")
mpy_get_diagonal = partial(muses_py_wrapper, "get_diagonal")

# Used in error_analysis, should replace
mpy_get_vector = partial(muses_py_wrapper, "get_vector")
mpy_my_total = partial(muses_py_wrapper, "my_total")
mpy_GetUniqueValues = partial(muses_py_util_wrapper, "GetUniqueValues")

# Used in cost_function. Note that this is only used in old_py_retrieve_wrapper, so we
# don't actually need to replace this. Fine that we depend on muses_py here.
mpy_radiance_data = partial(muses_py_wrapper_keep, "radiance_data")

# Used in muses_forward_model. We don't want to replace these, the dependency is ok.
mpy_fm_oss_stack = partial(muses_py_wrapper_keep, "fm_oss_stack")
mpy_tropomi_fm = partial(muses_py_wrapper_keep, "tropomi_fm")
mpy_omi_fm = partial(muses_py_wrapper_keep, "omi_fm")

# This is used in muses_levmar_solver. I'm not sure, it would be nice to have this
# in refractor so we don't require muses-py to solve. But at the same time, this is
# a pretty central function in muses-py. For now, we import this.
mpy_levmar_nllsq_elanor = partial(muses_py_wrapper_keep, "levmar_nllsq_elanor")

# Uses in muses_observation. It would be good to bring this over, but the input
# code is fairly lengthy. We'll want to look at this at some point.
mpy_read_airs = partial(muses_py_wrapper, "read_airs")
mpy_read_tes_l1b = partial(muses_py_wrapper, "read_tes_l1b")
mpy_radiance_apodize = partial(muses_py_wrapper, "radiance_apodize")
mpy_cdf_read_tes_frequency = partial(muses_py_wrapper, "cdf_read_tes_frequency")
mpy_read_noaa_cris_fsr = partial(muses_py_wrapper, "read_noaa_cris_fsr")
mpy_read_nasa_cris_fsr = partial(muses_py_wrapper, "read_nasa_cris_fsr")
mpy_read_tropomi = partial(muses_py_wrapper, "read_tropomi")
mpy_read_tropomi_surface_altitude = partial(
    muses_py_wrapper, "read_tropomi_surface_altitude"
)
mpy_read_omi = partial(muses_py_wrapper, "read_omi")

# muses_optical_depth
mpy_get_tropomi_o3xsec = partial(muses_py_wrapper, "get_tropomi_o3xsec")
mpy_get_omi_o3xsec = partial(muses_py_wrapper, "get_omi_o3xsec")

# muses_spectral_window - these are used for comparison code and can stay
mpy_table_get_spectral_filename = partial(
    muses_py_wrapper_keep, "table_get_spectral_filename"
)
mpy_table_new_mw_from_step = partial(muses_py_wrapper_keep, "table_new_mw_from_step")
mpy_radiance_get_indices = partial(muses_py_wrapper_keep, "radiance_get_indices")

# order_species, we can keep this as we have a work around for muses_py not being available
mpy_ordered_species_list = partial(muses_py_wrapper_keep, "ordered_species_list")
mpy_atmospheric_species_list = partial(
    muses_py_wrapper_keep, "atmospheric_species_list"
)

# osp_reader, should replace
mpy_make_interpolation_matrix_susan = partial(
    muses_py_wrapper, "make_interpolation_matrix_susan"
)
mpy_supplier_constraint_matrix_ssuba = partial(
    muses_py_wrapper, "supplier_constraint_matrix_ssuba"
)

# osswrapper, can keep
mpy_register_observer_function = partial(
    muses_py_wrapper_keep, "register_observer_function"
)
mpy_pyoss_dir = muses_py.pyoss_dir if have_muses_py else ""
mpy_fm_oss_init = partial(muses_py_wrapper_keep, "fm_oss_init")
mpy_fm_oss_windows = partial(muses_py_wrapper_keep, "fm_oss_windows")
mpy_fm_oss_delete = partial(muses_py_wrapper_keep, "fm_oss_delete")
mpy_struct_combine = partial(muses_py_wrapper, "struct_combine")

# qa_data_handle, should replace
mpy_write_quality_flags = partial(muses_py_wrapper, "write_quality_flags")

# used in state_element_climatology
mpy_make_interpolation_matrix_susan = partial(
    muses_py_wrapper, "make_interpolation_matrix_susan"
)
mpy_supplier_shift_profile = partial(muses_py_wrapper, "supplier_shift_profile")
mpy_supplier_nh3_type_cris = partial(muses_py_wrapper, "supplier_nh3_type_cris")
mpy_supplier_nh3_type_tes = partial(muses_py_wrapper, "supplier_nh3_type_tes")
mpy_supplier_nh3_type_airs = partial(muses_py_wrapper, "supplier_nh3_type_airs")
mpy_supplier_hcooh_type = partial(muses_py_wrapper, "supplier_hcooh_type")

# refractor_capture_directory, have handling for no muses_py so we can keep
mpy_cli_options = muses_py.cli_options if have_muses_py else None

# refractor_uip. We can keep all this, only use UIP for old muses-py forward models.
mpy_update_uip = partial(muses_py_wrapper_keep, "update_uip")
mpy_script_retrieval_ms = partial(muses_py_wrapper_keep, "script_retrieval_ms")
mpy_get_omi_radiance = partial(muses_py_wrapper_keep, "get_omi_radiance")
mpy_get_tropomi_radiance = partial(muses_py_wrapper_keep, "get_tropomi_radiance")
mpy_atmosphere_level = partial(muses_py_wrapper_keep, "atmosphere_level")
mpy_raylayer_nadir = partial(muses_py_wrapper_keep, "raylayer_nadir")
mpy_pressure_sigma = partial(muses_py_wrapper_keep, "pressure_sigma")
mpy_oco2_get_wavelength = partial(muses_py_wrapper_keep, "oco2_get_wavelength")
mpy_nir_match_wavelength_edges = partial(
    muses_py_wrapper_keep, "nir_match_wavelength_edges"
)
mpy_make_uip_master = partial(muses_py_wrapper_keep, "make_uip_master")
mpy_make_uip_airs = partial(muses_py_wrapper_keep, "make_uip_airs")
mpy_make_uip_cris = partial(muses_py_wrapper_keep, "make_uip_cris")
mpy_make_uip_tes = partial(muses_py_wrapper_keep, "make_uip_tes")
mpy_make_uip_omi = partial(muses_py_wrapper_keep, "make_uip_omi")
mpy_make_uip_tropomi = partial(muses_py_wrapper_keep, "make_uip_tropomi")
mpy_make_uip_oco2 = partial(muses_py_wrapper_keep, "make_uip_oco2")

# retrieval_array, should replace
mpy_make_maps = partial(muses_py_wrapper, "make_maps")

# retrieval_strategy - can keep as we only call with have_muses_py
mpy_register_replacement_function = partial(
    muses_py_wrapper_keep, "register_replacement_function"
)

# Used by the various output classes. There should all get replaced, but
# the code is a bit involved.
mpy_cdf_write_dict = partial(muses_py_wrapper, "cdf_write_dict")
mpy_plot_results = partial(muses_py_wrapper, "plot_results")
mpy_plot_radiance = partial(muses_py_wrapper, "plot_radiance")
mpy_cdf_write = partial(muses_py_wrapper, "cdf_write")
mpy_tai = partial(muses_py_wrapper, "tai")
mpy_cdf_var_add_strings = partial(muses_py_wrapper, "cdf_var_add_strings")
mpy_cdf_var_attributes = muses_py.cdf_var_attributes if have_muses_py else {}
mpy_cdf_var_names = partial(muses_py_wrapper, "cdf_var_names")
mpy_GetColumnFromList = partial(muses_py_util_wrapper, "GetColumnFromList")
mpy_cdf_var_map = partial(muses_py_wrapper, "cdf_var_map")
mpy_make_one_lite = partial(muses_py_wrapper, "make_one_lite")


def mpy_ManualArraySetsWithLHSRHSIndices(*args: Any, **kwargs: Any) -> Any:
    funcname = "ManualArraySetsWithLHSRHSIndices"
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return muses_py.UtilGeneral().ManualArraySetsWithLHSRHSIndices(*args, **kwargs)


mpy_specie_type = partial(muses_py_wrapper, "specie_type")
mpy_get_diagonal = partial(muses_py_wrapper, "get_diagonal")


# used in state_element_freq, should get replaced
def mpy_get_emis_dispatcher(*args: Any, **kwargs: Any) -> Any:
    funcname = "get_emis_dispatcher"
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return muses_py.get_emis_uwis.get_emis_dispatcher(*args, **kwargs)


def mpy_emis_source_citation(*args: Any, **kwargs: Any) -> Any:
    funcname = "emis_source_citation"
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return muses_py.get_emis_uwis.UwisCamelOptions.emis_source_citation(*args, **kwargs)


mpy_mw_frequency_needed = partial(muses_py_wrapper, "mw_frequency_needed")
mpy_idl_interpol_1d = partial(muses_py_wrapper, "mpy_idl_interpol_1d")

# used in state_element_gmao, should get replaced
mpy_supplier_surface_pressure = partial(muses_py_wrapper, "supplier_surface_pressure")
mpy_supplier_fm_pressures = partial(muses_py_wrapper, "supplier_fm_pressures")

# Used in state_element_single, should get replaced
mpy_my_interpolate = partial(muses_py_wrapper, "my_interpolate")

# Uses in tes file. We can keep this, it is optional reading using muses_py as a way to
# test our handling of the  tes files

mpy_read_all_tes = partial(muses_py_wrapper_keep, "read_all_tes")

# omi_fm_object_creator. Should replace, but is a bit of a long function.
mpy_get_omi_ils = partial(muses_py_wrapper, "get_omi_ils")
mpy_get_tropomi_ils = partial(muses_py_wrapper, "get_tropomi_ils")
mpy_get_omi_ils_fastconv = partial(muses_py_wrapper, "get_omi_ils_fastconv")
mpy_get_tropomi_ils_fastconv = partial(muses_py_wrapper, "get_tropomi_ils_fastconv")

__all__ = [
    "mpy_WhereEqualIndices",
    "mpy_get_one_map",
    "mpy_ccurve_jessica",
    "mpy_quality_deviation",
    "mpy_compute_cloud_factor",
    "mpy_column",
    "mpy_get_diagonal",
    "mpy_get_vector",
    "mpy_my_total",
    "mpy_GetUniqueValues",
    "mpy_fm_oss_stack",
    "mpy_tropomi_fm",
    "mpy_omi_fm",
    "mpy_levmar_nllsq_elanor",
    "mpy_read_airs",
    "mpy_read_tes_l1b",
    "mpy_radiance_apodize",
    "mpy_cdf_read_tes_frequency",
    "mpy_read_noaa_cris_fsr",
    "mpy_read_nasa_cris_fsr",
    "mpy_read_tropomi",
    "mpy_read_tropomi_surface_altitude",
    "mpy_read_omi",
    "mpy_read_all_tes",
    "mpy_my_interpolate",
    "mpy_supplier_surface_pressure",
    "mpy_supplier_fm_pressures",
    "mpy_get_emis_dispatcher",
    "mpy_emis_source_citation",
    "mpy_mw_frequency_needed",
    "mpy_idl_interpol_1d",
    "mpy_make_interpolation_matrix_susan",
    "mpy_supplier_shift_profile",
    "mpy_supplier_nh3_type_cris",
    "mpy_supplier_nh3_type_tes",
    "mpy_supplier_nh3_type_airs",
    "mpy_supplier_hcooh_type",
    "mpy_get_omi_o3xsec",
    "mpy_get_tropomi_o3xsec",
    "mpy_table_get_spectral_filename",
    "mpy_table_new_mw_from_step",
    "mpy_radiance_get_indices",
    "mpy_ordered_species_list",
    "mpy_atmospheric_species_list",
    "mpy_make_interpolation_matrix_susan",
    "mpy_supplier_constraint_matrix_ssuba",
    "mpy_register_observer_function",
    "mpy_pyoss_dir",
    "mpy_fm_oss_init",
    "mpy_fm_oss_windows",
    "mpy_fm_oss_delete",
    "mpy_struct_combine",
    "mpy_write_quality_flags",
    "mpy_cli_options",
    "mpy_register_replacement_function",
    "mpy_make_maps",
    "mpy_update_uip",
    "mpy_script_retrieval_ms",
    "mpy_get_omi_radiance",
    "mpy_get_tropomi_radiance",
    "mpy_atmosphere_level",
    "mpy_raylayer_nadir",
    "mpy_pressure_sigma",
    "mpy_oco2_get_wavelength",
    "mpy_nir_match_wavelength_edges",
    "mpy_make_uip_master",
    "mpy_make_uip_airs",
    "mpy_make_uip_cris",
    "mpy_make_uip_tes",
    "mpy_make_uip_omi",
    "mpy_make_uip_tropomi",
    "mpy_make_uip_oco2",
    "mpy_cdf_write_dict",
    "mpy_plot_results",
    "mpy_plot_radiance",
    "mpy_ManualArraySetsWithLHSRHSIndices",
    "mpy_cdf_write",
    "mpy_specie_type",
    "mpy_get_diagonal",
    "mpy_get_omi_ils",
    "mpy_get_tropomi_ils",
    "mpy_get_omi_ils_fastconv",
    "mpy_get_tropomi_ils_fastconv",
    "mpy_tai",
    "mpy_cdf_var_add_strings",
    "mpy_cdf_var_attributes",
    "mpy_cdf_var_names",
    "mpy_GetColumnFromList",
    "mpy_cdf_var_map",
    "mpy_make_one_lite",
]
