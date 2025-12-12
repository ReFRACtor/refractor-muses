from __future__ import annotations
import refractor.muses_py as muses_py  # type: ignore
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
# have_muses_py = False


def muses_py_wrapper(funcname: str, *args: Any, **kwargs: Any) -> Any:
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return getattr(muses_py, funcname)(*args, **kwargs)


# Synonym, just to make it clear where we have dependencies we intend to keep
muses_py_wrapper_keep = muses_py_wrapper

# Used in muses_forward_model. We don't want to replace these, the dependency is ok.
mpy_fm_oss_stack = partial(muses_py_wrapper_keep, "fm_oss_stack")
mpy_tropomi_fm = partial(muses_py_wrapper_keep, "tropomi_fm")
mpy_omi_fm = partial(muses_py_wrapper_keep, "omi_fm")

# Used by muses_py_call, have handling for no muses_py so we can keep
mpy_cli_options = muses_py.cli_options if have_muses_py else None
mpy_pyoss_dir = muses_py.pyoss_dir if have_muses_py else ""

# Used in osswrapper, can keep
mpy_register_observer_function = partial(
    muses_py_wrapper_keep, "register_observer_function"
)
mpy_pyoss_dir = muses_py.pyoss_dir if have_muses_py else ""
mpy_fm_oss_init = partial(muses_py_wrapper_keep, "fm_oss_init")
mpy_fm_oss_windows = partial(muses_py_wrapper_keep, "fm_oss_windows")
mpy_fm_oss_delete = partial(muses_py_wrapper_keep, "fm_oss_delete")

# Used by refractor_uip. We can keep all this, only use UIP for old muses-py forward models.
mpy_update_uip = partial(muses_py_wrapper_keep, "update_uip")
mpy_script_retrieval_ms = partial(muses_py_wrapper_keep, "script_retrieval_ms")
mpy_make_maps = partial(muses_py_wrapper, "make_maps")
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

# Functions we can keep
__all__ = [
    "mpy_atmosphere_level",
    "mpy_cli_options",
    "mpy_fm_oss_delete",
    "mpy_fm_oss_init",
    "mpy_fm_oss_stack",
    "mpy_fm_oss_windows",
    "mpy_get_omi_radiance",
    "mpy_get_tropomi_radiance",
    "mpy_make_maps",
    "mpy_make_uip_airs",
    "mpy_make_uip_cris",
    "mpy_make_uip_master",
    "mpy_make_uip_oco2",
    "mpy_make_uip_omi",
    "mpy_make_uip_tes",
    "mpy_make_uip_tropomi",
    "mpy_nir_match_wavelength_edges",
    "mpy_oco2_get_wavelength",
    "mpy_omi_fm",
    "mpy_pressure_sigma",
    "mpy_pyoss_dir",
    "mpy_raylayer_nadir",
    "mpy_register_observer_function",
    "mpy_script_retrieval_ms",
    "mpy_tropomi_fm",
    "mpy_update_uip",
]
