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

# Used in tes file. We can keep this, it is optional reading using muses_py as a way to
# test our handling of the  tes files

mpy_read_all_tes = partial(muses_py_wrapper_keep, "read_all_tes")


# Used in get_emis_uwis, but only for older UV1 format. We can pull this over if needed,
# but for now it is reasonable to rely on muses-py for older format data
def mpy_bilinear(*args: Any, **kwargs: Any) -> Any:
    funcname = "UtilMath().bilinear"
    if not have_muses_py:
        raise NameError(
            f"muses_py is not available, so we can't call the function {funcname}"
        )
    return muses_py.UtilMath().bilinear(*args, **kwargs)


# ---- Below are things to replace ----


# Used in muses_observation. It would be good to bring this over, but the input
# code is fairly lengthy. We'll want to look at this at some point.
mpy_radiance_data = partial(muses_py_wrapper, "radiance_data")
mpy_read_airs_l1b = partial(muses_py_wrapper, "read_airs_l1b")
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


# Functions we can keep
__all__ = [
    "mpy_bilinear",
    "mpy_read_all_tes",
]
# Ones we want to replace
__all__.extend(
    [
        "mpy_cdf_read_tes_frequency",
        "mpy_radiance_apodize",
        "mpy_radiance_data",
        "mpy_read_airs_l1b",
        "mpy_read_nasa_cris_fsr",
        "mpy_read_noaa_cris_fsr",
        "mpy_read_omi",
        "mpy_read_tes_l1b",
        "mpy_read_tropomi",
        "mpy_read_tropomi_surface_altitude",
    ]
)
