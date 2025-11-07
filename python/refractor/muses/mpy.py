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
mpy_radiance_data = partial(muses_py_wrapper, "radiance_data")

# Used in muses_forward_model. We don't want to replace these, the dependency is ok.
mpy_fm_oss_stack = partial(muses_py_wrapper, "fm_oss_stack")
mpy_tropomi_fm = partial(muses_py_wrapper, "tropomi_fm")
mpy_omi_fm = partial(muses_py_wrapper, "omi_fm")

# This is used in muses_levmar_solver. I'm not sure, it would be nice to have this
# in refractor so we don't require muses-py to solve. But at the same time, this is
# a pretty central function in muses-py. For now, we import this.
mpy_levmar_nllsq_elanor = partial(muses_py_wrapper, "levmar_nllsq_elanor")

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
]
