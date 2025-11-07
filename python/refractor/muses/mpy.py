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

__all__ = [
    "mpy_WhereEqualIndices",
    "mpy_get_one_map",
    "mpy_ccurve_jessica",
    "mpy_quality_deviation",
    "mpy_compute_cloud_factor",
]
