from __future__ import annotations
from contextlib import contextmanager
import tempfile
import os
import math
from typing import Generator


@contextmanager
def osp_setup(
    osp_dir: str | os.PathLike[str] | None = None,
) -> Generator[None, None, None]:
    """Some of the readers assume the OSP is available as "../OSP". We
    are trying to get away from assuming we are in a run directory
    whenever we do things, it limits using the code in various
    contexts.  So this handles things by taking the osp_dir and
    setting up a temporary directory so things look like muses_py
    assumes.

    We can perhaps just move the muses-py code over at some point and
    handle this more cleanly, but for now we do this.
    """
    if osp_dir is None:
        dname = os.path.abspath("../OSP")
    else:
        dname = os.path.abspath(str(osp_dir))
    curdir = os.path.abspath(os.path.curdir)
    try:
        with tempfile.TemporaryDirectory() as tname:
            os.chdir(tname)
            os.symlink(dname, "OSP")
            os.mkdir("subdir")
            os.chdir("./subdir")
            yield
    finally:
        os.chdir(curdir)


def greatcircle(
    lat_deg1: float, lon_deg1: float, lat_deg2: float, lon_deg2: float
) -> float:
    """Return great circle distance in meters"""
    x1 = math.radians(lat_deg1)
    y1 = math.radians(lon_deg1)
    x2 = math.radians(lat_deg2)
    y2 = math.radians(lon_deg2)

    # Compute using the Haversine formula.
    a = math.sin((x2 - x1) / 2.0) ** 2.0 + (
        math.cos(x1) * math.cos(x2) * (math.sin((y2 - y1) / 2.0) ** 2.0)
    )

    # Great circle distance in radians
    angle = 2.0 * math.asin(min(1.0, math.sqrt(a)))

    # Each degree on a great circle of Earth is 60 nautical miles.
    distance = 60.0 * math.degrees(angle)
    conversion_factor = 0.5390007480064188  # nautical miles -> kilometers
    distance = (distance / conversion_factor) * 1000
    return distance


__all__ = ["osp_setup", "greatcircle"]
