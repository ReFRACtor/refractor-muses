from __future__ import annotations
from contextlib import contextmanager
from .input_file_helper import InputFileHelper
import tempfile
import os
import math
from collections import UserDict
from collections.abc import MutableMapping
import numpy as np
import copy
from typing import Generator, Any, Iterator


# TODO Once we clean up input, get rid of this function
@contextmanager
def osp_setup(
    ifile_hlp: InputFileHelper | None = None,
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
    if ifile_hlp is None:
        ifile_hlp = InputFileHelper()
    curdir = os.path.abspath(os.path.curdir)
    try:
        with tempfile.TemporaryDirectory() as tname:
            os.chdir(tname)
            os.symlink(str(ifile_hlp.osp_dir.path_for_muses_py), "OSP")
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


class AttrDictAdapter(MutableMapping):
    """muses-py often uses ObjectView, which makes dictionary object access the
    values using attributes. This is just syntactic sugar, but it is used in a lot
    of places. This is essentially like the old attrdict library - but this has been
    deprecated. This is a simple adapter to do the same thing. This is used for example
    by the muses-py uip."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.__dict__ = data

    # ObjectView has these two functions, supply since lower level routines may look
    # for these.
    @classmethod
    def as_dict(cls, obj: AttrDictAdapter | dict[str, Any]) -> dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        return obj.__dict__

    @classmethod
    def as_object(cls, obj: AttrDictAdapter | dict[str, Any]) -> AttrDictAdapter:
        if isinstance(obj, dict):
            return AttrDictAdapter(obj)
        return obj

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getattr__(self, name: str) -> Any:
        # Just defined to make mypy happy. We shouldn't actually call this
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name: str, v: Any) -> None:
        # Just defined to make mypy happy. We don't actually do anything special
        super().__setattr__(name, v)


class ResultIrk(UserDict):
    """This holds the results of IRK. This is basically just a dict,
    with a few extra functions. We can perhaps create a proper class
    at some point, but right now there isn't much of a need for this.
    The old py-retrieve code that uses ResultIrk is expecting a dict."""

    def __getattr__(self, nm: str) -> Any:
        if nm in self.data:
            return self[nm]
        raise AttributeError()

    def __setattr__(self, nm: str, value: Any) -> None:
        if nm in ("data",):
            super().__setattr__(nm, value)
        if nm in self.data:
            self[nm] = value
        else:
            super().__setattr__(nm, value)

    def get_state(self) -> dict[str, Any]:
        res = dict(copy.deepcopy(self.data))
        for k in res.keys():
            if k in (
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
            ):
                res[k] = res[k].tolist()
            if k == "radiances":
                for k2 in ("radarr_fm", "freq_fm", "rad_L1b", "freq_L1b"):
                    res[k][k2] = res[k][k2].tolist()
            if k not in (
                "flux",
                "flux_l1b",
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
                "radiances",
            ):
                for k2 in (
                    "irfk",
                    "lirfk",
                    "pressure",
                    "irfk_segs",
                    "lirfk_segs",
                    "vmr",
                ):
                    if res[k][k2] is None:
                        res[k][k2] = None
                    else:
                        res[k][k2] = res[k][k2].tolist()
        return res

    def set_state(self, d: dict[str, Any]) -> None:
        self.data = d
        for k in self.data.keys():
            if k in (
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
            ):
                self.data[k] = np.array(self.data[k])
            if k == "radiances":
                for k2 in ("radarr_fm", "freq_fm", "rad_L1b", "freq_L1b"):
                    self.data[k][k2] = np.array(self.data[k][k2])
            if k not in (
                "flux",
                "flux_l1b",
                "fluxSegments",
                "freqSegments",
                "fluxSegments_l1b",
                "freqSegments_irk",
                "radiances",
            ):
                for k2 in (
                    "irfk",
                    "lirfk",
                    "pressure",
                    "irfk_segs",
                    "lirfk_segs",
                    "vmr",
                ):
                    self.data[k][k2] = np.array(self.data[k][k2])


__all__ = ["osp_setup", "greatcircle", "AttrDictAdapter", "ResultIrk"]
