from __future__ import annotations
from .templated_output import TemplatedOutput
import os
import numpy as np
from .declaritve_output import register_dataset, DeclarativeOutput
import typing
import xarray
from pathlib import Path
from functools import cache
from datetime import datetime
import astropy.time
if typing.TYPE_CHECKING:
    from .ml import MlPredictionClass


# Skip this for now. The file definition doesn't have this, and it doesn't currently
# work. We can possibly come back to this, so we will leave in place
class _DimColumn:
    """Make dim_column a enumeration"""

    def __init__(self):
        self.__self__ = self

    def _creator(self, t, ds, var_name) -> None:
        en = ds.createEnumType(
            np.uint8,
            "dim_column",
            {"Column": 0, "Trop": 1, "UpperTrop": 2, "LowerTrop": 3, "Strato": 4},
        )
        ds.createVariable(var_name, en, ("dim_column",))

    def __call__(self) -> np.ndarray:
        return np.array([0, 1, 2, 3, 4], dtype=np.uint8)


class ColumnCoFile(DeclarativeOutput):
    """We might try to get a general file rather than hardcoding CO, but start
    simply for now."""

    def __init__(
        self,
        prediction: MlPredictionClass,
        cris_input_file: list[str | os.PathLike[str]],
        output_filename: str | os.PathLike[str],
        pspec: str | os.PathLike[str],
    ) -> None:
        # We might pull the cris input stuff into a separate class. But isn't clear
        # yet exactly how general these files are, so now we keep this in here.
        # Once we have everything in place we can look at possibly redesigning things
        self.input_path = [Path(s) for s in cris_input_file]
        self.prediction = prediction
        self.output = TemplatedOutput(pspec, output_filename)
        self.output.register_instances((self,))
        # See comment above, for now skip the enumeration metadata
        # self.output.register_variable("dim_column", _DimColumn())

    # To fill in:
    # /altitude
    # /geolocation/cris_view_ang
    # /geophysical/day_night_flag
    # /geophysical/land_flag
    # /latitude
    # /level
    # /longitude
    # /observation_ops
    # /observation_ops/ak_col
    # /observation_ops/col_dof
    # /observation_ops/col_prior
    # /pressure
    # /surf_pres
    #
    # Also need to fill in global attributes 

    @register_dataset("/datetime_utc")
    @cache
    def datetime_utc(self, chop_milliseconds=True) -> np.ndarray:
        res = []
        for fname in self.input_path:
            ds = xarray.open_dataset(fname)
            # The :6 goes up to seconds, chopping off millisec and microseconds
            if chop_milliseconds:
                t = ds["obs_time_utc"].values[:,:,:6].astype(int)
            else:
                t = ds["obs_time_utc"].values.astype(int)
            # Data is atrack x xtrack. Repeat the data to add the pixel index, even
            # though the time is the same for all the pixels. Needed to we have
            # this for every sounding id.
            t = np.concatenate((t[:,:,np.newaxis,:],) * ds.sizes['fov'], axis=2)
            # Now flatten out, and add to result
            if chop_milliseconds:
                res.append(t.reshape((-1,6)))
            else:
                res.append(t.reshape((-1,8)))
        return np.concatenate(res, axis=0)

    @register_dataset("/time")
    def time_float(self) -> np.ndarray:
        utime = self.datetime_utc(chop_milliseconds=False)
        tepoch = astropy.time.Time("1993-01-01 00:00:00")
        tarray = [astropy.time.Time(datetime(*utime[i,:6], microsecond=utime[i,6] * 1000 + utime[i,7])) for i in range(utime.shape[0])]
        return np.array([(t-tepoch).to_value("s") for t in tarray])
    
    @register_dataset("/target_id")
    def sounding_id(self) -> np.ndarray:
        utime = self.datetime_utc()
        gnumber = self.granule_number()
        atrack = self.atrack()
        xtrack = self.xtrack()
        fov = self.fov()
        sidlist = [f"{utime[i,0]}{utime[i,1]:02d}{utime[i,2]:02d}_{gnumber[i]:03d}_{atrack[i]:02d}_{xtrack[i]:02d}_{fov[i]:01d}" for i in range(utime.shape[0])]
        # Convert to fixed character array that output wants
        return np.array([list(sid.ljust(36, '\0')) for sid in sidlist]).astype('S1')

    @register_dataset("/geolocation/cris_granule")
    @cache
    def granule_number(self) -> np.ndarray:
        res = []
        for fname in self.input_path:
            ds = xarray.open_dataset(fname)
            gnumber = ds.attrs["granule_number"]
            res.extend([gnumber,] * ds.sizes["atrack"] * ds.sizes["xtrack"] * ds.sizes["fov"])
        return np.array(res)

    @register_dataset("/geolocation/cris_atrack")
    @cache
    def atrack(self) -> np.ndarray:
        res = []
        for fname in self.input_path:
            ds = xarray.open_dataset(fname)
            res.extend([atrack for atrack in range(ds.sizes["atrack"]) for xtrack in range(ds.sizes["xtrack"]) for fov in range(ds.sizes["fov"])])
        return res

    @register_dataset("/geolocation/cris_xtrack")
    @cache
    def xtrack(self) -> np.ndarray:
        res = []
        for fname in self.input_path:
            ds = xarray.open_dataset(fname)
            res.extend([xtrack for atrack in range(ds.sizes["atrack"]) for xtrack in range(ds.sizes["xtrack"]) for fov in range(ds.sizes["fov"])])
        return res

    @register_dataset("/geolocation/cris_fov")
    @cache
    def fov(self) -> np.ndarray:
        res = []
        for fname in self.input_path:
            ds = xarray.open_dataset(fname)
            res.extend([fov for atrack in range(ds.sizes["atrack"]) for xtrack in range(ds.sizes["xtrack"]) for fov in range(ds.sizes["fov"])])
        return res
    
    
    @register_dataset("/year_fraction")
    def year_fraction(self) -> np.ndarray:
        utime = self.datetime_utc()
        return np.array([astropy.time.Time(datetime(*utime[i,:])).decimalyear for i in range(utime.shape[0])])
            
    @register_dataset("/col")
    def co_column(self) -> np.ndarray:
        # We get the columns to use from the file labels_order.txt
        return self.prediction.labels_pred[:, :5]

    @register_dataset("/col_error")
    def co_column_error(self) -> np.ndarray:
        # We get the columns to use from the file labels_order.txt
        return self.prediction.labels_pred[:, -5:]

    def write(self):
        self.output.write()


__all__ = [
    "ColumnCoFile",
]
