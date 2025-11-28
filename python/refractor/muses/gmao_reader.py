from __future__ import annotations
from functools import cache
import h5py  # type: ignore
import numpy as np
import scipy
import datetime
from .misc import greatcircle
from pathlib import Path
from loguru import logger
import typing

if typing.TYPE_CHECKING:
    from .sounding_metadata import SoundingMetadata


class GmaoReader:
    """This class does the same thing as mpy.gmao_read."""

    def __init__(
        self,
        smeta: SoundingMetadata,
        gmao_dir: Path,
        pressure_in: np.ndarray | None = None,
    ) -> None:
        """Note we could just take the time and latitude/longitude directly.
        But where we are using this, we have the SoundingMetadata structure so
        just used that. We can rework if needed
        """

        # Find the hour on a 3 hour boundary
        yr = smeta.year
        month = smeta.month
        day = smeta.day
        h = int(smeta.hour / 3.0 + 0.5) * 3
        # Handle roll over to next day
        if h == 24:
            h = 0
            next_day = datetime.date(yr, month, day) + datetime.timedelta(days=1)
            yr = next_day.year
            month = next_day.month
            day = next_day.day
        gmao_d = GmaoReader.read_gmao(gmao_dir, yr, month, day, h)
        lon = gmao_d["lon"]
        lat = gmao_d["lat"]

        # Find index of nearest lat/lon
        lon_ind = None
        lat_ind = None
        dmin = None
        for i, lonv in enumerate(lon):
            # Don't bother looking at latitude unless we are with 2.0 in longitude
            if lonv >= smeta.longitude.value - 2 and lonv <= smeta.longitude.value + 2:
                for j, latv in enumerate(lat):
                    d = greatcircle(
                        latv,
                        lonv,
                        smeta.latitude.value,
                        smeta.longitude.value,
                    )
                    if dmin is None or d < dmin:
                        dmin = d
                        lon_ind = i
                        lat_ind = j
        # Convert pa to hPa unit.
        pa_2_hpa = 1.0 / 100.0
        # Convert specified humidity to vmr unit.
        Md = 28.966  # molecular mass of air
        Mw = 18.016  # molecular mass of water
        qv_2_vmr = Md / Mw

        GMAOOrig = {
            "latitude": gmao_d["lat"][lat_ind],
            "longitude": gmao_d["lon"][lon_ind],
            "pressure": gmao_d["lev"],
            "TATM": gmao_d["T"][:, lat_ind, lon_ind],
            "QV": gmao_d["QV"][:, lat_ind, lon_ind],
            "H2O": gmao_d["QV"][:, lat_ind, lon_ind] * qv_2_vmr,
            "surfacePressure": gmao_d["SLP"][lat_ind, lon_ind] * pa_2_hpa,
            "surfaceTemperature": gmao_d["TS"][lat_ind, lon_ind],
            "tropopausePressure": gmao_d["TROPPT"][lat_ind, lon_ind] * pa_2_hpa,
        }

        # Look for bad data, and replace
        ind_bad = np.where(
            (
                (GMAOOrig["QV"] > 100.0)
                | (GMAOOrig["TATM"] > 500)
                | (GMAOOrig["TATM"] < 100)
            )
        )[0]

        if ind_bad.shape[0] > 0:
            first_value_to_use = GMAOOrig["H2O"][np.amax(ind_bad) + 1]
            GMAOOrig["H2O"][ind_bad] = first_value_to_use
            first_value_to_use = GMAOOrig["TATM"][np.amax(ind_bad) + 1]
            GMAOOrig["TATM"][ind_bad] = first_value_to_use

        # Look for bad water values, we just detect this
        if np.count_nonzero((GMAOOrig["QV"] < 1e-20) | (GMAOOrig["H2O"] < 1e-20)) > 0:
            raise RuntimeError("BAD water found.")

        pressure = GMAOOrig["pressure"]
        tatm = GMAOOrig["TATM"]
        h2o = GMAOOrig["H2O"]

        # Ensure that the GMAO grid includes the surface pressure, adding if needed.
        if GMAOOrig["surfacePressure"] < GMAOOrig["pressure"].max():
            logger.info(
                f"Surface pressure {GMAOOrig['surfacePressure']} is lower than max pressure {GMAOOrig['pressure'].max()}"
            )
        else:
            pressure = np.insert(pressure, 0, GMAOOrig["surfacePressure"])
            tatm = np.insert(tatm, 0, tatm[0])
            h2o = np.insert(h2o, 0, h2o[0])
            # Sort pressure in reverse order, and apply to tatm and h2o also
            ind = np.argsort(pressure)[::-1]
            pressure = pressure[ind]
            tatm = tatm[ind]
            h2o = h2o[ind]

        if pressure_in is None:
            pressure_in = pressure

        # Make sure pressure_in.max() is in pressure list
        if pressure_in.max() - pressure.max() > 0.1:
            pressure = np.insert(pressure, 0, pressure_in.max())
            tatm = np.insert(tatm, 0, tatm[0])
            h2o = np.insert(h2o, 0, h2o[0])
            # Sort pressure in reverse order, and apply to tatm and h2o also
            ind = np.argsort(pressure)[::-1]
            pressure = pressure[ind]
            tatm = tatm[ind]
            h2o = h2o[ind]

        tatm2 = self.interpol_1d(tatm, np.log(pressure), np.log(pressure_in))
        h2o2 = self.interpol_1d(h2o, np.log(pressure), np.log(pressure_in))

        # Ensure we are not extrapolating... should not occur because of above
        # precautions
        tatm2[(pressure_in - GMAOOrig["surfacePressure"]) > 0.1] = tatm[0]
        h2o2[(pressure_in - GMAOOrig["surfacePressure"]) > 0.1] = h2o[0]

        tatm = tatm2
        h2o = h2o2

        self.latitude = GMAOOrig["latitude"]
        self.longitude = GMAOOrig["longitude"]
        self.pressure = pressure
        self.tatm = tatm
        self.qv = gmao_d["QV"][:, :lat_ind, :lon_ind]
        self.h2o = h2o
        self.surface_pressure = GMAOOrig["surfacePressure"]
        self.surface_temperature = GMAOOrig["surfaceTemperature"]
        self.tropopause_pressure = GMAOOrig["tropopausePressure"]

    @classmethod
    @cache
    def read_gmao(
        cls, gmao_dir: Path, year: int, month: int, day: int, hour: int
    ) -> dict[str, np.ndarray]:
        # GMAO apparently has different directory structures, try each.
        gmao_fdir = gmao_dir / f"{year}/{month:02d}/{day:02d}"
        # Try year/month
        if not gmao_fdir.exists():
            gmao_fdir = gmao_fdir.parent
        # Try year
        if not gmao_fdir.exists():
            gmao_fdir = gmao_fdir.parent
        if not gmao_fdir.exists():
            gmao_fdir = gmao_dir / f"{year}/{month:02d}/{day:02d}/{hour:04d}"
            raise RuntimeError(
                f"Can not find GMAO directory {gmao_fdir} or any parents"
            )
        try:
            fname_3d = next(
                gmao_fdir.glob(
                    f"*.asm.inst3_3d_asm_Np.*.{year}{month:02d}{day:02d}_{hour:02d}00.V01.nc4"
                )
            )
            fname_2d = next(
                gmao_fdir.glob(
                    f"*.asm.inst3_2d_asm_Nx.*.{year}{month:02d}{day:02d}_{hour:02d}00.V01.nc4"
                )
            )
        except StopIteration:
            raise RuntimeError("GMAO file not found")
        res = {}
        with h5py.File(fname_3d, "r") as f:
            for v in ["QV", "T", "lat", "lev", "lon"]:
                res[v] = f[v][:]
                if res[v].shape[0] == 1:
                    res[v] = res[v][0]
        with h5py.File(fname_2d, "r") as f:
            for v in ["SLP", "TROPPT", "TS", "lat", "lon"]:
                res[v] = f[v][:]
                if res[v].shape[0] == 1:
                    res[v] = res[v][0]
        return res

    def interpol_1d(
        self, yin: np.ndarray, xin: np.ndarray, xout: np.ndarray
    ) -> np.ndarray:
        interpfunc = scipy.interpolate.interp1d(
            xin, yin, kind="linear", fill_value="extrapolate"
        )
        return interpfunc(xout)


__all__ = [
    "GmaoReader",
]
