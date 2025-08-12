from __future__ import annotations
from . import muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import typing
from typing import Self
from .identifier import InstrumentIdentifier
from .tes_file import TesFile

if typing.TYPE_CHECKING:
    from refractor.old_py_retrieve_wrapper import (  # type: ignore
        StateInfoOld,
    )
    from .muses_observation import MeasurementId, MusesObservation


class SoundingMetadata:
    """Metadata associated with a Sounding. Note that is similar to information
    in a MusesObservation, but for joint retrievals there isn't one clear observation
    that has this information.
    """

    def __init__(
        self,
        latitude: rf.DoubleWithUnit = rf.DoubleWithUnit(0, "deg"),
        longitude: rf.DoubleWithUnit = rf.DoubleWithUnit(0, "deg"),
        surface_altitude: rf.DoubleWithUnit = rf.DoubleWithUnit(0, "km"),
        day_flag: bool = False,
        surface_type: str = "",
        tai_time: float = 0.0,
        sounding_id: str = "",
        utc_time: str = "",
    ) -> None:
        """Note you normally call one of the creator functions rather
        than __init__. However, you can certainly just supply the
        information and create from __init__, for example if you have
        test data or something like that.
        """
        self._latitude = latitude
        self._longitude = longitude
        self._surface_altitude = surface_altitude
        self._day_flag = day_flag
        self._surface_type = surface_type
        self._tai_time = tai_time
        self._sounding_id = sounding_id
        self._utc_time = utc_time

    @classmethod
    def create_from_measurement_id(
        cls,
        measurement_id: MeasurementId,
        instrument: InstrumentIdentifier,
        obs: MusesObservation,
    ) -> Self:
        res = cls()
        instrument_name = str(instrument)
        if f"{instrument_name}_latitude" in measurement_id:
            res._latitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_latitude"]), "deg"
            )
        else:
            res._latitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_Latitude"]), "deg"
            )
        if f"{instrument_name}_longitude" in measurement_id:
            res._longitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_longitude"]), "deg"
            )
        else:
            res._longitude = rf.DoubleWithUnit(
                float(measurement_id[f"{instrument_name}_Longitude"]), "deg"
            )
        if "oceanFlag" in measurement_id:
            oceanflag = int(measurement_id["oceanflag"])
        else:
            oceanflag = int(measurement_id["OCEANFLAG"])
        res._surface_type = "OCEAN" if oceanflag == 1 else "LAND"
        res._sounding_id = measurement_id["key"]
        # Couple of things in the DateTime file
        f = TesFile(measurement_id["run_dir"] / "DateTime.asc")
        res._tai_time = float(f["TAI_Time_of_ZPD"])
        res._utc_time = f["UTC_Time"]

        date_struct = mpy.utc_from_string(res._utc_time)
        i_date_struct = {}
        i_date_struct["dateStruct"] = date_struct
        i_date_struct["year"] = date_struct["utctime"].year
        i_date_struct["month"] = date_struct["utctime"].month
        i_date_struct["day"] = date_struct["utctime"].day
        i_date_struct["hour"] = date_struct["utctime"].hour
        i_date_struct["minute"] = date_struct["utctime"].minute
        i_date_struct["second"] = date_struct["utctime"].second
        res._day_flag = bool(mpy.daytime(i_date_struct, res._longitude.value))
        res._surface_altitude = obs.surface_altitude
        return res

    @classmethod
    def create_from_old_state_info(
        cls, state_info: StateInfoOld, step: str = "current"
    ) -> Self:
        if step not in ("current", "initial", "initialInitial"):
            raise RuntimeError(
                "Don't support anything other than the current, initial, or initialInitial step"
            )
        res = cls()
        res._latitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["latitude"], "deg"
        )
        res._longitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["longitude"], "deg"
        )
        res._surface_altitude = rf.DoubleWithUnit(
            state_info.state_info_dict[step]["tsa"]["surfaceAltitudeKm"], "km"
        )
        res._day_flag = bool(state_info.state_info_dict[step]["tsa"]["dayFlag"])
        res._surface_type = state_info.state_info_dict[step]["surfaceType"].upper()
        res._tai_time = state_info._tai_time
        res._sounding_id = state_info._sounding_id
        res._utc_time = state_info._utc_time
        return res

    @property
    def latitude(self) -> rf.DoubleWithUnit:
        return self._latitude

    @property
    def longitude(self) -> rf.DoubleWithUnit:
        return self._longitude

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return self._surface_altitude

    @property
    def tai_time(self) -> float:
        return self._tai_time

    @property
    def utc_time(self) -> str:
        return self._utc_time

    @property
    def year(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].year

    @property
    def month(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].month

    @property
    def day(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].day

    @property
    def hour(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].hour

    @property
    def minute(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].minute

    @property
    def second(self) -> int:
        date_struct = mpy.utc_from_string(self.utc_time)
        return date_struct["utctime"].second

    @property
    def local_hour(self) -> int:
        timestruct = mpy.utc(self.utc_time)
        hour = timestruct["hour"] + self.longitude.convert("deg").value / 180.0 * 12
        if hour < 0:
            hour += 24
        if hour > 24:
            hour -= 24
        return hour

    @property
    def wrong_tai_time(self) -> float:
        """The muses-py function mpy.tai uses the wrong number of leapseconds, it
        doesn't include anything since 2006. To match old data, return the incorrect
        value so we can match the file. This should get fixed actually."""
        timestruct = mpy.utc(self.utc_time)
        if timestruct["yearfloat"] >= 2017.0:
            extraleapscond = 4
        elif timestruct["yearfloat"] >= 2015.5:
            extraleapscond = 3
        elif timestruct["yearfloat"] >= 2012.5:
            extraleapscond = 2
        elif timestruct["yearfloat"] >= 2009.0:
            extraleapscond = 1
        else:
            extraleapscond = 0
        return self._tai_time - extraleapscond

    @property
    def sounding_id(self) -> str:
        return self._sounding_id

    @property
    def surface_type(self) -> str:
        return self._surface_type

    @property
    def is_day(self) -> bool:
        return self._day_flag

    @property
    def is_ocean(self) -> bool:
        return self.surface_type == "OCEAN"

    @property
    def is_land(self) -> bool:
        return self.surface_type == "LAND"


__all__ = [
    "SoundingMetadata",
]
