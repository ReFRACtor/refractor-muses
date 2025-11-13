from __future__ import annotations
from .mpy import (
    mpy_products_get_maptype,
    mpy_products_cleanup,
    mpy_products_add_fields,
    mpy_products_map_pressures,
    mpy_products_combine_hdo,
    mpy_products_add_rtvmr,
    mpy_products_set_quality,
    mpy_read_all_tes_cache,
    mpy_tes_file_get_struct,
)
import numpy as np


class CdfWriteLiteTes:
    """Logically this fits into CdfWriteTes, but that class is already getting
    pretty big. We separate out the lite file part, just to reduce the size.

    Note that both these classes need a serious clean up, it is possible that we
    can shrink the size down a bit. If so, we can move this functionality back
    to CdfWriteTes - the only reason this is separated out is size."""

    def __init__(self) -> None:
        pass

    def make_one_lite(
        self,
        species_name: str,
        filenameIn: str,
        starttai: float,
        endtai: float,
        instrument: list[str],
        pressuresMax: list[float],
        liteDirectory: str,
        version: str,
        liteVersion: str,
        data1: dict,
        data2: dict | None,
        dataAnc: dict,
        step: int = 0,
    ) -> tuple[dict, list[float]]:
        mapType = mpy_products_get_maptype(data1, species_name)
        linear = 0
        if mapType == "Linear":
            linear = 1
        data1 = mpy_products_cleanup(data1, species_name)
        (data1, data2) = mpy_products_add_fields(
            data1,
            species_name,
            data2,
            dataAnc,
            version,
            step,
            linear,
            instrument,
            liteDirectory,
        )
        levelFilename = (
            liteDirectory
            + f"RetrievalLevels/Retrieval_Levels_Nadir_{mapType}_{species_name.upper()}"
        )
        (read_status, fileID) = mpy_read_all_tes_cache(levelFilename)
        infoFile = mpy_tes_file_get_struct(fileID)

        if "level" in infoFile:
            levels_text_list = infoFile["level"]
        elif "Level" in infoFile:
            levels_text_list = infoFile["Level"]
        elif "LEVEL" in infoFile:
            levels_text_list = infoFile["LEVEL"]
        else:
            raise RuntimeError(f"Trouble reading file {levelFilename}")

        levelsv = []
        for ii, level_text_value in enumerate(levels_text_list):
            levelsv.append(int(level_text_value))
        levels = np.asarray(levelsv)
        pressureFilename = liteDirectory + "TES_baseline_66.asc"
        (read_status, fileID) = mpy_read_all_tes_cache(pressureFilename)
        infoFile = mpy_tes_file_get_struct(fileID)

        if "pressure" in infoFile:
            pressures_text_list = infoFile["pressure"]
        elif "Pressure" in infoFile:
            pressures_text_list = infoFile["Pressure"]
        elif "PRESSURE" in infoFile:
            pressures_text_list = infoFile["PRESSURE"]
        else:
            raise RuntimeError(f"Trouble reading file {pressureFilename}")

        pressure0v = []
        for ii, pressure_text_value in enumerate(pressures_text_list):
            pressure0v.append(float(pressure_text_value))
        pressure0 = np.asarray(pressure0v)

        if linear == 1:
            addmap = True
            mapType = "Linear"
            no_cut = 0
            (dataNew, pressuresMax) = mpy_products_map_pressures(
                data1,
                levels,
                pressure0,
                mapType,
                addmap,
                None,
                no_cut,
                species_name,
            )
        else:
            addmap = True
            mapType = None
            no_cut = 0
            (dataNew, pressuresMax) = mpy_products_map_pressures(
                data1,
                levels,
                pressure0,
                mapType,
                addmap,
                pressuresMax,
                no_cut,
                species_name,
            )

        if species_name == "HDO":
            dataNew = mpy_products_combine_hdo(dataNew)

        if (
            species_name == "CH4"
            or species_name == "NH3"
            or species_name == "HCOOH"
            or species_name == "CH3OH"
        ):
            dataNew = mpy_products_add_rtvmr(dataNew, species_name)

        dataNew = mpy_products_set_quality(
            dataNew,
            species_name,
            filenameIn,
            None,
            instrument,
            version,
            liteVersion,
            step,
        )

        return (dataNew, pressuresMax)


__all__ = ["CdfWriteLiteTes"]
