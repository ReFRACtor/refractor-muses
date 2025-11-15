from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .mpy import (
    mpy_products_add_fields,
    mpy_products_map_pressures,
    mpy_products_combine_hdo,
    mpy_products_add_rtvmr,
    mpy_products_set_quality,
    mpy_read_all_tes_cache,
    mpy_tes_file_get_struct,
)
from .identifier import StateElementIdentifier
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState


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
        current_state: CurrentState,
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
        if species_name == "RH":
            # Special case, relative humidity isn't something we retrieve
            linear = 1
        else:
            smap = current_state.state_mapping(StateElementIdentifier(species_name))
            linear = 1 if isinstance(smap, rf.StateMappingLinear) else 0
        self.product_cleanup(data1, species_name)
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
            + f"RetrievalLevels/Retrieval_Levels_Nadir_{'Linear' if linear else 'Log'}_{species_name.upper()}"
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

    def product_cleanup(self, dataInOut: dict, species_name: str) -> None:
        for v in (
            "CALIBRATION_QA",
            "MAXNUMITERATIONSNUMBERITERPERFORMED",
            "RADIANCERESIDUALMAX",
            "SCAN_AVERAGED_COUNT",
            "SPECIESRETRIEVALCONVERGED",
            "SURFACEEMISSIONLAYER_QA",
            "DEVIATIONVSRETRIEVALCOVARIANCE",
            "BORESIGHTNADIRANGLEUNC",
            "VERTICALRESOLUTION",
        ):
            if v in dataInOut:
                del dataInOut[v]

        if species_name != "O3":
            if "FMOZONEBANDFLUX" in dataInOut:
                del dataInOut["FMOZONEBANDFLUX"]

            if "O3_CCURVE_QA" in dataInOut:
                del dataInOut["O3_CCURVE_QA"]

            if "OZONETROPOSPHERICCOLUMN" in dataInOut:
                if "ONETROPOSPHERICCOLUMN" in dataInOut:
                    del dataInOut["ONETROPOSPHERICCOLUMN"]

                if "ONETROPOSPHERICCOLUMNERROR" in dataInOut:
                    del dataInOut["ONETROPOSPHERICCOLUMNERROR"]

                if "ONETROPOSPHERICCOLUMNINITIAL" in dataInOut:
                    del dataInOut["ONETROPOSPHERICCOLUMNINITIAL"]

            if "O3TROPOSPHERICCOLUMN" in dataInOut:
                if "TROPOSPHERICCOLUMN" in dataInOut:
                    del dataInOut["TROPOSPHERICCOLUMN"]

                if "TROPOSPHERICCOLUMNERROR" in dataInOut:
                    del dataInOut["TROPOSPHERICCOLUMNERROR"]

                if "TROPOSPHERICCOLUMNINITIAL" in dataInOut:
                    del dataInOut["TROPOSPHERICCOLUMNINITIAL"]

            if "OZONEIRK" in dataInOut:
                if "ONEIRK" in dataInOut:
                    del dataInOut["ONEIRK"]

            if "OZONEIRFK" in dataInOut:
                if "ONEIRFK" in dataInOut:
                    del dataInOut["ONEIRFK"]

            if "L1BOZONEBANDATAFLUX" in dataInOut:
                if "BOZONEBANDFLUX" in dataInOut:
                    del dataInOut["BOZONEBANDFLUX"]

        if species_name != "TATM":
            if "SURFACETEMPVSATMTEMP_QA" in dataInOut:
                del dataInOut["SURFACETEMPVSATMTEMP_QA"]
        else:
            if "TEMPERATURE" in dataInOut:
                del dataInOut["TEMPERATURE"]
            if "TEMPERATUREPRECISION" in dataInOut:
                del dataInOut["TEMPERATUREPRECISION"]

        if dataInOut["averagingkernel".upper()][20, 20] < -990:
            dataInOut["species".upper()][:] = -999


__all__ = ["CdfWriteLiteTes"]
