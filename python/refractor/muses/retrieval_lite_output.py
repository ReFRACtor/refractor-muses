from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .mpy import (
    mpy_products_add_fields,
    mpy_products_map_pressures,
    mpy_products_add_rtvmr,
)
from .identifier import StateElementIdentifier
from .tes_file import TesFile
from .refractor_uip import AttrDictAdapter
from pathlib import Path
import numpy as np
from typing import Any
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
        pressure_max: list[float],
        lite_directory: Path,
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
            str(lite_directory) + "/",
        )
        level_filename = (
            lite_directory
            / f"RetrievalLevels/Retrieval_Levels_Nadir_{'Linear' if linear else 'Log'}_{species_name.upper()}"
        )
        fh = TesFile.create(level_filename)
        found = False
        for k in ("level", "Level", "LEVEL"):
            if k in fh.table:
                found = True
                levels = fh.table[k].array
        if not found:
            raise RuntimeError(f"Trouble reading file {level_filename}")

        pressure_filename = lite_directory / "TES_baseline_66.asc"
        fh = TesFile.create(pressure_filename)
        found = False
        for k in ("pressure", "Pressure", "PRESSURE"):
            if k in fh.table:
                found = True
                pressure0 = fh.table[k].array
        if not found:
            raise RuntimeError(f"Trouble reading file {pressure_filename}")
        addmap = True
        no_cut = 0
        (dataNew, pressuresMax) = mpy_products_map_pressures(
            data1,
            levels,
            pressure0,
            "Linear" if linear == 1 else None,
            addmap,
            pressure_max if linear == 1 else None,
            no_cut,
            species_name,
        )
        if species_name == "HDO":
            self.product_combine_hdo(dataNew)

        if (
            species_name == "CH4"
            or species_name == "NH3"
            or species_name == "HCOOH"
            or species_name == "CH3OH"
        ):
            dataNew = mpy_products_add_rtvmr(dataNew, species_name)

        self.product_set_quality(
            dataNew,
            species_name,
            instrument,
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

    def product_combine_hdo(self, dataNew: dict[str, Any]) -> None:
        if (
            len((dataNew["species".upper()]).shape) == 2
            and dataNew["species".upper()].shape[1] == 1
        ):
            dataNew["species".upper()] = np.reshape(
                dataNew["species".upper()], (dataNew["species".upper()].shape[0])
            )

        num_points = len(dataNew["species".upper()]) * 2
        nn = 1
        mySpecies = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myInitial = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myAK = np.zeros(shape=(num_points, num_points), dtype=np.float32) - 999
        myAKDiagonal = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myTotalError = np.zeros(shape=(num_points, num_points), dtype=np.float32) - 999
        myMeasError = np.zeros(shape=(num_points, num_points), dtype=np.float32) - 999
        myObsError = np.zeros(shape=(num_points, num_points), dtype=np.float32) - 999
        myXa = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myP = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myAirD = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myAlt = np.zeros(shape=(num_points), dtype=np.float32) - 999

        mySpeciesOrig = np.zeros(shape=(num_points), dtype=np.float32) - 999
        myCloudOD = np.zeros(shape=(nn), dtype=np.float32) - 999
        myCloudODError = np.zeros(shape=(nn), dtype=np.float32) - 999

        if (
            len(dataNew["pressure".upper()].shape) > 1
            and dataNew["pressure".upper()].shape[1] == 1
        ):
            if (
                len(dataNew["species".upper()].shape) > 1
                and dataNew["species".upper()].shape[1] == 1
            ):
                indp = np.where(
                    (dataNew["pressure".upper()] > 0) & (dataNew["species".upper()] > 0)
                )[0]

            if len(dataNew["species".upper()].shape) == 1:
                indp = np.where(
                    (dataNew["pressure".upper()] > 0) & (dataNew["species".upper()] > 0)
                )[0]

            if len(indp) == 0:
                raise RuntimeError("len(indp) is zero.")

        if len(dataNew["pressure".upper()].shape) == 1:
            if (
                len(dataNew["species".upper()].shape) > 1
                and dataNew["species".upper()].shape[1] == 1
            ):
                indp = np.where(
                    (dataNew["pressure".upper()] > 0) & (dataNew["species".upper()] > 0)
                )[0]

            if len(dataNew["species".upper()].shape) == 1:
                indp = np.where(
                    (dataNew["pressure".upper()] > 0) & (dataNew["species".upper()] > 0)
                )[0]

            if len(indp) == 0:
                raise RuntimeError("len(indp) is zero.")

        if len(indp) == 0:
            raise RuntimeError("len(indp) is zero.")

        if len(indp) > 0:
            start2 = num_points - len(indp)
            start1 = int(num_points / 2) - len(indp)
            npp = len(indp)

            mySpecies[start1 : start1 + npp] = dataNew["species".upper()][indp]
            mySpecies[start2:] = dataNew["h2o_species".upper()][indp]

            mySpeciesOrig[start1 : start1 + npp] = dataNew[
                "original_species_hdo".upper()
            ][indp]
            mySpeciesOrig[start2:] = dataNew["h2o_species".upper()][indp]

            myXa[start1 : start1 + npp] = dataNew["constraintVector".upper()][indp]
            myXa[start2:] = dataNew["h2o_constraintVector".upper()][indp]

            myInitial[start1 : start1 + npp] = dataNew["initial".upper()][indp]
            myInitial[start2:] = dataNew["h2o_initial".upper()][indp]

            myP[start1 : start1 + npp] = dataNew["pressure".upper()][indp]
            myP[start2:] = dataNew["pressure".upper()][indp]

            myAlt[start1 : start1 + npp] = dataNew["altitude".upper()][indp]
            myAlt[start2:] = dataNew["altitude".upper()][indp]

            myAirD[start1 : start1 + npp] = dataNew["airDensity".upper()][indp]
            myAirD[start2:] = dataNew["airDensity".upper()][indp]

            myAK[start1 : start1 + npp, start1 : start1 + npp] = dataNew[
                "averagingKernel".upper()
            ][indp, :][:, indp]
            myAK[start2 : start2 + npp, start2 : start2 + npp] = dataNew[
                "h2o_h2oaveragingKernel".upper()
            ][indp, :][:, indp]
            myAK[start2:, start1 : start1 + npp] = dataNew[
                "HDO_H2OAVERAGINGKERNEL".upper()
            ][indp, :][:, indp]
            myAK[start1 : start1 + npp, start2:] = dataNew[
                "H2O_HDOAVERAGINGKERNEL".upper()
            ][indp, :][:, indp]

            myAKDiagonal[start1 : start1 + npp] = np.diagonal(
                dataNew["averagingKernel".upper()][indp, :][:, indp]
            )
            myAKDiagonal[start2:] = np.diagonal(
                dataNew["h2o_h2oaveragingKernel".upper()][indp, :][:, indp]
            )

            myMeasError[start1 : start1 + npp, start1 : start1 + npp] = dataNew[
                "MEASUREMENTERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myMeasError[start2 : start2 + npp, start2 : start2 + npp] = dataNew[
                "h2o_h2oMEASUREMENTERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myMeasError[start2:, start1 : start1 + npp] = dataNew[
                "HDO_H2OMEASUREMENTERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myMeasError[start1 : start1 + npp, start2:] = np.transpose(
                dataNew["HDO_H2OMEASUREMENTERRORCOVARIANCE".upper()]
            )[indp, :][:, indp]

            # full observation error covariance
            # [D, D] = HDO
            # [H, H] = H2O
            # [H, D] = HDO_H2O
            # [D, H] = H2O_HDO

            myObsError[start1 : start1 + npp, start1 : start1 + npp] = dataNew[
                "ObservationERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myObsError[start2:, start2:] = dataNew[
                "h2o_h2oObservationERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myObsError[start2:, start1 : start1 + npp] = dataNew[
                "HDO_H2OObservationERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myObsError[start1 : start1 + npp, start2:] = np.transpose(
                dataNew["HDO_H2OObservationERRORCOVARIANCE".upper()]
            )[indp, :][:, indp]

            myTotalError[start1 : start1 + npp, start1 : start1 + npp] = dataNew[
                "TotalERRORCOVARIANCE".upper()
            ][indp, :][:, indp]
            myTotalError[start2:, start2:] = dataNew[
                "h2o_h2oTotalERRORCOVARIANCE".upper()
            ][indp, :][:, indp]

            myTotalError[start2:, start1 : start1 + npp] = dataNew[
                "HDO_H2OTotalERRORCOVARIANCE".upper()
            ][indp, :][:, indp]
            myTotalError[start1 : start1 + npp, start2:] = np.transpose(
                dataNew["HDO_H2OTotalERRORCOVARIANCE".upper()]
            )[indp, :][:, indp]

            # Not sure why we are getting element [9].
            myCloudOD[0] = dataNew["CLOUDEFFECTIVEOPTICALDEPTH".upper()][9]
            myCloudODError[0] = dataNew["CLOUDEFFECTIVEOPTICALDEPTHError".upper()][9]
        # end if len (indp) > 0:

        # take away old
        if "AVERAGINGKERNEL".upper() in dataNew:
            del dataNew["AVERAGINGKERNEL".upper()]
        if "AVERAGINGKERNELDIAGONAL".upper() in dataNew:
            del dataNew["AVERAGINGKERNELDIAGONAL".upper()]
        if "CONSTRAINTVECTOR".upper() in dataNew:
            del dataNew["CONSTRAINTVECTOR".upper()]
        if "SPECIES".upper() in dataNew:
            del dataNew["SPECIES".upper()]
        if "PRECISION".upper() in dataNew:
            del dataNew["PRECISION".upper()]
        if "INITIAL".upper() in dataNew:
            del dataNew["INITIAL".upper()]
        if "MEASUREMENTERRORCOVARIANCE".upper() in dataNew:
            del dataNew["MEASUREMENTERRORCOVARIANCE".upper()]
        if "OBSERVATIONERRORCOVARIANCE".upper() in dataNew:
            del dataNew["OBSERVATIONERRORCOVARIANCE".upper()]
        if "TOTALERROR".upper() in dataNew:
            del dataNew["TOTALERROR".upper()]
        if "TOTALERRORCOVARIANCE".upper() in dataNew:
            del dataNew["TOTALERRORCOVARIANCE".upper()]
        if "PRESSURE".upper() in dataNew:
            del dataNew["PRESSURE".upper()]
        if "H2O_H2OAVERAGINGKERNEL".upper() in dataNew:
            del dataNew["H2O_H2OAVERAGINGKERNEL".upper()]
        if "HDO_H2OAVERAGINGKERNEL".upper() in dataNew:
            del dataNew["HDO_H2OAVERAGINGKERNEL".upper()]
        if "H2O_HDOAVERAGINGKERNEL".upper() in dataNew:
            del dataNew["H2O_HDOAVERAGINGKERNEL".upper()]
        if "H2O_H2OMEASUREMENTERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_H2OMEASUREMENTERRORCOVARIANCE".upper()]
        if "HDO_H2OMEASUREMENTERRORCOVARIANCE".upper() in dataNew:
            del dataNew["HDO_H2OMEASUREMENTERRORCOVARIANCE".upper()]
        if "H2O_HDOMEASUREMENTERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_HDOMEASUREMENTERRORCOVARIANCE".upper()]
        if "H2O_H2OOBSERVATIONERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_H2OOBSERVATIONERRORCOVARIANCE".upper()]
        if "HDO_H2OOBSERVATIONERRORCOVARIANCE".upper() in dataNew:
            del dataNew["HDO_H2OOBSERVATIONERRORCOVARIANCE".upper()]
        if "H2O_HDOOBSERVATIONERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_HDOOBSERVATIONERRORCOVARIANCE".upper()]

        if "H2O_H2OTOTALERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_H2OTOTALERRORCOVARIANCE".upper()]
        if "HDO_H2OTOTALERRORCOVARIANCE".upper() in dataNew:
            del dataNew["HDO_H2OTOTALERRORCOVARIANCE".upper()]
        if "H2O_HDOTOTALERRORCOVARIANCE".upper() in dataNew:
            del dataNew["H2O_HDOTOTALERRORCOVARIANCE".upper()]

        if "ALTITUDE".upper() in dataNew:
            del dataNew["ALTITUDE".upper()]
        if "ORIGINAL_SPECIES_HDO".upper() in dataNew:
            del dataNew["ORIGINAL_SPECIES_HDO".upper()]
        if "AIRDENSITY".upper() in dataNew:
            del dataNew["AIRDENSITY".upper()]

        if "h2o_species".upper() in dataNew:
            del dataNew["h2o_species".upper()]
        if "h2o_constraintvector".upper() in dataNew:
            del dataNew["h2o_constraintvector".upper()]
        if "h2o_initial".upper() in dataNew:
            del dataNew["h2o_initial".upper()]
        # put in new
        dataNew["AVERAGINGKERNEL".upper()] = myAK
        dataNew["AVERAGINGKERNELDIAGONAL".upper()] = myAKDiagonal
        dataNew["initial".upper()] = myInitial
        dataNew["species".upper()] = mySpecies
        dataNew["original_species".upper()] = mySpeciesOrig
        dataNew["TOTALERRORCOVARIANCE".upper()] = myTotalError
        dataNew["MEASUREMENTERRORCOVARIANCE".upper()] = myMeasError
        dataNew["OBSERVATIONERRORCOVARIANCE".upper()] = myObsError
        dataNew["CONSTRAINTVECTOR".upper()] = myXa
        dataNew["PRESSURE".upper()] = myP
        dataNew["AIRDENSITY".upper()] = myAirD
        dataNew["ALTITUDE".upper()] = myAlt

        # add in separate vectors for H2O and HDO
        dataNew["HDO_H2O".upper()] = np.copy(dataNew["species".upper()])

    def product_set_quality(
        self, dataNew: dict[str, Any], species_name: str, instrument_list: list[str]
    ) -> None:
        dataInOut = AttrDictAdapter(dataNew)
        if "AIRS" in instrument_list and species_name == "CH4":
            indgood = (
                dataInOut.QUALITY == 1
                and dataInOut.DOFS > 1.1
                and dataInOut.CH4_DOFTROP > 0.7
                and dataInOut.CH4_DOFSTRAT <= 0.5
                and dataInOut.COLUMN750_ERROR <= 53
            )
            dataInOut.QUALITY = np.int16(1) if indgood else np.int16(0)
        if "TES" in instrument_list and species_name == "PAN":
            # comparing to idl, I don't see rad_residual_stdev_change here
            indgood = (
                dataInOut.QUALITY == 1
                and dataInOut.SURFACETEMPERATURE > 265
                and dataInOut.PAN_DESERT_QA == 1
                and dataInOut.RADIANCE_RESIDUAL_STDEV_CHANGE > -0.15
            )
            dataInOut.QUALITY = np.int16(1) if indgood else np.int16(0)

            #  check that cloud top pressure is below tropopause-20 hPa
            #  easier than tropopause
            if "TROPOPAUSEPRESSURE" in dataInOut.__dict__:
                indbad = dataInOut.TROPOPAUSEPRESSURE > dataInOut.CLOUDTOPPRESSURE + 20

                if indbad:
                    dataInOut.QUALITY = np.int16(0)

        # end if 'TES' in instrument_list and species_name == 'PAN':

        # Vivienne 7/19/2019
        # note desert not a problem for cris pan, so do not use pan_desert_qa
        if "CRIS" in instrument_list and species_name == "PAN":
            indgood = dataInOut.QUALITY == 1 and dataInOut.SURFACETEMPERATURE > 265

            dataInOut.QUALITY = np.int16(1) if indgood else np.int16(0)

            #  check that cloud top pressure is below tropopause-20 hPa
            #  easier than tropopause
            if "TROPOPAUSEPRESSURE" in dataInOut.__dict__:
                indbad = dataInOut.TROPOPAUSEPRESSURE > dataInOut.CLOUDTOPPRESSURE + 20

                if indbad:
                    dataInOut.QUALITY = np.int16(0)

        if ("TES" in instrument_list) and (
            species_name == "HCOOH" or species_name == "CH3OH"
        ):
            pass
            # checked IDL:  for HCOOH and CH3OH used O3 quality flags but only v6 and earlier.

        if species_name == "RH":
            # check RH between 0 and 200
            # if RH < 0 set to fill
            # if RH > 200 set to bad quality
            for iip in range(0, len(dataInOut.SPECIES)):
                if (dataInOut.SPECIES[iip] < 0) and (dataInOut.SPECIES[iip] > -990):
                    dataInOut.SPECIES[iip] = -999

                if dataInOut.SPECIES[iip] > 200:
                    dataInOut.QUALITY = np.int16(0)


__all__ = ["CdfWriteLiteTes"]
