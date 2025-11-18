from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .identifier import StateElementIdentifier
from .tes_file import TesFile
from .refractor_uip import AttrDictAdapter
from pathlib import Path
import numpy as np
import copy
import math
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
        instrument: list[str],
        lite_directory: Path,
        data1: dict[str, Any],
        data2: dict[str, Any] | None,
        dataAnc: dict[str, Any],
    ) -> tuple[dict, list[float]]:
        if species_name == "RH":
            # Special case, relative humidity isn't something we retrieve
            linear = True
        else:
            smap = current_state.state_mapping(StateElementIdentifier(species_name))
            linear = isinstance(smap, rf.StateMappingLinear)
        self.product_cleanup(data1, species_name)
        data1 = self.products_add_fields(
            data1,
            species_name,
            data2,
            dataAnc,
            instrument,
            lite_directory,
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
        (dataNew, pressuresMax) = self.products_map_pressures(
            data1,
            levels,
            pressure0,
            "Linear" if linear == 1 else "Log",
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
            self.products_add_rtvmr(dataNew, species_name)

        self.product_set_quality(
            dataNew,
            species_name,
            instrument,
        )

        return (dataNew, pressuresMax)

    def products_add_rtvmr(self, dataInOut: dict[str, Any], species_name: str) -> None:
        # Temp
        from .muses_py import make_maps, get_indices, get_rtvmr_grid

        dof_thr = 0.0
        if species_name == "CH4":
            dof_thr = 1.6

        if species_name == "NH3":
            dof_thr = 1.2

        if species_name == "HCOOH" or species_name == "CH3OH":
            dof_thr = 1.2

        if dof_thr > 0.1:
            # rtvmr on retrieval grid
            # same code for NH3 and CH4
            # Vivienne: set dof_thr=1.6 for CH4 (4/2012)

            dataInOut["rtvmr".upper()] = np.ndarray(shape=(2), dtype=np.float32)
            dataInOut["rtvmr".upper()].fill(-999.0)

            dataInOut["rtvmr_pressure".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_pressure".upper()].fill(-999.0)

            dataInOut["rtvmr_pressureboundupper".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_pressureboundupper".upper()].fill(-999.0)

            dataInOut["rtvmr_pressureboundlower".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_pressureboundlower".upper()].fill(-999.0)

            dataInOut["rtvmr_errortotal".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_errortotal".upper()].fill(-999.0)

            dataInOut["rtvmr_errormeasurement".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_errormeasurement".upper()].fill(-999.0)

            dataInOut["rtvmr_errorobservation".upper()] = np.ndarray(
                shape=(2), dtype=np.float32
            )
            dataInOut["rtvmr_errorobservation".upper()].fill(-999.0)

            dataInOut["rtvmr_map".upper()] = np.ndarray(
                shape=(5, len(dataInOut["pressure".upper()])), dtype=np.float32
            )
            dataInOut["rtvmr_map".upper()].fill(-999.0)

            dataInOut["rtvmr_mappressure".upper()] = np.ndarray(
                shape=(5), dtype=np.float32
            )
            dataInOut["rtvmr_mappressure".upper()].fill(-999.0)

            # calculate RTVMR
            indp = np.where(
                (dataInOut["pressure".upper()] > 0)
                & (dataInOut["species".upper()] > -999)
            )[0]
            if len(indp) > 0:
                ak = dataInOut["averagingkernel".upper()][indp[0] :, indp[0] :]

                (cgrid, fwhm, ntrop, plower, pupper) = get_rtvmr_grid(
                    ak, dataInOut["pressure".upper()][indp], species_name, dof_thr
                )

                if len(cgrid) > 1:
                    if cgrid[len(cgrid) - 1] == cgrid[len(cgrid) - 2]:
                        for ii in range(len(cgrid) - 1, 1, -1):
                            zz = cgrid[ii]
                            ind = np.where(dataInOut["pressure".upper()] == zz)[0]
                            dataInOut["pressure".upper()]
                            cgrid[ii - 1] = dataInOut["pressure".upper()][ind[0] - 1]
                        cgrid[0] = np.max(dataInOut["pressure".upper()])
                    pndex = np.where(dataInOut["pressure".upper()] > 0)[0]
                    press = dataInOut["pressure".upper()][pndex]
                    meas_cov = dataInOut["measurementerrorcovariance".upper()][
                        pndex[0] :, pndex[0] :
                    ]

                    my_index = get_indices(press, cgrid)
                    my_map = make_maps(press, my_index + 1)

                    new_prof = np.exp(
                        np.matmul(
                            np.log(dataInOut["species".upper()][pndex]),
                            my_map["toPars"],
                        )
                    )
                    error_arr = np.matmul(
                        np.matmul(np.transpose(my_map["toPars"]), meas_cov),
                        my_map["toPars"],
                    )
                    error_total_arr = np.matmul(
                        np.matmul(
                            np.transpose(my_map["toPars"]),
                            dataInOut["totalerrorcovariance".upper()][
                                pndex[0] :, pndex[0] :
                            ],
                        ),
                        my_map["toPars"],
                    )
                    error_observation_arr = np.matmul(
                        np.matmul(
                            np.transpose(my_map["toPars"]),
                            dataInOut["observationerrorcovariance".upper()][
                                pndex[0] :, pndex[0] :
                            ],
                        ),
                        my_map["toPars"],
                    )

                    dataInOut["rtvmr".upper()][2 - ntrop :] = new_prof[
                        1 : ntrop + 1
                    ]  # PYTHON_NOTE: We add +1 to the end of slice.
                    dataInOut["rtvmr_pressure".upper()][2 - ntrop :] = cgrid[
                        1 : ntrop + 1
                    ]
                    if fwhm[0] >= 0:
                        dataInOut["rtvmr_pressureboundupper".upper()][2 - ntrop :] = (
                            pupper  # full width half max upper bound
                        )
                        dataInOut["rtvmr_pressureboundlower".upper()][2 - ntrop :] = (
                            plower  # full width half max lower bound
                        )

                    dataInOut["rtvmr_errortotal".upper()][2 - ntrop :] = np.sqrt(
                        np.diagonal(error_total_arr[1 : ntrop + 1, 1 : ntrop + 1])
                    )
                    dataInOut["rtvmr_errormeasurement".upper()][2 - ntrop :] = np.sqrt(
                        np.diagonal(error_arr[1 : ntrop + 1, 1 : ntrop + 1])
                    )
                    dataInOut["rtvmr_errorobservation".upper()][2 - ntrop :] = np.sqrt(
                        np.diagonal(error_observation_arr[1 : ntrop + 1, 1 : ntrop + 1])
                    )
                    dataInOut["rtvmr_map".upper()][2 - ntrop :, pndex] = my_map[
                        "toState"
                    ][:, :]
                    dataInOut["rtvmr_mappressure".upper()][2 - ntrop :] = cgrid[:]
                # end if len(cgrid) > 1):
            # end if len(indp) > 0:

    def products_map_pressures(
        self,
        dataIn: dict[str, Any],
        levelsIn: np.ndarray,
        pressureIn: np.ndarray,
        mapType: str,
        species_name: str,
    ) -> tuple[dict[str, Any], list[float]]:
        # Temp
        from .muses_py import make_maps, supplier_retrieval_levels_tes

        o_dataOut = copy.deepcopy(dataIn)

        nocut = 1

        # get levels, with max pressure = 1040.
        pressureMax = [
            1040.000000,
            1000.000000,
            908.514000,
            825.402000,
            749.893000,
            681.291000,
            618.966000,
            562.342000,
            510.898000,
            464.160000,
            421.698000,
            383.117000,
            348.069000,
            316.227000,
            287.298000,
            261.016000,
            237.137000,
            215.444000,
            195.735000,
            177.829000,
            161.561000,
            146.779000,
            133.352000,
            121.152000,
            110.069000,
            100.000000,
            90.851800,
            82.540600,
            74.989600,
            68.129500,
            61.896300,
            56.233900,
            51.089600,
            46.415800,
            42.169600,
            38.311900,
            34.807100,
            31.622900,
            28.729900,
            26.101700,
            23.713600,
            21.544300,
            19.573400,
            17.782800,
            16.156000,
            14.678000,
            13.335200,
            12.115300,
            11.007000,
            10.000000,
            9.085140,
            8.254020,
            6.812910,
            5.108980,
            4.641600,
            3.162270,
            2.610160,
            2.154430,
            1.615600,
            1.333520,
            1.000000,
            0.681292,
            0.383118,
            0.215443,
            0.100000,
        ]

        levelsMax = supplier_retrieval_levels_tes(
            levelsIn, pressureIn, pressureMax, nocut
        )

        pressuresMax = np.asarray(pressureMax)[levelsMax - 1]

        d = dataIn
        npp = len(levelsMax)
        np0 = len(dataIn["pressure".upper()])

        names0 = list(d.keys())

        names_list = []

        for kk in range(len(names0)):
            if "LMRESULTS" in names0[kk] and names0[kk] != "LMRESULTS_DELTA":
                names_list.append(names0[kk])
            elif "MICROWINDOW" in names0[kk]:
                names_list.append(names0[kk])
            else:
                # find all single element values
                if (
                    isinstance(dataIn[names0[kk]], int)
                    or isinstance(dataIn[names0[kk]], float)
                    or isinstance(dataIn[names0[kk]], np.int32)
                    or isinstance(dataIn[names0[kk]], float)
                    or isinstance(dataIn[names0[kk]], np.float64)
                    or isinstance(dataIn[names0[kk]], np.float32)
                ):
                    names_list.append(names0[kk])  # Add integer or float variables.
                elif isinstance(dataIn[names0[kk]], str):
                    names_list.append(names0[kk])  # Add string variables
                elif isinstance(dataIn[names0[kk]], np.ndarray):
                    # These are either vector or matrix.
                    if len(dataIn[names0[kk]].shape) == 1:
                        names_list.append(names0[kk])  # Add vector variables
                    elif len(dataIn[names0[kk]].shape) == 2:
                        names_list.append(names0[kk])  # Add matrix variables
                    elif len(dataIn[names0[kk]].shape) == 3:
                        names_list.append(names0[kk])  # Add matrix variables
                else:
                    if isinstance(dataIn[names0[kk]], list):
                        names_list.append(names0[kk])  # Add list variables.
        # end for kk in range(len(names0)):

        nn = 1

        mymaps = np.zeros(shape=(npp, np0, nn), dtype=np.float32)
        o_dataOut["map".upper()] = mymaps
        o_dataOut["pressure_fm".upper()] = copy.deepcopy(dataIn["pressure".upper()])
        o_dataOut["altitude_fm".upper()] = copy.deepcopy(dataIn["altitude".upper()])

        indp0 = np.where(dataIn["pressure".upper()] >= 0)[0]

        # Get indices that will need to be set with fill values -999.0
        my_mask = np.ones(
            len(dataIn["pressure".upper()]), dtype=bool
        )  # Set everything to 1.
        my_mask[indp0] = False  # Set all values with indices indp0 to False.
        my_other_indp0 = np.where(my_mask == 1)[
            0
        ]  # Get all indices where the 1's remain.

        pressure_temp_array = copy.deepcopy(dataIn["pressure".upper()][indp0])
        if pressure_temp_array[0] == pressure_temp_array[1]:
            pressure_temp_array[0] = pressure_temp_array[1] + 0.1

        # get retrieval levels
        # nocut means don't cut out a level that is 'too close'.  For
        # mapping lite products want to SHOW every retrieval level that exists
        # no mater how close it is to the surface.
        levels = supplier_retrieval_levels_tes(
            levelsIn, pressureIn, pressure_temp_array, nocut
        )

        # make maps
        linearFlag = False
        averageFlag = None

        maps = make_maps(pressure_temp_array, levels, linearFlag, averageFlag)
        maps = AttrDictAdapter(maps)
        startt = npp - len(levels)

        if startt < 0:
            raise RuntimeError("Not expecting value of startt to be negative")

        for kk in range(len(names_list)):
            key_name = names_list[kk]

            # copy variables that do not need pressure mapping
            if "MICROWINDOW" in key_name:
                o_dataOut[key_name] = copy.deepcopy(dataIn[key_name])
                continue

            dataIn_shape = [
                0,
                0,
            ]  # We have to fake the value of dataIn_shape in case it is scalar.
            if not np.isscalar(dataIn[key_name]):
                if isinstance(dataIn[key_name], list):
                    dataIn_shape = [len(dataIn[key_name])]
                else:
                    dataIn_shape = dataIn[key_name].shape

            if len(dataIn_shape) == 1 or (
                len(dataIn_shape) == 2 and dataIn_shape[1] == 1
            ):
                val = copy.deepcopy(dataIn[key_name])
                if dataIn_shape[0] == np0:
                    val = np.array(val)
                    o_dataOut[key_name] = val
                    val = val[indp0]
                    o_dataOut[key_name][my_other_indp0] = -999.0
                    if key_name == "OZONEIRK":
                        # linear map
                        result = np.matmul(val, np.transpose(maps.toState))
                        o_dataOut[key_name][startt:] = result[:]
                        o_dataOut[key_name] = np.resize(
                            o_dataOut[key_name], result.size
                        )
                    elif key_name == "ALTITUDE" or key_name == "PRESSURE":
                        assert len(val.shape) == 1 and len(dataIn_shape) == 1
                        assert not (levels > val.size).any()

                        o_dataOut[key_name][startt : startt + len(levels)] = val[
                            levels - 1
                        ]
                        o_dataOut[key_name] = np.resize(
                            o_dataOut[key_name], startt + len(levels)
                        )
                    else:
                        if mapType == "Log":
                            result = np.exp(np.matmul(np.log(val), maps.toPars))
                        else:
                            result = np.matmul(val, maps.toPars)
                        o_dataOut[key_name][startt : startt + len(result)] = result
                        o_dataOut[key_name] = np.resize(
                            o_dataOut[key_name], (startt + len(result))
                        )
                    # end: if key_name == 'OZONEIRK':
                # end: if dataIn_shape[0] == np0:
            # end: if len(dataIn_shape) == 1 or (len(dataIn_shape) == 2 and dataIn_shape[1] == 1):

            dataIn_shape = [
                0,
                1,
            ]  # We have to fake the value of dataIn_shape in case it is scalar.
            if not np.isscalar(dataIn[key_name]):
                # A 2D array is definitely not a list.
                if not isinstance(dataIn[key_name], list):
                    dataIn_shape = dataIn[key_name].shape

            if (len(dataIn_shape) == 2 and dataIn_shape[1] != 1) or (
                len(dataIn_shape) > 2 and dataIn_shape[2] == 1
            ):
                val = copy.deepcopy(dataIn[key_name])
                val = val[
                    67 - len(pressure_temp_array) :, 67 - len(pressure_temp_array) :
                ]

                if len(val.shape) > 2 and val.shape[2] == 1:
                    val = np.reshape(val, (val.shape[0], val.shape[1]))
                    o_dataOut[key_name] = np.reshape(
                        o_dataOut[key_name],
                        (o_dataOut[key_name].shape[0], o_dataOut[key_name].shape[1]),
                    )
                o_dataOut[key_name].fill(-999)  # Fill all values to fill value.

                # AT_LINE 307 src_ms-2018-12-10/TOOLS/products_map_pressures.pro
                if "AVERAGINGKERNEL" in key_name and (
                    key_name.index("AVERAGINGKERNEL") >= 0
                ):
                    o_dataOut[key_name] = np.resize(
                        o_dataOut[key_name],
                        (startt + maps.toPars.shape[1], startt + maps.toPars.shape[1]),
                    )

                    o_dataOut[key_name][
                        startt : startt + maps.toPars.shape[1],
                        startt : startt + maps.toPars.shape[1],
                    ] = np.matmul(np.matmul(maps.toState, val), maps.toPars)

                    ind = np.where(dataIn["species".upper()] > 0)[0]
                    if len(ind) == 0:
                        o_dataOut[key_name][startt:, startt:] = -999
                elif key_name == "SYSESTIMATE":
                    raise RuntimeError("Can't map true data because of SYSESTIMATE")
                elif key_name == "LMRESULTS_ITERLIST":
                    o_dataOut[key_name] = val
                else:
                    mapped_val = maps.toPars.T @ val @ maps.toPars

                    # Do the resize before the assignment.
                    o_dataOut[key_name] = np.resize(
                        o_dataOut[key_name],
                        (startt + maps.toPars.shape[1], startt + maps.toPars.shape[1]),
                    )

                    o_dataOut[key_name][
                        startt : startt + maps.toPars.shape[1],
                        startt : startt + maps.toPars.shape[1],
                    ] = mapped_val
            # end: if (len(dataIn_shape) == 2 and dataIn_shape[1] != 1) or (len(dataIn_shape) > 2 and dataIn_shape[2] == 1):
        # end for kk in range(len(names_list)):

        if "AVERAGINGKERNELDIAGONAL" in list(o_dataOut.keys()):
            o_dataOut["AVERAGINGKERNELDIAGONAL"][:] = np.diagonal(
                o_dataOut["AVERAGINGKERNEL"]
            )

        o_dataOut["map".upper()][:, 0, 0] = np.diagonal(o_dataOut["AVERAGINGKERNEL"])

        if (len(o_dataOut["map".upper()].shape) == 3) and (
            o_dataOut["map".upper()].shape[2] == 1
        ):
            o_dataOut["map".upper()] = np.reshape(
                o_dataOut["map".upper()],
                (o_dataOut["map".upper()].shape[0], o_dataOut["map".upper()].shape[1]),
            )

        # mypy is confused by range as argument to np.ix_. This is actually fine, it is
        # a problem with mypy. So tell it to ignore what it thinks is an error.
        o_dataOut["map".upper()][
            np.ix_(range(startt, o_dataOut["map".upper()].shape[0]), indp0)  # type: ignore[arg-type]
        ] = maps.toState
        o_dataOut["pressure_fm".upper()] = np.copy(dataIn["pressure".upper()])
        o_dataOut["altitude_fm".upper()] = np.copy(dataIn["altitude".upper()])

        return (o_dataOut, pressuresMax)

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

    def products_add_fields(
        self,
        dataIn: dict[str, Any],
        species_name: str,
        data2: dict[str, Any] | None,
        dataAnc: dict[str, Any],
        instrument: list[str],
        lite_directory: Path,
    ) -> dict[str, Any]:
        # Temp
        from .muses_py import (
            products_bias_correct,
            column_integrate,
            add_column,
            products_add_pan_fields,
        )

        function_name = "products_add_fields: "

        len_dataIn = 1

        if "EMISSIVITY" not in dataIn and dataAnc is not None:
            if dataAnc["SURFACEEMISSIVITY"].shape[0] == 1:
                dataIn["EMISSIVITY"] = copy.deepcopy(dataAnc["SURFACEEMISSIVITY"])
                dataIn["EMISSIVITY_INITIAL"] = copy.deepcopy(
                    dataAnc["SURFACEEMISSINITIAL"]
                )
            else:
                dataIn["EMISSIVITY"] = copy.deepcopy(dataAnc["SURFACEEMISSIVITY"])
                dataIn["EMISSIVITY_INITIAL"] = copy.deepcopy(
                    dataAnc["SURFACEEMISSINITIAL"]
                )

        if species_name == "OCS":
            raise RuntimeError("Not implemented for OCS")

        # PAN Vivienne: So far, I have been working with the average between
        # 800 hPa and tropopause.  I confess I haven't really spent much time
        # digging into the RTVMR.  Literally a vmr average

        if species_name == "PAN":
            dataIn = products_add_pan_fields(str(lite_directory) + "/", dataIn)

        if "AIRS" in instrument and species_name == "CH4":
            # no n2o correction
            # bias correction

            # if some correction already done, go back to original
            if "original_species".upper() in dataIn:
                dataIn["species".upper()] = np.copy(dataIn["original_species".upper()])

            pressure = copy.deepcopy(dataIn["pressure".upper()])

            bias = np.zeros(shape=(pressure.shape[0]), dtype=np.float32)
            ind = np.where(pressure > 150)[0]

            bias[ind] = -0.038 - 0.006

            ind = np.where(dataIn["averagingkernel".upper()] > 10)[0]
            if len(ind) > 0:
                indp = np.where(dataIn["species".upper()] > -990)[0]
                indp_2d = np.ix_(indp, indp)
                dataIn["averagingkernel".upper()][indp_2d] = 0

            ind = np.where(
                (dataIn["averagingkernel".upper()] < -10)
                & (dataIn["averagingkernel".upper()] > -990)
            )[0]
            if len(ind) > 0:
                indp = np.where(dataIn["species".upper()] > -990)[0]
                indp_2d = np.ix_(indp, indp)
                dataIn["averagingkernel".upper()][indp_2d] = 0

            if "original_species".upper() not in dataIn:
                dataIn["original_species".upper()] = np.copy(dataIn["species".upper()])

            dataIn = products_bias_correct(dataIn, bias)
        # if 'AIRS' in instrument and species_name == 'CH4':

        if "CRIS" in instrument and species_name == "CH4":
            # no n2o correction
            # bias correction

            # if some correction already done, go back to original
            if "original_species".upper() in dataIn:
                dataIn["species".upper()] = np.copy(dataIn["original_species".upper()])

            pressure = copy.deepcopy(dataIn["pressure".upper()])

            bias = np.zeros(shape=(pressure.shape[0]), dtype=np.float32)
            ind = np.where(pressure > 150)[0]

            bias[ind] = -0.038 - 0.006

            ind = np.where(dataIn["averagingkernel".upper()] > 10)[0]
            if len(ind) > 0:
                indp = np.where(dataIn["species".upper()] > -990)[0]
                indp_2d = np.ix_(indp, indp)
                dataIn["averagingkernel".upper()][indp_2d] = 0

            ind = np.where(
                (dataIn["averagingkernel".upper()] < -10)
                & (dataIn["averagingkernel".upper()] > -990)
            )[0]
            if len(ind) > 0:
                indp = np.where(dataIn["species".upper()] > -990)[0]
                indp_2d = np.ix_(indp, indp)
                dataIn["averagingkernel".upper()][indp_2d] = 0

            if "original_species".upper() not in dataIn:
                dataIn["original_species".upper()] = np.copy(dataIn["species".upper()])
        # if 'CRIS' in instrument and species_name == 'CH4':

        if species_name == "CH4":
            # quality info not in standard product
            # high ev ratio is "better" when 1 DOF, should be primarily first EV
            if "ch4_evs".upper() in dataIn:
                # Not sure if correct.
                dataIn["ch4_evratio".upper()] = np.zeros(shape=(1), dtype=np.float32)
                dataIn["ch4_evratio".upper()] = abs(
                    dataIn["ch4_evs".upper()][0]
                ) / np.sum(np.abs(dataIn["ch4_evs".upper()][0:10]))

            # now add in N2O data to CH4 lite product
            if "n2o_species".upper() not in dataIn:
                if data2 is not None:
                    dataIn["n2o_species".upper()] = copy.deepcopy(data2["SPECIES"])

            if "n2o_constraintVector".upper() not in dataIn:
                if data2 is not None:
                    dataIn["n2o_constraintVector".upper()] = copy.deepcopy(
                        data2["CONSTRAINTVECTOR"]
                    )

            if "n2o_averagingKernel".upper() not in dataIn:
                if data2 is not None:
                    dataIn["n2o_averagingKernel".upper()] = copy.deepcopy(
                        data2["AVERAGINGKERNEL"]
                    )

            if "n2o_observationErrorCovariance".upper() not in dataIn:
                if data2 is not None:
                    dataIn["n2o_observationErrorCovariance".upper()] = copy.deepcopy(
                        data2["OBSERVATIONERRORCOVARIANCE"]
                    )

            if "n2o_dofs".upper() not in dataIn:
                if data2 is not None:
                    dataIn["n2o_dofs".upper()] = copy.deepcopy(
                        data2["degreesOfFreedomForSignal".upper()]
                    )

            # if original species exists, e.g. for AIRS above, then do not do n2o correction
            # n2o correction:
            if "ORIGINAL_SPECIES" not in dataIn:
                dataIn["ORIGINAL_SPECIES"] = copy.deepcopy(dataIn["SPECIES"])

                # calculated N2O-corrected value
                species = copy.deepcopy(dataIn["SPECIES"])
                indp = np.where(
                    (dataIn["pressure".upper()] > 0) & (dataIn["species"] > 0)
                )[0]

                if len(indp) > 0:
                    species[indp] = np.exp(
                        np.log(dataIn["species".upper()][indp])
                        + np.log(dataIn["n2o_constraintVector".upper()][indp])
                        - np.log(dataIn["n2o_species".upper()][indp])
                    )

                dataIn["species".upper()] = copy.deepcopy(species)
            # end if 'ORIGINAL_SPECIES' not in dataIn:

            if "variabilitych4_QA".upper() in dataIn:
                del dataIn["variabilitych4_QA".upper()]

            if "variabilityn2o_QA".upper() in dataIn:
                del dataIn["variabilityn2o_QA".upper()]

            if "variabilitych4_QA".upper() not in dataIn:
                dataIn["variabilitych4_QA".upper()] = copy.deepcopy(dataIn["LATITUDE"])
                indp = np.where(dataIn["pressure".upper()] > 0)[0]

                x = np.var(
                    dataIn["species".upper()][indp]
                    - dataIn["constraintVector".upper()][indp]
                )
                y = np.mean(dataIn["constraintVector".upper()][indp])
                dataIn["variabilitych4_QA".upper()] = math.sqrt(x) / y

            if "variabilityN2O_QA".upper() not in dataIn:
                dataIn["variabilityN2O_QA".upper()] = (
                    copy.deepcopy(dataIn["LATITUDE"]) * 0
                )

                # calculate variability_QA
                indp = np.where(dataIn["pressure".upper()] >= 0)[0]

                x = np.var(
                    dataIn["n2o_species".upper()][indp]
                    - dataIn["n2o_constraintVector".upper()][indp]
                )
                y = np.mean(dataIn["n2o_constraintVector".upper()][indp])
                dataIn["variabilityN2O_QA".upper()] = (
                    math.sqrt(x) / y
                )  # GOOD IF LESS THAN .005
            # end if 'variabilityN2O_QA'.upper() not in dataIn:

            # but don't do for fill
            name_list = ["species".upper()]
            pressureList = ["pressure".upper()]
            dataIn = add_column(
                dataIn, "column750".upper(), 750, 0.0, name_list, pressureList
            )

            # convert to ppb
            if np.amax(dataIn["column750".upper()]) < 1:
                dataIn["column750".upper()] = dataIn["column750".upper()] * 1e9
                dataIn["column750_constraintvector".upper()] = (
                    dataIn["column750_constraintvector".upper()] * 1e9
                )
                dataIn["column750_initial".upper()] = (
                    dataIn["column750_initial".upper()] * 1e9
                )
                dataIn["column750_ObservationError".upper()] = (
                    dataIn["column750_ObservationError".upper()] * 1e9
                )
            # end if np.amax(dataIn['column750'.upper()]) < 1:

            # AT_LINE 507 Lite/products_add_fields.pro
            if "CTINTERP_CH4".upper() in dataIn:
                dataIn["column750_CTINTERP_CH4".upper()] = (
                    np.zeros(shape=(len_dataIn), dtype=np.float32) - 999
                )  # FLTARR(N_ELEMENTS(data))-999)
                print(
                    function_name,
                    "TRUE:'CTINTERP_CH4'.upper() in dataIn",
                    ("CTINTERP_CH4".upper() in dataIn),
                )
                print(function_name, "len_dataIn", len_dataIn)
                indp = np.where(
                    (dataIn["CTINTERP_PRESSURE".upper()] > 0)
                    & (dataIn["CTINTERP_CH4".upper()] > 0)
                )[0]
                print(function_name, "indp, len(indp)", indp, len(indp))

                if len(indp) > 0:
                    VMR = dataIn["CTINTERP_CH4".upper()][indp] * 1e-9
                    pressure = dataIn["CTINTERP_PRESSURE".upper()][indp]
                    altitude = dataIn["altitude".upper()][indp]
                    airDensity = dataIn["airDensity".upper()][indp]

                    ind1 = np.amin(np.where(pressure <= 764)[0])
                    ind2 = len(pressure) - 1

                    if np.amin(altitude) / 1000 > 20:
                        raise RuntimeError("np.amin(altitude) / 1000 > 20")

                    minIndex = ind1
                    maxIndex = ind2
                    c1 = column_integrate(VMR, airDensity, altitude, minIndex, maxIndex)
                    a = column_integrate(
                        VMR * 0 + 1, airDensity, altitude, minIndex, maxIndex
                    )

                    dataIn["column750_CTINTERP_CH4".upper()] = (
                        c1["column"] / a["column"] * 1.0e9
                    )
                # end if len(indp) > 0:
            # end if 'CTINTERP_CH4' in dataIn:

            # additional screening parameters
            dofstrat = np.zeros(shape=(1), dtype=np.float32)
            doftrop = np.zeros(shape=(1), dtype=np.float32)

            indp = np.where(
                dataIn["pressure".upper()] > dataIn["tropopausePressure".upper()]
            )[0]
            doftrop = np.sum(dataIn["averagingkerneldiagonal".upper()][indp])

            indp = np.where(
                (dataIn["pressure".upper()] < dataIn["tropopausepressure".upper()])
                & (dataIn["pressure".upper()] >= 0)
            )[0]
            dofstrat = np.sum(dataIn["averagingkerneldiagonal".upper()][indp])

            if "ch4_doftrop".upper() not in dataIn:
                dataIn["ch4_doftrop".upper()] = doftrop

            if "ch4_dofstrat".upper() not in dataIn:
                dataIn["ch4_dofstrat".upper()] = dofstrat

            dataIn["ch4_doftrop".upper()] = doftrop
            dataIn["ch4_dofstrat".upper()] = dofstrat

            dataIn["ch4_stratosphere_QA".upper()] = copy.deepcopy(
                dataIn["latitude".upper()]
            )
            ind562 = np.asarray([np.argmin(np.abs(dataIn["pressure".upper()] - 562))])

            indT = np.where(
                dataIn["pressure".upper()] >= dataIn["tropopausePressure".upper()]
            )[0]
            indS = np.where(
                (dataIn["pressure".upper()] >= 0)
                & (dataIn["pressure".upper()] <= dataIn["tropopausePressure".upper()])
            )[0]

            if len(indT) > 0 and len(indS) > 0:
                totT = np.sum(dataIn["averagingKernel".upper()][indT, ind562])
                totS = np.sum(dataIn["averagingKernel".upper()][indS, ind562])
                dataIn["ch4_stratosphere_QA".upper()] = totS / (totT + totS)
        # end if species_name == 'CH4':

        if species_name == "HDO":
            # add in all fields needed up front
            # then only use data

            # add the ancillary matrices to d so that they can be mapped along
            # with the rest of the profiles and matrices in d.
            names_list = [
                "HDO_H2OAVERAGINGKERNEL",
                "H2O_HDOAVERAGINGKERNEL",
                "HDO_H2OMEASUREMENTERRORCOVARIANCE",
                "HDO_H2OObservationERRORCOVARIANCE".upper(),
                "HDO_H2OTotalERRORCOVARIANCE".upper(),
            ]

            for jj in range(len(names_list)):
                if names_list[jj] not in dataIn:
                    dataIn[names_list[jj]] = copy.deepcopy(dataAnc[names_list[jj]])

            # add h2o-h2o part from data2
            my_names = [
                "H2O_H2OAVERAGINGKERNEL".upper(),
                "H2O_H2OMEASUREMENTERRORCOVARIANCE".upper(),
                "H2O_H2OOBSERVATIONERRORCOVARIANCE".upper(),
                "H2O_H2OTOTALERRORCOVARIANCE".upper(),
            ]

            for jj in range(0, len(my_names)):
                if my_names[jj] not in dataIn:
                    if data2 is not None:
                        dataIn[my_names[jj]] = copy.deepcopy(data2[my_names[jj][7:]])

            my_names = [
                "H2O_INITIAL".upper(),
                "H2O_SPECIES".upper(),
                "H2O_CONSTRAINTVECTOR".upper(),
            ]

            for jj in range(0, len(my_names)):
                if my_names[jj] not in dataIn:
                    if data2 is not None:
                        dataIn[my_names[jj]] = copy.deepcopy(data2[my_names[jj][4:]])

            # add values
            if "H2O_SPECIES" not in dataIn:
                if data2 is not None:
                    dataIn["H2O_SPECIES"] = copy.deepcopy(data2["SPECIES"])

            if "H2O_CONSTRAINTVECTOR" not in dataIn:
                if data2 is not None:
                    dataIn["H2O_CONSTRAINTVECTOR"] = data2["CONSTRAINTVECTOR"]

            # v08 bias correction. Retain the retrieved HDO value as hdo_original.
            # AT_LINE 595 Lite/products_add_fields.pro
            dataIn["original_species_hdo".upper()] = copy.deepcopy(dataIn["SPECIES"])

            if not np.isscalar(dataIn["latitude".upper()]):
                raise RuntimeError(
                    "not implemented: not np.isscalar(dataIn['latitude'.upper()])"
                )
            else:
                pressure = dataIn["pressure".upper()]

            delta = 0.00019 * pressure - 0.067
            ind = np.where(pressure < 310)[0]
            delta[ind] = 0

            dcorr = products_bias_correct(dataIn, delta, hdoFlag=True)

            # Put the bias corrected value into species
            dataIn["species".upper()] = np.copy(dcorr["species".upper()])

            # there are a very small # of cases where the AK has too many fill values
            # don't take these out or will mess up alignment, but set VMR values to fill
            if (dataIn["averagingkernel".upper()][20, 20] < -990) or (
                data2 is not None and data2["averagingkernel".upper()][20, 20] < -990
            ):
                if "run".upper() in dataIn:
                    raise RuntimeError("Not implemented")

            # check that water adjacent values are not too different. A problem in GMAO
            count = 0

            indp = np.where(dataIn["h2o_species".upper()] > 0)[0]
            if len(indp) > 0:
                test1 = (
                    dataIn["h2o_constraintVector".upper()][indp[1:]]
                    / dataIn["h2o_constraintVector".upper()][indp[0 : len(indp) - 1]]
                )

                if np.amax(test1) > 1000:
                    dataIn["Quality".upper()] = 0
                    count = count + 1

            # check that trop water is > 1e-16.  A problem in GMAO.
            indp = np.where(
                (dataIn["h2o_species".upper()] > 0)
                & (dataIn["pressure".upper()] >= 200)
            )[0]
            if len(indp) > 0:
                test = np.where(dataIn["h2o_constraintVector".upper()][indp] <= 1e-16)[
                    0
                ]
                if len(test) > 0:
                    # AT_LINE 656 Lite/products_add_fields.pro
                    dataIn["Quality".upper()] = 0
                    count = count + 1
        # end if species_name == 'HDO':

        if species_name == "RH":
            # check that water adjacent values are not too different. A problem in GMAO

            count = 0

            indp = np.where(dataIn["SPECIES"] > 0)[0]
            if len(indp) > 0:
                test1 = (
                    dataIn["constraintVector".upper()][indp[1:]]
                    / dataIn["constraintVector".upper()][indp[0 : len(indp) - 1]]
                )
                if np.amax(test1) > 1000:
                    if data2 is not None:
                        data2["Quality".upper()] = 0
                    dataIn["Quality".upper()] = 0
                    count = count + 1

            # check that trop water is > 1e-16.  A problem in GMAO.
            indp = np.where(
                (dataIn["species".upper()] > 0) & (dataIn["pressure".upper()] >= 200)
            )[0]
            if len(indp) > 0:
                test = np.where(dataIn["constraintVector".upper()][indp] <= 1e-16)[0]
                if len(test) > 0:
                    if data2 is not None:
                        data2["Quality".upper()] = 0
                    dataIn["Quality".upper()] = 0
                    count = count + 1

            # have to calculate RH on 66-level grid, as T and H2O have different
            # retrieval levels.
            assert data2 is not None
            dataIn["tatm".upper()] = copy.deepcopy(data2["species".upper()])
            dataIn["h2o".upper()] = copy.deepcopy(dataIn["species".upper()])
            indp = np.where(data2["pressure".upper()] > 0)[0]
            TATM = data2["species".upper()][indp]
            pressure = data2["pressure".upper()][indp]
            H2O = dataIn["species".upper()][indp]
            num_pressures = len(indp)
            RH = np.zeros(shape=(num_pressures), dtype=np.float32)
            RH_xa = np.zeros(shape=(num_pressures), dtype=np.float32)
            dRHdT = np.zeros(shape=(num_pressures), dtype=np.float64)
            TATM_xa = data2["constraintVector".upper()][indp]
            H2O_xa = dataIn["constraintVector".upper()][indp]

            ind = np.where(TATM > 273.15)[0]
            if len(ind) > 0:
                x = 0.0415 * (TATM[ind] - 218.8)
                tanhx = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
                # for water,
                # the saturation vapor pressure es (hPa) is:
                es = 0.01 * np.exp(
                    54.842763
                    - 6763.22 / TATM[ind]
                    - 4.21 * np.log(TATM[ind])
                    + 0.000367 * TATM[ind]
                    + tanhx
                    * (
                        53.878
                        - 1331.22 / TATM[ind]
                        - 9.44523 * np.log(TATM[ind])
                        + 0.014025 * TATM[ind]
                    )
                )
                # and the relative humidity
                RH[ind] = 100.0 * pressure[ind] * H2O[ind] / es
                # where pressure is in hPa and VMR is the H2O vmr.
                dataIn["species".upper()][indp[ind]] = RH[ind]

                # prior
                x = 0.0415 * (TATM_xa[ind] - 218.8)
                tanhx = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
                es = 0.01 * np.exp(
                    54.842763
                    - 6763.22 / TATM[ind]
                    - 4.21 * np.log(TATM_xa[ind])
                    + 0.000367 * TATM_xa[ind]
                    + tanhx
                    * (
                        53.878
                        - 1331.22 / TATM_xa[ind]
                        - 9.44523 * np.log(TATM_xa[ind])
                        + 0.014025 * TATM_xa[ind]
                    )
                )

                RH_xa[ind] = 100.0 * pressure[ind] * H2O_xa[ind] / es[:]
                dataIn["constraintVector".upper()][indp[ind]] = RH_xa[ind]
            # end if len(ind) > 0:

            ind = np.where(TATM <= 273.15)[0]
            if len(ind) > 0:
                # for ice,
                # the saturation vapor pressure (hPa) is:
                es = 0.01 * np.exp(
                    9.550426
                    - 5723.265 / TATM[ind]
                    + 3.53068 * np.log(TATM[ind])
                    - 0.00728332 * TATM[ind]
                )

                # and the relative humidity
                RH[ind] = 100.0 * pressure[ind] * H2O[ind] / es[:]

                # where pressure is in hPa and VMR is the H2O vmr.
                dataIn["species".upper()][indp[ind]] = RH[ind]

                # prior
                x = 0.0415 * (TATM_xa[ind] - 218.8)
                tanhx = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
                es = 0.01 * np.exp(
                    54.842763
                    - 6763.22 / TATM[ind]
                    - 4.21 * np.log(TATM_xa[ind])
                    + 0.000367 * TATM_xa[ind]
                    + tanhx
                    * (
                        53.878
                        - 1331.22 / TATM_xa[ind]
                        - 9.44523 * np.log(TATM_xa[ind])
                        + 0.014025 * TATM_xa[ind]
                    )
                )
                RH_xa[ind] = 100.0 * pressure[ind] * H2O_xa[ind] / es[:]
                dataIn["constraintVector".upper()][indp[ind]] = RH_xa[ind]
            # end if len(ind) > 0:

            ind = np.where(TATM > 273.15)[0]
            if len(ind) > 0:
                dRHdT[ind] = RH[ind] * (
                    -1730.63 * np.log(10) / np.square(-39.724 + TATM[ind])
                )

            ind = np.where(TATM <= 273.15)[0]
            if len(ind) > 0:
                dRHdT[ind] = RH[ind] * (-2663.5 * np.log(10) / np.square(TATM[ind]))

            H2OTOTALERRORCOVARIANCE = dataIn["TOTALERRORCOVARIANCE"][indp, :][
                :, indp
            ].astype(np.float64)
            H2OOBSERVATIONERRORCOVARIANCE = dataIn["OBSERVATIONERRORCOVARIANCE"][
                indp, :
            ][:, indp].astype(np.float64)
            H2OMEASUREMENTERRORCOVARIANCE = dataIn["MEASUREMENTERRORCOVARIANCE"][
                indp, :
            ][:, indp].astype(np.float64)
            TATMTOTALERRORCOVARIANCE = data2["TOTALERRORCOVARIANCE"][indp, :][
                :, indp
            ].astype(np.float64)
            TATMOBSERVATIONERRORCOVARIANCE = data2["OBSERVATIONERRORCOVARIANCE"][
                indp, :
            ][:, indp].astype(np.float64)
            TATMMEASUREMENTERRORCOVARIANCE = data2["MEASUREMENTERRORCOVARIANCE"][
                indp, :
            ][:, indp].astype(np.float64)

            # H2O and TATM propagated errors
            dRHdTMatrix = np.diag(dRHdT)
            dRHdlnH2OMatrix = np.diag(RH)

            temp_sum = np.matmul(
                np.matmul(dRHdlnH2OMatrix, H2OTOTALERRORCOVARIANCE), dRHdlnH2OMatrix
            ) + np.matmul(np.matmul(dRHdTMatrix, TATMTOTALERRORCOVARIANCE), dRHdTMatrix)

            dataIn["TOTALERRORCOVARIANCE"][np.ix_(indp, indp)] = temp_sum

            temp_sum = np.matmul(
                np.matmul(dRHdlnH2OMatrix, H2OOBSERVATIONERRORCOVARIANCE),
                dRHdlnH2OMatrix,
            ) + np.matmul(
                np.matmul(dRHdTMatrix, TATMOBSERVATIONERRORCOVARIANCE), dRHdTMatrix
            )

            dataIn["OBSERVATIONERRORCOVARIANCE"][np.ix_(indp, indp)] = temp_sum

            temp_sum = np.matmul(
                np.matmul(dRHdlnH2OMatrix, H2OMEASUREMENTERRORCOVARIANCE),
                dRHdlnH2OMatrix,
            ) + np.matmul(
                np.matmul(dRHdTMatrix, TATMMEASUREMENTERRORCOVARIANCE), dRHdTMatrix
            )

            dataIn["MEASUREMENTERRORCOVARIANCE"][np.ix_(indp, indp)] = temp_sum

            dataIn["TOTALERROR"][indp] = np.sqrt(
                dataIn["TOTALERRORCOVARIANCE"][indp, indp]
            )

            # RH AK:  dRH/dRH = dRH/dlnH2O dlnH2O/dlnH2O dlnH2O/dRH = dRH/dlnH2O AK(lnH2O) dlnH2O/dRH
            # plus TATM term... this has DOF of ~10 and doesn't seem right
            # use H2O AK for RH AK.

            if "AVERAGINGKERNELDIAGONAL" in dataIn:
                dataIn["AVERAGINGKERNELDIAGONAL"] = np.diagonal(
                    dataIn["averagingKernel".upper()]
                )

            dataIn["Dofs".upper()] = np.sum(dataIn["AVERAGINGKERNELDIAGONAL"][indp])
        # end if species_name == 'RH':

        # surface altitude
        if "surfaceAltitude".upper() not in dataIn:
            dataIn["surfaceAltitude".upper()] = (
                np.zeros(shape=(len_dataIn), dtype=np.float32) - 999
            )

        indp = np.where(dataIn["altitude".upper()] > 0)[0]
        if len(indp) > 0:
            dataIn["surfaceAltitude".upper()] = np.amin(
                dataIn["altitude".upper()][indp]
            )

        # add land flag.  Make land flag so land==1 or land==0, otherwise have
        # to worry about ocean/lake, etc.
        if "landflag".upper() not in dataIn:
            if "SURFACETYPEFOOTPRINT" in dataIn:
                dataIn["landflag".upper()] = (
                    np.zeros(shape=(1), dtype=np.int32) - 999
                )  # INTARR(nn)-999)

                ind = np.where(dataIn["SURFACETYPEFOOTPRINT"] == 2)[0]
                dataIn["LANDFLAG".upper()][ind] = 0

                ind = np.where(dataIn["SURFACETYPEFOOTPRINT"] == 1)[0]
                dataIn["LANDFLAG".upper()][ind] = 0

                ind = np.where(dataIn["SURFACETYPEFOOTPRINT"] == 3)[0]
                dataIn["LANDFLAG".upper()][ind] = 1

                ind = np.where(dataIn["SURFACETYPEFOOTPRINT"] == 4)[0]
                dataIn["LANDFLAG".upper()][ind] = 1
            # end if 'SURFACETYPEFOOTPRINT' in dataIn:
        # end if 'landflag' not in dataIn:
        return dataIn


__all__ = ["CdfWriteLiteTes"]
