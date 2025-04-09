from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .identifier import StateElementIdentifier
import numpy as np
from scipy.linalg import block_diag  # type: ignore
from pathlib import Path
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .error_analysis import ErrorAnalysis
    from .muses_strategy_executor import CurrentStrategyStep
    from .retrieval_result import RetrievalResult


class RetrievalInfo:
    """Not sure if we'll keep this or not, but pull out RetrievalInfo stuff so
    we can figure out the interface and if we should replace this.

    A few functions seem sort of like member functions, we'll just make a list
    of these to sort out later but not try to get the full interface in place.
    """

    def __init__(
        self,
        error_analysis: ErrorAnalysis,
        species_dir: Path,
        current_strategy_step: CurrentStrategyStep,
        current_state: CurrentState,
    ):
        self.retrieval_dict = self.init_data(
            error_analysis, species_dir, current_strategy_step, current_state
        )
        self.retrieval_dict = self.retrieval_dict.__dict__
        self._map_type_systematic = mpy.constraint_get_maptype(
            error_analysis.error_current, self.species_list_sys
        )

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix to go from forward model grid to retrieval
        grid.  By convention, None if we don't actually have any
        retrieval parameters.

        """
        if self.n_totalParameters == 0 or self.n_totalParametersFM == 0:
            return None
        mmm = self.n_totalParameters
        nnn = self.n_totalParametersFM
        return self.retrieval_dict["mapToState"][0:mmm, 0:nnn]

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        # TODO Sort this out
        # Not clear why this isn't just the transpose of basis_matrix. But it isn't,
        # so just have this in place
        if self.n_totalParameters == 0 or self.n_totalParametersFM == 0:
            return None
        mmm = self.n_totalParameters
        nnn = self.n_totalParametersFM
        return self.retrieval_dict["mapToParameters"][0:nnn, 0:mmm]

    @property
    def retrieval_info_obj(self) -> mpy.ObjectView:
        return mpy.ObjectView(self.retrieval_dict)

    @property
    def initial_guess_list(self) -> np.ndarray:
        """This is the initial guess for the state vector (not the full state)"""
        return self.retrieval_dict["initialGuessList"]

    @property
    def constraint_vector(self) -> np.ndarray:
        """This is the initial guess for the state vector (not the full state)"""
        return self.retrieval_dict["constraintVector"]

    def species_results(
        self,
        results: RetrievalResult,
        spcname: str,
        FM_Flag: bool = True,
        INITIAL_Flag: bool = False,
    ) -> np.ndarray:
        return mpy.get_vector(
            results.results_list,
            self.retrieval_info_obj,
            spcname,
            FM_Flag,
            INITIAL_Flag,
        )

    def species_initial(self, spcname: str, FM_Flag: bool = True) -> np.ndarray:
        return mpy.get_vector(
            self.initial_guess_list, self.retrieval_info_obj, spcname, FM_Flag, True
        )

    def species_constraint(self, spcname: str, FM_Flag: bool = True) -> np.ndarray:
        return mpy.get_vector(
            self.constraint_vector, self.retrieval_info_obj, spcname, FM_Flag, True
        )

    @property
    def initialGuessListFM(self) -> np.ndarray:
        """This is the initial guess for the FM state vector"""
        return self.retrieval_dict["initialGuessListFM"]

    @property
    def initial_guess_list_fm(self) -> np.ndarray:
        """This is the initial guess for the FM state vector"""
        return self.retrieval_dict["initialGuessListFM"]

    @property
    def parameter_start_fm(self) -> list[int]:
        return self.retrieval_dict["parameterStartFM"]

    @property
    def parameter_end_fm(self) -> list[int]:
        return self.retrieval_dict["parameterEndFM"]

    @property
    def map_type(self) -> list[str]:
        return self.retrieval_dict["mapType"]

    @property
    def type(self) -> str:
        return self.retrieval_dict["type"]

    @property
    def apriori_cov(self) -> np.ndarray:
        return self.retrieval_dict["Constraint"][
            0 : self.n_totalParameters, 0 : self.n_totalParameters
        ]

    @property
    def apriori(self) -> np.ndarray:
        return self.retrieval_dict["constraintVector"][0 : self.n_totalParameters]

    @property
    def apriori_fm(self) -> np.ndarray:
        return self.retrieval_dict["constraintVectorListFM"][
            0 : self.n_totalParametersFM
        ]

    @property
    def true_value(self) -> np.ndarray:
        """Apriori value"""
        return self.retrieval_dict["trueParameterList"][0 : self.n_totalParameters]

    @property
    def true_value_fm(self) -> np.ndarray:
        """Apriori value"""
        return self.retrieval_dict["trueParameterListFM"][0 : self.n_totalParametersFM]

    @property
    def species_names(self) -> list[str]:
        return list(
            self.retrieval_dict["species"][0 : self.retrieval_dict["n_species"]]
        )

    @property
    def species_names_sys(self) -> list[str]:
        return list(
            self.retrieval_dict["speciesSys"][0 : self.retrieval_dict["n_speciesSys"]]
        )

    @property
    def species_list(self) -> list[str]:
        return list(self.retrieval_dict["speciesList"][0 : self.n_totalParameters])

    @property
    def species_list_sys(self) -> list[str]:
        return list(
            self.retrieval_dict["speciesListSys"][0 : self.n_totalParametersSys]
        )

    @property
    def map_type_systematic(self) -> list[str]:
        return list(self._map_type_systematic)

    def retrieval_info_systematic(self) -> mpy.ObjectView:
        """Version of retrieval info to use for a creating a systematic UIP"""
        return mpy.ObjectView(
            {
                "parameterStartFM": self.retrieval_info_obj.parameterStartSys,
                "parameterEndFM": self.retrieval_info_obj.parameterEndSys,
                "species": self.species_names_sys,
                "n_species": self.n_totalParametersSys,
                "speciesList": self.species_list_sys,
                "speciesListFM": self.species_list_sys,
                "mapTypeListFM": self.map_type_systematic,
                "initialGuessListFM": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "constraintVectorListFM": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "initialGuessList": np.zeros(
                    shape=(self.n_totalParametersSys,), dtype=np.float32
                ),
                "n_totalParametersFM": self.n_totalParametersSys,
            }
        )

    @property
    def species_list_fm(self) -> list[str]:
        return self.retrieval_dict["speciesListFM"][0 : self.n_totalParametersFM]

    @property
    def pressure_list_fm(self) -> list[float]:
        return self.retrieval_dict["pressureListFM"][0 : self.n_totalParametersFM]

    @property
    def surface_type(self) -> str:
        return self.retrieval_dict["surfaceType"]

    @property
    def is_ocean(self) -> bool:
        return self.surface_type == "OCEAN"

    # Synonyms used in the muses-py code.
    @property
    def speciesListFM(self) -> list[str]:
        return self.species_list_fm

    @property
    def pressureListFM(self) -> list[float]:
        return self.pressure_list_fm

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def minimumList(self) -> list[float]:
        return list(self.retrieval_dict["minimumList"][0 : self.n_totalParameters])

    @property
    def maximumList(self) -> list[float]:
        return list(self.retrieval_dict["maximumList"][0 : self.n_totalParameters])

    @property
    def maximumChangeList(self) -> list[float]:
        return list(
            self.retrieval_dict["maximumChangeList"][0 : self.n_totalParameters]
        )

    @property
    def n_totalParameters(self) -> int:
        # Might be a better place to get this, but start by getting from
        # initial guess
        return self.initial_guess_list.shape[0]

    @property
    def n_totalParametersSys(self) -> int:
        return self.retrieval_dict["n_totalParametersSys"]

    @property
    def n_totalParametersFM(self) -> int:
        return self.retrieval_dict["n_totalParametersFM"]

    @property
    def n_speciesSys(self) -> int:
        return self.retrieval_dict["n_speciesSys"]

    def init_interferents(
        self,
        current_strategy_step: CurrentStrategyStep,
        current_state: CurrentState,
        o_retrievalInfo: mpy.ObjectView,
        error_analysis: ErrorAnalysis,
    ) -> None:
        """Update the various "Sys" stuff in o_retrievalInfo to add in
        the error analysis interferents"""
        sys_tokens = [str(i) for i in current_strategy_step.error_analysis_interferents]
        o_retrievalInfo.n_speciesSys = len(sys_tokens)
        o_retrievalInfo.speciesSys.extend(sys_tokens)
        if len(sys_tokens) == 0:
            return
        myspec = list(
            mpy.constraint_get_species(error_analysis.error_initial, sys_tokens)
        )
        o_retrievalInfo.n_totalParametersSys = len(myspec)
        for tk in sys_tokens:
            cnt = sum(t == tk for t in myspec)
            if cnt > 0:
                pstart = myspec.index(tk)
                o_retrievalInfo.parameterStartSys.append(pstart)
                o_retrievalInfo.parameterEndSys.append(pstart + cnt - 1)
                o_retrievalInfo.speciesListSys.extend([tk] * cnt)
            else:
                o_retrievalInfo.parameterStartSys.append(-1)
                o_retrievalInfo.parameterEndSys.append(-1)

    def add_species(
        self,
        species_name: str,
        current_strategy_step: CurrentStrategyStep,
        current_state: CurrentState,
        o_retrievalInfo: mpy.ObjectView,
    ) -> None:
        selem = current_state.full_state_element_old(
            StateElementIdentifier(species_name)
        )

        row = o_retrievalInfo.n_totalParameters
        rowFM = o_retrievalInfo.n_totalParametersFM
        mm = len(selem.initialGuessList)
        nn = len(selem.initialGuessListFM)
        o_retrievalInfo.pressureList.extend(selem.pressureList)
        o_retrievalInfo.altitudeList.extend(selem.altitudeList)
        o_retrievalInfo.speciesList.extend([species_name] * mm)
        o_retrievalInfo.pressureListFM.append(selem.pressureListFM)
        o_retrievalInfo.altitudeListFM.append(selem.altitudeListFM)
        o_retrievalInfo.speciesListFM.extend([species_name] * nn)
        o_retrievalInfo.constraintVector.append(selem.constraintVector)
        o_retrievalInfo.initialGuessList.append(selem.initialGuessList)
        o_retrievalInfo.initialGuessListFM.append(selem.initialGuessListFM)
        o_retrievalInfo.constraintVectorListFM.append(selem.constraintVectorFM)
        o_retrievalInfo.minimumList.append(selem.minimum)
        o_retrievalInfo.maximumList.append(selem.maximum)
        o_retrievalInfo.maximumChangeList.append(selem.maximum_change)
        o_retrievalInfo.trueParameterList.append(selem.trueParameterList)
        o_retrievalInfo.trueParameterListFM.append(selem.trueParameterListFM)
        o_retrievalInfo.mapToState.append(selem.mapToState)
        o_retrievalInfo.mapToParameters.append(selem.mapToParameters)
        o_retrievalInfo.parameterStart.append(row)
        o_retrievalInfo.parameterEnd.append(row + mm - 1)
        o_retrievalInfo.n_parameters.append(mm)
        o_retrievalInfo.n_parametersFM.append(nn)
        o_retrievalInfo.mapTypeList.extend([selem.mapType] * mm)
        o_retrievalInfo.mapTypeListFM.extend([selem.mapType] * nn)
        o_retrievalInfo.mapType.append(selem.mapType)

        o_retrievalInfo.Constraint = block_diag(
            o_retrievalInfo.Constraint, selem.constraintMatrix
        )
        o_retrievalInfo.parameterStartFM.append(rowFM)
        o_retrievalInfo.parameterEndFM.append(rowFM + nn - 1)
        o_retrievalInfo.n_totalParameters = row + mm
        o_retrievalInfo.n_totalParametersFM = rowFM + nn

    def init_joint(
        self,
        o_retrievalInfo: mpy.ObjectView,
        species_dir: Path,
        current_state: CurrentState,
    ) -> None:
        """This should get cleaned up somehow"""
        index_H2O = -1
        index_HDO = -1
        if "H2O" in o_retrievalInfo.species:
            index_H2O = o_retrievalInfo.species.index("H2O")

        if "HDO" in o_retrievalInfo.species:
            index_HDO = o_retrievalInfo.species.index("HDO")

        locs = [index_H2O, index_HDO]

        i_nh3type = current_state.full_state_value_str(
            StateElementIdentifier("nh3type")
        )
        i_ch3ohtype = current_state.full_state_value_str(
            StateElementIdentifier("ch3ohtype")
        )

        if locs[0] >= 0 and locs[1] >= 0:
            # HDO and H2O both retrieved in this step
            # only allow PREMADE type?
            names = ["H2O", "HDO"]

            loop_count = 0
            for xx in range(2):
                for yy in range(2):
                    loop_count += 1
                    specie1 = names[xx]
                    specie2 = names[yy]

                    filename = species_dir / f"{specie1}_{specie2}.asc"
                    if not filename.exists():
                        # If cannot find file, look for one with the species names swapped.
                        filename = species_dir / f"{specie2}_{specie1}.asc"

                    (_, fileID) = mpy.read_all_tes_cache(str(filename), "asc")

                    # AT_LINE 877 Get_Species_Information.pro
                    speciesInformationFile = mpy.tes_file_get_struct(fileID)

                    # get indices of location of where to place this matrix
                    # AT_LINE 880 Get_Species_Information.pro
                    loc11 = o_retrievalInfo.parameterStart[locs[xx]]
                    loc12 = o_retrievalInfo.parameterEnd[
                        locs[xx]
                    ]  # Note the spelling of this key 'parameterEnd'
                    loc21 = o_retrievalInfo.parameterStart[locs[yy]]
                    loc22 = o_retrievalInfo.parameterEnd[locs[yy]]

                    # AT_LINE 887 Get_Species_Information.pro
                    mm = o_retrievalInfo.n_parameters[
                        locs[0]
                    ]  # Note the spelling of this key 'n_parameters'

                    # We look for the tag 'constraintFilename' and it may not have the same case.
                    # So we will make everything lower case, find the index of that tag and use it to get to the value
                    # of 'constraintFilename' key in the tagNames.  Get the exact name of the tag using the index and then use
                    # that exact name to get to the value.
                    preferences = fileID["preferences"]
                    tagNames = [
                        x for x in preferences.keys()
                    ]  # Convert the dict keys into a regular list.
                    lowerTags = [x.lower() for x in tagNames]
                    ind_to_lower = -1
                    if "constraintFilename".lower() in lowerTags:
                        ind_to_lower = lowerTags.index("constraintFilename".lower())
                        actual_tag = tagNames[ind_to_lower]

                    if ind_to_lower < 0:
                        raise RuntimeError(
                            f"Name not found for constraintFilename. In file {speciesInformationFile}"
                        )

                    # AT_LINE 902 Get_Species_Information.pro
                    filename = preferences[actual_tag]

                    # At this point, the file name may not actually exist since it may contain the '_87' in the name.
                    # ../OSP/Constraint/H2O_HDO/Constraint_Matrix_H2O_NADIR_LOG_90S_90N_87.asc
                    # The next function will remove it and will attempt to read it.

                    constraint_species = specie1 + "_" + specie2

                    constraintMatrix, pressurex = (
                        mpy.supplier_constraint_matrix_premade(
                            constraint_species,
                            filename,
                            mm,
                            i_nh3type=i_nh3type,
                            i_ch3ohtype=i_ch3ohtype,
                        )
                    )
                    # PYTHON_NOTE: We add 1 to the end of the slice since Python does not include the slice end value.
                    o_retrievalInfo.Constraint[loc11 : loc12 + 1, loc21 : loc22 + 1] = (
                        constraintMatrix[:, :]
                    )
                    if loc11 != loc21:
                        o_retrievalInfo.Constraint[
                            loc21 : loc22 + 1, loc11 : loc12 + 1
                        ] = np.transpose(constraintMatrix)[:, :]
                # end for yy in range(0,1):
                # AT_LINE 921
            # end for xx in range(0,1):
            # AT_LINE 922
        # end if (locs[0] >= 0 and locs[1] >= 0):

    def init_data(
        self,
        error_analysis: ErrorAnalysis,
        species_dir: Path,
        current_strategy_step: CurrentStrategyStep,
        current_state: CurrentState,
    ) -> mpy.ObjectView:
        # This is a reworking of get_species_information in muses-py

        # errors propagated from step to step - possibly used as covariances
        #                                       but perhaps also as propagated
        #                                       constraints.

        # get retrieval parameters, including a list of retrieved species,
        # initial values for each parameter, true values for each parameter,
        # constraints for all parameters, maps for each parameter

        smeta = current_state.sounding_metadata
        o_retrievalInfod: dict[str, Any] = {
            # Info by retrieval parameter
            "surfaceType": "OCEAN" if smeta.is_ocean else "LAND",
            "speciesList": [],
            "pressureList": [],
            "altitudeList": [],
            "mapTypeList": [],
            "initialGuessList": [],
            "constraintVector": [],
            "trueParameterList": [],
            # optional allowed range and maximum stepsize during retrieval, set to -999 if not used
            "minimumList": [],
            "maximumList": [],
            "maximumChangeList": [],
            "doUpdateFM": None,
            "speciesListFM": [],
            "pressureListFM": [],
            "altitudeListFM": [],
            "initialGuessListFM": [],
            "constraintVectorListFM": [],
            "trueParameterListFM": [],
            "n_totalParametersFM": 0,
            "parameterStartFM": [],
            "parameterEndFM": [],
            "mapTypeListFM": [],
            # Info by species
            "n_speciesSys": 0,
            "speciesSys": [],
            "parameterStartSys": [],
            "parameterEndSys": [],
            "speciesListSys": [],
            "n_totalParametersSys": 0,
            "n_species": 0,
            "species": [],
            "parameterStart": [],
            "parameterEnd": [],
            "n_parametersFM": [],
            "n_parameters": [],
            "mapType": [],
            "mapToState": [],
            "mapToParameters": [],
            # Constraint & SaTrue, & info for all parameters
            "n_totalParameters": 0,
            "Constraint": np.zeros((0, 0), dtype=np.float64),
            "type": None,
        }
        # o_retrievalInfo OBJECT_TYPE dict

        # AT_LINE 83 Get_Species_Information.pro
        o_retrievalInfod["type"] = str(current_strategy_step.retrieval_type)

        o_retrievalInfo = mpy.ObjectView(
            o_retrievalInfod
        )  # Convert to object so we can use '.' to access member variables.

        # map types for all species
        # now in strategy table

        if str(current_strategy_step.retrieval_type).lower() in ("bt", "forwardmodel"):
            pass
        else:
            o_retrievalInfo.species = [
                str(i) for i in current_strategy_step.retrieval_elements
            ]
            o_retrievalInfo.n_species = len(o_retrievalInfo.species)

            for species_name in o_retrievalInfo.species:
                self.add_species(
                    species_name,
                    current_strategy_step,
                    current_state,
                    o_retrievalInfo,
                )

            self.init_interferents(
                current_strategy_step,
                current_state,
                o_retrievalInfo,
                error_analysis,
            )

        self.init_joint(o_retrievalInfo, species_dir, current_state)

        # Convert to numpy arrays
        for key in (
            "initialGuessList",
            "constraintVector",
            "trueParameterList",
            "trueParameterListFM",
            "pressureListFM",
            "altitudeListFM",
            "initialGuessListFM",
            "constraintVectorListFM",
            "minimumList",
            "maximumList",
            "maximumChangeList",
        ):
            if len(o_retrievalInfo.__dict__[key]) > 0:
                o_retrievalInfo.__dict__[key] = np.concatenate(
                    [a.flatten() for a in o_retrievalInfo.__dict__[key]]
                )
            else:
                o_retrievalInfo.__dict__[key] = np.zeros(0)
        # Few block diagonal matrixes
        for key in ("mapToState", "mapToParameters"):
            o_retrievalInfo.__dict__[key] = block_diag(*o_retrievalInfo.__dict__[key])
        o_retrievalInfo.Constraint = o_retrievalInfo.Constraint[
            0 : o_retrievalInfo.n_totalParameters, 0 : o_retrievalInfo.n_totalParameters
        ]
        o_retrievalInfo.doUpdateFM = np.zeros(o_retrievalInfo.n_totalParametersFM)
        o_retrievalInfo.speciesListSys = np.array(o_retrievalInfo.speciesListSys)
        # Not sure if these empty values are important or not, but for now
        # match what the existing muses-py code does.
        for k in (
            "speciesList",
            "speciesListFM",
            "mapTypeList",
            "speciesSys",
            "speciesListSys",
            "mapTypeListFM",
        ):
            if len(o_retrievalInfo.__dict__[k]) == 0:
                o_retrievalInfo.__dict__[k] = [
                    "",
                ]
        for k in (
            "altitudeList",
            "altitudeListFM",
            "constraintVector",
            "constraintVectorListFM",
            "initialGuessList",
            "initialGuessListFM",
            "maximumChangeList",
            "maximumList",
            "minimumList",
            "pressureList",
            "pressureListFM",
            "trueParameterList",
            "trueParameterListFM",
        ):
            if len(o_retrievalInfo.__dict__[k]) == 0:
                o_retrievalInfo.__dict__[k] = np.array(
                    [
                        0.0,
                    ]
                )
        for k in (
            "doUpdateFM",
            "parameterStartSys",
            "parameterEndSys",
        ):
            if len(o_retrievalInfo.__dict__[k]) == 0:
                o_retrievalInfo.__dict__[k] = np.array(
                    [
                        0,
                    ]
                )
        for k in ("mapToParameters", "mapToState", "Constraint"):
            if o_retrievalInfo.__dict__[k].shape[1] == 0:
                o_retrievalInfo.__dict__[k] = np.array(
                    [
                        [
                            0.0,
                        ]
                    ]
                )

        # Check the constaint vector for sanity.
        if np.all(np.isfinite(o_retrievalInfo.constraintVector)) == False:
            raise RuntimeError(
                f"NaN's in constraint vector!! Constraint vector is: {o_retrievalInfo.constraintVector}. Check species {o_retrievalInfo.speciesList}"
            )

        # Check the constaint matrix for sanity.
        if np.all(np.isfinite(o_retrievalInfo.Constraint)) == False:
            raise RuntimeError(
                f"NaN's in constraint matrix!! Constraint matrix is: {o_retrievalInfo.Constraint}. Check species {o_retrievalInfo.speciesList}"
            )

        return o_retrievalInfo


__all__ = [
    "RetrievalInfo",
]
