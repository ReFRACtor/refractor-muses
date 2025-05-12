from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
from .identifier import StateElementIdentifier
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .retrieval_result import RetrievalResult
    from .error_analysis import ErrorAnalysis
    from .current_state import CurrentState


# Needs a lot of cleanup, we are just shoving stuff into place
class ColumnResultSummary:
    def __init__(
        self, current_state: CurrentState, error_analysis: ErrorAnalysis
    ) -> None:
        self.current_state = current_state
        utilList = mpy.UtilList()
        stateInfo = FakeStateInfo(current_state)
        retrievalInfo = FakeRetrievalInfo(current_state)

        if np.max(stateInfo.true["values"]) > 0:
            have_true = True
        else:
            have_true = False
        num_species = retrievalInfo.n_species
        # This really is exactly 5. See the column calculation. This is
        # ["Column", "Trop", "UpperTrop", "LowerTrop", ""Strato
        num_cols = 5
        # AT_LINE 255 Write_Retrieval_Summary.pro
        # Now get species dependent preferences
        self._columnSpecies = []
        self._column = np.zeros((num_cols, num_species))
        self._columnDOFS = np.zeros(self._column.shape)
        self._columnPriorError = np.full(self._column.shape, -999.0)
        self._columnInitial = np.full(self._column.shape, -999.0)
        self._columnInitialInitial = np.full(self._column.shape, -999.0)
        self._columnError = np.full(self._column.shape, -999.0)
        self._columnPrior = np.full(self._column.shape, -999.0)
        self._column = np.full(self._column.shape, -999.0)
        self._columnTrue = np.full(self._column.shape, -999.0)

        self._columnAir = np.full((self._column.shape[0],), -999.0)
        self._columnPressureMax = np.zeros(self._columnAir.shape)
        self._columnPressureMin = np.zeros(self._columnAir.shape)
        self._H2O_H2OQuality = 0.0
        self._O3_columnErrorDU = 0.0
        self._O3_tropo_consistency = 0.0

        for ispecie in range(0, num_species):
            # AT_LINE 292 Write_Retrieval_Summary.pro
            species_name = retrievalInfo.species[ispecie]
            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]

            # AT_LINE 294 Write_Retrieval_Summary.pro
            if (loc >= 0) and (species_name != "TATM"):
                # Add the species_name to the current index.
                self._columnSpecies.append(species_name)
                indcol = len(self._columnSpecies) - 1

                # AT_LINE 301 Write_Retrieval_Summary.pro

                # EM NOTE - Adding stratosphere to the column for analysis
                for ij in range(0, 5):
                    # AT_LINE 303 Write_Retrieval_Summary.pro
                    if ij == 0:
                        my_type = "Column"

                        minPressure = 0.0
                        minIndex = np.int64(len(stateInfo.current["pressure"]) - 1)

                        maxPressure = np.amax(stateInfo.current["pressure"])
                    elif ij == 1:
                        my_type = "Trop"

                        minPressure = self.tropopause_pressure
                        minIndex = np.argmin(
                            np.abs(stateInfo.current["pressure"] - minPressure)
                        )

                        maxPressure = np.amax(stateInfo.current["pressure"])
                    elif ij == 2:
                        # upper tropopause
                        my_type = "UpperTrop"

                        maxPressure = 500

                        minPressure = self.tropopause_pressure
                        minIndex = np.argmin(
                            np.abs(stateInfo.current["pressure"] - minPressure)
                        )
                    elif ij == 3:
                        # lower troposphere
                        my_type = "LowerTrop"

                        minPressure = 500
                        minIndex = np.argmin(
                            np.abs(stateInfo.current["pressure"] - minPressure)
                        )

                        maxPressure = np.amax(stateInfo.current["pressure"])
                    elif ij == 4:
                        # Stratosphere
                        my_type = "Strato"
                        minPressure = 0
                        minIndex = np.int64(len(stateInfo.current["pressure"]) - 1)

                        maxPressure = self.tropopause_pressure
                        if maxPressure == 0:
                            maxPressure = 200.0
                    else:
                        raise RuntimeError("Type not found")
                    # end if (ij == 0):

                    # AT_LINE 336 Write_Retrieval_Summary.pro
                    self._columnPressureMin[ij] = minPressure
                    self._columnPressureMax[ij] = maxPressure

                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    my_map = retrievalInfo.mapToState
                    if my_map is None:
                        raise RuntimeError("basis matrix missing")

                    mapType = retrievalInfo.mapType[ispecie]

                    linear = 0
                    if mapType.lower() == "linear":
                        linear = 1
                    if mapType.lower() == "linearpca":
                        linear = 1

                    indSpecie = loc
                    indH2O = utilList.WhereEqualIndices(stateInfo.species, "H2O")[0]
                    indTATM = utilList.WhereEqualIndices(stateInfo.species, "TATM")[0]

                    # AT_LINE 357 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.constraint["values"][indSpecie, :],
                        stateInfo.constraint["pressure"],
                        stateInfo.constraint["values"][indTATM, :],
                        stateInfo.constraint["values"][indH2O, :],
                        stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                        stateInfo.current["latitude"],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None,
                    )

                    self._columnPrior[ij, indcol] = x["column"]

                    # AT_LINE 368 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.initial["values"][indSpecie, :],
                        stateInfo.initial["pressure"],
                        stateInfo.initial["values"][indTATM, :],
                        stateInfo.initial["values"][indH2O, :],
                        stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                        stateInfo.current["latitude"],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None,
                    )

                    self._columnInitial[ij, indcol] = x["column"]

                    # AT_LINE 379 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.initialInitial["values"][indSpecie, :],
                        stateInfo.initialInitial["pressure"],
                        stateInfo.initialInitial["values"][indTATM, :],
                        stateInfo.initialInitial["values"][indH2O, :],
                        stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                        stateInfo.current["latitude"],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None,
                    )

                    self._columnInitialInitial[ij, indcol] = x["column"]

                    # AT_LINE 390 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.current["values"][indSpecie, :],
                        stateInfo.current["pressure"],
                        stateInfo.current["values"][indTATM, :],
                        stateInfo.current["values"][indH2O, :],
                        stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                        stateInfo.current["latitude"],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None,
                    )

                    self._column[ij, indcol] = x["column"]

                    # AT_LINE 400 Write_Retrieval_Summary.pro
                    # air column
                    x = mpy.column(
                        stateInfo.current["values"][indSpecie, :] * 0 + 1,
                        stateInfo.current["pressure"],
                        stateInfo.current["values"][indTATM, :],
                        stateInfo.current["values"][indH2O, :],
                        stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                        stateInfo.current["latitude"],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None,
                    )

                    self._columnAir[ij] = x["columnAir"]

                    # AT_LINE 411 Write_Retrieval_Summary.pro
                    if species_name == "O3" and my_type == "Trop":
                        # compare initial gues for this step to retrieved.
                        ret = self._column[ij, indcol]
                        ig = self._columnInitial[ij, indcol]
                        ratio = (ret / ig) - 1.0
                        self._O3_tropo_consistency = ratio
                    # end if species_name == 'O3' and my_type == 'Trop':

                    # AT_LINE 420 Write_Retrieval_Summary.pro
                    # true values
                    # only for synthetic data
                    if have_true:
                        x = mpy.column(
                            stateInfo.true["values"][indSpecie, :],
                            stateInfo.true["pressure"],
                            stateInfo.true["values"][indTATM, :],
                            stateInfo.true["values"][indH2O, :],
                            stateInfo.current["tsa"]["surfaceAltitudeKm"] * 1000,
                            stateInfo.current["latitude"],
                            minPressure,
                            maxPressure,
                            linear,
                            pge=None,
                        )

                        self._columnTrue[ij, indcol] = x["column"]

                    # AT_LINE 435 Write_Retrieval_Summary.pro
                    Sx = error_analysis.Sx[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]

                    derivativeFinal = np.copy(x["derivative"])

                    mapType = mapType.lower()
                    if mapType == "log":
                        # PYTHON_NOTE: It is possible that the length of x['derivative'] is greater than stateInfo.current['values'][indSpecie,:]
                        #              In that case, we make sure both terms below on the right hand side are the same sizes.
                        rhs_term_sizes = len(stateInfo.current["values"][indSpecie, :])
                        derivativeFinal = (
                            x["derivative"][0:rhs_term_sizes]
                            * stateInfo.current["values"][indSpecie, 0:rhs_term_sizes]
                        )

                    # IDL:
                    # error = SQRT(derivativeFinal[0:minIndex] ## Sx[0:minIndex,0:minIndex] ## TRANSPOSE(derivativeFinal[0:minIndex])

                    error = np.sqrt(
                        derivativeFinal[0 : minIndex + 1].T
                        @ Sx[0 : minIndex + 1, 0 : minIndex + 1]
                        @ derivativeFinal[0 : minIndex + 1]
                    )
                    self._columnError[ij, indcol] = error

                    # AT_LINE 446 Write_Retrieval_Summary.pro
                    # multipy prior covariance to calc predicted prior error
                    if mapType == "log":
                        # PYTHON_NOTE: It is possible that the length of x['derivative'] is greater than stateInfo.current['values'][indSpecie,:]
                        #              In that case, we make sure both terms below on the right hand side are the same sizes.
                        rhs_term_sizes = len(stateInfo.initial["values"][indSpecie, :])
                        derivativeFinal = (
                            x["derivative"][0:rhs_term_sizes]
                            * stateInfo.initial["values"][indSpecie, 0:rhs_term_sizes]
                        )

                    # IDL:
                    # error = SQRT(derivativeFinal[0:minIndex] ## results.Sa[0:minIndex,0:minIndex] ## TRANSPOSE(derivativeFinal[0:minIndex])

                    error = np.sqrt(
                        derivativeFinal[0 : minIndex + 1].T
                        @ error_analysis.Sa[0 : minIndex + 1, 0 : minIndex + 1]
                        @ derivativeFinal[0 : minIndex + 1]
                    )

                    self._columnPriorError[ij, indcol] = error

                    # AT_LINE 455 Write_Retrieval_Summary.pro
                    if species_name == "O3" and my_type == "Column":
                        self._O3_columnErrorDU = self._columnError[ij, indcol] / 2.69e16
                    # end if species_name == 'O3' and my_type == 'Column':

                    if my_type == "Column" and species_name == "H2O":
                        ind4 = utilList.WhereEqualIndices(
                            retrievalInfo.speciesListFM, "H2O"
                        )
                        ind5 = utilList.WhereEqualIndices(
                            retrievalInfo.speciesListFM, "HDO"
                        )

                        if len(ind4) > 0 and len(ind5) > 0:
                            # in H2O/HDO step, check H2O column - H2O column from O3 step / error
                            self._H2O_H2OQuality = (
                                self._column[ij, indcol]
                                - self._columnInitial[ij, indcol]
                            ) / self._columnPriorError[ij, indcol]
                        # end if len(ind4) > 0 and len(ind5) > 0
                    # end if my_type == 'Column' and species_name == 'H2O':

                    # AT_LINE 474 Write_Retrieval_Summary.pro
                    # calculate DOFs for different ranges
                    # based on layers
                    # each level corresponds to a layer which ranges to 1/2
                    # between it and level below to halfway between it and
                    # level above.  So if range was 1000-100 hPa, and there
                    # was a level at 100 hPa, only half of the AK at 100 hPa would be
                    # included because only half of the above described layer is inclu

                    ispecie = utilList.WhereEqualIndices(
                        retrievalInfo.species, species_name
                    )[0]
                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    ak = mpy.get_diagonal(error_analysis.A)
                    ak = ak[ind1FM : ind2FM + 1]
                    na = len(ak)

                    pressureLayers = np.asarray(stateInfo.current["pressure"][0])
                    pressureLayers = np.append(
                        pressureLayers,
                        (
                            stateInfo.current["pressure"][1:]
                            + stateInfo.current["pressure"][0 : na - 1]
                        )
                        / 2,
                    )

                    indp = np.where(
                        (pressureLayers >= (minPressure - 0.0001))
                        & (pressureLayers < (maxPressure + 0.0001))
                    )[0]

                    dof = np.sum(ak[indp[0 : len(indp) - 1]])

                    # PYTHON_NOTE: It is possible with the where() function below, the array returned is empty.
                    #              If it is empty, we cannot use np.amax() so we have to do two separated steps.
                    #              as opposed to IDL which does it it one step: indp1 = max(where(pressureLayers GT maxPressure))
                    max_indices = np.where(pressureLayers > maxPressure)[0]

                    indp1 = np.int64(-1)
                    if len(max_indices) > 0:
                        indp1 = np.amax(max_indices)

                    if indp1 != -1:
                        fraction1 = (maxPressure - pressureLayers[indp1 + 1]) / (
                            pressureLayers[indp1] - pressureLayers[indp1 + 1]
                        )
                        dof = dof + fraction1 * ak[indp1]

                    indp2 = np.int64(-1)
                    min_indices = np.where(pressureLayers < minPressure)[0]
                    if len(min_indices) > 0:
                        indp2 = np.amin(min_indices)

                    if indp2 != -1:
                        fraction2 = (minPressure - pressureLayers[indp2 - 1]) / (
                            pressureLayers[indp2] - pressureLayers[indp2 - 1]
                        )
                        dof = dof + fraction2 * ak[indp2]

                    self._columnDOFS[ij, indcol] = dof
                # end for ij in range(0, 5):

                # AT_LINE 505 Write_Retrieval_Summary.pro
            # end if (loc >= 0) and (species_name != 'TATM'):
            continue
        # end for ispecie in range(0,num_species):

    def state_value(self, state_name: str) -> float:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))[
            0
        ]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))

    @property
    def tropopause_pressure(self) -> float:
        res = self.state_value("gmaoTropopausePressure")
        if res <= -990:
            raise RuntimeError("GMA tropopause pressure is not defined")
        return res
    
    @property
    def H2O_H2OQuality(self) -> float:
        return self._H2O_H2OQuality

    @property
    def O3_columnErrorDU(self) -> float:
        return self._O3_columnErrorDU

    @property
    def O3_tropo_consistency(self) -> float:
        return self._O3_tropo_consistency

    @property
    def columnDOFS(self) -> np.ndarray:
        return self._columnDOFS

    @property
    def columnPriorError(self) -> np.ndarray:
        return self._columnPriorError

    @property
    def columnInitial(self) -> np.ndarray:
        return self._columnInitial

    @property
    def columnInitialInitial(self) -> np.ndarray:
        return self._columnInitialInitial

    @property
    def columnError(self) -> np.ndarray:
        return self._columnError

    @property
    def columnPrior(self) -> np.ndarray:
        return self._columnPrior

    @property
    def column(self) -> np.ndarray:
        return self._column

    @property
    def columnAir(self) -> np.ndarray:
        return self._columnAir

    @property
    def columnTrue(self) -> np.ndarray:
        return self._columnTrue

    @property
    def columnPressureMax(self) -> np.ndarray:
        return self._columnPressureMax

    @property
    def columnPressureMin(self) -> np.ndarray:
        return self._columnPressureMin

    @property
    def columnSpecies(self) -> list[str]:
        return self._columnSpecies


__all__ = [
    "ColumnResultSummary",
]
