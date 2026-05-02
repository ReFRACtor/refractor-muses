from __future__ import annotations
from .muses_altitude_pge import MusesAltitudePge
from .identifier import StateElementIdentifier
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .error_analysis import ErrorAnalysis
    from .current_state import CurrentState


# Needs a lot of cleanup, we are just shoving stuff into place
class ColumnResultSummary:
    def __init__(
        self, current_state: CurrentState, error_analysis: ErrorAnalysis
    ) -> None:
        # Temp, we want to remove this
        from refractor.muses_py_fm import FakeRetrievalInfo

        self.current_state = current_state
        retrievalInfo = FakeRetrievalInfo(current_state)

        # I don't think this is fully supported, so we just always say
        # have_true is False
        have_true = False
        num_species = retrievalInfo.n_species
        # This really is exactly 5. See the column calculation. This is
        # ["Column", "Trop", "UpperTrop", "LowerTrop", ""Strato
        num_cols = 5
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

        self.pselem = self.current_state.state_element("pressure")
        self.hselem = self.current_state.state_element("H2O")
        self.tselem = self.current_state.state_element("TATM")
        # Set of columns we process for. This gives a name, min pressure
        # and max pressure
        column_set = [
            ["Column", 0, np.amax(self.pselem.value_fm)],
            ["Trop", self.tropopause_pressure, np.amax(self.pselem.value_fm)],
            ["UpperTrop", self.tropopause_pressure, 500],
            ["LowerTrop", 500, np.amax(self.pselem.value_fm)],
            ["Strato", 0, self.tropopause_pressure],
        ]
        for ispecie in range(0, num_species):
            species_name = retrievalInfo.species[ispecie]
            selem_name = StateElementIdentifier(species_name)
            self.selem = self.current_state.state_element(selem_name)
            if selem_name.is_atmospheric_species and species_name != "TATM":
                self._columnSpecies.append(species_name)
                indcol = len(self._columnSpecies) - 1
                for ij, (my_type, self.min_pressure, self.max_pressure) in enumerate(
                    column_set
                ):
                    if self.min_pressure == 0.0:
                        min_index = np.int64(len(self.pselem.value_fm) - 1)
                    else:
                        min_index = np.argmin(
                            np.abs(self.pselem.value_fm - self.min_pressure)
                        )
                    if self.max_pressure == 0:
                        self.max_pressure = 200.0

                    self._columnPressureMin[ij] = self.min_pressure
                    self._columnPressureMax[ij] = self.max_pressure

                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    map_type = retrievalInfo.mapType[ispecie]
                    self.linear = map_type.lower() in ("linear", "linearpca")

                    self._columnPrior[ij, indcol] = self.column_calc(
                        "constraint_vector_fm"
                    )
                    self._columnInitial[ij, indcol] = self.column_calc(
                        "step_initial_fm"
                    )
                    self._columnInitialInitial[ij, indcol] = self.column_calc(
                        "retrieval_initial_fm"
                    )
                    self._column[ij, indcol] = self.column_calc("value_fm")
                    self._columnAir[ij] = self.column_calc("value_fm", do_air=True)
                    # only for synthetic data
                    if have_true:
                        self._columnTrue[ij, indcol] = self.column_calc("true_value_fm")

                    if species_name == "O3" and my_type == "Trop":
                        # compare initial gues for this step to retrieved.
                        ret = self._column[ij, indcol]
                        ig = self._columnInitial[ij, indcol]
                        ratio = (ret / ig) - 1.0
                        self._O3_tropo_consistency = ratio

                    Sx = error_analysis.Sx[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]

                    xder = self.column_calc("value_fm", do_air=True, do_deriv=True)
                    derivativeFinal = np.copy(xder)

                    map_type = map_type.lower()
                    if map_type == "log":
                        rhs_term_sizes = len(self.selem.value_fm)
                        derivativeFinal = (
                            xder[0:rhs_term_sizes]
                            * self.selem.value_fm[0:rhs_term_sizes]
                        )

                    error = np.sqrt(
                        derivativeFinal[0 : min_index + 1].T
                        @ Sx[0 : min_index + 1, 0 : min_index + 1]
                        @ derivativeFinal[0 : min_index + 1]
                    )
                    self._columnError[ij, indcol] = error

                    if map_type == "log":
                        rhs_term_sizes = len(self.selem.step_initial_fm)
                        derivativeFinal = (
                            xder[0:rhs_term_sizes]
                            * self.selem.step_initial_fm[0:rhs_term_sizes]
                        )

                    error = np.sqrt(
                        derivativeFinal[0 : min_index + 1].T
                        @ error_analysis.Sa[0 : min_index + 1, 0 : min_index + 1]
                        @ derivativeFinal[0 : min_index + 1]
                    )

                    self._columnPriorError[ij, indcol] = error

                    if species_name == "O3" and my_type == "Column":
                        self._O3_columnErrorDU = self._columnError[ij, indcol] / 2.69e16

                    if my_type == "Column" and species_name == "H2O":
                        if (
                            "H2O" in retrievalInfo.speciesListFM
                            and "HDO" in retrievalInfo.speciesListFM
                        ):
                            # in H2O/HDO step, check H2O column - H2O
                            # column from O3 step / error
                            self._H2O_H2OQuality = (
                                self._column[ij, indcol]
                                - self._columnInitial[ij, indcol]
                            ) / self._columnPriorError[ij, indcol]

                    # calculate DOFs for different ranges
                    # based on layers
                    # each level corresponds to a layer which ranges to 1/2
                    # between it and level below to halfway between it and
                    # level above.  So if range was 1000-100 hPa, and there
                    # was a level at 100 hPa, only half of the AK at 100 hPa would be
                    # included because only half of the above described layer is inclu

                    ispecie = retrievalInfo.species.index(species_name)
                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    ak = np.diagonal(error_analysis.A)
                    ak = ak[ind1FM : ind2FM + 1]
                    na = len(ak)

                    pressure_layer = np.asarray(self.pselem.value_fm[0])
                    pressure_layer = np.append(
                        pressure_layer,
                        (self.pselem.value_fm[1:] + self.pselem.value_fm[0 : na - 1])
                        / 2,
                    )

                    indp = np.where(
                        (pressure_layer >= (self.min_pressure - 0.0001))
                        & (pressure_layer < (self.max_pressure + 0.0001))
                    )[0]

                    dof = np.sum(ak[indp[0 : len(indp) - 1]])

                    # PYTHON_NOTE: It is possible with the where() function below, the array returned is empty.
                    #              If it is empty, we cannot use np.amax() so we have to do two separated steps.
                    #              as opposed to IDL which does it it one step: indp1 = max(where(pressure_layer GT self.max_pressure))
                    max_indices = np.where(pressure_layer > self.max_pressure)[0]

                    indp1 = np.int64(-1)
                    if len(max_indices) > 0:
                        indp1 = np.amax(max_indices)

                    if indp1 != -1:
                        fraction1 = (self.max_pressure - pressure_layer[indp1 + 1]) / (
                            pressure_layer[indp1] - pressure_layer[indp1 + 1]
                        )
                        dof = dof + fraction1 * ak[indp1]

                    indp2 = np.int64(-1)
                    min_indices = np.where(pressure_layer < self.min_pressure)[0]
                    if len(min_indices) > 0:
                        indp2 = np.amin(min_indices)

                    if indp2 != -1:
                        fraction2 = (self.min_pressure - pressure_layer[indp2 - 1]) / (
                            pressure_layer[indp2] - pressure_layer[indp2 - 1]
                        )
                        dof = dof + fraction2 * ak[indp2]

                    self._columnDOFS[ij, indcol] = dof

    def column_calc(
        self, ctype: str, do_air: bool = False, do_deriv: bool = False
    ) -> np.ndarray:
        x = MusesAltitudePge.column(
            getattr(self.selem, ctype)
            if not do_air
            else np.full_like(getattr(self.selem, ctype), 1),
            getattr(self.pselem, ctype),
            getattr(self.tselem, ctype),
            getattr(self.hselem, ctype),
            self.current_state.sounding_metadata.surface_altitude.convert("m").value,
            self.current_state.sounding_metadata.latitude.value,
            self.min_pressure,
            self.max_pressure,
            self.linear,
        )
        if do_deriv:
            return x["derivative"]
        if do_air:
            return x["columnAir"]
        return x["column"]

    def state_value(self, state_name: str) -> float:
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        return self.current_state.state_value(StateElementIdentifier(state_name))

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
