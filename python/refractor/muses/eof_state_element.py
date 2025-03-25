from __future__ import annotations
from .state_info import RetrievableStateElement, StateInfo
from .identifier import StateElementIdentifier
import numpy as np
from typing import Tuple
import typing

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MeasurementId
    from .muses_strategy_executor import CurrentStrategyStep


class OmiEofStateElement(RetrievableStateElement):
    """Note that we may rework this. Not sure how much we need
    specific StateElement vs. handling a class of them. But for now,
    we have the EOF as a separate StateElement as we work out what
    exactly we want to do with new ReFRACtor only StateElement.

    We can use the SingleSpeciesHandle to add this in, e.g.,

    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("OMIEOFUV1",
         EofStateElement, pass_state=False))

    """

    def __init__(
        self,
        state_info: StateInfo,
        name: StateElementIdentifier = StateElementIdentifier("OMIEOFUV1"),
        number_eof: int = 3,
    ) -> None:
        super().__init__(state_info, name)
        self._value = np.zeros(number_eof)
        self._constraint = self._value.copy()
        self.number_eof = number_eof

    def sa_covariance(self) -> Tuple[np.ndarray, list[float]]:
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        # TODO, Double check this. Not sure of the connection between this
        # and the constraintMatrix. Note the pressure is right, this
        # indicates we aren't on levels so we don't need a pressure
        return np.diag([10 * 10.0] * self.number_eof), [-999.0] * self.number_eof

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def apriori_value(self) -> np.ndarray:
        return np.array(
            [
                0.0,
            ]
        )

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        if current is not None:
            self._value = current
        if (
            apriori is not None
            or initial is not None
            or initial_initial is not None
            or true is not None
        ):
            raise NotImplementedError

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
        do_update_fm: np.ndarray,
    ) -> None:
        self._value = results_list[
            np.array(retrieval_info.species_list) == str(self._name)
        ]

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        self.mapType = "linear"
        self.pressureList = np.full((self.number_eof), -2.0)
        self.altitudeList = np.full((self.number_eof), -2.0)
        self.pressureListFM = self.pressureList
        self.altitudeListFM = self.altitudeList
        # Apriori
        self.constraintVector = self._constraint.copy()
        # Normally the same as apriori, but doesn't have to be
        self.initialGuessList = self.value.copy()
        self.trueParameterList = np.zeros((self.number_eof))
        self.constraintVectorFM = self.constraintVector
        self.initialGuessListFM = self.initialGuessList
        self.trueParameterListFM = self.trueParameterList
        self.minimum = np.full((self.number_eof), -999.0)
        self.maximum = np.full((self.number_eof), -999.0)
        self.maximum_change = np.full((self.number_eof), -999.0)
        self.mapToState = np.eye(self.number_eof)
        self.mapToParameters = np.eye(self.number_eof)
        # Not sure if the is covariance, or sqrt covariance. Note this
        # does not seem to the be the same as the Sa used in the error
        # analysis. I think muses-py uses the constraintMatrix sort of
        # like a weighting that is independent of apriori covariance.
        self.constraintMatrix = np.diag(np.full((self.number_eof,), 10 * 10.0))


__all__ = [
    "OmiEofStateElement",
]
