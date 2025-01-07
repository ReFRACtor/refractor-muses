from __future__ import annotations
from .state_info import RetrievableStateElement, StateInfo
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .retrieval_configuration import RetrievalConfiguration
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

    def __init__(self, state_info: StateInfo, name="OMIEOFUV1", number_eof=3):
        super().__init__(state_info, name)
        self._value = np.zeros(number_eof)
        self._constraint = self._value.copy()
        self.number_eof = number_eof

    def sa_covariance(self) -> np.ndarray:
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        # TODO, Double check this. Not sure of the connection between this
        # and the constraintMatrix. Note the pressure is right, this
        # indicates we aren't on levels so we don't need a pressure
        return np.diag([10 * 10.0] * self.number_eof), [-999.0] * self.number_eof

    @property
    def value(self):
        return self._value

    def should_write_to_l2_product(self, instruments) -> bool:
        if "OMI" in instruments:
            return True
        return False

    def net_cdf_variable_name(self) -> str:
        # Want names like OMI_EOF_UV1
        return self.name.replace("EOF", "_EOF_")

    def net_cdf_struct_units(self) -> dict:
        """Returns the attributes attached to a netCDF write out of this
        StateElement."""
        return {
            "Longname": "OMI EOF scale factors",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        }

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.array,
        update_next: bool,
        retrieval_config: RetrievalConfiguration,
        step: int,
        do_update_fm: np.array,
    ):
        # If we are requested not to update the next step, then save a copy
        # of this to reset the value
        if not update_next:
            self.state_info.next_state[self.name] = self.clone_for_other_state()
        self._value = results_list[retrieval_info.species_list == self._name]

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
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
