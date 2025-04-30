from __future__ import annotations
from .identifier import StateElementIdentifier
from .state_element import StateElementImplementation, StateElementHandle
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MeasurementId
    from .muses_strategy_executor import CurrentStrategyStep


class OmiEofStateElement(StateElementImplementation):
    """Sample new StateElement
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier = StateElementIdentifier("OMIEOFUV1"),
        number_eof: int = 3,
    ) -> None:
        self.number_eof = number_eof
        value = np.zeros(number_eof)
        apriori_value = value.copy()
        constraint_matrix = np.diag([10 * 10.0] * self.number_eof)
        apriori_cov_fm = np.diag([0.1 * 0.1] * self.number_eof)
        super().__init__(state_element_id, value, apriori_value, apriori_cov_fm,
                         constraint_matrix)

class OmiEofStateElementHandle(StateElementHandle):
    def __init__(self, state_element_id: StateElementIdentifier):
        self._state_element_id = state_element_id
        
    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self._state_element_id:
           return None
        return OmiEofStateElement(self._state_element_id)
        
__all__ = [
    "OmiEofStateElement", "OmiEofStateElementHandle"
]
