from __future__ import annotations
from .identifier import StateElementIdentifier
from .state_element import StateElementImplementation, StateElementHandle, StateElement
from .current_state import FullGridMappedArray, RetrievalGrid2dArray, FullGrid2dArray
import numpy as np
import typing

if typing.TYPE_CHECKING:
    pass


class OmiEofStateElement(StateElementImplementation):
    """Sample new StateElement"""

    def __init__(
        self,
        state_element_id: StateElementIdentifier = StateElementIdentifier("OMIEOFUV1"),
        number_eof: int = 3,
    ) -> None:
        self.number_eof = number_eof
        value = np.zeros(number_eof).view(FullGridMappedArray)
        constraint_value = value.copy().view(FullGridMappedArray)
        constraint_matrix = np.diag([10 * 10.0] * self.number_eof).view(
            RetrievalGrid2dArray
        )
        apriori_cov_fm = np.diag([0.1 * 0.1] * self.number_eof).view(FullGrid2dArray)
        super().__init__(
            state_element_id, value, constraint_value, apriori_cov_fm, constraint_matrix
        )


class OmiEofStateElementHandle(StateElementHandle):
    def __init__(self, state_element_id: StateElementIdentifier):
        self._state_element_id = state_element_id

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self._state_element_id:
            return None
        return OmiEofStateElement(self._state_element_id)


__all__ = ["OmiEofStateElement", "OmiEofStateElementHandle"]
