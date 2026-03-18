from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
from typing import Self


class StateMappingUpdateArray(rf.StateMapping):
    """This is similar to rf.StateMappingAtIndexes, but the indices
    that aren't mapped are actually in the state vector as
    placeholders. This is the way py-retrieve handled the state
    elements that depend on frequency (e.g., EmisState).

    It would probably be cleaner just to not have the unused parts of
    the state vector not there, i.e., just replace this mapping with a
    rf.StateMappingAtIndexes. However it isn't clear yet what parts of
    the code depend on having the full state vector including
    placeholders.

    For now, we duplicate what py-retrieve does. We may come back and clean
    this up.

    TODO - consider removing unused element from the StateVector.
    """

    def __init__(self, update_array: np.ndarray | None) -> None:
        super().__init__()
        self.update_array = update_array
        if self.update_array is not None and self.update_array.shape[0] == 0:
            self.update_array = None
        self.initial_value: None | np.ndarray = None

    def clone(self) -> Self:
        return StateMappingUpdateArray(self.update_array.copy() if self.update_array is not None else None)

    def _v_name(self) -> str:
        return "update array"

    def mapped_state(
        self, retrieval_values: rf.ArrayAd_double_1
    ) -> rf.ArrayAd_double_1:
        if self.initial_value is None:
            raise RuntimeError("Initial state has not yet been initialized")
        if (
            self.update_array is not None
            and self.initial_value.shape[0] != self.update_array.shape[0]
        ):
            raise RuntimeError(
                "initial_value and update_array need to be the same size"
            )
        if retrieval_values.rows != self.initial_value.shape[0]:
            raise RuntimeError(
                "retrieval_values and initial_value need to be the same size"
            )
        if self.update_array is None:
            return retrieval_values
        res = rf.ArrayAd_double_1(
            retrieval_values.rows, retrieval_values.number_variable
        )
        for i in range(self.update_array.shape[0]):
            if self.update_array[i]:
                res[i] = retrieval_values[i]
            else:
                # TODO Look into this
                # Note it actually seems wrong that we have a nonzero jacobian here,
                # but this is what py-retrieve does.
                #
                # However, I did try removing this, and things changed a lot. It is
                # possible the error analysis etc. depends on the jacobian (e.g. it is
                # a bit like the systematic jacobians). Somebody smarter than me
                # will need to look into this
                res[i] = rf.AutoDerivativeDouble(
                    self.initial_value[i], retrieval_values[i].gradient
                )
        return res

    def retrieval_state(
        self, initial_value: rf.ArrayAd_double_1
    ) -> rf.ArrayAd_double_1:
        self.initial_value = initial_value.value.copy()
        if (
            self.update_array is not None
            and self.initial_value.shape[0] != self.update_array.shape[0]
        ):
            raise RuntimeError(
                "initial_value and update_array need to be the same size"
            )
        return initial_value


__all__ = [
    "StateMappingUpdateArray",
]
