from __future__ import annotations
from .state_info import StateElement, StateElementHandle
import refractor.framework as rf  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .current_state import CurrentStateStateInfoOld

class StateElementOldWrapper(StateElement):
    """This wraps around a CurrentStateStateInfoOld and presents the
    data like a StateElement. This is used for backwards testing against
    our StateInfoOld that wraps around the old py-retrieve state info code.
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        current_state_old: CurrentStateStateInfoOld,
    ) -> None:
        super().__init__(state_element_id)
        self._current_state_old = current_state_old

    def sa_cross_covariance(self, selem2: StateElement) -> np.ndarray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        selem_old = self._current_state_old.full_state_element(self.state_element_id)
        selem2_old = self._current_state_old.full_state_element(selem2.state_element_id)
        return selem_old.sa_cross_covariance(selem2_old)

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return rf.SpectralDomain(self.spectral_domain_wavelength, rf.Unit("nm"))

    @property
    def spectral_domain_wavelength(self) -> np.ndarray | None:
        """Short cut to return the spectral domain in units of nm."""
        return self._current_state_old.full_state_spectral_domain_wavelength(
            self.state_element_id
        )

    @property
    def value(self) -> np.ndarray:
        """Current value of StateElement"""
        return self._current_state_old.full_state_value(self.state_element_id)

    @property
    def value_str(self) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like value, but we
        return a str instead. For most StateElement, this returns "None"
        instead which indicates we don't have a str value.

        It isn't clear that this is the best interface, on the other hand these
        str values don't have an obvious other place to go. And these value are
        at least in the spirit of other StateElement. So at least for now we
        will support this, possibly reworking this in the future.
        """
        return self._current_state_old.full_state_value_str(self.state_element_id)

    @property
    def apriori(self) -> np.ndarray:
        """Apriori value of StateElement"""
        return self._current_state_old.full_state_apriori_value(
            self.state_element_id
        )

    @property
    def apriori_cov(self) -> np.ndarray:
        """Apriori Covariance"""
        return self._current_state_old.full_state_apriori_covariance(
            self.state_element_id
        )

    @property
    def retrieval_initial_value(self) -> np.ndarray:
        """Value StateElement had at the start of the retrieval."""
        return self._current_state_old.full_state_retrieval_initial_value(
            self.state_element_id
        )

    @property
    def step_initial_value(self) -> np.ndarray:
        """Value StateElement had at the start of the retrieval step."""
        return self._current_state_old.full_state_step_initial_value(
            self.state_element_id
        )

    @property
    def true_value(self) -> np.ndarray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return self._current_state_old.full_state_true_value(self.state_element_id)

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        """Update the value of the StateElement. This function updates
        each of the various values passed in.  A value of 'None' (the
        default) means skip updating that part of the StateElement.
        """
        self._current_state_old.update_full_state_element(
            self.state_element_id,
            current,
            apriori,
            step_initial,
            retrieval_initial,
            true_value,
        )


class StateElementOldWrapperHandle(StateElementHandle):
    def __init__(self) -> None:
        from .current_state import CurrentStateStateInfoOld
        self._current_state_old = CurrentStateStateInfoOld(None)

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        return StateElementOldWrapper(state_element_id, self._current_state_old)


__all__ = [
    "StateElementOldWrapper",
    "StateElementOldWrapperHandle"
]
