from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
import numpy as np
import abc
import typing
import copy
from collections import UserDict

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .muses_observation import MeasurementId


class StateElement(object, metaclass=abc.ABCMeta):
    """This handles a single StateElement, identified by
    a StateElementIdentifier (e.g., the name "TATM" or
    "pressure".

    A StateElement has a current value, a apriori value, a
    retrieval_initial_value (at the start of a full multi-step retrieval)
    and step_initial_value (at the start the current retrieval step).
    In addition, we possibly have "true" value, e.g, we know the answer
    because we are simulating data or something like that. Not
    every state element has a "true" value, is None in those cases.

    Note as a convention, we always return the values as a np.ndarray,
    even for single scalar values. This just saves on needing to have lots
    of "if ndarray else if scalar" code. A scalar is returned as a np.ndarray
    of size 1.
    """

    def __init__(self, state_element_id: StateElementIdentifier):
        self._state_element_id = state_element_id

    @property
    def state_element_id(self) -> StateElementIdentifier:
        return self._state_element_id

    def sa_cross_covariance(self, selem2: StateElement) -> np.ndarray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return None

    @property
    def spectral_domain_wavelength(self) -> np.ndarray | None:
        """Short cut to return the spectral domain in units of nm."""
        sd = self.spectral_domain
        if sd is None:
            return None
        return sd.convert_wave(rf.Unit("nm"))

    @abc.abstractproperty
    def value(self) -> np.ndarray:
        """Current value of StateElement"""
        raise NotImplementedError

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
        return None

    @abc.abstractproperty
    def apriori(self) -> np.ndarray:
        """Apriori value of StateElement"""
        raise NotImplementedError

    @property
    def apriori_cov(self) -> np.ndarray:
        """Apriori Covariance"""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_initial_value(self) -> np.ndarray:
        """Value StateElement had at the start of the retrieval."""
        raise NotImplementedError

    @abc.abstractproperty
    def step_initial_value(self) -> np.ndarray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError

    @property
    def true_value(self) -> np.ndarray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return None

    @abc.abstractmethod
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
        raise NotImplementedError


class StateElementHandle(CreatorHandle):
    """Return StateElement objects, for a given StateElementIdentifier

    Note StateElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        raise NotImplementedError


class StateElementHandleSet(CreatorHandleSet):
    """This maps a StateElementIdentifier to a StateElement object that handles it.

    Note StatElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def __init__(self) -> None:
        super().__init__("state_element")

    def state_element(self, state_element_id: StateElementIdentifier) -> StateElement:
        return self.handle(state_element_id)

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(MeasurementId)


class StateInfo(UserDict):
    """This class maintains the full state as we perform a retrieval.
    This class is closely tied to CurrentStateStateInfo, it isn't clear
    if we perhaps want to merge theses classes at some point. But right
    now StateInfo focuses on maintaining the state info, and CurrentState
    on making that state info available to other parts of software.

    This is basically a dict like object, where we create StateElement from
    a StateElementHandleSet on first usage.
    """

    def __init__(
        self, state_element_handle_set: StateElementHandleSet | None = None
    ) -> None:
        super().__init__()
        if state_element_handle_set is not None:
            self.state_element_handle_set = state_element_handle_set
        else:
            self.state_element_handle_set = copy.deepcopy(
                StateElementHandleSet.default_handle_set()
            )
        self._state_element: dict[StateElementIdentifier, StateElement] = {}

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        self.state_element_handle_set.notify_update_target(measurement_id)
        self.data = {}

    def __missing__(self, state_element_id: StateElementIdentifier) -> StateElement:
        self.data[state_element_id] = self.state_element_handle_set.state_element(
            state_element_id
        )
        return self.data[state_element_id]


__all__ = [
    "StateElement",
    "StateElementHandle",
    "StateElementHandleSet",
]
