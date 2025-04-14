from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
from .current_state import PropagatedQA

import numpy as np
import abc
import typing
import copy
from collections import UserDict

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .muses_observation import MeasurementId

# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray


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

    See the documentation at the start of CurrentState for a discussion
    of the various state vectors.

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

    @abc.abstractproperty
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to forward model
        vector. Would be nice to replace this with a general
        rf.StateMapping, but for now this is assumed in a lot of
        muses-py code."""
        raise NotImplementedError()

    @abc.abstractproperty
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from the basis matrix, going from
        the forward model vector the retrieval vector."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def sys_sv_length(self) -> int:
        raise NotImplementedError()
    
    @abc.abstractproperty
    def forward_model_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def map_type(self) -> str:
        """For ReFRACtor we use a general rf.StateMapping, which can mostly
        replace the map type py-retrieve uses. However there are some places
        where old code depends on the map type strings (for example, writing
        metadata to an output file). It isn't clear what we will need to do if
        we have a more general mapping type like a scale retrieval or something like
        that. But for now, supply the old map type. The string will be something
        like "log" or "linear" """
        raise NotImplementedError()
    
    @abc.abstractproperty
    def altitude_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        raise NotImplementedError()

    @abc.abstractproperty
    def altitude_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        raise NotImplementedError()
    
    @abc.abstractproperty
    def pressure_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        raise NotImplementedError()

    @abc.abstractproperty
    def pressure_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        raise NotImplementedError()
    
    @abc.abstractproperty
    def value(self) -> RetrievalGridArray:
        """Current value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def value_fm(self) -> ForwardModelGridArray:
        """Current value of StateElement"""
        raise NotImplementedError()

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
    def apriori_value(self) -> RetrievalGridArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def apriori_value_fm(self) -> ForwardModelGridArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @property
    def apriori_cov(self) -> RetrievalGridArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_initial_value(self) -> RetrievalGridArray:
        """Value StateElement had at the start of the retrieval."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_initial_value(self) -> RetrievalGridArray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError()

    @property
    def true_value(self) -> RetrievalGridArray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return None

    @property
    def true_value_fm(self) -> ForwardModelGridArray | None:
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
        raise NotImplementedError()


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
        raise NotImplementedError()


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
                h.notify_update_target(measurement_id)


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
        self.propagated_qa = PropagatedQA()
        self.brightness_temperature_data: dict[int, dict[str, float | None]] = {}

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        self.state_element_handle_set.notify_update_target(measurement_id)
        self.data = {}
        self.propagated_qa = PropagatedQA()
        self.brightness_temperature_data = {}

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
