from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
from .current_state import PropagatedQA

import numpy as np
import abc
import typing
from typing import Any
import copy
from collections import UserDict

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .muses_observation import MeasurementId, ObservationHandleSet
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .error_analysis import ErrorAnalysis
    from .current_state import SoundingMetadata


# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray
RetrievalGrid2dArray = np.ndarray
ForwardModelGrid2dArray = np.ndarray


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

    def assert_equal(self, other: StateElement) -> None:
        """Simple test to make sure two StateElement are the same, intended for
        initial development and testing. This doesn't check that all the function
        call are identical, just the various property items."""
        assert self.state_element_id == other.state_element_id
        sd1 = self.spectral_domain
        sd2 = self.spectral_domain
        assert (sd1 is None and sd2 is None) or (
            sd1 is not None
            and sd2 is not None
            and np.allclose(
                sd1.convert_wave(rf.Unit("nm")), sd2.convert_wave(rf.Unit("nm"))
            )
        )
        assert self.retrieval_sv_length == other.retrieval_sv_length
        assert self.sys_sv_length == other.sys_sv_length
        assert self.forward_model_sv_length == other.forward_model_sv_length
        assert self.map_type == other.map_type
        assert (
            self.value_str is None and other.value_str is None
        ) or self.value_str == other.value_str
        for param in (
            "basis_matrix",
            "map_to_parameter_matrix",
            "altitude_list",
            "altitude_list_fm",
            "pressure_list",
            "pressure_list_fm",
            "value",
            "value_fm",
            "apriori_value",
            "apriori_value_fm",
            "apriori_cov",
            "apriori_cov_fm",
            "retrieval_initial_value",
            "step_initial_value",
            "step_initial_value_fm",
            "true_value",
            "true_value_fm",
        ):
            # Not all the functions of implemented in StateElementOld, these don't actuall
            # get called in our test cases so this ok. Just skip this - it is possible a
            # failure is a real problem but for now just skip this. Ok, since this is only
            # used in testing which can only do so much until we do full runs.
            try:
                v1 = getattr(self, param)
                v2 = getattr(other, param)
                assert (v1 is None and v2 is None) or np.allclose(v1, v2)
            except (RuntimeError, NotImplementedError):
                pass

    @property
    def metadata(self) -> dict[str, Any]:
        """Some StateElement have extra metadata. There is really only one example
        now, emissivity has camel_distance and prior_source. It isn't clear the best
        way to handle this, but the current design just returns a dictionary with
        any extra metadata values. We can perhaps rework this if needed in the future.
        For most StateElement this will just be a empty dict."""
        res: dict[str, Any] = {}
        return res

    @property
    def state_element_id(self) -> StateElementIdentifier:
        return self._state_element_id

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        """Called with the subset of parameters for this StateElement
        when the cost function changes."""
        pass

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

    @property
    def altitude_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def altitude_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def pressure_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        return None

    @property
    def pressure_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        return None

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
    def apriori_value_fm(self) -> ForwardModelGrid2dArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def apriori_cov(self) -> RetrievalGrid2dArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    def apriori_cross_covariance(
        self, selem2: StateElement
    ) -> RetrievalGrid2dArray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    @abc.abstractproperty
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    def apriori_cross_covariance_fm(
        self, selem2: StateElement
    ) -> ForwardModelGrid2dArray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

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
    def update_state_element(
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

    # TODO Perhaps this can go away, replaced with being a StateVector observer?
    @abc.abstractmethod
    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> ForwardModelGridArray | None:
        """Update the state based on results, and return a boolean array
        indicating which coefficients were updated."""
        raise NotImplementedError()

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        pass

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        pass


class StateElementHandle(CreatorHandle):
    """Return StateElement objects, for a given StateElementIdentifier

    Note StateElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
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

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(
                    measurement_id, retrieval_config, strategy, observation_handle_set
                )


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
        # Temp, clumsy but this will go away
        for p in sorted(self.state_element_handle_set.handle_set.keys(), reverse=True):
            for h in self.state_element_handle_set.handle_set[p]:
                if hasattr(h, "_current_state_old"):
                    self._current_state_old = h._current_state_old

    @property
    def brightness_temperature_data(self) -> dict[int, dict[str, float | None]]:
        # Right now, need the old brightness_temperature_data.
        # We can probably straighten this out later, but as we forward stuff
        # to the old current_state_old we need to have the data there
        return self._current_state_old.state_info.brightness_temperature_data

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        # Right now, use the old SoundingMetadata. We'll want to move this over,
        # but that can wait a bit
        return self._current_state_old.sounding_metadata

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self.state_element_handle_set.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )
        self.data = {}
        self.propagated_qa = PropagatedQA()

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        # TODO, we want to remove this
        self._current_state_old.notify_new_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )
        # Since we aren't actually doing the init stuff yet in our
        # new StateElement, make sure everything get created (since
        # this happens on first use)
        for sid in self._current_state_old.full_state_element_id:
            _ = self[sid]
        for selem in self.values():
            selem.notify_start_retrieval(
                current_strategy_step,
                retrieval_config,
            )

    def restart(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        # TODO, we want to remove this
        self._current_state_old.restart(current_strategy_step, retrieval_config)
        # Since we aren't actually doing the init stuff yet in our
        # new StateElement, make sure everything get created (since
        # this happens on first use)
        for sid in self._current_state_old.full_state_element_id:
            _ = self[sid]
        for selem in self.values():
            selem.notify_start_retrieval(
                current_strategy_step,
                retrieval_config,
            )

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
