from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
import abc
import typing

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier


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

    def __init__(self, state_element_identifier: StateElementIdentifier):
        self._state_element_identifier = state_element_identifier

    @property
    def name(self) -> StateElementIdentifier:
        return self._state_element_identifier

    @abc.abstractproperty
    def sa_covariance(self) -> np.ndarray:
        """Return sa covariance matrix."""
        raise NotImplementedError()

    def sa_cross_covariance(self, selem2: StateElement):
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
        retrieval_initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ):
        """Update the value of the StateElement. This function updates
        each of the various values passed in.  A value of 'None' (the
        default) means skip updating that part of the StateElement.
        """
        raise NotImplementedError
