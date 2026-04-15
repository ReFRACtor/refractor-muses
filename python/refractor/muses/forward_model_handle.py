from __future__ import annotations
from .creator_handle import CreatorHandleWithContextSet, CreatorHandleWithContext
from .creator_dict import CreatorDict
import refractor.framework as rf  # type: ignore
import abc
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation
    from .current_state import CurrentState
    from .identifier import InstrumentIdentifier
    from .muses_strategy_context import MusesStrategyContext


class ForwardModelHandle(CreatorHandleWithContext, metaclass=abc.ABCMeta):
    """Base class for ForwardModelHandle. Note we use duck typing, so
    you don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents that
    a class is intended for this.

    This can do caching based on assuming the target is the same
    between calls, see CreatorHandle for a discussion of this.

    However, when forward_model is called a "newish" object should be
    created. Specifically we want to be able to attach each object to
    a separate StateVector and have different SpectralWindowRange set
    - we want to be able to have more than one CostFunction active at
    one time and we don't want updates in one CostFunction to affect
    the others. So this can be thought of as a shallow copy, if that
    make sense for the object. Things that don't depend on the
    StateVector can be shared (e.g., data read from a file), but state
    related parts should be independent.

    """

    @abc.abstractmethod
    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel | None:
        """Return ForwardModel if we can process the given
        instrument_name, or None if we can't.

        The forward model state vector is passed in, in case we want
        to attach anything to it as an observer (so object state gets
        updated as we update for forward model state vector).

        Because we sometimes need the metadata we also pass in the
        MusesObservation that goes with the given instrument name.
        """
        raise NotImplementedError()


class ForwardModelHandleSet(CreatorHandleWithContextSet):
    """This takes the instrument name and RefractorUip, and creates a
    ForwardModel and Observation for that instrument.

    """

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("forward_model", strategy_context)

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        **kwargs: Any,
    ) -> rf.ForwardModel | None:
        """Create a ForwardModel for the given instrument.

        The forward model state vector is passed in, in case we want
        to attach anything to it as an observer (so object state gets
        updated as we update for forward model state vector).

        Because we sometimes need the metadata we also pass in the
        MusesObservation that goes with the given instrument name.
        """

        return self.handle(instrument_name, current_state, obs, fm_sv, **kwargs)


# Register creator set
CreatorDict.register(rf.ForwardModel, ForwardModelHandleSet)

__all__ = [
    "ForwardModelHandle",
    "ForwardModelHandleSet",
]
