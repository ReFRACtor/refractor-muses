from __future__ import annotations
from .priority_handle_set import PriorityHandleSet
from .muses_strategy_context import (
    MusesStrategyContext,
    MusesStrategyContextMixin,
    MusesStrategyContextProxy,
)
import copy
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .creator_dict import CreatorDict


class CreatorHandleSet(PriorityHandleSet):
    """We have a few objects that we want to be able to create to
    support a retrieval, e.g, a ForwardModel for a particular
    instrument.

    This is the primary place where we extend the retrieval strategy,
    e.g., have a new ForwardModel for a new instrument.

    We use a CreatorHandleSet, which is a specialization of a
    PriorityHandleSet.  New functionality can be added by adding new
    handles to create the new objects, this plugs into the existing
    RetrievalStategy for doing a retrieval.

    The design of the PriorityHandleSet is a bit overkill for this
    class, we could probably get away with a simple dictionary mapping
    instrument name to functions that handle it or something like
    that. However the PriorityHandleSet was already available from
    another library with a much more complicated set of handlers where
    a dictionary isn't sufficient (see
    https://github.jpl.nasa.gov/Cartography/pynitf and
    https://cartography-jpl.github.io/pynitf/design.html#priority-handle-set).
    The added flexibility can be nice here, and since the code was
    already written we make use of it.

    In practice you create a simple CreatorHandle class that just
    creates something like a ForwardModel or Observation, and
    register with the CreatorHandleSet, e.g., ForwardModelHandleSet or
    ObservationHandleSet.  Take a look at the existing examples
    (e.g. the unit tests) - the design seems complicated but is
    actually pretty simple to use in practice.
    """

    def __init__(self, creator_func_name: str) -> None:
        """Constructor, takes the name of the creator function (e.g.,
        "observation")."""
        super().__init__()
        self.creator_func_name = creator_func_name

    # Temp, we will remove this once we have everything moved over
    def notify_update_target(self, *args: Any, **kwargs: Any) -> None:
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(*args, **kwargs)

    def handle_h(self, h: CreatorHandle, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        """Process a registered function"""
        res = getattr(h, self.creator_func_name)(*args, **kwargs)
        if res is None:
            return (False, None)
        return (True, res)


class CreatorHandleWithContextSet(CreatorHandleSet):
    """This is CreatorHandleSet that also has a
    MusesStrategyContext. This is pretty common, a lot of our objects
    handles need the context to produce objects.

    Note a CreatorHandleWithContext can assume that it called for the
    same MusesStrategyContext, until MusesStrategyContext sends a
    notify_update_strategy_context message. So if it makes sense,
    these objects can do internal caching for things that don't change
    when the MusesStrategyContext is the same. These objects should
    register as observers of MusesStrategyContext to get the
    notify_update_strategy_context message if needed. See for example
    MusesPyQaDataHandle.

    We had considered just passing the MusesStrategyContext to handle_h
    like any other argument. However it makes sense to instead treat this
    as different because the lifecycles are different. We tend to call
    handle_h for each retrieval step, but the MusesStrategyContext only
    changes at the very beginning of a retrieval. So it makes sense to
    treat these as different, although this means MusesStrategyContext
    is like a "hidden" argument. I think the trade off is reasonable,
    but know that the handle_h has access to the MusesStrategyContext as
    well as whatever argument get passed to handle_h.
    """

    def __init__(
        self,
        creator_func_name: str,
        strategy_context: MusesStrategyContext | None = None,
    ) -> None:
        """Constructor, takes the name of the creator function (e.g.,
        "observation")."""
        super().__init__(creator_func_name)
        self.strategy_context = MusesStrategyContextProxy(strategy_context)

    @classmethod
    def default_handle_set_with_context(
        cls, strategy_context: MusesStrategyContext
    ) -> Self:
        """Like default_handle_set, but also set of the strategy_context. This is
        a copy, so it can be modified without changing default_handle_set"""
        res = copy.deepcopy(cls.default_handle_set())
        res.strategy_context.reset_context(strategy_context)
        return res

    def notify_add_creator_dict(self, cdict: CreatorDict) -> None:
        self.strategy_context.reset_context(cdict.strategy_context)


class CreatorHandle:
    """Base class for handles used by CreatorHandleSet. Note we use
    duck typing, so you don't actually need to derive from this
    object. But it can be useful because it 1) provides the interface
    and 2) documents that a class is intended for this.
    """

    def _dispatch(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        """It can be useful sometimes to have a handle with multiple
        functions that we can call. This support function just takes
        the name of the function to call and then calls that function
        with the arguments.

        See for example SpectralWindowHandleSet, where we have both
        filter_name_dict and spectral_window_dict.
        """
        return getattr(self, func_name)(*args, **kwargs)

    def notify_update_target(self, *args: Any) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    # Derived classes should add a creator function, e.g. observation.
    # This should either return an object if we can create it, or None
    # if we can't (and the CreatorHandleSet goes on to try the next
    # handle.  def observation(self, *args):


class CreatorHandleWithContext(CreatorHandle, MusesStrategyContextMixin):
    """This with a CreatorHandle with the addition of support for the
    MusesStrategyContext.

    Note a CreatorHandleWithContext can assume that it called for the same
    MusesStrategyContext, until notify_update_strategy_context is
    called. So if it makes sense, these objects can do internal
    caching for things that don't change when the MusesStrategyContext
    is the same from one call to the next (e.g., read a L1B file
    once).

    notify_update_strategy_context will also be called before the
    first time the objects are created. So Objects can defer
    initialization that needs the MusesStrategyContext, and a
    guaranteed to get this message. Objects should register as
    observers to get the message if the need it.

    We had considered just passing the MusesStrategyContext to
    handle_h like any other argument. However it makes sense to
    instead treat this as different because the lifecycles are
    different. We tend to call handle_h for each retrieval step, but
    the MusesStrategyContext only changes at the very beginning of a
    retrieval. So it makes sense to treat these as different, although
    this means MusesStrategyContext is like a "hidden" argument. I
    think the trade off is reasonable, but know that the handle_h has
    access to the MusesStrategyContext as well as whatever argument
    get passed to handle_h

    """

    def __init__(self, add_as_context_observer: bool = False) -> None:
        super(MusesStrategyContextMixin, self).__init__()
        self.add_as_context_observer = add_as_context_observer

    def notify_add_handle(self, hset: CreatorHandleWithContextSet) -> None:
        self._strategy_context = hset.strategy_context
        # Because it is a common case, we just handle adding this as
        # an observer if the derived class requested this.
        # Note that the class should add a notify_update_strategy_context
        # if it is an observer.
        if self.has_strategy_context and self.add_as_context_observer:
            self.strategy_context.add_observer(self)


__all__ = [
    "CreatorHandleSet",
    "CreatorHandleWithContextSet",
    "CreatorHandle",
    "CreatorHandleWithContext",
]
