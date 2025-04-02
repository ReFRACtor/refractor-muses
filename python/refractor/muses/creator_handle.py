from __future__ import annotations
from .priority_handle_set import PriorityHandleSet
from typing import Any


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
    creates the something like a ForwardModel or Observation, and
    register with the CreatorHandleSet, e.g., ForwardModelHandleSet or
    ObservationHandleSet.  Take a look at the existing examples
    (e.g. the unit tests) - the design seems complicated but is
    actually pretty simple to use in pratice.

    Note a CreatorHandle can assume that it called for the same
    target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next (e.g., read a L1B file once).

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and the creator_func_name
    because they have different scopes (notify_update_target for the
    full retrieval, creator_func_name for a retrieval step). If a
    CreatorHandle doesn't care about the target, it can just ignore
    this function call.
    """

    def __init__(self, creator_func_name: str) -> None:
        """Constructor, takes the name of the creator function (e.g.,
        "observation")."""
        super().__init__()
        self.creator_func_name = creator_func_name

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


class CreatorHandle:
    """Base class for handles used by CreatorHandleSet. Note we use
    duck typing, so you don't actually need to derive from this
    object. But it can be useful because it 1) provides the interface
    and 2) documents that a class is intended for this.

    Note a CreatorHandle can assume that it called for the same
    target, until notify_update_target is called. So if it makes
    sense, these objects can do internal caching for things that don't
    change when the target being retrieved is the same from one call
    to the next (e.g., read a L1B file once).

    notify_update_target will also be called before the first time the
    objects are created - basically it makes sense to separate the
    arguments for notify_update_target and the creator_func_name
    because they have different scopes (notify_update_target for the
    full retrieval, creator_func_name for a retrieval step). If a
    CreatorHandle doesn't care about the target, it can just ignore
    this function call.

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


__all__ = ["CreatorHandleSet", "CreatorHandle"]
