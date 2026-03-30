from __future__ import annotations
from .priority_handle_set import PriorityHandleSet
import copy
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .muses_strategy_executor import MusesStrategyContext
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_observation import MeasurementId


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

    Right now, every CreatorHandleSet has a MusesStrategyContext. I
    think this is what we want, however we might in the future run
    into CreatorHandleSet that don't have or need a
    MusesStrategyContext (they are separate concepts).  However, for
    now we have this baked in. We can revisit this in the future if
    and when we have an example of how we would want to separate this.

    Note a CreatorHandle can assume that it called for the same
    MusesStrategyContext, until MusesStrategyContext sends a
    notify_update_strategy_executor message. So if it makes sense,
    these objects can do internal caching for things that don't change
    when the MusesStrategyContext is the same. These objects should
    register as observers of MusesStrategyContext to get the
    notify_update_strategy_executor message. See for example
    MusesPyQaDataHandle.

    We had considered just passing the MusesStrategyContext to handle_h
    like any other argument. However it makes sense to instead treat this
    as different because the lifecycles are different. We tend to call
    handle_h for each retrieval step, but the MusesStrategyContext only
    changes at the very beginning of a retrieval. So it makes sense to
    treat these as different, although this means MusesStrategyContext
    is like a "hidden" argument. I think the trade off is reasonable,
    but know that the handle_h has access to the MusesStrategyContext as
    well as whatever argument get passed to handle_h

    """

    # We don't initialize this here, so we can use the "trick" of
    # doing this add_default_handle. When we do this we attach this to
    # the derived class, rather than the base class. This is based on
    # cls._default_handle = blah creating a copy in cls rather than
    # CreatorHandle._default_handle
    _default_handle: None | dict[Any, int] = None

    # Temp, None here just so we can skip needing to update all the handles
    # at once. This way we can test them separately.
    def __init__(
        self,
        creator_func_name: str,
        strategy_context: MusesStrategyContext | None = None,
    ) -> None:
        """Constructor, takes the name of the creator function (e.g.,
        "observation")."""
        super().__init__()
        self.creator_func_name = creator_func_name
        self.strategy_context = strategy_context

    # Temp, we'll remove this
    def notify_update_target(self, *args: Any, **kwargs: Any) -> None:
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                if hasattr(h, "notify_update_target"):
                    h.notify_update_target(*args, **kwargs)

    @classmethod
    def default_handle_set(cls) -> Self:
        # We handle the default handle set a little differently, because we
        # need to add the MusesStrategyContext. Stub this out so it isn't
        # accidentally called

        # Temp, just so we can test before everything has changed
        return cls.default_handle_set_with_context(None)
        raise RuntimeError(
            "default_handle_set isn't implemented. Call default_handle_set_with_context instead"
        )

    @classmethod
    def default_handle_set_with_context(
        cls, strategy_context: MusesStrategyContext
    ) -> Self:
        # Temp, until we get everything moved over
        try:
            res = cls(strategy_context)
        except TypeError:
            res = cls()
        if cls._default_handle is not None:
            for h, priority_order in cls._default_handle.items():
                res.add_handle(copy.deepcopy(h), priority_order)
        return res

    @classmethod
    def add_default_handle(cls, h: Any, priority_order: int = 0) -> None:
        """Add the given handle to the default set of handlers.  The
        higher priority_order (larger number) items are tried first."""
        if cls._default_handle is None:
            cls._default_handle = dict()
        cls._default_handle[h] = priority_order

    @classmethod
    def discard_default_handle(cls, h: Any) -> None:
        if cls._default_handle is None:
            return
        # Remove is present, no error if not found
        cls._default_handle.pop(h, None)

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
    MusesStrategyContext, until notify_update_strategy_executor is
    called. So if it makes sense, these objects can do internal
    caching for things that don't change when the MusesStrategyContext
    is the same from one call to the next (e.g., read a L1B file
    once).

    notify_update_strategy_executor will also be called before the
    first time the objects are created. Objects should register as
    observers to get the message if the need it.

    We had considered just passing the MusesStrategyContext to handle_h
    like any other argument. However it makes sense to instead treat this
    as different because the lifecycles are different. We tend to call
    handle_h for each retrieval step, but the MusesStrategyContext only
    changes at the very beginning of a retrieval. So it makes sense to
    treat these as different, although this means MusesStrategyContext
    is like a "hidden" argument. I think the trade off is reasonable,
    but know that the handle_h has access to the MusesStrategyContext as
    well as whatever argument get passed to handle_h
    """

    def __init__(self, add_as_context_observer: bool = False) -> None:
        self.strategy_context: None | MusesStrategyContext = None
        self.add_as_context_observer = add_as_context_observer

    def notify_add_handle(self, hset: CreatorHandleSet) -> None:
        # Derived classes can override this to do whatever they need to.
        self.strategy_context = hset.strategy_context
        # Because it is a common case, we just handle adding this as
        # an observer if the derived class requested this.
        # Note that the class should add a notify_update_strategy_context
        # if it is an observer.
        # Temp, until we get all the classes moved over
        if not hasattr(self, "add_as_context_observer"):
            return
        if self.strategy_context is not None and self.add_as_context_observer:
            self.strategy_context.add_observer(self)

    @property
    def has_measurement_id(self) -> bool:
        if (
            self.strategy_context is None
            or self.strategy_context.measurement_id is None
        ):
            return False
        return True

    # Temp name, so we don't conflict with code we haven't moved over yet.
    # We'll rename this is a bit
    @property
    def measurement_id_new(self) -> MeasurementId:
        """We often want to get the measurement id from the strategy_context,
        having an error if either the strategy_context or the measurement id
        is None. This function is a short cut for that, throwing an exception
        if we can't get the measurement_id. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        if (
            self.strategy_context is None
            or self.strategy_context.measurement_id is None
        ):
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return self.strategy_context.measurement_id

    @property
    def has_retrieval_config(self) -> bool:
        if (
            self.strategy_context is None
            or self.strategy_context.retrieval_config is None
        ):
            return False
        return True

    # Temp name, so we don't conflict with code we haven't moved over yet.
    # We'll rename this is a bit
    @property
    def retrieval_config_new(self) -> RetrievalConfiguration:
        """We often want to get the retrieval_config from the strategy_context,
        having an error if either the strategy_context or the retrieval_config
        is None. This function is a short cut for that, throwing an exception
        if we can't get the retrieval_config. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        if (
            self.strategy_context is None
            or self.strategy_context.retrieval_config is None
        ):
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return self.strategy_context.retrieval_config

    @property
    def has_stac_catalog(self) -> bool:
        if self.strategy_context is None or self.strategy_context.stac_catalog is None:
            return False
        return True

    def stac_catalog(self) -> RetrievalConfiguration:
        """We often want to get the stac_catalog from the strategy_context,
        having an error if either the strategy_context or the stac_catalog
        is None. This function is a short cut for that, throwing an exception
        if we can't get the stac_catalog. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        if self.strategy_context is None or self.strategy_context.stac_catalog is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return self.strategy_context.stac_catalog

    def _dispatch(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        """It can be useful sometimes to have a handle with multiple
        functions that we can call. This support function just takes
        the name of the function to call and then calls that function
        with the arguments.

        See for example SpectralWindowHandleSet, where we have both
        filter_name_dict and spectral_window_dict.
        """
        return getattr(self, func_name)(*args, **kwargs)

    # Derived classes should add a creator function, e.g. observation.
    # This should either return an object if we can create it, or None
    # if we can't (and the CreatorHandleSet goes on to try the next
    # handle.  def observation(self, *args):


__all__ = ["CreatorHandleSet", "CreatorHandle"]
