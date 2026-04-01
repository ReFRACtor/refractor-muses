from __future__ import annotations
from collections import UserDict
from typing import Any
import copy
import typing

if typing.TYPE_CHECKING:
    from .creator_handle import CreatorHandleSet
    from .muses_strategy_context import MusesStrategyContext


class CreatorDict(UserDict):
    """We would like to have a central place to get CreatorHandleSet,
    so we have one place to add updates etc. We originally created
    this in various classes (e.g., CostFunctionCreator had the
    ForwardModelHandleSet), but it wasn't always obvious where this
    was located. Rather than needing "special" knowledge about where
    we happened to have stuck this, we have this central class that
    maps a type (e.g. rf.ForwardModel) to the set that creates it (e.g
    ForwardModelHandleSet).

    This also gives a central place where we can add new creators for
    new retrieval types (e.g., if machine learning ends up having
    creators).

    We create the CreatorHandleSet on first use, so if a
    CreatorHandleSet isn't used we never bother creating it. This was
    added when we put the machine learning stuff in, and has none of
    the object we use for the Optimal Estimation stuff. We use the
    default_handle_set_with_context to create this - we can possibly
    extend this if we need other ways to handling this.

    Note that the CreatorHandleSet can be completely replaced if
    desired. Usually we can to start wit the
    default_handle_set_with_context but there might be instances where
    we just want a completely different set. Just assign to this class
    like a dict, e.g., creator_dict[MyType] = MyCreatorHandleSet

    We also have a MusesStrategyContext for each of our
    CreatorHandleSet. I'm not 100% sure that this will always be the
    case - they are two distinct concepts. But I'm not sure how this
    would work with a CreatorHandleSet that doesn't need a
    MusesStrategyContext. We can revisit this if/when we come up with
    an example where having the MusesStrategyContext is a problem.

    This class handles the MusesStrategyContext and passing to the
    CreatorHandleSet.

    CreatorHandleSet that want to be part of refractor muses should
    register themselves with this class, so we know how to create
    them.
    """

    _creator_class: dict[Any, type[CreatorHandleSet]] = {}

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        from .muses_strategy_context import MusesStrategyContext

        super().__init__()
        self.strategy_context = (
            strategy_context if strategy_context is not None else MusesStrategyContext()
        )

    @classmethod
    def register(
        self, created_type: Any, handle_set_class: type[CreatorHandleSet]
    ) -> None:
        self._creator_class[created_type] = handle_set_class

    def __missing__(self, created_type: Any) -> CreatorHandleSet:
        if created_type not in self._creator_class:
            raise KeyError(
                f"The type {created_type} has not been registered with CratotrDict"
            )
        self.data[created_type] = copy.deepcopy(
            self._creator_class[created_type].default_handle_set()
        )
        if hasattr(self.data[created_type], "notify_add_creator_dict"):
            self.data[created_type].notify_add_creator_dict(self)
        return self.data[created_type]


__all__ = ["CreatorDict"]
