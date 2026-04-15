from __future__ import annotations
from .identifier import ProcessLocation
from typing import Any, Type


class ProcessLocationObservable:
    """We don't directly produce output in our RetrievalStrategyStep
    and related classes. Instead, we use an Observer/Observable
    pattern to decouple go generation of output, and in some cases
    logging. This was a lesson learned on developing OCO-2. The output
    code tends to became complicated - intrinsically, rather than just
    having bad code. Separating this from the actual retrieval is
    desirable, the retrieval can worry about running the forward model
    and producing a optimal estimation, and the output code about writing output
    files and generating related quantities (e.g., error analysis).

    This class provides a central place for handling these interconnections.
    Object can register an observers to be notified when we reach a particular
    point in the processing (e.g., "retrieval step"), and the retrieval code
    can emit notifications when things when things occur.

    Objects that are observers should have a notify_process_location function,
    that should take a variable number of kwargs, so that different locations
    can include different information (e.g RetrievalStrategyStep includes
    a retrieval_strategy_step argument).

    They should also have a observing_process_location function that returns
    a list of ProcessLocations that the object wants to be notified about. Or
    if they leave that function off, we notify the objects about every ProcessLocation
    event.
    """

    default_observer_list: set[Type[object]] = set()
    default_debug_observer_list: set[Type[object]] = set()
    default_plot_observer_list: set[Type[object]] = set()

    def __init__(self) -> None:
        self._observers: dict[Any, set[ProcessLocation] | None] = {}

    def add_observer(self, obs: Any) -> None:
        if not hasattr(obs, "notify_process_location"):
            raise RuntimeError(f"Bad observer added {obs}")
        if hasattr(obs, "observing_process_location"):
            self._observers[obs] = set(obs.observing_process_location)
        else:
            self._observers[obs] = None
        if hasattr(obs, "notify_add"):
            obs.notify_add(self)

    def remove_observer(self, obs: Any) -> None:
        self._observers.pop(obs, None)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self) -> None:
        # We change self._observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._observers.keys())
        for obs in lobs:
            self.remove_observer(obs)

    def notify_process_location(
        self, location: ProcessLocation | str, **kwargs: Any
    ) -> None:
        loc = location
        if not isinstance(loc, ProcessLocation):
            loc = ProcessLocation(loc)
        lobs = list(self._observers.items())
        for obs, pset in lobs:
            if pset is None or loc in pset:
                obs.notify_process_location(loc, **kwargs)

    def add_default_observer(
        self,
        write_debug_output: bool = False,
        write_plots: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add the default observers that have been registered. These can then
        be modified, but most of the time we want to start with the default set and
        possible add extra.

        We have a set of observers for when we are writing debug output
        and writing plots. These just are useful to group the observers into things
        we always want, thing we want when debugging, and things we want with generating
        plots."""
        for cls in self.default_observer_list:
            self.add_observer(cls(**kwargs))
        if write_debug_output:
            for cls in self.default_debug_observer_list:
                self.add_observer(cls(**kwargs))
        if write_plots:
            for cls in self.default_plot_observer_list:
                self.add_observer(cls(**kwargs))

    @classmethod
    def register_default_observer(cls, obs_cls: Type[object]) -> None:
        cls.default_observer_list.add(obs_cls)

    @classmethod
    def register_default_debug_observer(cls, obs_cls: Type[object]) -> None:
        cls.default_debug_observer_list.add(obs_cls)

    @classmethod
    def register_default_plot_observer(cls, obs_cls: Type[object]) -> None:
        cls.default_plot_observer_list.add(obs_cls)


__all__ = [
    "ProcessLocationObservable",
]
