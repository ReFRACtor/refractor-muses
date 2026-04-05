from __future__ import annotations
from .identifier import ProcessLocation
from typing import Any


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

    def __init__(self) -> None:
        self._observers: dict[Any, set[ProcessLocation] | None] = {}

    def add_observer(self, obs: Any) -> None:
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
        for obs, pset in self._observers.items():
            if pset is None or loc in pset:
                obs.notify_process_location(self, loc, **kwargs)
