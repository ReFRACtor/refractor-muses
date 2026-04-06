from __future__ import annotations
import abc
from loguru import logger
from .creator_dict import CreatorDict
from .creator_handle import CreatorHandleWithContext, CreatorHandleWithContextSet
from .identifier import RetrievalType, ProcessLocation
from .muses_strategy_context import MusesStrategyContextMixin
import json
import gzip
import os
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .muses_strategy_context import MusesStrategyContext
    from .current_state import CurrentState
    from .process_location_observable import ProcessLocationObservable


class RetrievalStrategyStepSet(CreatorHandleWithContextSet):
    """This takes the retrieval_type and determines a
    RetrievalStrategyStep to handle this, returning it so it can
    be called.
    """

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("retrieval_step", strategy_context)

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        **kwargs: Any,
    ) -> RetrievalStrategyStep:
        return self.handle(
            retrieval_type,
            creator_dict,
            current_state,
            process_location_observable,
            **kwargs,
        )


class RetrievalStrategyStepHandle(CreatorHandleWithContext):
    """Right now our strategy steps just key off of the retrieval_type
    being in a set (or None to match any). We have this handle for
    this case. We can certainly create other CreatorHandle if we have
    RetrievalStrategyStep with more complicated selection logic."""

    def __init__(
        self,
        cls: type[RetrievalStrategyStep],
        retrieval_type_set: set[RetrievalType] | None = None,
    ) -> None:
        super().__init__()
        self._create_cls = cls
        self._retrieval_type_set = retrieval_type_set

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        **kwargs: Any,
    ) -> RetrievalStrategyStep | None:
        if (
            self._retrieval_type_set is None
            or retrieval_type in self._retrieval_type_set
        ):
            return self._create_cls(
                creator_dict, current_state, process_location_observable, **kwargs
            )
        return None


class RetrievalStrategyStep(MusesStrategyContextMixin, metaclass=abc.ABCMeta):
    """Do the retrieval step indicated by retrieval_type

    We *only* maintain state between steps in CurrentState (other than
    possibly internal caching for performance). If the same
    RetrievalStrategyStep is called with the same CurrentState, it
    will produce the same output (possibly slower, if internal caching
    isn't the same).

    A RetrievalStrategyStep *only* modifies CurrentState and produces
    output through side effects. Note that the RetrievalStrategyStep
    doesn't directly produce output, instead we use the
    Observer/Observable pattern to decouple generation of output. A
    RetrievalStrategyStep calls ProcessLocationObserver.notify_process_location
    at points during the processing, where the Observer can do
    whatever with the processing (e.g, produce an output file).

    """

    def __init__(
        self,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        **kwargs: Any,
    ) -> None:
        super().__init__(creator_dict.strategy_context)
        self._saved_state: None | dict[str, Any] = None
        self.creator_dict = creator_dict
        self.current_state = current_state
        self.process_location_observable = process_location_observable
        self.kwargs = kwargs

    def do_retrieval(
        self,
    ) -> None:
        self.set_state(self.kwargs.get("ret_state", None))
        self.retrieval_step_body()
        self.notify_process_location(ProcessLocation("end_retrieval_step"))

    @abc.abstractmethod
    def retrieval_step_body(self) -> None:
        """Actual do the retrieval step"""

    def get_state(self) -> dict[str, Any]:
        """Return a dictionary of values that can be used by
        set_state.  This allows us to skip pieces of the retrieval
        step. This is similar to a pickle serialization (which we also
        support), but only saves the things that change when we update
        the parameters.

        This is useful for unit tests of side effects of doing the retrieval
        step (e.g., generating output files) without needing to actually
        run the retrieval.

        """
        # Default, no state
        return {}

    def set_state(self, d: dict[str, Any] | None) -> None:
        """Set the state previously saved by get_state"""
        # Default, just put this into the _saved_state for use
        # in the rest of this object
        self._saved_state = d

    def notify_process_location(self, ploc: ProcessLocation) -> None:
        self.process_location_observable.notify_process_location(
            ploc, current_state=self.current_state, retrieval_strategy_step=self
        )


class RetrievalStrategyStepNotImplemented(RetrievalStrategyStep):
    """There seems to be a few retrieval types that aren't implemented
    in py-retrieve. It might also be that we just don't have this
    implemented right in ReFRACtor, but in any case we don't have a
    test case for this.

    Throw an exception to indicate this, we can look at implementing
    this in the future - particularly if we have a test case to
    validate the code.

    """

    def retrieval_step_body(self) -> None:
        raise RuntimeError(
            f"We don't currently support retrieval_type {self.retrieval_type}"
        )


RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(
        RetrievalStrategyStepNotImplemented,
        {RetrievalType("forwardmodel"), RetrievalType("omi_radiance_calibration")},
    )
)

CreatorDict.register(RetrievalStrategyStep, RetrievalStrategyStepSet)


class RetrievalStepCaptureObserver:
    """Helper class, saves results of retrieval step so we can rerun skipping
    much of the calculation. Data saved when notify_process_location is called.
    Intended for unit tests and other kinds of debugging.

    Note this only saves the RetrievalStepCapture.get_state(),
    not the StateInfo. You will probably want to use this with
    something like StateInfoCaptureObserver.

    Note we can also save full pickles of the RetrievalStep, however
    since we are modifying the code this tends to be fragile. This might
    change in the future when things are more stable, but particularly for
    unit tests the json state is more fundamental and should be more stable.
    """

    def __init__(
        self, basefname: str, location_to_capture: str = "end_retrieval_step"
    ) -> None:
        self.basefname = basefname
        self.location_to_capture = ProcessLocation(location_to_capture)

    @classmethod
    def load_retrieval_state(cls, fname: str | os.PathLike[str]) -> dict[str, Any]:
        return json.loads(gzip.open(fname, mode="rb").read().decode("utf-8"))

    def notify_process_location(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        if location != self.location_to_capture:
            return
        if retrieval_strategy_step is None:
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        fname = (
            f"{self.basefname}_{retrieval_strategy.strategy_step.step_number}.json.gz"
        )
        with gzip.open(fname, mode="wb") as fh:
            fh.write(
                json.dumps(
                    retrieval_strategy_step.get_state(), sort_keys=True, indent=4
                ).encode("utf-8")
            )


__all__ = [
    "RetrievalStrategyStepSet",
    "RetrievalStrategyStep",
    "RetrievalStrategyStepNotImplemented",
    "RetrievalStepCaptureObserver",
]
