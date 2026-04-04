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
    from .current_state import CurrentState
    from .muses_strategy_executor import CurrentStrategyStep
    from .retrieval_strategy import RetrievalStrategy
    from .muses_strategy_context import MusesStrategyContext
    from .identifier import StrategyStepIdentifier


class RetrievalStrategyStepSet(CreatorHandleWithContextSet):
    """This takes the retrieval_type and determines a
    RetrievalStrategyStep to handle this, returning it so it can
    be called.

    Note RetrievalStrategyStep can assume that they are called for the
    same MusesStrategyContext, until notify_update_strategy_context is
    called. So if it makes sense, these objects can do internal
    caching for things that don't change when the context being
    retrieved is the same from one call to the next.
    """

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("retrieval_step", strategy_context)

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> RetrievalStrategyStep:
        return self.handle(retrieval_type, rs, creator_dict, **kwargs)


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
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> RetrievalStrategyStep | None:
        if (
            self._retrieval_type_set is None
            or retrieval_type in self._retrieval_type_set
        ):
            return self._create_cls(retrieval_type, rs, creator_dict, **kwargs)
        return None


class RetrievalStrategyStep(MusesStrategyContextMixin, metaclass=abc.ABCMeta):
    """Do the retrieval step indicated by retrieval_type

    We *only* maintain state between steps in CurrentState (other than
    possibly internal caching for performance). If the same
    RetrievalStrategyStep is called with the same CurrentState, it
    will produce the same output.

    A RetrievalStrategyStep *only* modifies CurrentState and produces
    output through side effects. Note that the RetrievalStrategyStep
    doesn't directly produce output, instead we use the
    Observer/Observable pattern to decouple generation of output. A
    RetrievalStrategyStep calls RetrievalStrategy.notify_update_target
    at points during the processing, where the Observer can do
    whatever with the processing (e.g, produce an output file).

    Note this current class has a number of things related to optimal
    estimation (e.g., forward model) that don't really apply to a
    different kind of retrieval (e.g., machine learning). We may want
    to separate this out. Right now, it seems harmless - we just have
    a bunch of functions we don't need for machine learning. But it
    might be better to separate this out, once we have a clearer
    picture of the differences between the two.

    """

    def __init__(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> None:
        super().__init__(creator_dict.strategy_context)
        self._saved_state: None | dict[str, Any] = None
        self.retrieval_type = retrieval_type
        # TODO I think we will want to remove the direct use of RetrievalStrategy,
        # although I'm not sure. For now we use this.
        self.rs = rs
        self.creator_dict = creator_dict
        self.kwargs = kwargs

    def do_retrieval(
        self,
    ) -> None:
        self.set_state(self.kwargs.get("ret_state", None))
        self.retrieval_step_body()
        self.notify_update(ProcessLocation("end_retrieval_step"))

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

    def notify_update(self, ploc: ProcessLocation) -> None:
        self.rs.notify_update(ploc, retrieval_strategy_step=self)

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        res = self.strategy.current_strategy_step()
        if res is None:
            raise RuntimeError("Need current_strategy_step")
        return res

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        return self.current_strategy_step.strategy_step

    @property
    def current_state(self) -> CurrentState:
        return self.rs.current_state


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
    much of the calculation. Data saved when notify_update is called.
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

    def notify_update(
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
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
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
