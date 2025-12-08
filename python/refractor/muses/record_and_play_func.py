from __future__ import annotations
import pickle
import typing
import os
from typing import Any

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import MusesStrategy


class RecordAndPlayFunc:
    """It can be useful for testing to be able to record the function calls to
    an object, and then play them back.

    It is useful to have some arguments "special" and not actually saved, but rather
    have a placeholder than gets replaced in the call back."""

    class PlaceHolder:
        def __init__(self, nm: str) -> None:
            self.nm = nm

    def __init__(self) -> None:
        self._record: list[tuple[str, tuple[Any, ...]]] = []

    def record(self, funcname: str, *args: Any) -> None:
        self._record.append((funcname, args))

    def replay(self, obj: Any, placeholder: dict[str, Any]) -> None:
        for funcname, avlist in self._record:
            alist = []
            for av in avlist:
                if isinstance(av, RecordAndPlayFunc.PlaceHolder):
                    alist.append(placeholder[av.nm])
                else:
                    alist.append(av)
            getattr(obj, funcname)(*alist)


class CurrentStateRecordAndPlay:
    """Add step information and interaction with CurrentState for recording and playing
    back to a certain step."""

    def __init__(self, fname: str | os.PathLike[str] | None = None) -> None:
        self._full_record: list[RecordAndPlayFunc] = []
        self._next_record: RecordAndPlayFunc | None = None
        if fname is not None:
            self.load_pickle(fname)

    def notify_start_retrieval(self) -> None:
        self._full_record = []
        self._next_record = None

    def notify_start_step(self) -> None:
        if self._next_record is not None:
            self._full_record.append(self._next_record)
        self._next_record = RecordAndPlayFunc()

    def record(self, funcname: str, *args: Any) -> None:
        if self._next_record is None:
            raise RuntimeError("Need to call notify_start_step before record")
        self._next_record.record(funcname, *args)

    def save_pickle(self, fname: str | os.PathLike[str]) -> None:
        """Save the record of the functions called to the given file."""
        if self._next_record is not None:
            self._full_record.append(self._next_record)
        self._next_record = RecordAndPlayFunc()
        pickle.dump(self._full_record, open(fname, "wb"))

    def load_pickle(self, fname: str | os.PathLike[str]) -> None:
        """Load the record of the functions from the given file."""
        self._full_record = pickle.load(open(fname, "rb"))
        self._next_record = RecordAndPlayFunc()

    def replay(
        self,
        current_state: CurrentState,
        strategy: MusesStrategy,
        retrieval_config: RetrievalConfiguration,
        step_number: int,
        at_start_step: bool = False,
    ) -> None:
        """Play back the function calls and also take the strategy to the given
        step number. We can either stop at the start of the step, and play back
        everything for the step and stop before the next step."""
        strategy.restart()
        current_state.notify_start_retrieval(
            strategy.current_strategy_step(), retrieval_config
        )
        while not strategy.is_done():
            current_state.notify_start_step(
                strategy.current_strategy_step(), retrieval_config
            )
            cstrat = strategy.current_strategy_step()
            if cstrat is None:
                raise RuntimeError("This shouldn't be able to happen")
            i = cstrat.strategy_step.step_number
            if i == step_number and at_start_step:
                return
            self._full_record[i].replay(
                current_state,
                {"current_strategy_step": strategy.current_strategy_step()},
            )
            if i == step_number:
                return
            strategy.next_step(current_state)


__all__ = ["RecordAndPlayFunc", "CurrentStateRecordAndPlay"]
