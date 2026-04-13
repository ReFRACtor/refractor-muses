from __future__ import annotations
from .creator_handle import CreatorHandleWithContext, CreatorHandleWithContextSet
from .creator_dict import CreatorDict
from .identifier import (
    InstrumentIdentifier,
    StateElementIdentifier,
    RetrievalType,
    FilterIdentifier,
    IdentifierSortByWaveLength,
)
from .input_file_helper import InputFileHelper, InputFilePath
from .spectral_window_handle import SpectralWindowHandleSet, MusesSpectralWindowDict
from .current_state import CurrentState
from .retrieval_array import FullGridMappedArray
from .muses_strategy_context import MusesStrategyContext
from .current_strategy_step import (
    CurrentStrategyStep,
    CurrentStrategyStepOEImp,
)
import os
import abc
import typing
import yaml  # type: ignore
import pyaml  # type: ignore
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Any, cast, Hashable

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .retrieval_strategy import RetrievalStrategy
    from .state_element import StateElementWithCreate
    from .retrieval_configuration import RetrievalConfiguration
    from .input_file_helper import InputFileHelper


class TesStrategyTableReader:
    """The '.asc' format for the strategy table needs to a bit of massaging
    to read cleanly. The table uses the convention of '-' as a None/Null entry.
    Also, some of the fields are arrays, which isn't always clear (so no marker
    for a single entry, or that '-' actually means and empty array).

    This class returns the massaged data. We also provide functions to convert
    this to and from YAML. The YAML is just sugar for the old strategy tables.
    I find it much easier to read and write, but has exactly the same content
    as the '.asc' Tes files. However for non OE retrievals, we really do need
    a YAML like file format because the content needed for a strategy step are
    completely different for something like a ML retrieval strategy step.
    """

    def __init__(
        self,
        fname: str | os.PathLike[str] | InputFilePath,
        ifile_help: InputFileHelper | None = None,
    ) -> None:
        if ifile_help is None:
            ifile_help = InputFileHelper()
        self._tes_file = ifile_help.open_tes(fname)

    @property
    def table(self) -> list[dict[Hashable, Any]]:
        res = []
        for _, row in self._tes_file.checked_table.iterrows():
            rdict = row.to_dict()
            for k, v in dict(rdict).items():
                if k in (
                    "retrievalElements",
                    "errorAnalysisInterferents",
                    "donotupdate",
                ):
                    rdict[k] = [] if v == "-" else v.split(",")
                else:
                    rdict[k] = None if v == "-" else v
            res.append(rdict)
        return res

    def to_yaml(self, fname: str | os.PathLike[str]) -> None:
        """Rewrite the table as a YAML file"""
        with open(fname, "w") as fh:
            print(
                pyaml.dump({"strategy": self.table}, sort_keys=False, indent=4), file=fh
            )

    @classmethod
    def from_yaml(
        self, yaml_fname: str | os.PathLike[str], tes_fname: str | os.PathLike[str]
    ) -> None:
        # TODO This creates a readable file, but it doesn't space the columns to
        # align. We could probably work up something to do that, but I'm not sure
        # how often we will actually use this function
        with open(yaml_fname, "r") as fh:
            d = yaml.safe_load(fh)["strategy"]
        d2 = []
        for row in d:
            rdict = {}
            for k, v in row.items():
                if k in (
                    "retrievalElements",
                    "errorAnalysisInterferents",
                    "donotupdate",
                ):
                    rdict[k] = "-" if len(v) == 0 else ",".join(v)
                else:
                    rdict[k] = "-" if v is None else str(v)
            d2.append(rdict)
        df = pd.DataFrame(d2)
        with open(tes_fname, "w") as fh:
            print("TES_File_ID = L2: Strategy Table", file=fh)
            print(f"Data_Size = {len(d)} x {len(d[0].keys())}", file=fh)
            print(
                "End_of_Header  ****  End_of_Header  ****  End_of_Header  ****  End_of_Header",
                file=fh,
            )
            print(" ".join(d[0].keys()), file=fh)
            print(
                " ".join(
                    [
                        "na",
                    ]
                    * len(d[0].keys())
                ),
                file=fh,
            )
            df.to_csv(fh, index=False, sep=" ", header=False)


class MusesStrategyHandle(CreatorHandleWithContext, metaclass=abc.ABCMeta):
    """Base class for MusesStrategyHandle. Note we use duck typing, so
    you don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents that
    a class is intended for this.

    This can do caching based on assuming the MusesStrategyContext is
    the same between calls, see CreatorHandle for a discussion of
    this.
    """

    @abc.abstractmethod
    def muses_strategy(
        self,
        strategy_context: MusesStrategyContext,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        """Return MusesStrategy if we can process the given context,
        or None if we can't.
        """
        raise NotImplementedError()


class MusesStrategyHandleSet(CreatorHandleWithContextSet):
    """This takes the MusesStrategyContext and creates a MusesStrategy for
    processing it."""

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("muses_strategy", strategy_context)

    def muses_strategy(
        self,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> MusesStrategy:
        """Create a MusesStrategy for the given strategy_context."""
        return self.handle(self.strategy_context, creator_dict, **kwargs)


class MusesStrategy(object, metaclass=abc.ABCMeta):
    """A MusesStrategy is a list of steps to be executed by a
    MusesStrategyExecutor.  Each step is represented by a
    CurrentStrategyStep, which give the list of retrieval elements, spectral
    windows, etc.

    The canonical MusesStrategy is to read a Table.asc file (e.g.,
    StrategyTable) to get the information about each step.

    Note that later steps can be changed based off of the results of
    previous steps. This part of the code may well need to be more
    developed, the only example we have right now is choosing steps
    based off of brightness temperature of the observation. If we get
    more examples of this, we may want to rework how conditional
    processing is handled.

    """

    @abc.abstractmethod
    def is_next_bt(self) -> bool:
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        # TODO Possibly rework this interface by changing
        # RetrievalStrategyStepBT
        raise NotImplementedError()

    @abc.abstractproperty
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps). This is needed by
        MusesObservation. Perhaps we can remove this, it would be
        reasonable to just read filters on first use in the
        MusesObservation.  But for now, this is needed.

        Note there is a assumption in py-retrieve that the
        list[FilterIdentifier] is sorted by the starting wavelength
        for each of filters, so we do that here.

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all
        retrieval steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all
        retrieval steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval
        steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        There is an assumption in the py-retrieve code that the instruments
        are sorted by whatever the smallest wavelength in the SpectralWindow
        for that instrument is, so this list is sorted by that logic.
        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractmethod
    def restart(self) -> None:
        """Set step to the first one."""
        raise NotImplementedError()

    @abc.abstractmethod
    def next_step(self, current_state: CurrentState | None) -> None:
        """Advance to the next step."""
        raise NotImplementedError()

    def set_step(self, step_number: int, current_state: CurrentState | None) -> None:
        """Set the step number to the given value"""
        # Default is just to restart and proceed through the given number of steps.
        self.restart()
        for i in range(step_number):
            self.next_step(current_state)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """True if we have reached the last step"""
        raise NotImplementedError()

    @abc.abstractmethod
    def current_strategy_step(self) -> CurrentStrategyStep | None:
        """Return the CurrentStrategyStep for the current
        step. Returns None is self.is_done()"""
        raise NotImplementedError

    def retrieval_initial_fm_from_cycle(
        self, selem: StateElementWithCreate, retrieval_config: RetrievalConfiguration
    ) -> None:
        """This cycles a state element through all the strategy steps,
        and uses the final value_fm to set the retrieval_initial_fm.

        Note that the actual steps used in a strategy table might
        depend on the results of previous steps. It isn't clear what
        we want to do in general here, but what we do is pass through
        all possible steps.

        According to Susan, historically the initial guess stuff was
        translated to and from the retrieval grid. This was done so
        that paired retrievals are in sync, so if we retrieve H2O with
        O3 held fixed and then O3 we don't want the O3 to jump a bunch
        as it goes to the retrieval grid. She said this is less
        important now where a lot of stuff is retrieved at the same
        time.

        But py-retrieve cycled through all the strategy table, which had
        the side effect of calling get_initial_guess() and taking
        values to and from the retrieval grid (so
        FullGridMappedArrayFromRetGrid). This function tries to
        duplicate this.

        We might 1) decide not to continue doing this or 2) We can't
        actually do this with a general MusesStrategy (we don't know
        all the steps until we actually process them). But for now, we
        will have this code in place.

        Note that we update selem in place. We don't generally want to
        have side effects, that update arguments, but in this case the
        entire job of the function is to update selem.

        TODO - Decide if this is something we want to continue doing

        """
        # Don't check against the old state element values while doing this, the
        # whole point is to update stuff to match what py-retrieve did
        original_value = CurrentState.check_old_state_element_value
        try:
            CurrentState.check_old_state_element_value = False
            if self.is_done():
                raise RuntimeError(
                    "Can't call retrieval_initial_fm_from_cycle id we are done"
                )
            cstep = self.current_strategy_step()
            if cstep is None:
                raise RuntimeError("This can't happen")
            cstepnum = cstep.strategy_step.step_number
            self.restart()
            while not self.is_done():
                cstep = self.current_strategy_step()
                if cstep is None:
                    raise RuntimeError("This can't happen")
                selem.notify_start_step(
                    cstep,
                    retrieval_config,
                    skip_initial_guess_update=True,
                )
                # For state element that are retrieved, replace value_fm
                # with value_fmprime. Don't do this for the state elements
                # with a spectral_domain - they handle this themselves as
                # a special case.
                if selem.retrieved_this_step and not selem.spectral_domain:
                    value_fm = selem.value_fm.to_fmprime(
                        selem.state_mapping_retrieval_to_fm(),
                        selem.state_mapping(include_subset=False),
                        selem.should_fix_negative,
                    ).view(FullGridMappedArray)
                    selem.update_state_element(value_fm=value_fm)
                selem.update_state_element(next_step_initial_fm=selem.value_fm)
                if False:
                    # We've need this debugging enough time that just keep the code
                    # here. If we haven't used this in a while, we can delete this block
                    print(cstep.strategy_step.step_number)
                    print(selem.value_fm[-10])
                self.next_step(None)
            self.set_step(cstepnum, None)
            # Note, rightly or wrongly we don't always update constraint_vector.
            # This is to match what py-retrieve does
            # TODO Determine if this is the correct behavior
            selem.update_state_element(
                retrieval_initial_fm=selem.value_fm.copy(),
                step_initial_fm=selem.value_fm.copy(),
                next_step_initial_fm=None,
            )

            # StateElementWithCreate has a
            # notify_done_retrieval_initial_fm_from_cycle. Call if the
            # StateElement has this attribute - no error if it
            # doesn't. Just marks so we know this has already been run
            if hasattr(selem, "notify_done_retrieval_initial_fm_from_cycle"):
                selem.notify_done_retrieval_initial_fm_from_cycle()
        finally:
            CurrentState.check_old_state_element_value = original_value

    def notify_update_strategy_context(
        self, strategy_context: MusesStrategyContext
    ) -> None:
        pass


class MusesStrategyImp(MusesStrategy):
    """Base class for the way we generally implement a MusesStrategy"""

    def __init__(self, creator_dict: CreatorDict) -> None:
        self._creator_dict = creator_dict

    @property
    def spectral_window_handle_set(self) -> SpectralWindowHandleSet:
        """The SpectralWindowHandleSet to use for getting the MusesSpectralWindow."""
        return self._creator_dict[MusesSpectralWindowDict]

    @property
    def creator_dict(self) -> CreatorDict:
        """The CreatorDict"""
        return self._creator_dict


class MusesStrategyFileHandle(MusesStrategyHandle):
    def muses_strategy(
        self,
        strategy_context: MusesStrategyContext,
        creator_dict: CreatorDict,
        strategy_table_filename: str | os.PathLike[str] | None = None,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        if (self.retrieval_config["run_dir"] / "strategy.yaml").exists():
            fname = self.retrieval_config["run_dir"] / "strategy.yaml"
        else:
            fname = self.retrieval_config["run_dir"] / "Table.asc"

        res = MusesStrategyStepList.create_from_strategy_file(
            strategy_table_filename if strategy_table_filename is not None else fname,
            self.retrieval_config.input_file_helper,
            strategy_context,
            creator_dict,
        )
        self.strategy_context.add_observer(res)
        return res


class MusesStrategyStepList(MusesStrategyImp):
    """This implementation uses a list of CurrentStrategyStep."""

    def __init__(
        self,
        strategy_context: MusesStrategyContext,
        creator_dict: CreatorDict,
    ) -> None:
        """This uses a list of CurrentStrategyStep,
        self.current_strategy_list.  Note that there is a bit of
        information that depends on the MusesStrategyContext, we fill
        that in when notify_update_strategy_context. Also, we may end
        up skipping some of the steps. The current_strategy_list can
        get modified (e.g., a derived class or outside process changes
        this), which can be useful to modify a StrategyTable. This can
        be useful for doing a variation of a StrategyTable for a test
        or something like that, without needing to modify the original
        StrategyTable.

        The various StateElement lists (like retrieval_elements) need
        to be ordered - however you don't need to worry about that when
        adding to the list. We do the sorting automatically in
        CurrentStrategyStepDict.

        """
        super().__init__(creator_dict)
        self._filter_list_dict: (
            dict[InstrumentIdentifier, list[FilterIdentifier]] | None
        ) = None
        self.current_strategy_list: list[CurrentStrategyStep] = []
        self._cur_step_index = 0
        self._cur_step_count = 0
        self.strategy_context = strategy_context

    @classmethod
    def create_from_strategy_file(
        cls,
        filename: str | os.PathLike[str],
        ifile_hlp: InputFileHelper,
        strategy_context: MusesStrategyContext,
        creator_dict: CreatorDict,
    ) -> MusesStrategyStepList:
        """Create a MusesStrategyStepList from a strategy table file."""
        res = cls(strategy_context, creator_dict)
        if Path(filename).suffix == ".yaml":
            table = ifile_hlp.open_yaml(filename)["strategy"]
        elif Path(filename).suffix == ".asc":
            table = TesStrategyTableReader(filename, ifile_hlp).table
        i2 = -1
        for i, row in enumerate(table):
            i2 += 1
            cstep = creator_dict[CurrentStrategyStep].create_current_strategy_step(
                i2, row, creator_dict[MusesSpectralWindowDict]
            )
            res.current_strategy_list.append(cstep)
        return res

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps). This is needed by
        MusesObservation. Perhaps we can remove this, it would be
        reasonable to just read filters on first use in the
        MusesObservation.  But for now, this is needed.

        Note there is a assumption in py-retrieve that the
        list[FilterIdentifier] is sorted by the starting wavelength
        for each of filters, so we do that here.

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.
        """
        if self._filter_list_dict is None:
            filter_list_dict_t: dict[
                InstrumentIdentifier, dict[FilterIdentifier, float]
            ] = defaultdict(dict)
            for cstep in self.current_strategy_list:
                sdict = self.spectral_window_handle_set.spectral_window_dict(
                    cstep, None
                )
                for k, v in sdict.items():
                    for fid, swav in v.filter_name_list():
                        filter_list_dict_t[k][fid] = swav

            self._filter_list_dict = {}
            for k2, v2 in filter_list_dict_t.items():
                sflist = IdentifierSortByWaveLength()
                for fid, swav in v2.items():
                    sflist.add(fid, swav)
                self._filter_list_dict[k2] = cast(
                    list[FilterIdentifier], sflist.sorted_identifer()
                )
        return self._filter_list_dict

    def _next_step_peek(self) -> CurrentStrategyStep | None:
        """Return the value of the next step, without actually changing the current step.
        Returns None if we hit done before getting the next step"""
        i = 1
        while (
            self._cur_step_index + i < len(self.current_strategy_list)
            and self.current_strategy_list[self._cur_step_index + i].is_skipped
        ):
            i += 1
        if self._cur_step_index + i < len(self.current_strategy_list):
            return self.current_strategy_list[self._cur_step_index + i]
        return None

    def is_next_bt(self) -> bool:
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        nstep = self._next_step_peek()
        if nstep is None:
            return False
        return nstep.retrieval_type == RetrievalType("BT")

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all
        retrieval steps)"""
        res: set[StateElementIdentifier] = set()
        for cstep in self.current_strategy_list:
            if hasattr(cstep, "retrieval_elements"):
                res.update(cstep.retrieval_elements)
        return StateElementIdentifier.sort_identifier(list(res))

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all
        retrieval steps)"""
        res: set[StateElementIdentifier] = set()
        for cstep in self.current_strategy_list:
            if hasattr(cstep, "error_analysis_interferents"):
                res.update(cstep.error_analysis_interferents)
        return StateElementIdentifier.sort_identifier(list(res))

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval
        steps)"""
        res: set[InstrumentIdentifier] = set()
        for cstep in self.current_strategy_list:
            if hasattr(cstep, "instrument_name"):
                res.update(cstep.instrument_name)
        return InstrumentIdentifier.sort_identifier(list(res))

    def restart(self) -> None:
        """Set step to the first one."""
        self._cur_step_index = 0
        self._cur_step_count = 0

    def next_step(self, current_state: CurrentState | None) -> None:
        """Advance to the next step."""
        if current_state is not None:
            self._handle_bt(current_state)
        self._cur_step_count += 1
        self._cur_step_index += 1
        while (
            not self.is_done()
            and self.current_strategy_list[self._cur_step_index].is_skipped
        ):
            self._cur_step_index += 1

    def _handle_bt(self, current_state: CurrentState) -> None:
        # Nothing to do if this isn't a BT step, or if we don't have the
        # brightness_temperature_data
        cstep = self.current_strategy_step()
        btdata = current_state.brightness_temperature_data(self._cur_step_count)
        if (
            cstep is None
            or cstep.retrieval_type != RetrievalType("BT")
            or btdata is None
        ):
            return
        # List of species determine in RetrievalStrategyStepBT
        species_igr = btdata["species_igr"]
        if species_igr is None:
            return

        # Look at all the bt_ig_refine, and only activate the one that contains species_igr
        found = False
        for cstate in self.current_strategy_list[self._cur_step_index + 1 :]:
            if cstate.retrieval_type != RetrievalType("bt_ig_refine") or not hasattr(
                cstate, "retrieval_elements"
            ):
                break
            if ",".join([str(s) for s in cstate.retrieval_elements]) != species_igr:
                cstate.is_skipped = True
            else:
                found = True
                cstate.is_skipped = False
        # Note species_igr of "-" is a special value that mean we
        # should skip all the BT_IG_Refine steps. So in that case, not
        # finding the step matching species_igr isn't actually an
        # error - we just mark at the bt_ig_refine as skipped
        if not found and species_igr != "-":
            raise RuntimeError(
                "Specified IG refinement not found (MUST be retrievalType BT_IG_Refine AND species listed in correct order)"
            )

    def is_done(self) -> bool:
        """True if we have reached the last step"""
        return self._cur_step_index >= len(self.current_strategy_list)

    def current_strategy_step(self) -> CurrentStrategyStep | None:
        """Return the CurrentStrategyStep for the current
        step. Returns None is self.is_done()"""
        if self.is_done():
            return None
        cstate = self.current_strategy_list[self._cur_step_index]
        # Update the step number, which may be different than the initial step number
        # we had before we marked anything skip
        cstate.strategy_step.step_number = self._cur_step_count
        return cstate

    def _parse_state_elements(self, s: str) -> list[StateElementIdentifier]:
        """Small logic used to handle the state element in the file"""
        # We need to handle empty lists (which get expressed as "-", and also make
        # sure the elements get put in the right order.
        r = s.split(",")
        if r[0] == "-":
            return []
        return [StateElementIdentifier(i) for i in r]

    def notify_update_strategy_context(
        self, strategy_context: MusesStrategyContext
    ) -> None:
        # Mark all the steps as available again. This gets modified as the
        # actual retrieval takes place
        for cstep in self.current_strategy_list:
            cstep.is_skipped = False
        # The filter_list_dict need to be regenerated, so mark as not available
        # yet
        self._filter_list_dict = None


class MusesStrategyModifyHandle(MusesStrategyHandle):
    """This is a handle useful for a quick test. We read in an
    existing Table.asc, but then modify the given step_number to have
    the given retrieval_elements and optionally change the max_iter.

    This is limited of course, but handles the bulk of what we have
    run into in our pytests. You can do a similar more complicated
    handle if useful, or even derive from MusesStrategyStepList for
    more complicated stuff.
    """

    def __init__(
        self,
        step_number: int,
        retrieval_elements: list[StateElementIdentifier],
        max_iter: int | None = None,
    ):
        super().__init__()
        self._step_number_value = step_number
        self.retrieval_elements = retrieval_elements
        self.max_iter = max_iter

    def muses_strategy(
        self,
        strategy_context: MusesStrategyContext,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        if (self.retrieval_config["run_dir"] / "strategy.yaml").exists():
            fname = self.retrieval_config["run_dir"] / "strategy.yaml"
        else:
            fname = self.retrieval_config["run_dir"] / "Table.asc"
        s = MusesStrategyStepList.create_from_strategy_file(
            fname,
            self.retrieval_config.input_file_helper,
            strategy_context,
            creator_dict,
        )
        t = s.current_strategy_list[self._step_number_value]
        assert isinstance(t, CurrentStrategyStepOEImp)
        t.retrieval_elements = StateElementIdentifier.sort_identifier(
            self.retrieval_elements
        )
        if self.max_iter is not None:
            t.retrieval_step_parameters["max_iter"] = self.max_iter
        self.strategy_context.add_observer(s)
        return s


def modify_strategy_table(
    rs: RetrievalStrategy,
    step_number: int,
    retrieval_elements: list[StateElementIdentifier],
    max_iter: int | None = None,
) -> None:
    """Simple function to modify the strategy table in a RetrievalStrategy. Meant for
    things like unit tests."""
    h = MusesStrategyModifyHandle(step_number, retrieval_elements, max_iter)
    rs.creator_dict[MusesStrategy].add_handle(h, 100)


MusesStrategyHandleSet.add_default_handle(MusesStrategyFileHandle())
# Register creator set
CreatorDict.register(MusesStrategy, MusesStrategyHandleSet)

__all__ = [
    "MusesStrategy",
    "MusesStrategyImp",
    "MusesStrategyHandle",
    "MusesStrategyHandleSet",
    "MusesStrategyModifyHandle",
    "modify_strategy_table",
    "MusesStrategyStepList",
    "TesStrategyTableReader",
]
