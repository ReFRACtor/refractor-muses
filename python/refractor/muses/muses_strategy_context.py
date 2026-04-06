from __future__ import annotations
import os
from pathlib import Path
import typing
from dataclasses import dataclass, field
from typing import Any, Self

if typing.TYPE_CHECKING:
    from .identifier import (
        InstrumentIdentifier,
        FilterIdentifier,
        RetrievalType,
        StrategyStepIdentifier,
    )

    from .input_file_helper import InputFileHelper
    from .muses_observation import MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .creator_dict import CreatorDict
    import pystac


@dataclass
class _ContextData:
    """One level of indirection, to support merge"""

    measurement_id: None | MeasurementId = None
    stac_catalog: None | pystac.Catalog = None
    retrieval_config: None | RetrievalConfiguration = None
    strategy: None | MusesStrategy = None
    filter_list_dict: None | dict[InstrumentIdentifier, list[FilterIdentifier]] = None
    need_notify: dict[Any, bool] = field(default_factory=dict)
    observers: set[Any] = field(default_factory=set)
    # Marker if notify_update_strategy_context has been called. If it
    # has, we want to immediately call this on new observers that have
    # been added
    has_been_updated = False


class MusesStrategyContext:
    """The MusesStrategyExecutor works in a context for doing one
    strategy execution. This currently just is the input data we are
    running on, either a MeasurementId or a pystac.Collection
    (currently), a RetrievalConfiguration, a MusesStrategy

    Note that a context is only for one MusesStrategyExecutor. We can
    certainly have multiple MusesStrategyExecutor in a python
    environment (e.g., we are running a test comparing two
    retrievals), but they each get their own MusesStrategyContext.

    Things such as CreatorHandleSet can assume that the
    MusesStrategyContext is unchanged as we move through the
    MusesStrategyExecutor, so if it makes sense internal caching based
    on the MusesStrategyExecutor can be done (e.g., read a L1B input
    file once).

    Objects can register as observers to be notified when something in
    the MusesStrategyContext changes (e.g., we move to the next target
    to process).

    The notify_update_strategy_context is guaranteed to be called
    before the first MusesStrategyExecutor starts, so any delayed
    initialization can depend on this being called (e.g. some setup
    that depends on the the MeasurementId). This is useful so we can
    do setup before pointing to a particular context. If the context
    is already set up, notify_update_strategy_context is called when
    the observer is attached. Otherwise this is delayed until the
    MusesStrategyContext is populated.

    I'm not sure how general this class will end up being. Right now
    we just read fixed files to populate this.

    """

    def __init__(
        self,
        strategy_table_filename: str | os.PathLike[str] | None = None,
        measurement_id_fname: str | os.PathLike[str] = "./Measurement_ID.asc",
        retrieval_config_fname: str | os.PathLike[str] = "./Table.asc",
        ifile_hlp: InputFileHelper | None = None,
        creator_dict: CreatorDict | None = None,
    ) -> None:
        """Create a empty MusesStrategyContext. Because it is common in testing,
        you can optionally pass a strategy_table_filename and we then call
        create_from_table_filename. This is just a short cut, because we do this
        a lot.
        """
        # We have one level of indirection here to support the merge function below
        self._context_data = _ContextData()
        # Be careful with adding any other variables here. The parent MusesStrategyContext
        # may be different in different places, even if the self._context_data
        # should be shared.
        self._observers: set[Any] = set()
        if strategy_table_filename is not None:
            self.create_from_table_filename(
                strategy_table_filename,
                measurement_id_fname=measurement_id_fname,
                retrieval_config_fname=retrieval_config_fname,
                ifile_hlp=ifile_hlp,
                creator_dict=creator_dict,
            )

    def merge(self, other: MusesStrategyContext) -> Self:
        """Replace our context data with other._context_data so the are both
        pointing at the same data. Merge the observer lists."""
        # We may already point to the same data, in which case this is a noop
        if id(self._context_data) == id(other._context_data):
            return self
        new_observers = self._context_data.observers | other._context_data.observers
        old_observers = set(self._context_data.observers)
        self._context_data = other._context_data
        self._context_data.observers = new_observers
        for obs in old_observers:
            self._context_data.need_notify[obs] = True
        if self._context_data.has_been_updated:
            lobs = list(self._context_data.need_notify.keys())
            for obs in lobs:
                if self._context_data.need_notify[obs]:
                    obs.notify_update_strategy_context(self)
                    self._context_data.need_notify[obs] = False
        return self

    def create_from_table_filename(
        self,
        strategy_table_filename: str | os.PathLike[str],
        measurement_id_fname: str | os.PathLike[str] = "./Measurement_ID.asc",
        retrieval_config_fname: str | os.PathLike[str] = "./Table.asc",
        ifile_hlp: InputFileHelper | None = None,
        creator_dict: CreatorDict | None = None,
    ) -> None:
        from .muses_observation import MeasurementIdFile
        from .retrieval_configuration import RetrievalConfiguration
        from .creator_dict import CreatorDict

        cdict = creator_dict if creator_dict is not None else CreatorDict(self)

        table_filename = Path(strategy_table_filename).absolute()
        rconf_filename = table_filename.parent / retrieval_config_fname
        mid_filename = table_filename.parent / measurement_id_fname
        rconf = RetrievalConfiguration.create_from_strategy_file(
            rconf_filename, ifile_hlp
        )
        mid = MeasurementIdFile(mid_filename, rconf)
        self.update_strategy_context(
            measurement_id=mid,
            retrieval_config=rconf,
            creator_dict=cdict,
            strategy_table_filename=strategy_table_filename,
        )

    def add_observer(self, obs: Any) -> None:
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._context_data.observers.add(obs)
        if hasattr(obs, "notify_add"):
            obs.notify_add(self)
        # Go ahead and call notify_update_strategy_context if we have
        # already been updated. This way the observer can do any initialization
        # needed for the first time the context is available. Otherwise, this
        # will get called in self.notify_update_strategy_context when that
        # happens later.
        if self._context_data.has_been_updated:
            obs.notify_update_strategy_context(self)

    def remove_observer(self, obs: Any) -> None:
        self._context_data.observers.discard(obs)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self) -> None:
        # We change self._context_data.observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._context_data.observers)
        for obs in lobs:
            self.remove_observer(obs)

    def notify_if_needed(self, obs: Any) -> None:
        """We can get race conditions when doing
        update_strategy_context, when we notify the various observers
        there is no guaranteed order of notifications.  A object may
        turn around and use another object (e.g., calling a
        CreatorHandleWithContextSet for a SpectralWindow or something
        like that). To avoid this, we have CreatorHandleWithContextSet
        check before using any object if it is on the list of items to
        be updated, and if so go ahead an send the
        notify_update_strategy_context message. If we aren't in the
        process of passing this out or if the object is already
        notified, this is a noop. Otherwise we go ahead and notify
        that object.

        Note it is safe to call this with objects that aren't actually
        observers. So CreatorHandleWithContextSet doesn't need to
        determine if a particular handle is an observer or not. This
        is a noop if obs isn't even an observer.
        """
        if self._context_data.need_notify.get(obs, False):
            obs.notify_update_strategy_context(self)
            self._context_data.need_notify[obs] = False

    def notify_update_strategy_context(self) -> None:
        self._context_data.need_notify.clear()
        for obs in self._context_data.observers:
            self._context_data.need_notify[obs] = True
        lobs = list(self._context_data.need_notify.keys())
        for obs in lobs:
            if self._context_data.need_notify[obs]:
                obs.notify_update_strategy_context(self)
                self._context_data.need_notify[obs] = False
        self._context_data.need_notify.clear()

    def update_strategy_context(
        self,
        creator_dict: CreatorDict,
        measurement_id: None | MeasurementId = None,
        stac_catalog: None | pystac.Catalog = None,
        retrieval_config: None | RetrievalConfiguration = None,
        strategy: None | MusesStrategy = None,
        strategy_table_filename: str | os.PathLike[str] | None = None,
        filter_list_dict: None
        | dict[InstrumentIdentifier, list[FilterIdentifier]] = None,
    ) -> None:
        from .muses_strategy import MusesStrategy
        from .spectral_window_handle import MusesSpectralWindowDict

        # Note, it is important these get created before we start the
        # update here.  These may results in calls to merge, which
        # interacts with the update. If these are already in cdict, then doing
        # this outside of the update doesn't hurt, but it is necessary if
        # they are created on first use here.
        strategy_creator = creator_dict[MusesStrategy]
        swin_creator = creator_dict[MusesSpectralWindowDict]
        # We are assuming out creators are pointing at the same context. Catch
        # this is not, we have some logic error at a higher level
        if id(strategy_creator.strategy_context._context_data) != id(  # noqa: SLF001
            self._context_data
        ):
            raise RuntimeError(
                "strategy_creator isn't looking at the same context data"
            )
        if id(swin_creator.strategy_context._context_data) != id(self._context_data):  # noqa: SLF001
            raise RuntimeError("swin_creator isn't looking at the same context data")
        self._context_data.measurement_id = measurement_id
        self._context_data.stac_catalog = stac_catalog
        self._context_data.retrieval_config = retrieval_config
        self._context_data.strategy = strategy
        self._context_data.need_notify.clear()
        for obs in self._context_data.observers:
            self._context_data.need_notify[obs] = True
        self._context_data.has_been_updated = True
        # Most of the time, the strategy isn't passed in. Instead, we create
        # this once we have the measurement_id/stac_catalog/retrieval_config.
        if self._context_data.strategy is None:
            self._context_data.strategy = strategy_creator.muses_strategy(
                swin_creator,
                strategy_table_filename=strategy_table_filename,
            )
        if filter_list_dict is not None:
            self._context_data.filter_list_dict = filter_list_dict
        else:
            self._context_data.strategy.notify_update_strategy_context(self)
            self._context_data.filter_list_dict = (
                self._context_data.strategy.filter_list_dict
            )
        lobs = list(self._context_data.need_notify.keys())
        for obs in lobs:
            if self._context_data.need_notify[obs]:
                obs.notify_update_strategy_context(self)
                self._context_data.need_notify[obs] = False
        self._context_data.need_notify.clear()

    @property
    def measurement_id(self) -> None | MeasurementId:
        return self._context_data.measurement_id

    @property
    def stac_catalog(self) -> None | pystac.Catalog:
        return self._context_data.stac_catalog

    @property
    def retrieval_config(self) -> None | RetrievalConfiguration:
        return self._context_data.retrieval_config

    @property
    def strategy(self) -> None | MusesStrategy:
        return self._context_data.strategy

    @property
    def filter_list_dict(
        self,
    ) -> None | dict[InstrumentIdentifier, list[FilterIdentifier]]:
        return self._context_data.filter_list_dict


class MusesStrategyContextMixin:
    """Mixin to provide some common functionality"""

    def __init__(
        self,
        strategy_context: MusesStrategyContext | None = None,
    ) -> None:
        self.strategy_context = (
            strategy_context if strategy_context is not None else MusesStrategyContext()
        )

    @property
    def has_measurement_id(self) -> bool:
        return self.strategy_context.measurement_id is not None

    @property
    def measurement_id(self) -> MeasurementId:
        """We often want to get the measurement id from the strategy_context,
        having an error if either the strategy_context or the measurement id
        is None. This function is a short cut for that, throwing an exception
        if we can't get the measurement_id. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        res = self.strategy_context.measurement_id
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res

    @property
    def has_retrieval_config(self) -> bool:
        return self.strategy_context.retrieval_config is not None

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        """We often want to get the retrieval_config from the strategy_context,
        having an error if either the strategy_context or the retrieval_config
        is None. This function is a short cut for that, throwing an exception
        if we can't get the retrieval_config. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        res = self.strategy_context.retrieval_config
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res

    @property
    def has_stac_catalog(self) -> bool:
        return self.strategy_context.stac_catalog is not None

    @property
    def stac_catalog(self) -> pystac.Catalog:
        """We often want to get the stac_catalog from the strategy_context,
        having an error if either the strategy_context or the stac_catalog
        is None. This function is a short cut for that, throwing an exception
        if we can't get the stac_catalog. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        res = self.strategy_context.stac_catalog
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res

    @property
    def has_filter_list_dict(self) -> bool:
        return self.strategy_context.filter_list_dict is not None

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """We often want to get the filter_list_dict from the strategy_context,
        having an error if either the strategy_context or the filter_list_dict
        is None. This function is a short cut for that, throwing an exception
        if we can't get the filter_list_dict. If you just want to check if this
        is available, you can get that from the strategy_context directly."""
        res = self.strategy_context.filter_list_dict
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res

    @property
    def has_strategy(self) -> bool:
        return self.strategy_context.strategy is not None

    @property
    def strategy(self) -> MusesStrategy:
        res = self.strategy_context.strategy
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res

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
    def step_number(self) -> int:
        return self.strategy_step.step_number

    @property
    def step_name(self) -> str:
        return self.strategy_step.step_name

    @property
    def retrieval_type(self) -> RetrievalType:
        return self.current_strategy_step.retrieval_type


__all__ = [
    "MusesStrategyContext",
    "MusesStrategyContextMixin",
]
