from __future__ import annotations
import os
from pathlib import Path
import typing
from dataclasses import dataclass
from typing import Any, Self

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFileHelper
    from .muses_observation import MeasurementId
    from .muses_strategy import MusesStrategy
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
    ) -> None:
        """Create a empty MusesStrategyContext. Because it is common in testing,
        you can optionally pass a strategy_table_filename and we then call
        create_from_table_filename. This is just a short cut, because we do this
        a lot.
        """
        # We have one level of indirection here to support the merge function below
        self._context_data = _ContextData()
        self._observers: set[Any] = set()
        if strategy_table_filename is not None:
            self.create_from_table_filename(strategy_table_filename)

    def merge(self, other: MusesStrategyContext) -> Self:
        """Replace our context data with other._context_data so the are both
        pointing at the same data. Merge the observer lists."""
        # We may already point to the same data, in which case this is a noop
        if id(self._context_data) == id(other._context_data):
            return self
        self._context_data = other._context_data
        # Notify observers of this current object of an update to the data.
        # We don't notify the observers of other since from their viewpoint
        # nothing has changed.
        if self._context_data.has_been_updated:
            self.notify_update_strategy_context()
        new_observers = self._observers | other._observers
        self._observers = new_observers
        # Rare time where updating a passed in argument is actually correct,
        # so although this looks funny it is actually intended.
        other._observers = new_observers
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

        table_filename = Path(strategy_table_filename).absolute()
        rconf_filename = table_filename.parent / retrieval_config_fname
        mid_filename = table_filename.parent / measurement_id_fname
        rconf = RetrievalConfiguration.create_from_strategy_file(
            rconf_filename, ifile_hlp
        )
        # TODO Hopefully we can remove the filter list dict stuff
        mid = MeasurementIdFile(mid_filename, rconf, {})
        self.update_strategy_context(
            measurement_id=mid,
            retrieval_config=rconf,
            creator_dict=creator_dict,
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
        self._observers.add(obs)
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
        self._observers.discard(obs)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self) -> None:
        # We change self._observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)

    def notify_update_strategy_context(self) -> None:
        # The list of observers might change in our loop, so grab a copy
        # of the list before we start
        lobs = list(self._observers)
        for obs in lobs:
            obs.notify_update_strategy_context(self)

    def update_strategy_context(
        self,
        measurement_id: None | MeasurementId = None,
        stac_catalog: None | pystac.Catalog = None,
        retrieval_config: None | RetrievalConfiguration = None,
        strategy: None | MusesStrategy = None,
        strategy_table_filename: str | os.PathLike[str] | None = None,
        creator_dict: CreatorDict | None = None,
    ) -> None:
        self._context_data.measurement_id = measurement_id
        self._context_data.stac_catalog = stac_catalog
        self._context_data.retrieval_config = retrieval_config
        self._context_data.strategy = strategy
        # Most of the time, the strategy isn't passed in. Instead, we create
        # this once we have the measurement_id/stac_catalog/retrieval_config.
        # A bit of a chicken an egg here, we need to use the spectral window
        # and muses strategy creators, which depend potentially on
        # notify_update_strategy_context having been called. So we go through
        # and notify these first. This is a bit clumsy, but I'm not sure how
        # to avoid this. We have this buried in the function, so hopefully this
        # hidden order dependency won't cause any problems.
        if self._context_data.strategy is None:
            from .muses_strategy import MusesStrategyHandle, MusesStrategy
            from .spectral_window_handle import (
                MusesSpectralWindowDict,
                SpectralWindowHandle,
            )
            from .creator_dict import CreatorDict

            cdict = creator_dict if creator_dict is not None else CreatorDict(self)
            strategy_creator = cdict[MusesStrategy]
            swin_creator = cdict[MusesSpectralWindowDict]
            lobs = list(self._observers)
            # Do spectral window first, since MusesStrategy depends on this
            for obs in lobs:
                if isinstance(obs, SpectralWindowHandle):
                    obs.notify_update_strategy_context(self)  # type: ignore
            # Then MusesStrategyHandle since we want to call the creator in the
            # next step
            for obs in lobs:
                if isinstance(obs, MusesStrategyHandle):
                    obs.notify_update_strategy_context(self)  # type: ignore
            self._context_data.strategy = strategy_creator.muses_strategy(
                swin_creator,
                strategy_table_filename=strategy_table_filename,
            )

        self._context_data.has_been_updated = True
        self.notify_update_strategy_context()

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
    def has_strategy(self) -> bool:
        return self.strategy_context.strategy is not None

    @property
    def strategy(self) -> MusesStrategy:
        res = self.strategy_context.strategy
        if res is None:
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return res


__all__ = [
    "MusesStrategyContext",
    "MusesStrategyContextMixin",
]
