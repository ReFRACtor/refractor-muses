from __future__ import annotations
import os
from pathlib import Path
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFileHelper
    from .muses_observation import MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .creator_dict import CreatorDict
    import pystac


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
        self._measurement_id: None | MeasurementId = None
        self._stac_catalog: None | pystac.Catalog = None
        self._retrieval_config: None | RetrievalConfiguration = None
        self._strategy: None | MusesStrategy = None
        self._observers: set[Any] = set()
        # Marker if notify_update_strategy_context has been called. If it
        # has, we want to immediately call this on new observers that have
        # been added
        self._have_been_updated = False
        if strategy_table_filename is not None:
            self.create_from_table_filename(strategy_table_filename)

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
        if self._have_been_updated:
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
        self._measurement_id = measurement_id
        self._stac_catalog = stac_catalog
        self._retrieval_config = retrieval_config
        self._strategy = strategy
        # Most of the time, the strategy isn't passed in. Instead, we create
        # this once we have the measurement_id/stac_catalog/retrieval_config.
        # A bit of a chicken an egg here, we need to use the spectral window
        # and muses strategy creators, which depend potentially on
        # notify_update_strategy_context having been called. So we go through
        # and notify these first. This is a bit clumsy, but I'm not sure how
        # to avoid this. We have this buried in the function, so hopefully this
        # won't cause any problems.
        if self._strategy is None:
            from .muses_strategy import MusesStrategyHandle, MusesStrategy
            from .spectral_window_handle import (
                MusesSpectralWindowDict,
                SpectralWindowHandle,
            )
            from .creator_dict import CreatorDict

            cdict = creator_dict if creator_dict is not None else CreatorDict(self)
            lobs = list(self._observers)
            for obs in lobs:
                if isinstance(obs, MusesStrategyHandle) or isinstance(
                    obs, SpectralWindowHandle
                ):
                    obs.notify_update_strategy_context(self)  # type: ignore
            self._strategy = cdict[MusesStrategy].muses_strategy(
                cdict[MusesSpectralWindowDict],
                strategy_table_filename=strategy_table_filename,
            )

        self._have_been_updated = True
        self.notify_update_strategy_context()

    @property
    def measurement_id(self) -> None | MeasurementId:
        return self._measurement_id

    @property
    def stac_catalog(self) -> None | pystac.Catalog:
        return self._stac_catalog

    @property
    def retrieval_config(self) -> None | RetrievalConfiguration:
        return self._retrieval_config

    @property
    def strategy(self) -> None | MusesStrategy:
        return self._strategy


__all__ = [
    "MusesStrategyContext",
]
