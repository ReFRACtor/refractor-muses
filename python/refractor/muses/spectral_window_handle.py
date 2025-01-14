from __future__ import annotations
from .creator_handle import CreatorHandleSet, CreatorHandle
from .filter_metadata import FileFilterMetadata
from .muses_spectral_window import MusesSpectralWindow
from loguru import logger
import abc
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import MeasurementId
    from .muses_strategy_executor import CurrentStrategyStep


class SpectralWindowHandle(CreatorHandle, metaclass=abc.ABCMeta):
    """Base class for SpectralWindowHandle. Note we use duck typing,
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents that
    a class is intended for this.

    """

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    def filter_name_dict(
        self, current_strategy_step: CurrentStrategyStep
    ) -> dict[str, list[str]] | None:
        """Return a dictionary that goes from instrument name to a
        list of filter names.  This is needed when are initially
        reading the data. This can be gotten from
        spectral_window_dict, but a handler might have a more
        efficient way to calculate just this value. The list of filter
        names might be empty, if all the values are None in the
        filter_name data.

        """
        # Default is to get this from the spectral_window_dict.
        swin_dict = self.spectral_window_dict(current_strategy_step)
        if swin_dict is None:
            return None
        res = {}
        for iname, swin in swin_dict.items():
            if swin.filter_name is not None:
                res[iname] = [
                    i
                    for i in list(dict.fromkeys(swin.filter_name.flatten().tolist()))
                    if i is not None
                ]
            else:
                res[iname] = []
        return res

    @abc.abstractmethod
    def spectral_window_dict(
        self, current_strategy_step: CurrentStrategyStep
    ) -> dict[str, MusesSpectralWindow] | None:
        """Return a dictionary that goes from instrument name to the
        MusesSpectralWindow for that instrument. Note because of the
        extra metadata and bad sample/full band handing we need we
        currently require a MusesSpectralWindow. We could perhaps
        relax this in the future if we have another way of handling
        this extra functionality.

        Note that the spectral windows don't have the bad samples set
        yet, because we create the MusesSpectralWindow before the
        MusesObservation, but the MusesObservation get passed the
        MusesSpectralWindow and update the bad pixel mask then.

        """
        raise NotImplementedError()


class SpectralWindowHandleSet(CreatorHandleSet):
    """This takes a CurrentStrategyStep and maps that to a dict. The
    dict in turn maps a instrument name to the MusesSpectralWindow to
    use for that instrument.

    """

    def __init__(self):
        super().__init__("_dispatch")

    def filter_name_dict(
        self, current_strategy_step: CurrentStrategyStep
    ) -> dict[str, list[str]] | None:
        """Return a dictionary that goes from instrument name to a
        list of filter names.  This is needed when are initially
        reading the data. This can be gotten from
        spectral_window_dict, but a handler might have a more
        efficient way to calculate just this value. The list of filter
        names might be empty, if all the values are None in the
        filter_name data.

        """
        return self.handle("filter_name_dict", current_strategy_step)

    def spectral_window_dict(
        self, current_strategy_step: CurrentStrategyStep
    ) -> dict[str, MusesSpectralWindow]:
        """Return a dictionary that goes from instrument name to the
        MusesSpectralWindow for that instrument. Note because of the
        extra metadata and bad sample/full band handing we need we
        currently require a MusesSpectralWindow. We could perhaps
        relax this in the future if we have another way of handling
        this extra functionality.

        """
        return self.handle("spectral_window_dict", current_strategy_step)


class MusesPySpectralWindowHandle(SpectralWindowHandle):
    """This wraps the old muses-py code for determining the spectral
    window. Note the logic used in this code is a bit complicated,
    this looks like something that has been extended and had special
    cases added over time. We should probably replace this with newer
    code, but this older wrapper is useful for doing testing if
    nothing else.

    """

    def __init__(self):
        self.viewing_mode = None
        self.spectral_window_directory = None
        self.filter_metadata = None

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # We'll add grabbing the stuff out of RetrievalConfiguration in a bit
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.spectral_window_directory = measurement_id["spectralWindowDirectory"]
        self.viewing_mode = measurement_id["viewingMode"]
        self.filter_metadata = FileFilterMetadata(
            measurement_id["defaultSpectralWindowsDefinitionFilename"]
        )
        self.filter_list_dict = measurement_id.filter_list_dict

    def spectral_window_dict(
        self, current_strategy_step: CurrentStrategyStep
    ) -> dict[str, MusesSpectralWindow] | None:
        """Return a dictionary that goes from instrument name to the MusesSpectralWindow
        for that instrument. Note because of the extra metadata and bad sample/full band
        handing we need we currently require a MusesSpectralWindow. We could perhaps
        relax this in the future if we have another way of handling this extra functionality."""
        fname = MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            self.viewing_mode,
            self.spectral_window_directory,
            current_strategy_step.retrieval_elements,
            current_strategy_step.step_name,
            current_strategy_step.retrieval_type,
            current_strategy_step.microwindow_file_name_override,
        )
        logger.debug(
            f"Creating spectral_window_dict using MusesSpectralWindow by reading file {fname}"
        )
        return MusesSpectralWindow.create_dict_from_file(
            fname, self.filter_list_dict, self.filter_metadata
        )


# For now, just fall back to the old muses-py code.
SpectralWindowHandleSet.add_default_handle(MusesPySpectralWindowHandle())

__all__ = [
    "SpectralWindowHandle",
    "SpectralWindowHandleSet",
    "MusesPySpectralWindowHandle",
]
