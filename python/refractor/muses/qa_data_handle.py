from __future__ import annotations
from .creator_handle import CreatorHandleSet, CreatorHandle
from .tes_file import TesFile
from .fake_state_info import FakeStateInfo
import refractor.muses.muses_py as mpy  # type: ignore
from loguru import logger
import abc
import os
from pathlib import Path
import typing

if typing.TYPE_CHECKING:
    from .retrieval_result import RetrievalResult
    from .muses_strategy import CurrentStrategyStep
    from .muses_observation import MeasurementId


class QaFlagValue(object, metaclass=abc.ABCMeta):
    """This class has the values needed to calculate QA flag values.
    This is what is used by mpy.write_quality_flags and mpy.calculate_quality_flags.
    """

    @abc.abstractproperty
    def qa_flag_name(self) -> list[str]:
        """Return list of QA flags the other values apply to."""
        raise NotImplementedError()

    @abc.abstractproperty
    def cutoff_min(self) -> list[float]:
        """Minimum cutoff value for flag."""
        raise NotImplementedError()

    @abc.abstractproperty
    def cutoff_max(self) -> list[float]:
        """Maximum cutoff value for flag."""
        raise NotImplementedError()

    @abc.abstractproperty
    def use_for_master(self) -> list[int]:
        """Indicate of QA flag is used for master quality."""
        raise NotImplementedError()


class QaFlagValueFile(QaFlagValue):
    """Implementation that uses a file to get the values."""

    def __init__(self, fname: str | os.PathLike[str]):
        self.d = TesFile(fname)

    @property
    def qa_flag_name(self) -> list[str]:
        """Return list of QA flags the other values apply to."""
        return self.d["Flag"]

    @property
    def cutoff_min(self) -> list[float]:
        """Minimum cutoff value for flag."""
        return self.d["CutoffMin"]

    @abc.abstractproperty
    def cutoff_max(self) -> list[float]:
        """Maximum cutoff value for flag."""
        return self.d["CutoffMax"]

    @abc.abstractproperty
    def use_for_master(self) -> list[int]:
        """Indicate of QA flag is used for master quality."""
        return self.d["Use_For_Master"]


class QaDataHandle(CreatorHandle, metaclass=abc.ABCMeta):
    """Base class for QaDatawHandle. Note we use duck typing,
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.
    """

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def qa_update_retrieval_result(
        self,
        retrieval_result: RetrievalResult,
        current_strategy_step: CurrentStrategyStep,
    ) -> str | None:
        """This does the QA calculation, and updates the given
        RetrievalResult.  Returns the master quality flag results

        """
        raise NotImplementedError()


class QaDataHandleSet(CreatorHandleSet):
    """This takes a RetrievalResult and updates it with QA data."""

    def __init__(self):
        super().__init__("qa_update_retrieval_result")

    def qa_update_retrieval_result(
        self,
        retrieval_result: RetrievalResult,
        current_strategy_step: CurrentStrategyStep,
    ) -> str | None:
        """This does the QA calculation, and updates the given RetrievalResult.
        Returns the master quality flag results"""
        return self.handle(retrieval_result, current_strategy_step)


class MusesPyQaDataHandle(QaDataHandle):
    """This wraps the old muses-py code for determining the qa file
    name and then using to calculate QA information.

    Note the logic used in this code is a bit complicated, this looks
    like something that has been extended and had special cases added
    over time. We should probably replace this with newer code, but
    this older wrapper is useful for doing testing if nothing else.

    """

    def __init__(self):
        self.viewing_mode = None
        self.qa_flag_directory = None

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # We'll add grabbing the stuff out of RetrievalConfiguration
        # in a bit
        logger.debug(f"Call to {self.__class__.__name__}::notify_update_target")
        self.run_dir = Path(
            measurement_id["outputDirectory"], measurement_id["sessionID"]
        )
        self.viewing_mode = measurement_id["viewingMode"]
        self.qa_flag_directory = measurement_id["QualityFlagDirectory"]

    def qa_update_retrieval_result(
        self,
        retrieval_result: RetrievalResult,
        current_strategy_step: CurrentStrategyStep,
    ) -> str | None:
        """This does the QA calculation, and updates the given
        RetrievalResult.  Returns the master quality flag results

        """
        logger.debug(f"Doing QA calculation using {self.__class__.__name__}")
        # Name is derived from the microwindows file name
        mwfname = current_strategy_step.muses_microwindows_fname()
        quality_fname = os.path.basename(mwfname)
        quality_fname = quality_fname.replace("Microwindows_", "QualityFlag_Spec_")
        quality_fname = quality_fname.replace("Windows_", "QualityFlag_Spec_")
        quality_fname = f"{self.qa_flag_directory}/{quality_fname}"
        # if this does not exist use generic nadir / limb quality flag
        if not os.path.isfile(quality_fname):
            logger.warning(f"Could not find quality flag file: {quality_fname}")
            viewMode = self.viewing_mode.lower().capitalize()
            quality_fname = (
                f"{os.path.dirname(quality_fname)}/QualityFlag_Spec_{viewMode}.asc"
            )
            logger.warning(f"Using generic quality flag file: {quality_fname}")
            # One last check.
            if not os.path.isfile(quality_fname):
                raise RuntimeError(f"Quality flag filename not found: {quality_fname}")
        quality_fname = os.path.abspath(quality_fname)
        qa_outname = Path(
            self.run_dir,
            f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}",
            "StepAnalysis",
            "QualityFlags.asc",
        )
        fstate_info = FakeStateInfo(retrieval_result.current_state)
        master = mpy.write_quality_flags(
            qa_outname,
            quality_fname,
            retrieval_result,
            fstate_info,
            writeOutput=False,
        )
        retrieval_result.masterQuality = 1 if master == "GOOD" else 0
        logger.info(f"Master Quality: {retrieval_result.masterQuality} ({master})")
        return master


# For now, just fall back to the old muses-py code.
QaDataHandleSet.add_default_handle(MusesPyQaDataHandle())

__all__ = [
    "QaDataHandle",
    "QaDataHandleSet",
    "MusesPyQaDataHandle",
    "QaFlagValue",
    "QaFlagValueFile",
]
