from __future__ import annotations
from .creator_handle import CreatorHandleSet, CreatorHandle
from .fake_state_info import FakeStateInfo
from loguru import logger
import abc
import os
import numpy as np
from pathlib import Path
from typing import Any
import typing
import pandas as pd
import texttable  # type: ignore

if typing.TYPE_CHECKING:
    from .retrieval_result import RetrievalResult
    from .muses_strategy import CurrentStrategyStep
    from .muses_observation import MeasurementId
    from .retrieval_configuration import RetrievalConfiguration
    from .input_file_helper import InputFileHelper


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
    def use_for_master(self) -> list[bool]:
        """Indicate of QA flag is used for master quality."""
        raise NotImplementedError()


class QaFlagValueFile(QaFlagValue):
    """Implementation that uses a file to get the values."""

    # TODO Note that the data is a pandas table. We convert to lists, but
    # it might be easier just to leave this as a pandas table. Right now this
    # is just a placeholder, we are using the old muses-py to do the QA calculation
    # but we can revisit this if needed, and perhaps change this interface.
    def __init__(self, fname: str | os.PathLike[str], ifile_hlp: InputFileHelper):
        self.d = ifile_hlp.open_tes(fname)
        self.tbl: pd.DataFrame = self.d.checked_table

    @property
    def qa_flag_name(self) -> list[str]:
        """Return list of QA flags the other values apply to."""
        return self.tbl["Flag"].tolist()

    @property
    def cutoff_min(self) -> list[float]:
        """Minimum cutoff value for flag."""
        return self.tbl["CutoffMin"].tolist()

    @property
    def cutoff_max(self) -> list[float]:
        """Maximum cutoff value for flag."""
        return self.tbl["CutoffMax"].tolist()

    @property
    def use_for_master(self) -> list[bool]:
        """Indicate of QA flag is used for master quality."""
        return self.tbl["Use_For_Master"].astype(bool).tolist()


class QaDataHandle(CreatorHandle, metaclass=abc.ABCMeta):
    """Base class for QaDatawHandle. Note we use duck typing,
    don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents
    that a class is intended for this.
    """

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def qa_flag(
        self,
        retrieval_result: RetrievalResult,
        current_strategy_step: CurrentStrategyStep,
    ) -> str | None:
        """This does the QA calculation, and returns the master quality flag
        results. A good result returns "GOOD". None if this handle can't process the
        flag.
        """
        raise NotImplementedError()


class QaDataHandleSet(CreatorHandleSet):
    """This takes a RetrievalResult and updates it with QA data."""

    def __init__(self) -> None:
        super().__init__("qa_flag")

    def qa_flag(
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

    def __init__(self) -> None:
        self.viewing_mode = None
        self.qa_flag_directory = None
        self.ifile_hlp: InputFileHelper | None = None

    def notify_update_target(
        self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration
    ) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # We'll add grabbing the stuff out of RetrievalConfiguration
        # in a bit
        logger.debug(f"Call to {self.__class__.__name__}::notify_update_target")
        self.run_dir = (
            retrieval_config["outputDirectory"] / retrieval_config["sessionID"]
        )
        self.viewing_mode = retrieval_config["viewingMode"]
        self.qa_flag_directory = measurement_id["QualityFlagDirectory"]
        self.ifile_hlp = retrieval_config.input_file_helper

    def quality_flag_file_name(
        self, current_strategy_step: CurrentStrategyStep
    ) -> Path:
        """Return the quality file name."""
        if self.viewing_mode is None or self.ifile_hlp is None:
            raise RuntimeError("Need to call notify_update_target first")

        # Name is derived from the microwindows file name
        mwfname = current_strategy_step.muses_microwindows_fname()
        quality_fname = mwfname.name
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
        return Path(quality_fname)

    def qa_flag(
        self,
        retrieval_result: RetrievalResult,
        current_strategy_step: CurrentStrategyStep,
    ) -> str:
        """This does the QA calculation, and returns the master quality flag
        results. A good result returns "GOOD".
        """
        logger.debug(f"Doing QA calculation using {self.__class__.__name__}")
        if self.ifile_hlp is None:
            raise RuntimeError("Need to call notify_update_target first")
        fstate_info = FakeStateInfo(retrieval_result.current_state)
        master = self.write_quality_flags(
            QaFlagValueFile(
                self.quality_flag_file_name(current_strategy_step), self.ifile_hlp
            ),
            retrieval_result,
            fstate_info,
        )
        logger.info(f"Master Quality: {master}")
        return master

    def write_quality_flags(
        self,
        qa_flag_value: QaFlagValue,
        results: RetrievalResult,
        stateInfo: FakeStateInfo,
    ) -> str:
        strs = [
            "radianceResidualRMS",
            "radianceResidualMean",
            "TSUR-Tatm[0]",
            "TSUR_vs_Apriori",
            "CLOUD_MEAN",
            "PCLOUD",
            "Desert_Emiss_QA",
            "EMIS_MEAN",
            "CLOUD_VAR",
            "STOPCODE",
            "KdotDL",
            "LdotDL",
            "Calscale_mean",
            "H2O_H2O_Quality",
            "Emission_Layer_Flag",
            "Ozone_Ccurve_Flag",
            "ResidualNormFinal",
            "ResidualNormInitial",
            "radianceMaximumSNR",
            "TATM_Propagated",
            "O3_Propagated",
            "H2O_Propagated",
            "Deviation_QA",
            "Ozone_Slope_QA",
            "OMI_cloudFraction",
            "TROPOMI_cloudFraction",
            "O3_columnErrorDU",
            "O3_tropo_consistency",
        ]

        ind = np.where(results.deviation_QA > -990)[0]
        if len(ind) == 0:
            deviation_QA = 1.0
        else:
            deviation_QA = int(np.sum(results.deviation_QA[ind])) / len(ind)

        values_list = [
            results.radianceResidualRMS[0],
            results.radianceResidualMean[0],
            0,
            0,
            results.cloudODAve,
            stateInfo.current["PCLOUD"][0],
            results.Desert_Emiss_QA,
            results.emisDev,
            results.cloudODVar,
            results.stopCode,
            results.KDotDL,
            results.LDotDL,
            results.calscaleMean,
            results.H2O_H2OQuality,
            results.emissionLayer,
            results.ozoneCcurve,
            results.residualNormFinal,
            results.residualNormInitial,
            results.radianceMaximumSNR,
            results.propagatedTATMQA,
            results.propagatedO3QA,
            results.propagatedH2OQA,
            deviation_QA,
            results.ozone_slope_QA,
            results.omi_cloudfraction,
            results.tropomi_cloudfraction,
            results.O3_columnErrorDU,
            results.O3_tropo_consistency,
        ]

        if stateInfo.current["TSUR"] >= 1:
            values_list[2] = (
                stateInfo.current["TSUR"]
                - stateInfo.current["values"][stateInfo.species.index("TATM"), 0]
            )
            values_list[3] = stateInfo.current["TSUR"] - stateInfo.constraint["TSUR"]

        ind_radiance = np.where(results.radianceResidualRMS > -990)[0]
        ind_both = []
        for ii in range(0, len(ind_radiance)):
            if results.filter_list[ii] != "ALL":
                # Only keep the index if it is not 'ALL'
                ind_both.append(ind_radiance[ii])

        # The values in ind_both are the indices we want.
        ind = np.asarray(ind_both)
        if len(ind) > 1:
            for jj in range(0, len(ind)):
                my_filter = results.filter_list[ind[jj]]
                strs.append(f"radianceResidualRMS_{my_filter}")
                strs.append(f"radianceResidualMean_{my_filter}")
                values_list.append(results.radianceResidualRMS[ind[jj]])
                values_list.append(results.radianceResidualMean[ind[jj]])

        # read in quality flags from file
        cutoffMin = []
        cutoffMax = []
        useForMaster = []

        col = qa_flag_value.qa_flag_name
        minn = qa_flag_value.cutoff_min
        maxx = qa_flag_value.cutoff_max
        use_v = qa_flag_value.use_for_master
        col = [s.lower() for s in col]

        # Convert col to lowercase, so we can do a case insensitive search for our str
        for s in strs:
            if s == "STOPCODE":
                cutoffMin.append(0.0)
                cutoffMax.append(0.0)
                useForMaster.append(False)
            else:
                i = col.index(s.lower())
                cutoffMin.append(minn[i])
                cutoffMax.append(maxx[i])
                useForMaster.append(use_v[i])

        resultsQuality = self.calculate_quality_flags(
            values_list, cutoffMin, cutoffMax, useForMaster, strs
        )
        return resultsQuality["master"]

    def calculate_quality_flags(
        self,
        values: list[float],
        cutoffMin: list[float],
        cutoffMax: list[float],
        useForMaster: list[bool],
        strs_data: list[str],
    ) -> dict[str, Any]:
        # pass in values, cutoffs, and whether used for master flag
        # returns flags (GOOD/BAD), and master flag (GOOD/BAD)
        # need propagated flag

        n = len(values)
        flag_list = ["    "] * n
        master = 0

        for ii in range(0, n):
            if (values[ii] < cutoffMin[ii] or values[ii] > cutoffMax[ii]) and cutoffMax[
                ii
            ] >= -998:
                if useForMaster[ii]:
                    master = master + 1

                flag_list[ii] = "BAD"
            else:
                flag_list[ii] = "GOOD"

        print_table = texttable.Texttable()
        print_table.set_deco(texttable.Texttable.HEADER)
        print_table.set_cols_dtype(["t", "f", "f", "f", "t", "i"])
        print_table.set_cols_align(["l", "r", "r", "r", "l", "r"])
        print_table.set_cols_valign(["m", "m", "m", "m", "m", "m"])

        rows: list[Any] = [
            ["name", "value", "cutoffMin", "cutoffMax", "flag", "use for master"],
        ]

        for ii in range(0, n):
            if useForMaster[ii]:
                rows.append(
                    [
                        strs_data[ii],
                        values[ii],
                        cutoffMin[ii],
                        cutoffMax[ii],
                        flag_list[ii],
                        useForMaster[ii],
                    ]
                )

        print_table.add_rows(rows)

        logger.info(f"Quality flag table\n{print_table.draw()}")
        # end if strs_data is not None:

        masterFlag = "GOOD"
        if master >= 1:
            masterFlag = "BAD"

        result = {"flag": flag_list, "master": masterFlag}

        return result


# For now, just fall back to the old muses-py code.
QaDataHandleSet.add_default_handle(MusesPyQaDataHandle())

__all__ = [
    "QaDataHandle",
    "QaDataHandleSet",
    "MusesPyQaDataHandle",
    "QaFlagValue",
    "QaFlagValueFile",
]
