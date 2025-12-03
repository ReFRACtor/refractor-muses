from __future__ import annotations
from .retrieval_strategy_step import RetrievalStrategyStep, RetrievalStrategyStepSet
from .tes_file import TesFile
from .identifier import RetrievalType, StateElementIdentifier
import numpy as np
from loguru import logger
import math
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .current_state import CurrentState
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import MusesStrategy


class RetrievalStrategyStepBT(RetrievalStrategyStep):
    """Brightness Temperature strategy step. This handles steps with
    the retrieval type "BT". This then selects one of the following
    BT_IG_Refine steps to execute.

    If the table indicates, we update the cloud effective extinction
    (CLOUDEXT) and/or tsurface (TSUR).

    """

    def __init__(self) -> None:
        super().__init__()
        self.notify_update_target(None)

    def notify_update_target(self, rs: RetrievalStrategy | None) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")

    def retrieval_step_body(
        self, retrieval_type: RetrievalType, rs: RetrievalStrategy, **kwargs: dict
    ) -> bool:
        if retrieval_type != RetrievalType("bt"):
            return False
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        logger.info("Running run_forward_model ...")
        self.fm = rs.create_forward_model_combine()
        self.calculate_bt(
            rs.retrieval_config,
            rs.strategy,
            rs.strategy_step.step_number,
            rs.current_state,
        )
        logger.info(f"Step: {rs.strategy_step}")
        return True

    def calculate_bt(
        self,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        step: int,
        cstate: CurrentState,
    ) -> None:
        """Calculate brightness temperature, and use to update
        cstate.brightness_temperature_data. We also update TSUR and
        CLOUDEXT in current state."""
        # Note from py-retrieve: issue with negative radiances, so take mean
        #
        # I'm not actually sure that is true, we filter out bad samples. But
        # regardless, use the mean like py-retrieve does.
        frequency = self.fm.spectral_domain_all().data
        radiance_bt_obs = self.bt(
            np.mean(frequency), np.mean(self.fm.obs_radiance_all())
        )
        radiance_bt_fit = self.bt(
            np.mean(frequency), np.mean(self.fm.radiance_all().spectral_range.data)
        )
        btdata: dict[str, Any] = {}
        btdata["diff"] = radiance_bt_fit - radiance_bt_obs
        btdata["obs"] = radiance_bt_obs
        btdata["fit"] = radiance_bt_fit
        btdata["species_igr"] = None

        bt_diff = btdata["diff"]

        # If next step is NOT BT, evaluate what to do with "cloud". Otherwise,
        # we are done.
        if strategy.is_next_bt():
            cstate.set_brightness_temperature_data(step, btdata)
            return

        cfile = TesFile(retrieval_config["CloudParameterFilename"])
        BTLow = np.array(cfile.checked_table["BT_low"])
        BTHigh = np.array(cfile.checked_table["BT_high"])
        # This is either 0 (for don't update) or 1 (for update)
        tsurIG = np.array(cfile.checked_table["TSUR_IG"])

        cloudIG = np.array(cfile.checked_table["CLOUDEXT_IG"])
        row = np.where((bt_diff >= BTLow) & (bt_diff <= BTHigh))[0]
        if row.size == 0:
            raise RuntimeError(
                f"No entry in file, {cfile.file_name} For BT difference of {bt_diff}"
            )

        btdata["species_igr"] = np.array(cfile.checked_table["SPECIES_IGR"])[row]

        # for IGR and TSUR modification for TSUR, must be daytime land
        if (
            not cstate.sounding_metadata.is_day
            or cstate.sounding_metadata.surface_type in ("OCEAN", "FRESH")
        ) and "TSUR" in btdata["species_igr"]:
            logger.info("Must be land, daytime for TSUR IGR")
            btdata["species_igr"] = None
            tsurIG[row] = 0

        if cloudIG[row] > 0:
            newv = cstate.state_value("CLOUDEXT").copy()
            newv[:] = cloudIG[row][0]
            cstate.update_full_state_element(
                StateElementIdentifier("CLOUDEXT"),
                step_initial_fm=newv.copy(),
                value_fm=newv.copy(),
                constraint_vector_fm=newv.copy(),
            )
        if tsurIG[row] != 0:
            # use difference in observed - fit to change TSUR.  Note, we
            # assume weak clouds.
            newv = cstate.state_value("TSUR")
            newv = newv + btdata["obs"] - btdata["fit"]
            cstate.update_full_state_element(
                StateElementIdentifier("TSUR"),
                step_initial_fm=newv.copy(),
                value_fm=newv.copy(),
                constraint_vector_fm=newv.copy(),
            )
        cstate.set_brightness_temperature_data(step, btdata)

    def bt(self, frequency: float, rad: float) -> float:
        """converts from radiance (W/cm2/cm-1/sr) to BT (erg/sec/cm2/cm-1/sr)"""
        planck = 6.626176e-27
        clight = 2.99792458e10
        boltz = 1.380662e-16
        radcn1 = 2.0 * planck * clight * clight * 1.0e-07
        radcn2 = planck * clight / boltz
        return radcn2 * frequency / math.log(1 + (radcn1 * frequency**3 / rad))


RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepBT())
__all__ = [
    "RetrievalStrategyStepBT",
]
