from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .retrieval_strategy_step import RetrievalStrategyStep, RetrievalStrategyStepSet
from .tes_file import TesFile
from .identifier import RetrievalType
import numpy as np
import copy
from loguru import logger
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .state_info import StateInfo
    from .current_state import CurrentState
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import MusesStrategy


class RetrievalStrategyStepBT(RetrievalStrategyStep):
    """Brightness Temperature strategy step. This handles steps with
    the retrieval type "BT". This then selects one of the following
    BT_IG_Refine steps to execute.

    If the table indicates, we update the cloud effective extinction
    (cloudEffExt) and/or tsurface (TSUR).

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
        jacobian_speciesNames = ["H2O"]
        logger.info("Running run_forward_model ...")
        self.cfunc = rs.create_cost_function(
            fix_apriori_size=True, jacobian_speciesIn=jacobian_speciesNames
        )
        self.calculate_bt(
            rs.retrieval_config,
            rs.strategy,
            rs.strategy_step.step_number,
            rs.state_info,
            rs.current_state(),
        )
        logger.info(f"Step: {rs.strategy_step}")
        rs.state_info.next_state_dict = copy.deepcopy(
            rs.state_info.state_info_dict["current"]
        )
        return True

    def calculate_bt(
        self,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        step: int,
        state_info: StateInfo,
        cstate: CurrentState,
    ) -> None:
        """Calculate brightness temperature, and use to update
        cstate.brightness_temperature_data. We also TSUR and
        cloudEffExt in state_info."""
        # Note from py-retrieve: issue with negative radiances, so take mean
        #
        # I'm not actually sure that is true, we filter out bad samples. But
        # regardless, use the mean like py-retrieve does.
        frequency = np.concatenate(
            [
                fm.spectral_domain_all().data
                for fm in self.cfunc.max_a_posteriori.forward_model
            ]
        )
        radiance_bt_obs = mpy.bt(
            np.mean(frequency), np.mean(self.cfunc.max_a_posteriori.measurement)
        )
        radiance_bt_fit = mpy.bt(
            np.mean(frequency), np.mean(self.cfunc.max_a_posteriori.model)
        )

        btdata = cstate.brightness_temperature_data
        btdata[step] = {}
        btdata[step]["diff"] = radiance_bt_fit[0] - radiance_bt_obs[0]
        btdata[step]["obs"] = radiance_bt_obs[0]
        btdata[step]["fit"] = radiance_bt_fit[0]
        btdata[step]["species_igr"] = None

        bt_diff = btdata[step]["diff"]

        # If next step is NOT BT, evaluate what to do with "cloud". Otherwise,
        # we are done.
        if strategy.is_next_bt():
            return

        cfile = TesFile(retrieval_config["CloudParameterFilename"])
        if cfile.table is None:
            raise RuntimeError(
                f"File {retrieval_config['CloudParameterFilename']} has no data table"
            )
        BTLow = np.array(cfile.table["BT_low"])
        BTHigh = np.array(cfile.table["BT_high"])
        # This is either 0 (for don't update) or 1 (for update)
        tsurIG = np.array(cfile.table["TSUR_IG"])

        cloudIG = np.array(cfile.table["CLOUDEXT_IG"])
        row = np.where((bt_diff >= BTLow) & (bt_diff <= BTHigh))[0]
        if row.size == 0:
            raise RuntimeError(
                f"No entry in file, {cfile.file_name} For BT difference of {bt_diff}"
            )

        btdata[step]["species_igr"] = np.array(cfile.table["SPECIES_IGR"])[row]

        # for IGR and TSUR modification for TSUR, must be daytime land
        stateInfo = state_info.state_info_dict
        if (
            stateInfo["current"]["tsa"]["dayFlag"] == 0
            or stateInfo["surfaceType"].upper() in ("OCEAN", "FRESH")
        ) and "TSUR" in btdata[step]["species_igr"]:
            logger.info("Must be land, daytime for TSUR IGR")
            btdata[step]["species_igr"] = None
            tsurIG[row] = 0

        if cloudIG[row] > 0:
            stateInfo["initial"]["cloudEffExt"][:] = cloudIG[row]
            stateInfo["current"]["cloudEffExt"][:] = cloudIG[row]
            stateInfo["constraint"]["cloudEffExt"][:] = cloudIG[row]

        if tsurIG[row] != 0:
            # use difference in observed - fit to change TSUR.  Note, we
            # assume weak clouds.
            stateInfo["initial"]["TSUR"] = (
                stateInfo["initial"]["TSUR"] + btdata[step]["obs"] - btdata[step]["fit"]
            )
            stateInfo["current"]["TSUR"] = stateInfo["initial"]["TSUR"]
            stateInfo["constraint"]["TSUR"] = stateInfo["initial"]["TSUR"]


RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepBT())

__all__ = [
    "RetrievalStrategyStepBT",
]
