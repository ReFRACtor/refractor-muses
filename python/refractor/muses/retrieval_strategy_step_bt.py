from __future__ import annotations
from .retrieval_strategy_step import (
    RetrievalStrategyStepSet,
    RetrievalStrategyStepHandle,
)
from .retrieval_strategy_step_oe import RetrievalStrategyStepOEBase
from .identifier import RetrievalType, StateElementIdentifier
import numpy as np
from loguru import logger
import math
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .creator_dict import CreatorDict


class RetrievalStrategyStepBT(RetrievalStrategyStepOEBase):
    """Brightness Temperature strategy step. This handles steps with
    the retrieval type "BT". This then selects one of the following
    BT_IG_Refine steps to execute.

    If the table indicates, we update the cloud effective extinction
    (CLOUDEXT) and/or tsurface (TSUR).

    """

    def __init__(
        self,
        retrieval_type: RetrievalType,
        rs: RetrievalStrategy,
        creator_dict: CreatorDict,
        **kwargs: Any,
    ) -> None:
        super().__init__(retrieval_type, rs, creator_dict, **kwargs)
        self.frequency: None | np.ndarray = None
        self.obs_rad_all: None | np.ndarray = None
        self.rad_all: None | np.ndarray = None

    def get_state(self) -> dict[str, Any]:
        """Return a dictionary of values that can be used by
        set_state.  This allows us to skip pieces of the retrieval
        step. This is similar to a pickle serialization (which we also
        support), but only saves the things that change when we update
        the parameters.

        This is useful for unit tests of side effects of doing the
        retrieval step (e.g., generating output files) without needing
        to actually run the retrieval.

        """
        res: dict[str, Any] = {"frequency": None, "obs_rad_all": None, "rad_all": None}
        if self.frequency is not None:
            res["frequency"] = self.frequency.tolist()
        if self.obs_rad_all is not None:
            res["obs_rad_all"] = self.obs_rad_all.tolist()
        if self.rad_all is not None:
            res["rad_all"] = self.rad_all.tolist()
        return res

    def retrieval_step_body(self) -> None:
        """Calculate brightness temperature, and use to update
        cstate.brightness_temperature_data. We also update TSUR and
        CLOUDEXT in current state."""
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        logger.info("Running run_forward_model ...")
        # Note from py-retrieve: issue with negative radiances, so take mean
        #
        # I'm not actually sure that is true, we filter out bad samples. But
        # regardless, use the mean like py-retrieve does.
        if self._saved_state is not None:
            # Skip forward model if we have a saved state.
            self.frequency = np.array(self._saved_state["frequency"])
            self.obs_rad_all = np.array(self._saved_state["obs_rad_all"])
            self.rad_all = np.array(self._saved_state["rad_all"])
        else:
            fm = self.create_forward_model_combine()
            self.frequency = fm.spectral_domain_all().data
            self.obs_rad_all = fm.obs_radiance_all().spectral_range.data
            # True here means skip the jacobian calculation. This doesn't actually matter
            # for the OSS FM that always calculate the jacobian, but might matter in the future
            # with different forward models
            self.rad_all = fm.radiance_all(True).spectral_range.data

        radiance_bt_obs = self.bt(np.mean(self.frequency), np.mean(self.obs_rad_all))
        radiance_bt_fit = self.bt(np.mean(self.frequency), np.mean(self.rad_all))
        btdata: dict[str, Any] = {}
        btdata["diff"] = radiance_bt_fit - radiance_bt_obs
        btdata["obs"] = radiance_bt_obs
        btdata["fit"] = radiance_bt_fit
        btdata["species_igr"] = None

        bt_diff = btdata["diff"]

        # If next step is NOT BT, evaluate what to do with "cloud". Otherwise,
        # we are done.
        if self.strategy.is_next_bt():
            self.current_state.set_brightness_temperature_data(
                self.strategy_step.step_number, btdata
            )
            return

        cfile = self.retrieval_config.input_file_helper.open_tes(
            self.retrieval_config["CloudParameterFilename"],
        )
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
            not self.current_state.sounding_metadata.is_day
            or self.current_state.sounding_metadata.surface_type in ("OCEAN", "FRESH")
        ) and "TSUR" in btdata["species_igr"]:
            logger.info("Must be land, daytime for TSUR IGR")
            btdata["species_igr"] = None
            tsurIG[row] = 0

        if cloudIG[row] > 0:
            newv = self.current_state.state_value("CLOUDEXT").copy()
            newv[:] = cloudIG[row][0]
            self.current_state.update_full_state_element(
                StateElementIdentifier("CLOUDEXT"),
                step_initial_fm=newv.copy(),
                value_fm=newv.copy(),
                constraint_vector_fm=newv.copy(),
            )
        if tsurIG[row] != 0:
            # use difference in observed - fit to change TSUR.  Note, we
            # assume weak clouds.
            newv = self.current_state.state_value("TSUR")
            newv = newv + btdata["obs"] - btdata["fit"]
            self.current_state.update_full_state_element(
                StateElementIdentifier("TSUR"),
                step_initial_fm=newv.copy(),
                value_fm=newv.copy(),
                constraint_vector_fm=newv.copy(),
            )
        self.current_state.set_brightness_temperature_data(
            self.strategy_step.step_number, btdata
        )
        logger.info(f"Step: {self.strategy_step}")

    def bt(self, frequency: float, rad: float) -> float:
        """converts from radiance (W/cm2/cm-1/sr) to BT (erg/sec/cm2/cm-1/sr)"""
        planck = 6.626176e-27
        clight = 2.99792458e10
        boltz = 1.380662e-16
        radcn1 = 2.0 * planck * clight * clight * 1.0e-07
        radcn2 = planck * clight / boltz
        return radcn2 * frequency / math.log(1 + (radcn1 * frequency**3 / rad))


RetrievalStrategyStepSet.add_default_handle(
    RetrievalStrategyStepHandle(
        RetrievalStrategyStepBT,
        {
            RetrievalType("bt"),
        },
    )
)
__all__ = [
    "RetrievalStrategyStepBT",
]
