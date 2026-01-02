from __future__ import annotations
from refractor.muses import (
    CurrentState,
    CurrentStrategyStep,
    CurrentStrategyStepDict,
    FilterIdentifier,
    InstrumentIdentifier,
    InputFileHelper,
    MeasurementId,
    MusesStrategy,
    MusesStrategyHandle,
    MusesStrategyImp,
    RetrievalType,
    SpectralWindowHandleSet,
    StateElementIdentifier,
    StrategyStepIdentifier,
)
from .strategy_table import StrategyTable  # type: ignore
from typing import Any
import os


class MusesStrategyOldStrategyTable(MusesStrategyImp):
    """This wraps the old py-retrieve StrategyTable code as a
    MusesStrategy.  Note that this class has largely been replaced
    with MusesStrategyTable, but we leave this in place for backwards
    testings.

    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        ifile_hlp: InputFileHelper | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
    ):
        super().__init__(spectral_window_handle_set)
        self._stable = StrategyTable(filename, ifile_hlp=ifile_hlp)

    def is_next_bt(self) -> bool:
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        return self._stable.is_next_bt()

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        Note there is a assumption in muses-py that the
        list[FilterIdentifier] is sorted by the starting wavelength
        for each of filters, so we do that here.
        """
        return self._stable.filter_list_all()

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all retrieval steps)"""
        return self._stable.retrieval_elements_all_step

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all retrieval steps)"""
        return self._stable.error_analysis_interferents_all_step

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval steps)"""
        return InstrumentIdentifier.sort_identifier(
            self._stable.instrument_name(all_step=True)
        )

    def restart(self) -> None:
        """Set step to the first one."""
        self._stable.table_step = 0

    def next_step(self, current_state: CurrentState | None) -> None:
        """Advance to the next step."""
        self._stable.next_step(current_state)

    def is_done(self) -> bool:
        """True if we have reached the last step"""
        return self._stable.is_done()

    def current_strategy_step(self) -> CurrentStrategyStep:
        if self.is_done():
            raise RuntimeError("Past end of strategy")
        if self.measurement_id is None:
            raise RuntimeError(
                "Need to call notify_update_target before calling this function."
            )

        # Various convergence criteria for solver. This is the MusesLevmarSolver. Note the
        # different convergence depending on the step type. The chi2_tolerance is calculated
        # in RetrievalStrategyStepRetrieve if we don't fill it in - this depends on the
        # size of the radiance data
        cost_function_params = {
            "max_iter": int(self._stable.max_num_iterations),
            "delta_value": int(self.measurement_id["LMDelta"].split()[0]),
            "conv_tolerance": [
                float(self.measurement_id["ConvTolerance_CostThresh"]),
                float(self.measurement_id["ConvTolerance_pThresh"]),
                float(self.measurement_id["ConvTolerance_JacThresh"]),
            ],
            "chi2_tolerance": None,  # Filled in by RetrievalStrategyStepRetrieve
        }
        if self._stable.retrieval_type == RetrievalType("bt_ig_refine"):
            cost_function_params["conv_tolerance"] = [0.00001, 0.00001, 0.00001]
            cost_function_params["chi2_tolerance"] = 0.00001
        cstepdict = {
            "retrieval_elements": [
                StateElementIdentifier(s) for s in self._stable.retrieval_elements()
            ],
            "instrument_name": self._stable.instrument_name(),
            "strategy_step": StrategyStepIdentifier(
                self._stable.table_step, self._stable.step_name
            ),
            "retrieval_step_parameters": {
                "cost_function_params": cost_function_params,
            },
            "retrieval_type": self._stable.retrieval_type,
            "error_analysis_interferents": [
                StateElementIdentifier(s)
                for s in self._stable.error_analysis_interferents()
            ],
            "spectral_window_dict": None,
            "do_not_update_list": [
                StateElementIdentifier(s) for s in self._stable.do_not_update_list
            ],
        }
        cstep = CurrentStrategyStepDict(cstepdict, self.measurement_id)
        cstep.current_strategy_step_dict["spectral_window_dict"] = (
            self.spectral_window_handle_set.spectral_window_dict(
                cstep, self.filter_list_dict
            )
        )
        return cstep


class MusesStrategyOldStrategyTableHandle(MusesStrategyHandle):
    def muses_strategy(
        self,
        measurement_id: MeasurementId,
        ifile_hlp: InputFileHelper,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        """Return MusesStrategy if we can process the given
        measurement_id, or None if we can't.
        """
        return MusesStrategyOldStrategyTable(
            measurement_id["run_dir"] / "Table.asc",
            ifile_hlp,
            spectral_window_handle_set,
        )


# Can turn on if needed for doing a test, but normally don't use this.
# MusesStrategyHandleSet.add_default_handle(MusesStrategyOldStrategyTableHandle(), -100)

__all__ = ["MusesStrategyOldStrategyTable", "MusesStrategyOldStrategyTableHandle"]
