from __future__ import annotations
from refractor.muses import (
    CurrentStrategyStepImp,
    CreatorHandleWithContext,
    CurrentStrategyStepHandleSet,
    SpectralWindowHandleSet,
    StrategyStepIdentifier,
    RetrievalType,
    CurrentStrategyStep,
    MusesStrategyContext,
)


class CurrentStrategyStepMl(CurrentStrategyStepImp):
    def __init__(
        self,
        strategy_context: MusesStrategyContext,
        retrieval_type: RetrievalType,
        strategy_step: StrategyStepIdentifier,
        instrument_name: str,
        species_name: str,
    ) -> None:
        super().__init__(strategy_context, retrieval_type, strategy_step)
        self._instrument_name = instrument_name
        self._species_name = species_name

    @property
    def has_spectral_window(self) -> bool:
        return False

    @property
    def instrument_name(self) -> str:
        """Note this maps to the weights files. This isn't a InstrumentIdentifier.
        We should probably align these, so that the same thing is used in both
        places. But for now, match what Frank is using in the weights file names.
        """
        return self._instrument_name

    @property
    def species_name(self) -> str:
        """Note this maps to the weights files. This isn't a StateElementIdentifier.
        We should probably align these, so that the same thing is used in both
        places. But for now, match what Frank is using in the weights file names.
        """
        return self._species_name


class CurrentStrategyStepHandleMl(CreatorHandleWithContext):
    def create_current_strategy_step(
        self,
        index: int,
        table_row: dict,
        spectral_window_handle_set: SpectralWindowHandleSet,
    ) -> CurrentStrategyStep | None:
        if table_row["retrievalType"] != "ML":
            return None
        strategy_step = StrategyStepIdentifier(index, table_row["stepName"])
        retrieval_type = RetrievalType(table_row["retrievalType"])
        return CurrentStrategyStepMl(
            self.strategy_context,
            retrieval_type,
            strategy_step,
            table_row["instrument"],
            table_row["species"],
        )


CurrentStrategyStepHandleSet.add_default_handle(CurrentStrategyStepHandleMl())

__all__ = ["CurrentStrategyStepHandleMl", "CurrentStrategyStepMl"]
