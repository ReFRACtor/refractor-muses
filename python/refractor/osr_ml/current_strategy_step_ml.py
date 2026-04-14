from __future__ import annotations
from refractor.muses import (
    CurrentStrategyStepImp,
    CreatorHandleWithContext,
    CurrentStrategyStepHandleSet,
    SpectralWindowHandleSet,
    StrategyStepIdentifier,
    RetrievalType,
    CurrentStrategyStep,
)


class CurrentStrategyStepMl(CurrentStrategyStepImp):
    @property
    def has_spectral_window(self) -> bool:
        return False


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
        )


CurrentStrategyStepHandleSet.add_default_handle(CurrentStrategyStepHandleMl())

__all__ = ["CurrentStrategyStepHandleMl", "CurrentStrategyStepMl"]
