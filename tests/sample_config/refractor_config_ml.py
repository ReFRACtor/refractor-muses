from __future__ import annotations
from refractor.muses import RetrievalStrategy
from refractor.osr_ml import (
    DummySpectralWindowHandle,
    RetrievalStrategyStepMl,
    RetrievalMlOutput,
)

rs = RetrievalStrategy(None)

rs.spectral_window_handle_set.add_handle(DummySpectralWindowHandle(), priority_order=1)
rs.retrieval_strategy_step_set.add_handle(RetrievalStrategyStepMl(), priority_order=1)

# Not sure if there is a cleaner way to handle this. But delete all the OE
# output observers, and add the ML observer for output
rs.clear_observers()
rs.add_observer(RetrievalMlOutput())
