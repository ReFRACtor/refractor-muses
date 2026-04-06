from __future__ import annotations
from refractor.muses import (
    RetrievalStrategy,
    MusesSpectralWindowDict,
    RetrievalStrategyStep,
    RetrievalStrategyStepHandle,
    RetrievalType,
)
from refractor.osr_ml import (
    DummySpectralWindowHandle,
    RetrievalStrategyStepMl,
    RetrievalMlOutput,
)

rs = RetrievalStrategy(None)

rs.creator_dict[MusesSpectralWindowDict].add_handle(
    DummySpectralWindowHandle(), priority_order=1
)
rs.creator_dict[RetrievalStrategyStep].add_handle(
    RetrievalStrategyStepHandle(RetrievalStrategyStepMl, {RetrievalType("ml")}),
    priority_order=1,
)

# Not sure if there is a cleaner way to handle this. But delete all the OE
# output observers, and add the ML observer for output
rs.clear_observers()
rs.add_observer(RetrievalMlOutput())
