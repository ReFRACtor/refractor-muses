from __future__ import annotations
from refractor.muses import (
    RetrievalStrategyStep,
    RetrievalStrategyStepHandle,
    ProcessLocation,
    ProcessLocationObservable,
    CurrentState,
    CreatorDict,
    RetrievalType,
)
from pathlib import Path
from .cris_io import read_l1b
from .ml import prediction, features_l1b # type: ignore
from loguru import logger
from typing import Any
import os


class RetrievalStrategyStepMl(RetrievalStrategyStep):
    def __init__(
        self,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        ml_model_path: Path,
        instrument: str,
        species: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(creator_dict, current_state, process_location_observable)
        self.ml_model_path = ml_model_path
        self.instrument = instrument
        self.species = species

    def retrieval_step_body(self) -> None:
        """Returns True if we handle the retrieval step, False otherwise"""
        # self.stac_catalog.describe()
        logger.info("Running ML retrieval step")
        # We may need to learn more about reading pystac, but this basic thing
        # just gets all the "item" links
        l1b_file = []
        for lnk in self.stac_catalog.get_item_links():
            l1b_file.extend([i.get_absolute_href() for i in lnk.resolve_stac_object().target.get_assets(role="data").values()]) # type:ignore[union-attr]
        self.l1b = read_l1b(
            l1b_file
        )
        self.features = features_l1b(
            l1b=self.l1b, prior=None, ml_model_path=self.ml_model_path
        )
        self.pred = prediction(
            mdl_api="sequential",
            path=self.ml_model_path,
            prefix=self.instrument + "_" + self.species + "_ret_col",
            # Until we get weights sorted out
            # suffix="keras-ANN",
            suffix="keras-ANN_new",
            features=self.features,
            batch_size_in=8192 * 2,
            evaluate=False,
            save_evaluate=False,
        )
        self.notify_process_location(ProcessLocation("ML step"))


class RetrievalStrategyStepMlHandle(RetrievalStrategyStepHandle):
    def __init__(
        self,
        cls: type[RetrievalStrategyStep],
        retrieval_type_set: set[RetrievalType],
        instrument: str,
        species: str,
    ) -> None:
        super().__init__(cls, retrieval_type_set)
        # These should perhaps come from current step
        self.instrument = instrument
        self.species = species
        # May want to get this from a different place, but for now use
        # environment variable
        self.ml_model_path = Path(os.environ["MUSES_ML_PATH"])

    def retrieval_step(
        self,
        retrieval_type: RetrievalType,
        creator_dict: CreatorDict,
        current_state: CurrentState,
        process_location_observable: ProcessLocationObservable,
        **kwargs: Any,
    ) -> RetrievalStrategyStep | None:
        if (
            self._retrieval_type_set is None
            or retrieval_type in self._retrieval_type_set
        ):
            return self._create_cls(
                creator_dict,
                current_state,
                process_location_observable,
                ml_model_path=self.ml_model_path,
                instrument=self.instrument,
                species=self.species,
                **kwargs,
            )
        return None


__all__ = [
    "RetrievalStrategyStepMl",
    "RetrievalStrategyStepMlHandle",
]
