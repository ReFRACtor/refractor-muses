from __future__ import annotations
from refractor.muses import (
    ProcessLocation,
    RetrievalStrategyStep,
    CurrentState,
    ProcessLocationObservable,
    MusesStrategyContextMixin,
)
from loguru import logger
from typing import Any
import shutil
import pystac
from datetime import datetime

class RetrievalMlOutput(MusesStrategyContextMixin):
    def __init__(self, creator_dict: CreatorDict, **kwargs: Any):
        super().__init__(creator_dict.strategy_context)

    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [
            ProcessLocation("ML step"),
        ]

    def notify_process_location(
        self,
        location: ProcessLocation,
        current_state: CurrentState | None = None,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        self.retrieval_strategy_step = retrieval_strategy_step
        inst = self.current_strategy_step.instrument_name.split('-')[0]
        spec = self.current_strategy_step.species_name
        pspect = self.retrieval_config["product_spec_path"] / f"columns_{inst}_{spec}.nc"
        props = self.stac_catalog.get_item_links()[0].resolve_stac_object().target.properties
        sensor_set = props['sensor_set']
        dstring  = props['datetime'].split('T')[0].replace('-','')
        foutname = self.retrieval_config["output_directory"] / f"{sensor_set}_{spec}_{dstring}.nc"
        shutil.copy(pspect, foutname)
        outcat = pystac.Catalog(id="myid", description="TROPESS Machine Learning product")
        item = pystac.Item(id=f"{sensor_set}_{spec}_{dstring}",
                           geometry={"type": "MultiPoint", "coordinates" : [[-180,90],[-180,-90],[180,-90],[180,90],[-180,90]]}, bbox=[-180,90,180,-90], properties={}, datetime=datetime.utcnow())
        outcat.add_item(item)
        outcat.normalize_hrefs(str(self.retrieval_config["output_directory"]))
        outcat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        logger.info("Fake output")


ProcessLocationObservable.register_default_observer(RetrievalMlOutput)

__all__ = [
    "RetrievalMlOutput",
]
