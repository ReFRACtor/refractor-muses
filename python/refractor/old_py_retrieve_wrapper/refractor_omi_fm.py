# Don't both typechecking the file. This is old code, only used for backwards testing.
# Silence mypy, just so we don't get a lot of noise in the output
# type: ignore

from __future__ import annotations
from .muses_py_forward_model import RefractorTropOrOmiFmMusesPy, RefractorTropOrOmiFm
from refractor.muses_py_fm import CurrentStateUip
from refractor.omi import OmiFmObjectCreator
import typing

if typing.TYPE_CHECKING:
    from refractor.muses import MusesObservation, MeasurementId

# ============================================================================
# This set of classes replace the lower level call to omi_fm in
# muses-py. This was used when initially comparing ReFRACtor and muses-py.
# This has been replaced with RefractorResidualFmJacobian which is higher
# in the call chain and has a cleaner interface.
# We'll leave these classes here for now, since it can be useful to do
# lower level comparisons. But these should largely be considered deprecated
# ============================================================================


class RefractorOmiFmMusesPy(RefractorTropOrOmiFmMusesPy):
    def __init__(self, **kwargs: dict) -> None:
        super().__init__(func_name="omi_fm", **kwargs)


class RefractorOmiFm(RefractorTropOrOmiFm):
    """
    NOTE - this is deprecated

    Use a ReFRACtor ForwardModel as a replacement for omi_fm."""

    def __init__(
        self, obs: MusesObservation, measurement_id: MeasurementId, **kwargs: dict
    ) -> None:
        super().__init__(func_name="omi_fm", **kwargs)
        self._obs = obs
        self.measurement_id = measurement_id

    @property
    def observation(self) -> MusesObservation:
        return self._obs

    @property
    def have_obj_creator(self) -> bool:
        return "omi_fm_object_creator" in self.rf_uip.refractor_cache

    @property
    def obj_creator(self) -> OmiFmObjectCreator:
        """Object creator using to generate forward model. You can use
        this to get various pieces we use to create the forward model."""
        if "omi_fm_object_creator" not in self.rf_uip.refractor_cache:
            self.rf_uip.refractor_cache["omi_fm_object_creator"] = OmiFmObjectCreator(
                CurrentStateUip(self.rf_uip),
                self.measurement_id,
                self._obs,
                rf_uip_func=lambda instrument_name: self.rf_uip,
                match_py_retrieve=True,
                **self.obj_creator_args,
            )
        return self.rf_uip.refractor_cache["omi_fm_object_creator"]


__all__ = ["RefractorOmiFmMusesPy", "RefractorOmiFm"]
