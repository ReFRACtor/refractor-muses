from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .refractor_uip import RefractorUip
from loguru import logger

# The information in the ReFRACtor StateVector and the py-retrieve UIP
# are redundant - basically we have two copies of everything because
# of the differences in design between the two systems.
#
# When working with ReFRACtor and py-retrieve, we need to make sure
# that things are in sync. Depending on the processing we are doing,
# we can either have calls the update_uip (which is done in a few
# places such as mpy.residual_fm_jacobian) also update the
# StateVector, or we can go the other way around have have updates to
# the StateVector reflected in the UIP.


class MaxAPosterioriSqrtConstraintUpdateUip(rf.ObserverMaxAPosterioriSqrtConstraint):
    def __init__(self, rf_uip: RefractorUip):
        super().__init__()
        self.rf_uip = rf_uip

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint):
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.rf_uip.update_uip(mstand.parameters)


__all__ = [
    "MaxAPosterioriSqrtConstraintUpdateUip",
]
