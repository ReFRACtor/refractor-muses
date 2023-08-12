import refractor.framework as rf
from .refractor_uip import RefractorUip

# The information in the ReFRACtor StateVector and the py-retrieve UIP are
# redundant - basically we have two copies of everything because of the
# differences in design between the two systems.
#
# When working with ReFRACtor and py-retrieve, we need to make sure that
# things are in sync. Depending on the processing we are doing, we can
# either have calls the update_uip (which is done in a few places such
# as mpy.residual_fm_jacobian) also update the StateVector, or we can
# go the other way around have have updates to the StateVector reflected
# in the UIP.
#
# The various classes in this file do the mapping from StateVector/ReFRACtor
# objects to the UIP.

class AbsorberVmrToUip(rf.CacheInvalidatedObserver):
    def __init__(self, rf_uip, pressure, absorber_vmr, species_name):
        super().__init__()
        self.rf_uip = rf_uip
        self.pressure = pressure
        self.absorber_vmr = absorber_vmr
        self.species_name = species_name
        self.absorber_vmr.add_cache_invalidated_observer(self)
        # Make sure data is synchronized initially
        self.invalidate_cache()

    def invalidate_cache(self):
        '''Called with self.absorber_vmr changes'''
        # Get the VMR, and put into the right place in the UIP.
        # Note that the UIP always has just the VMR, so even if absorber_vmr
        # uses a state vector of log(vmr) or something like that we want
        # the UIP to have the VMR.
        # Note the UIP goes from surface to TOA, so we want
        # the DECREASING_PRESSURE order here.
        vgrid = self.absorber_vmr.vmr_grid(self.pressure,
                                           rf.Pressure.DECREASING_PRESSURE)
        self.rf_uip.atmosphere_column(self.species_name)[:] = vgrid.value
        self.cache_valid_flag = True        

class StateVectorUpdateUip(rf.StateVectorObserver):
    def __init__(self, rf_uip : RefractorUip):
        super().__init__()
        self.rf_uip = rf_uip

    def notify_update(self, sv : rf.StateVector):
        self.rf_uip.update_uip(sv.state)
        
__all__ = ["AbsorberVmrToUip", "StateVectorUpdateUip"]
