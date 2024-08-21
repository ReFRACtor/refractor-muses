import numpy as np
import refractor.framework as rf
import logging
import abc

logger = logging.getLogger("py-retrieve")

class SurfaceAlbedo(object, metaclass=abc.ABCMeta):
    '''MusesRaman needs a surface albedo. This class supplies that, however
    is appropriate for a particular forward model set up. This should handle
    cloud vs clear forward models, similar to rf.GroundWithCloudHandling and
    other classes.
    '''
    # TODO Note in generate surface albedo has a jacobian wrt the state vector.
    # We aren't currently handling propagating that in MusesRaman.
    @abc.abstractmethod
    def surface_albedo(self) -> float:
        '''Return the surface albedo'''
        raise NotImplementedError
    
class MusesRaman(rf.RamanSiorisEffect):

    def __init__(self, surface_albedo : SurfaceAlbedo,
                 ray_info : 'MusesRayInfo',
                 Solar_and_odepth_spec_domain : rf.SpectralDomain,
                 scale_factor : float,
                 sensor_index : int,
                 solar_zenith : rf.DoubleWithUnit,
                 observation_zenith : rf.DoubleWithUnit,
                 relative_azimuth : rf.DoubleWithUnit,
                 atmosphere : rf.AtmosphereStandard,
                 solar_model : rf.SolarModel,
                 mapping : rf.StateMapping):

        # Note sensor_index is only used here for naming the state vector element
        super().__init__(Solar_and_odepth_spec_domain, scale_factor, sensor_index,
                         solar_zenith, observation_zenith, relative_azimuth, 
                         atmosphere, solar_model, mapping)
        self.surface_albedo = surface_albedo
        self.ray_info = ray_info
        # Save range of Solar_and_odepth_spec_domain, to give clearer error
        # message
        self.solar_model_sd_min = Solar_and_odepth_spec_domain.data.min()
        self.solar_model_sd_max = Solar_and_odepth_spec_domain.data.max()

    def apply_effect(self, spec: rf.Spectrum, fm_grid: rf.ForwardModelSpectralGrid):
        temp_layers = self.ray_info.tbar()
        surf_alb = self.surface_albedo.surface_albedo()
        # Sanity check - the error messages at the C++ level are pretty obscure
        sd_min = spec.spectral_domain.data.min()
        sd_max = spec.spectral_domain.data.max()
        if(sd_min < self.solar_model_sd_min or sd_min > self.solar_model_sd_max):
            raise RuntimeError(f"Spectral domain is out of range sd_min = {sd_min} sd_max = {sd_max} raman_sd_min = {self.solar_model_sd_min} raman_sd_max = {self.solar_model_sd_max}")
        self.apply_raman_effect(spec, temp_layers, surf_alb)

    def desc(self):
        return "MusesRaman"
        
__all__ = ["MusesRaman", "SurfaceAlbedo"]
