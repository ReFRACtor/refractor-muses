import numpy as np
import refractor.framework as rf

class MusesRaman(rf.RamanSiorisEffect):

    def __init__(self, rf_uip, instrument_name,
                 Solar_and_odepth_spec_domain : rf.SpectralDomain,
                 scale_factor : float,
                 fm_idx : int,
                 ii_mw: int,
                 solar_zenith : rf.DoubleWithUnit,
                 observation_zenith : rf.DoubleWithUnit,
                 relative_azimuth : rf.DoubleWithUnit,
                 atmosphere : rf.AtmosphereStandard,
                 solar_model : rf.SolarModel,
                 mapping : rf.StateMapping):

        super().__init__(Solar_and_odepth_spec_domain, scale_factor, fm_idx,
                         solar_zenith, observation_zenith, relative_azimuth, 
                         atmosphere, solar_model, mapping)

        self.rf_uip = rf_uip
        self._pressure = atmosphere.pressure
        self._ii_mw = ii_mw
        self.instrument_name = instrument_name

    def apply_effect(self, spec: rf.Spectrum, fm_grid: rf.ForwardModelSpectralGrid):

        nlay = self._pressure.number_layer

        temp_layers = self.rf_uip.ray_info(self.instrument_name)['tbar'][::-1][:nlay]

        filter_name = self.rf_uip.filter_name(self._ii_mw)

        if filter_name in ("UV1", "UV2"):
            if self._pressure.do_cloud:
                # Replicate py-retrieve behavior in omi/print_ring_input.py
                surf_alb = 0.80
            else:
                surf_alb = self.rf_uip.omi_params[f'surface_albedo_{str.lower(filter_name)}']
        elif filter_name in ("BAND1", "BAND2", "BAND3", "BAND7"):
            if self._pressure.do_cloud:
                surf_alb = self.rf_uip.tropomi_params["cloud_Surface_Albedo"]
            else:
                surf_alb = self.rf_uip.tropomi_params[f'surface_albedo_{filter_name}']
        else:
            raise RuntimeError(f"Unrecognized filter_name {filter_name}")

        self.apply_raman_effect(spec, temp_layers, surf_alb)

    def desc(self):
        return "MusesRaman"
        
__all__ = ["MusesRaman",]
