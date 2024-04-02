import numpy as np
import refractor.framework as rf

class MusesRaman(rf.RamanSiorisEffect):

    def __init__(self, rf_uip, instrument_name,
                 Solar_and_odepth_spec_domain : rf.SpectralDomain,
                 scale_factor : float,
                 fm_idx : int,
                 filter_name : str,
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
        self.filter_name = filter_name
        self.instrument_name = instrument_name
        # Save range of Solar_and_odepth_spec_domain, to give clearer error
        # message
        self.solar_model_sd_min = Solar_and_odepth_spec_domain.data.min()
        self.solar_model_sd_max = Solar_and_odepth_spec_domain.data.max()

    def apply_effect(self, spec: rf.Spectrum, fm_grid: rf.ForwardModelSpectralGrid):

        nlay = self._pressure.number_layer

        temp_layers = self.rf_uip.ray_info(self.instrument_name)['tbar'][::-1][:nlay]

        if self.filter_name in ("UV1", "UV2"):
            if self._pressure.do_cloud:
                # Replicate py-retrieve behavior in omi/print_ring_input.py
                surf_alb = 0.80
            else:
                surf_alb = self.rf_uip.omi_params[f'surface_albedo_{str.lower(self.filter_name)}']
        elif self.filter_name in ("BAND1", "BAND2", "BAND3", "BAND7"):
            if self._pressure.do_cloud:
                surf_alb = self.rf_uip.tropomi_params["cloud_Surface_Albedo"]
            else:
                surf_alb = self.rf_uip.tropomi_params[f'surface_albedo_{self.filter_name}']
        else:
            raise RuntimeError(f"Unrecognized filter_name {self.filter_name}")
        # Sanity check - the error messages at the C++ level are pretty obscure
        sd_min = spec.spectral_domain.data.min()
        sd_max = spec.spectral_domain.data.max()
        if(sd_min < self.solar_model_sd_min or sd_min > self.solar_model_sd_max):
            raise RuntimeError(f"Spectral domain is out of range sd_min = {sd_min} sd_max = {sd_max} raman_sd_min = {self.solar_model_sd_min} raman_sd_max = {self.solar_model_sd_max}")
        self.apply_raman_effect(spec, temp_layers, surf_alb)

    def desc(self):
        return "MusesRaman"
        
__all__ = ["MusesRaman",]
