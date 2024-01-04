import numpy as np

from refractor.framework import refractor_config
from refractor import framework as rf

from .base_config import base_config_definition, num_channels, channel_names

class OmiSimConfig(rf.MusesUipSimConfig):

    grid_units = "nm"

    def __init__(self, muses_uip_file, config_def, atm_gas_list=None):
        super().__init__(muses_uip_file, config_def, num_channels, atm_gas_list=atm_gas_list)

        self.uip_omi = self.uip['uip_omi'][0]

    def configure_scenario(self):
        omi_obs = self.uip_omi['omi_obs_table'][0]

        # Just UV1 and UV2 info
        where_channels = ((0, 1),)

        self.setup_scenario(
            latitude = omi_obs['latitude'][0][where_channels],
            obs_azimuth = omi_obs['viewingazimuthangle'][0][where_channels]% 360.0, 
            obs_zenith = omi_obs['viewingzenithangle'][0][where_channels],
            solar_azimuth = omi_obs['solarazimuthangle'][0][where_channels] % 360.0, 
            solar_zenith = omi_obs['solarzenithangle'][0][where_channels],
            surface_height = omi_obs['terrainheight'][0][where_channels],
            relative_azimuth = omi_obs['relativeazimuthangle'][0][where_channels])

        self.config_def['scenario']['across_track_indexes'] = self.uip_omi['omi_obs_table'][0]['xtrack'][0]
        
        self.config_def['solar_model']['across_track_indexes'] = self.uip_omi['omi_obs_table'][0]['xtrack'][0]

    def configure_micro_windows(self):
        return super().configure_micro_windows(desired_instrument='OMI')

    def configure_sample_grid(self):
        
        all_freq = self.uip_omi['fullbandfrequency'][0]
        filt_loc = np.array([ val.decode('UTF-8') for val in self.uip_omi['frequencyfilterlist'][0] ])

        sample_grid = []
        for band_name in channel_names:
            sample_grid.append( rf.SpectralDomain(all_freq[np.where(filt_loc == band_name)], rf.Unit(self.grid_units)) )

        self.setup_sample_grid(sample_grid)

    def configure_albedo(self):
        omipars = self.uip['omipars'][0]

        albedo = np.zeros((num_channels, 2))

        albedo[0, 0] = omipars['surface_albedo_uv1']
        albedo[1, 0] = omipars['surface_albedo_uv2']
        albedo[1, 1] = omipars['surface_albedo_slope_uv2']

        self.setup_surface_albedo(albedo)

    def configure(self):
        super().configure()

        self.configure_albedo()

@refractor_config
def uip_config(uip_filename):

    config_def = base_config_definition()

    sim_config = OmiSimConfig(uip_filename, config_def)
    sim_config.configure()

    return config_def
