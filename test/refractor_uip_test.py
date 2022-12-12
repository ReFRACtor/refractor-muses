from test_support import *
from refractor.muses import RefractorUip

@require_muses_py
def test_refractor_uip(uip_step_1):
    m = RefractorUip(uip=uip_step_1.uip)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    ii_mw = 0
    print(m.atmosphere_column("O3"))
    print(m.omi_params)
    print(m.channel_indexes(ii_mw))
    print(m.observation_zenith_with_unit(ii_mw))
    print(m.observation_azimuth_with_unit(ii_mw))
    print(m.solar_azimuth_with_unit(ii_mw))
    print(m.solar_zenith_with_unit(ii_mw))
    print(m.relative_azimuth_with_unit(ii_mw))
    print(m.latitude(ii_mw))
    print(m.longitude(ii_mw))
    print(m.surface_height(ii_mw))
    print(m.across_track_indexes(ii_mw))
    print(m.atm_params)
    print(m.ray_info)
    
