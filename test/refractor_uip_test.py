from test_support import *
from refractor.muses import RefractorUip

@require_muses_py
def test_refractor_omi_uip(isolated_dir):
    m = RefractorUip.load_uip(f"{test_base_path}/omi/in/sounding_1/uip_step_1.pkl",
                              change_to_dir=False)
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
    print(m.solar_irradiance(ii_mw))

@require_muses_py
def test_refractor_joint_uip(isolated_dir):
    # UIP  that has both AIRS and OMI
    m = RefractorUip.load_uip(f"{test_base_path}/airs_omi/in/sounding_1/uip_step_7.pkl",
                              change_to_dir=False)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    ii_mw = 10 # First index that is OMI
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
    
@require_muses_py
def test_refractor_tropomi_uip(isolated_dir):
    m = RefractorUip.load_uip(f"{test_base_path}/tropomi/in/sounding_1/uip_step_1.pkl",
                              change_to_dir=False)
    # We just want to make sure we can access everything, so just call
    # each of the functions and print the results out
    ii_mw = 0
    print(m.atmosphere_column("O3"))
    print(m.tropomi_params)
    print(m.channel_indexes(ii_mw))
    print(m.observation_zenith_with_unit(ii_mw))
    # For some reason, not actually in the tropomi UIP. Really
    # isn't there, not an error in our processing. I don't think
    # this actually matters though
    # print(m.observation_azimuth_with_unit(ii_mw))
    print(m.solar_azimuth_with_unit(ii_mw))
    print(m.solar_zenith_with_unit(ii_mw))
    print(m.relative_azimuth_with_unit(ii_mw))
    print(m.latitude(ii_mw))
    print(m.longitude(ii_mw))
    print(m.surface_height(ii_mw))
    print(m.across_track_indexes(ii_mw))
    print(m.atm_params)
    print(m.ray_info)
