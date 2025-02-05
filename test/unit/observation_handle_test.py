from refractor.muses import mpy_radiance_from_observation_list, FilterIdentifier


def test_radiance(joint_tropomi_step_12):
    rs, _, _ = joint_tropomi_step_12
    oset = rs.observation_handle_set
    olist = [oset.observation(iname, None, None, None) for iname in [FilterIdentifier("CRIS"), FilterIdentifier("TROPOMI")]]
    rad = mpy_radiance_from_observation_list(olist, full_band=True)
    print(rad)
