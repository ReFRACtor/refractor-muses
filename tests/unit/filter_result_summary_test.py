from refractor.muses import (
    mpy_radiance_from_observation_list,
    FilterIdentifier,
    FilterResultSummary,
    AttrDictAdapter,
)


def test_filter_result_summary(joint_tropomi_step_12):
    rs, _, _ = joint_tropomi_step_12
    oset = rs.observation_handle_set
    olist = [
        oset.observation(iname, None, None, None)
        for iname in [FilterIdentifier("CRIS"), FilterIdentifier("TROPOMI")]
    ]
    rad = mpy_radiance_from_observation_list(olist, full_band=True)
    fsummary = FilterResultSummary(AttrDictAdapter(rad))
    assert fsummary.filter_index == [0, 11, 13, 14, 4]
    assert fsummary.filter_list == ["ALL", "TIR1", "TIR3", "TIR4", "UVIS"]
    assert fsummary.filter_start == [0, 0, 717, 1586, 2223]
    assert fsummary.filter_end == [2719, 716, 1585, 2222, 2719]
    assert fsummary.filter_slice == [
        slice(0, 2720),
        slice(0, 717),
        slice(717, 1586),
        slice(1586, 2223),
        slice(2223, 2720),
    ]
