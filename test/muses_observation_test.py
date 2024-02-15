from refractor.muses import (MusesAirsObservationNew, MusesRunDir, MusesAirsObservation,
                             StrategyTable)
import refractor.framework as rf
from test_support import *

def test_muses_airs_observation(isolated_dir, osp_dir, gmao_dir):
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    stable = StrategyTable(f"{joint_omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesAirsObservationNew(fname, xtrack, atrack, channel_list, osp_dir=osp_dir)
    step_number = 8
    iteration = 2
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    obs_old = MusesAirsObservation(rf_uip, rrefractor.params["ret_info"]["obs_rad"],
                                   rrefractor.params["ret_info"]["meas_err"])
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = stable.spectral_window("AIRS", stp=step_number+1)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    print(obs_old.radiance(0).spectral_range.data)
    print([obs_old.rf_uip.uip['microwindows_all'][i] for i in
           range(len(obs_old.rf_uip.uip['microwindows_all']))
           if obs_old.rf_uip.uip['microwindows_all'][i]['instrument'] == "AIRS"])
