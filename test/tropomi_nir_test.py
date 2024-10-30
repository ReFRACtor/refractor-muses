# This is similar to the band 7 data, but is based on a test case from Josh. The
# band 7 test case was a major kludge, and I expect it may go away. This is more
# likely to be the replacement for this.

import numpy as np
import numpy.testing as npt
from pathlib import Path
from test_support import *
from test_support.refwrapper_lite import absco_files as rw_absco_files, build_state as rw_state, clouds as rw_clouds, simulation as rw_sim
import refractor.framework as rf
from refractor.tropomi import (TropomiFmObjectCreator, TropomiForwardModelHandle)
from refractor.muses import MusesRunDir
from refractor.old_py_retrieve_wrapper import RefractorMusesIntegration

class SaveSpectrum(rf.ObserverPtrNamedSpectrum):

    def __init__(self, filename):
        super().__init__()

        self.filename = filename

    def notify_update(self, o):
        import pickle
        fn = self.filename.format(name=o.name.replace(" ", "_"))
        with open(fn, "wb") as out:
            print(f"Saving {o.name} to {fn}")
            data = { "name": o.name,
                     "wavelength": o.spectral_domain.wavelength("nm"),
                     "radiance": o.spectral_range.data,
                    }
            pickle.dump(data, out)

@long_test    
@require_muses_py            
def test_nir_retrieval(isolated_dir, osp_dir, gmao_dir, vlidort_cli,
                       clean_up_replacement_function):
    # NOTE - This depends on the newer OSP directory. If you don't have this
    # data, you should set the environment variable MUSES_OSP_PATH to
    # "/tb/sandbox17/laughner/OSP-mine/OSP"
    
    r = RefractorMusesIntegration()
    r.forward_model_handle_set.add_handle(TropomiForwardModelHandle(use_pca=True,
                                       use_raman=False,
                                       use_lrad=False, lrad_second_order=False),
                                       priority_order=100)
    r.register_with_muses_py()
    r = MusesRunDir(tropomi_test_in_dir2, osp_dir, gmao_dir)
    r.run_retrieval(vlidort_cli=vlidort_cli)


    
@long_test
@require_muses_py            
def test_nir_simulation(isolated_dir, josh_osp_dir, gmao_dir, clean_up_replacement_function):
    r = MusesRunDir(tropomi_band7_test_in_dir2, josh_osp_dir, gmao_dir)

    atm_state_file = Path(tropomi_band7_test_state_dir2) / 'State_AtmProfiles.asc'
    tropomi_state_file = Path(tropomi_band7_test_state_dir2) / 'State_TROPOMI.asc'
    l1b_file = Path(tropomi_band7_test_top) / 'S5P_RPRO_L1B_RA_BD7_20220628T185806_20220628T203935_24394_03_020100_20230104T092546.nc'
    isrf_file = Path(josh_osp_dir) / 'TROPOMI' / 'isrf_release' / 'isrf' / 'binned_uvn_swir_sampled' / 'S5P_OPER_AUX_ISRF___00000101T000000_99991231T235959_20180320T084215.nc'

    sounding_state = rw_state.build_state_from_sounding(r.run_dir, atm_state_file=atm_state_file, tropomi_state_file=tropomi_state_file)
    absco_files = rw_absco_files.absco_file_list(josh_osp_dir)

    cloud_props = rw_clouds.MusesCloudProperties(
        cloud_frac=sounding_state['ancillary']['cloud_frac'],
        cloud_pres=sounding_state['ancillary']['cloud_pres'],
        cloud_albedo=sounding_state['ancillary']['cloud_albedo']
    )
    sounding_atm, sounding_sv, grid_points, grid_units = rw_sim.setup_atmosphere_from_tropomi(
        sounding_state, absco_files, albedo_key='albedo', n_alb_terms=3, cloud_props=cloud_props
    )
    band7_ils = rw_sim.setup_tropomi_ils(
        xtrack_index=int(sounding_state['file_id']['TROPOMI_XTrack_Index_BAND7']),
        l1b_file=l1b_file,
        isrf_file=isrf_file
    )
    spectrum_instr_no_solar, _ = rw_sim.run_simulation(sounding_state, sounding_atm, grid_points, grid_units, solar_model_file=None, ils=band7_ils, cloud_props=cloud_props)
    # TODO: actually check against expected spectrum? Right now this just needs to succeed.