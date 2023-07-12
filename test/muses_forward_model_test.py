from test_support import *
from refractor.muses import RefractorUip, MusesCrisForwardModel

@require_muses_py
def test_muses_cris_forward_model(isolated_dir, osp_dir, gmao_dir):
    rf_uip = RefractorUip.load_uip(f"{test_base_path}/cris_tropomi/in/sounding_1/uip_step_10.pkl", change_to_dir=True, osp_dir=osp_dir, gmao_dir=gmao_dir)
    fm = MusesCrisForwardModel(rf_uip)
    s = fm.radiance(0)
    rad = s.spectral_range.data
    jac = s.spectral_range.data_ad.jacobian
    print(rad)
    print(jac)
    print(rad.shape)
    print(jac.shape)
    
