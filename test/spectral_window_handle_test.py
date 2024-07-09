from test_support import *
from refractor.muses import (MusesPySpectralWindowHandle)

@require_muses_py
def test_muses_py_spectral_window_handle(osp_dir):
    default_fname = f"{osp_dir}/Strategy_Tables/ops/Defaults/Default_Spectral_Windows_Definition_File_Filters_CrIS_TROPOMI.asc"
    viewing_mode = "nadir"
    spectral_dir = f"{osp_dir}/Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions"
    retrieval_elements = ["H2O","O3","CLOUDEXT","PCLOUD","TSUR","EMIS","TROPOMIRINGSFBAND3","TROPOMISOLARSHIFTBAND3","TROPOMIRADIANCESHIFTBAND3","TROPOMISURFACEALBEDOBAND3","TROPOMISURFACEALBEDOSLOPEBAND3","TROPOMISURFACEALBEDOSLOPEORDER2BAND3"]
    step_name = "H2O,O3,EMIS_TROPOMI"
    retrieval_type = "joint"
    spec_fname = None
    shandle = MusesPySpectralWindowHandle()
