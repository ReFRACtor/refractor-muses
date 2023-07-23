from test_support import *
from refractor.muses import MusesRunDir

@require_muses_py
def test_muses_run_dir(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(f"{test_base_path}/omi/in/sounding_1",
                    osp_dir, gmao_dir)

@long_test    
@require_muses_py
def test_muses_full_run(isolated_dir, osp_dir, gmao_dir, vlidort_cli):
    r = MusesRunDir(f"{test_base_path}/omi/in/sounding_1",
                    osp_dir, gmao_dir)
    r.run_retrieval(vlidort_cli=vlidort_cli)
    
    
    
