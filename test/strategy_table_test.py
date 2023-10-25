from test_support import *
from refractor.muses import StrategyTable, MusesRunDir
from pprint import pprint

@require_muses_py
def test_strategy_table(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(f"{test_base_path}/omi/in/sounding_1",
                    osp_dir, gmao_dir)
    s = StrategyTable(f"{r.run_dir}/Table.asc")

    s.table_step = 0
    print(s.spectral_filename)
    assert os.path.basename(s.spectral_filename) == "Windows_Nadir_OMICLOUDFRACTION_OMICLOUD_IG_Refine.asc"
    assert os.path.basename(s.cloud_parameters_filename) == "CloudParameters.asc"
    assert s.table_step == 0
    assert s.number_table_step == 2
    assert s.step_name == "OMICLOUDFRACTION"
    assert s.output_directory == os.path.abspath("./20160414_23_394_11_23") 
    
    s.table_step = 1
    assert os.path.basename(s.spectral_filename) == "Windows_Nadir_O3.asc"
    assert os.path.basename(s.cloud_parameters_filename) == "CloudParameters.asc"
    assert s.table_step == 1
    assert s.number_table_step == 2
    assert s.step_name == "O3_OMI"
    assert s.output_directory == os.path.abspath("./20160414_23_394_11_23") 

    
    
    


