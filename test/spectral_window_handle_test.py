from test_support import *
from refractor.muses import (MusesPySpectralWindowHandle, MusesRunDir, RetrievalConfiguration,
                             StrategyTable, SpectralWindowHandleSet,
                             CurrentStrategyStepDict, MeasurementIdFile)

@require_muses_py
def test_muses_py_spectral_window_handle(osp_dir, isolated_dir, gmao_dir):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc",
                                                               osp_dir=osp_dir)
    flist = {'OMI': ['UV1', 'UV2'], 'AIRS': ['2B1', '1B2', '2A1', '1A1']}
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig, flist)
    swin_handle_set = SpectralWindowHandleSet.default_handle_set()
    swin_handle_set.notify_update_target(mid)
    stable = StrategyTable(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    stable.table_step = 8+1
    current_strategy_step = CurrentStrategyStepDict({'retrieval_elements' : stable.retrieval_elements(),
                                                     'step_name' : stable.step_name,
                                                     'step_number' : 8,
                                                     'max_num_iterations' : stable.max_num_iterations,
                                                     'retrieval_type' : stable.retrieval_type})
    swin_dict = swin_handle_set.spectral_window_dict(current_strategy_step)
    breakpoint()
