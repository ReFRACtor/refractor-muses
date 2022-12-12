from test_support import *
import numpy as np
import numpy.testing as npt
import os
from refractor.omi import RefractorRtfOmi, RefractorRunRetrieval
import refractor.muses.muses_py as mpy


@require_muses_py
@long_test
@pytest.mark.parametrize("step_number", [1, 2])
def test_base_retrieval(clean_up_replacement_function, osp_dir,
                        isolated_dir, step_number):
    uip = load_uip(step_number, osp_dir=osp_dir)
    mpy.run_retrieval(**run_retrieval_parm(step_number=step_number))

@require_muses_py
@long_test
@pytest.mark.parametrize("step_number", [1, 2])
def test_refractor_retrieval(clean_up_replacement_function, osp_dir,
                             isolated_dir, step_number):
    uip = load_uip(step_number, osp_dir=osp_dir)
    mpy.register_replacement_function("rtf_omi", RefractorRtfOmi())
    mpy.run_retrieval(**run_retrieval_parm(step_number=step_number))

# Collect test data for mpy.run_retrieval, the first iteration for
# both strategy step 1 and 2.
@capture_test
@pytest.mark.parametrize("step_number", [1, 2])
def test_capture_retrieval_data(clean_up_replacement_function, step_number):
    from click.testing import CliRunner
    from py_retrieve.cli import cli
    
    mpy.register_replacement_function("run_retrieval",
            mpy.PickleFunctionArgument(test_in_dir +
                                   "run_retrieval_step_%d.pkl" % step_number,
                                   func_count=step_number))
    target_dir = mpy.Path("~/output_py/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_23").expanduser()
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--targets", target_dir
    ])
    print(result.output)

# Collect test data for a RefractorUip, including the directory structure.
# This is for first iteration for both strategy step 1 and 2.
@capture_test
@pytest.mark.parametrize("step_number", [1, 2])
def test_capture_uip(clean_up_replacement_function, step_number):
    fname = os.path.expanduser("~/output_py/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_23/Table.asc")
    rf_uip = RefractorUip.create_from_table(fname, step=step_number,
                                            capture_directory=True)
    pname = test_in_dir + "uip_step_%d.pkl" % step_number
    pickle.dump(rf_uip, open(pname, "wb"))
    
    
@require_muses_py
@long_test
def test_full_run(clean_up_replacement_function, original_run_dir):
    '''Set up with the input test data, and do a full muse-py run'''
    from click.testing import CliRunner
    from py_retrieve.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--targets", original_run_dir
    ])
    print(result.output)

@require_muses_py
@long_test
def test_refractor_full_run(clean_up_replacement_function, original_run_dir):
    '''Use the ReFRACtor version of run_retrieval. This just turns around
    and call mpy.run_retrieval, but we test the interface of replacing this.'''
    from click.testing import CliRunner
    from py_retrieve.cli import cli
    t = RefractorRunRetrieval()
    t.register_with_muses_py()
    runner = CliRunner()
    result = runner.invoke(cli, [
        "--targets", original_run_dir
    ])
    print(result.output)
    
