# This defines fixtures that gives the paths to various directories with
# test data

import os
import pytest
from pathlib import Path
from refractor.muses import InputFileHelper, InputFileSave, InputFilePathDelta


@pytest.fixture(scope="session")
def test_base_path():
    """Base directory for much of our test data. We generally get this
    from the environment variable REFRACTOR_TEST_DATA (set up in the
    conda environment from refractor-build-supplement), but we also
    have a default relative path that is parallel to the the
    refractor-muses directory"""
    if "REFRACTOR_TEST_DATA" in os.environ:
        return Path(os.environ["REFRACTOR_TEST_DATA"])
    return (Path(os.path.dirname(__file__)) / "../../../refractor_test_data").resolve()


@pytest.fixture(scope="session")
def osp_real_dir():
    """Location of "real" OSP directory. We generally use our saved version in
    test_data_dir / "OSP", but it can be useful to use the real actual OSP in
    some cases, particularly when developing something new that we haven't already
    saved."""
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        pytest.skip(
            "test requires OSP directory set by through the MUSES_OSP_PATH environment variable"
        )
    return Path(osp_path)

@pytest.fixture(scope="session")
def osp_dir(test_base_path, osp_real_dir):
    """Location of OSP directory. We generally use the OSP that we have saved, so our
    unit tests don't break from updates to the real OSP.
    This is based off of v1.23.0 of the OSP (github.jpl.nasa.gov:MUSES-Processing/OSP.git).
    """
    if False:
        return osp_real_dir
    else:
        return test_base_path / "OSP"

@pytest.fixture(scope="session")
def osp_delta_dir(test_base_path):
    """Location of OSP delta directory. Changes to osp_dir.
    """
    return test_base_path / "OSP_delta"
    

@pytest.fixture(scope="function")
def josh_osp_dir():
    """Location of Josh's newer OSP directory. Eventually stuff will get merged into
    the real OSP directory, but for now keep this separate"""
    osp_path = os.environ.get(
        "MUSES_JOSH_OSP_PATH", "/tb/sandbox17/laughner/OSP-mine/OSP"
    )
    if osp_path is None or not os.path.exists(osp_path):
        pytest.skip(
            "test requires Josh's OSP directory set by through the MUSES_JOSH_OSP_PATH environment variable"
        )
    return Path(osp_path)


@pytest.fixture(scope="session")
def gmao_real_dir():
    """Location of "real" GMAO directory. We generally use our saved version in
    test_data_dir / "OSP", but it can be useful to use the real actual OSP in
    some cases, particularly when developing something new that we haven't already
    saved."""
    gmao_path = os.environ.get("MUSES_GMAO_PATH", None)
    if gmao_path is None or not os.path.exists(gmao_path):
        pytest.skip(
            "test requires GMAO directory set by through the MUSES_GMAO_PATH environment variable"
        )
    return Path(gmao_path)

@pytest.fixture(scope="session")
def gmao_dir(test_base_path, gmao_real_dir):
    """Location of GMAO directory. We generally use the GMAO that we have saved, so our
    unit tests don't break from updates to the real GMAO.
    """
    if False:
        return gmao_real_dir
    else:
        return test_base_path / "GMAO"

@pytest.fixture(scope="session")
def ifile_hlp(osp_dir, osp_delta_dir, gmao_dir, osp_real_dir, gmao_real_dir, test_base_path):
    # Quick way to grab all the needed test data. Note if you run this, you should
    # run this serially. I don't think the InputFileSave works in parallel (e.g.,
    # two processes try to copy the same file write over each other)
    if False:
        ifile_hlp = InputFileHelper(osp_dir=osp_real_dir, gmao_dir=gmao_real_dir)
        ifile_hlp.add_observer(InputFileSave(Path(str(ifile_hlp.osp_dir)),
                                             test_base_path / "OSP"))
        ifile_hlp.add_observer(InputFileSave(Path(str(ifile_hlp.gmao_dir)),
                                             test_base_path / "GMAO"))
        return ifile_hlp
    return InputFileHelper(osp_dir=InputFilePathDelta(osp_dir, osp_delta_dir), gmao_dir=gmao_dir)


@pytest.fixture(scope="session")
def omi_test_in_dir(test_base_path):
    return test_base_path / "omi/in/sounding_1"


@pytest.fixture(scope="session")
def omi_test_expected_results_dir(test_base_path):
    return test_base_path / "omi/expected"


@pytest.fixture(scope="session")
def tropomi_test_in_dir(test_base_path):
    return test_base_path / "tropomi/in/sounding_1"


@pytest.fixture(scope="session")
def cris_test_in_dir(test_base_path):
    return test_base_path / "cris/in/sounding_1"


@pytest.fixture(scope="session")
def cris_ml_dir(test_base_path):
    return test_base_path / "cris/in/ml_weight/225"


@pytest.fixture(scope="session")
def tes_test_in_dir(test_base_path):
    return test_base_path / "tes/in/sounding_1"


@pytest.fixture(scope="session")
def tes_test_expected_dir(test_base_path):
    return test_base_path / "tes/expected/sounding_1"


@pytest.fixture(scope="session")
def tes_test_expected_retrieval_output_dir(test_base_path):
    return test_base_path / "tes/expected/sounding_1_retrieval_output"


@pytest.fixture(scope="session")
def tropomi_band7_test_in_dir(test_base_path):
    return test_base_path / "tropomi_band7/in/sounding_1"


@pytest.fixture(scope="session")
def tropomi_band7_test_in_dir2(test_base_path):
    return test_base_path / "tropomi_band7/in/sounding_2"


@pytest.fixture(scope="session")
def tropomi_band7_test_in_state_dir2(test_base_path):
    return test_base_path / "tropomi_band7/in/sounding_2_state"


@pytest.fixture(scope="session")
def tropomi_band7_test_expected_results_dir(test_base_path):
    return test_base_path / "tropomi_band7/expected"


@pytest.fixture(scope="session")
def tropomi_band7_swir_step_test_in_dir(test_base_path):
    return test_base_path / "tropomi_band7/in/sounding_one_step"


@pytest.fixture(scope="session")
def tropomi_band7_sim_alb_dir(test_base_path):
    return test_base_path / "tropomi_band7/in/synth_alb_0_9"


@pytest.fixture(scope="session")
def joint_tropomi_test_in_dir(test_base_path):
    return test_base_path / "cris_tropomi/in/sounding_1"


@pytest.fixture(scope="session")
def tropomi_test_in_dir3(test_base_path):
    return test_base_path / "tropomi/in/sounding_3"


@pytest.fixture(scope="session")
def joint_tropomi_test_expected_dir(test_base_path):
    return test_base_path / "cris_tropomi/expected/sounding_1"


@pytest.fixture(scope="session")
def joint_tropomi_test_expected_retrieval_output_dir(test_base_path):
    return test_base_path / "cris_tropomi/expected/sounding_1_retrieval_output"


@pytest.fixture(scope="session")
def joint_tropomi_test_refractor_expected_dir(test_base_path):
    return test_base_path / "cris_tropomi/expected/sounding_1_refractor"


@pytest.fixture(scope="session")
def joint_omi_test_in_dir(test_base_path):
    return test_base_path / "airs_omi/in/sounding_1"


@pytest.fixture(scope="session")
def joint_omi_test_expected_dir(test_base_path):
    return test_base_path / "airs_omi/expected/sounding_1"


@pytest.fixture(scope="session")
def joint_omi_test_expected_retrieval_output_dir(test_base_path):
    return test_base_path / "airs_omi/expected/sounding_1_retrieval_output"


@pytest.fixture(scope="session")
def joint_omi_eof_test_in_dir(test_base_path):
    return test_base_path / "airs_omi/in/sounding_2_eof"


@pytest.fixture(scope="session")
def joint_omi_test_refractor_expected_dir(test_base_path):
    return test_base_path / "airs_omi/expected/sounding_1_refractor"


@pytest.fixture(scope="session")
def airs_irk_test_in_dir(test_base_path):
    return test_base_path / "airs_omi/in/sounding_1_irk"


@pytest.fixture(scope="session")
def airs_irk_test_expected_dir(test_base_path):
    return test_base_path / "airs_omi/expected/sounding_1_irk"


@pytest.fixture(scope="session")
def airs_irk_test_expected_retrieval_output_dir(test_base_path):
    return test_base_path / "airs_omi/expected/sounding_1_irk_retrieval_output"


@pytest.fixture(scope="session")
def omi_config_dir():
    return Path(os.path.dirname(__file__)).parent.parent / "python/refractor/omi/config"


@pytest.fixture(scope="session")
def end_to_end_run_dir():
    res = Path(os.path.dirname(__file__)).parent.parent / "end_to_end_run"
    # Create directory if it isn't already there
    res.mkdir(parents=True, exist_ok=True)
    return res


@pytest.fixture(scope="session")
def refractor_config_file():
    res = (
        Path(os.path.dirname(__file__)).parent.parent
        / "muses_config"
        / "refractor_config.py"
    )
    return res


@pytest.fixture(scope="session")
def unit_test_expected_dir():
    res = Path(os.path.dirname(__file__)).parent / "expected"
    # Create directory if it isn't already there
    res.mkdir(parents=True, exist_ok=True)
    return res
