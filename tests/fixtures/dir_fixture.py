# This defines fixtures that gives the paths to various directories with
# test data

import os
import pytest
from pathlib import Path


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
def osp_dir():
    """Location of OSP directory."""
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        pytest.skip(
            "test requires OSP directory set by through the MUSES_OSP_PATH environment variable"
        )
    return Path(osp_path)


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
def gmao_dir():
    """Location of GAMO directory."""
    gmao_path = os.environ.get("MUSES_GMAO_PATH", None)
    if gmao_path is None or not os.path.exists(gmao_path):
        pytest.skip(
            "test requires GMAO directory set by through the MUSES_GMAO_PATH environment variable"
        )
    return Path(gmao_path)


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
