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
    return (
        Path(os.path.abspath(os.path.dirname(__file__)))
        / "../../../refractor_test_data"
    )


@pytest.fixture(scope="session")
def osp_dir():
    """Location of OSP directory."""
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        raise pytest.skip(
            "test requires OSP directory set by through the MUSES_OSP_PATH environment variable"
        )
    return Path(osp_path)


@pytest.fixture(scope="session")
def omi_test_in_dir(test_base_path):
    return test_base_path / "omi/in/sounding_1"
