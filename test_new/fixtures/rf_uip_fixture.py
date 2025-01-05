# This is has some old py-retrieve UIP fixtures
import pytest
from refractor.muses import RefractorUip


def load_uip(dir_in, step_number=1, osp_dir=None, gmao_dir=None):
    return RefractorUip.load_uip(
        f"{dir_in}/uip_step_{step_number}.pkl",
        change_to_dir=True,
        osp_dir=osp_dir,
        gmao_dir=gmao_dir,
    )


@pytest.fixture(scope="function")
def omi_uip_step_2(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    """Return a RefractorUip for strategy step 2, and also unpack all the
    support files into a directory"""
    return load_uip(omi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir)


@pytest.fixture(scope="function")
def tropomi_uip_step_2(isolated_dir, osp_dir, gmao_dir, tropomi_test_in_dir):
    """Return a RefractorUip for strategy step 2, and also unpack all the
    support files into a directory"""
    return load_uip(
        tropomi_test_in_dir, step_number=2, osp_dir=osp_dir, gmao_dir=gmao_dir
    )


@pytest.fixture(scope="function")
def joint_omi_uip_step_8(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    """Return a RefractorUip for strategy step 1, and also unpack all the
    support files into a directory"""
    return load_uip(
        joint_omi_test_in_dir, step_number=8, osp_dir=osp_dir, gmao_dir=gmao_dir
    )


@pytest.fixture(scope="function")
def joint_tropomi_uip_step_12(
    isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir
):
    """Return a RefractorUip for strategy step 1, and also unpack all the
    support files into a directory"""
    return load_uip(
        joint_tropomi_test_in_dir, step_number=12, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
