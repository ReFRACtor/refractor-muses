from refractor.muses import MusesRunDir
import pytest


def test_muses_run_dir(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    _ = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir)


@pytest.mark.long_test
def test_muses_full_run(isolated_dir, osp_dir, gmao_dir, vlidort_cli, omi_test_in_dir):
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir)
    r.run_retrieval(vlidort_cli=vlidort_cli)
