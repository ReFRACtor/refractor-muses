from refractor.muses import MusesRunDir
import pytest


def test_muses_run_dir(isolated_dir, ifile_hlp, omi_test_in_dir):
    _ = MusesRunDir(omi_test_in_dir, ifile_hlp)


# We can skip this. the end to end tests already check this, no reason to have a
# retrieval just to make sure we can. Can turn on for specific debugging if needed.
@pytest.mark.skip
@pytest.mark.long_test
def test_muses_full_run(isolated_dir, ifile_hlp, omi_test_in_dir):
    r = MusesRunDir(omi_test_in_dir, ifile_hlp)
    r.run_retrieval()
