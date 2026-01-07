# Top level executables. Don't need to normally run these, we check all the calculations
# at a lower level. But useful every once in a while to make sure this isn't a problem with
# the cli or something like that.
#
# Note, unlike most of our tests these shouldn't be run in parallel (e.g., using -n 10 or
# whatever). Since these are full runs, each one uses a lot of resources. We were regularly
# seeing load of > 150. Instead, just run this like
#
# pytest -s tests/executable --run-long-executable
#
# or something like that
import subprocess
import pytest
from refractor.muses import MusesRunDir


@pytest.mark.long_executable_test
def test_py_retrieve_airs_omi(ifile_hlp, joint_omi_test_in_dir, end_to_end_run_dir):
    """Full run of the old py-retrieve code, used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking the output.
    That is done at lower levels of testing.
    """
    dir = end_to_end_run_dir / "py_retrieve_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir, ifile_hlp, path_prefix=dir, osp_sym_link=True
    )
    subprocess.run(
        f"py-retrieve --targets {r.run_dir}",
        shell=True,
        check=True,
    )


@pytest.mark.long_executable_test
def test_py_retrieve_cris_tropomi(
    ifile_hlp, joint_tropomi_test_in_dir, end_to_end_run_dir
):
    """Full run of the old py-retrieve code, used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking the output.
    That is done at lower levels of testing.
    """
    dir = end_to_end_run_dir / "py_retrieve_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir, ifile_hlp, path_prefix=dir, osp_sym_link=True
    )
    subprocess.run(
        f"py-retrieve --targets {r.run_dir}",
        shell=True,
        check=True,
    )


@pytest.mark.long_executable_test
def test_refractor_py_retrieve_airs_omi(
    ifile_hlp, joint_omi_test_in_dir, end_to_end_run_dir
):
    """Full run of the refractor-retrieve code, using the old
    py-retrieve forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    """
    dir = end_to_end_run_dir / "refractor_py_retrieve_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    subprocess.run(f"refractor-retrieve --targets {r.run_dir}", shell=True, check=True)


@pytest.mark.long_executable_test
def test_refractor_py_retrieve_cris_tropomi(
    ifile_hlp, joint_tropomi_test_in_dir, end_to_end_run_dir
):
    """Full run of the refractor-retrieve code, using the old
    py-retrieve forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    """
    dir = end_to_end_run_dir / "refractor_py_retrieve_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    subprocess.run(f"refractor-retrieve --targets {r.run_dir}", shell=True, check=True)


@pytest.mark.long_executable_test
def test_refractor_retrieve_airs_omi(
    ifile_hlp, joint_omi_test_in_dir, end_to_end_run_dir, refractor_config_file
):
    """Full run of the refractor-retrieve code, using the
    ReFRACtor forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    """
    dir = end_to_end_run_dir / "refractor_retrieve_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    subprocess.run(
        f"refractor-retrieve --refractor-config {refractor_config_file} --targets {r.run_dir}",
        shell=True,
        check=True,
    )


@pytest.mark.long_executable_test
def test_refractor_retrieve_cris_tropomi(
    ifile_hlp,
    joint_tropomi_test_in_dir,
    end_to_end_run_dir,
    refractor_config_file,
):
    """Full run of the refractor-retrieve code, using the
    ReFRACtor forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    """
    dir = end_to_end_run_dir / "refractor_retrieve_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    subprocess.run(
        f"refractor-retrieve --refractor-config {refractor_config_file} --targets {r.run_dir}",
        shell=True,
        check=True,
    )


@pytest.mark.long_executable_test
def test_py_retrieve_refractor_cris_tropomi(
    ifile_hlp,
    joint_tropomi_test_in_dir,
    end_to_end_run_dir,
    refractor_config_file,
):
    """Full run of the py-retrieve code, configure to use ReFRACtor with the
    ReFRACtor forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    We don't normally run like this, instead we just use refractor-retrieve. But
    at least short term, make sure this functionality is still working.
    """
    dir = end_to_end_run_dir / "py_retrieve_refractor_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    subprocess.run(
        f"py-retrieve --refractor --refractor-config {refractor_config_file} --targets {r.run_dir}",
        shell=True,
        check=True,
    )
