import os
import io
import tarfile
from contextlib import contextmanager
from . import muses_py as mpy

@contextmanager
def muses_py_call(rundir,
                  vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli",
                  debug=False,
                  vlidort_nstokes=2,
                  vlidort_nstreams=4):
    '''There is some cookie cutter code needed to call a py_retrieve function.
    We collect that here as a context manager, so you can just do something
    like:
    
    with muses_py_call(rundir):
       mpy.run_retrieval(...)

    without all the extra stuff. Note that we handle changing to the rundir
    before calling, so you don't need to do that before hand (or just
    pass "." if for whatever reason you have already done that).

    The "debug" flag turns on the per iteration directory diagnostics in omi and
    tropomi. I don't think it actually changes anything else, as far as I can tell
    only this get changed by the flags. The per iteration stuff is needed for some
    of the diagnostics (e.g., RefractorTropOmiFmMusesPy.surface_albedo).'''
    curdir = os.getcwd()
    old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
    # Temporary, make sure libgfortran.so.4 is in path. See
    # https://jpl.slack.com/archives/CVBUUE5T5/p1664476320620079.
    # Note that currently omi uses muses-vlidort repository build, which
    # doesn't have this problem any longer. But tropomi does still
    old_ld_library_path = None
    old_vlidort_cli = mpy.cli_options.get("vlidort_cli")
    old_debug = mpy.cli_options.get("debug")
    old_vlidort_nstokes = mpy.cli_options.vlidort.get("nstokes")
    old_vlidort_nstreams = mpy.cli_options.vlidort.get("nstreams")
    if('CONDA_PREFIX' in os.environ):
        old_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
        if old_ld_library_path:
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:{old_ld_library_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
    try:
        os.environ["MUSES_DEFAULT_RUN_DIR"] = os.path.abspath(rundir)
        os.environ["MUSES_PYOSS_LIBRARY_DIR"] = mpy.pyoss_dir
        os.chdir(rundir)
        if(vlidort_cli is not None):
            mpy.cli_options.vlidort_cli=vlidort_cli
        mpy.cli_options.debug=debug
        mpy.cli_options.vlidort.nstokes = vlidort_nstokes
        mpy.cli_options.vlidort.nstreams = vlidort_nstreams
        yield
    finally:
        os.chdir(curdir)
        mpy.cli_options.vlidort_cli=old_vlidort_cli
        mpy.cli_options.debug=old_debug
        mpy.cli_options.vlidort.nstokes = old_vlidort_nstokes
        mpy.cli_options.vlidort.nstreams = old_vlidort_nstreams
        if(old_run_dir):
            os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
        else:
            del os.environ["MUSES_DEFAULT_RUN_DIR"]
        if(old_ld_library_path):
            os.environ["LD_LIBRARY_PATH"] = old_ld_library_path
        else:
            del os.environ["LD_LIBRARY_PATH"]
   
class RefractorCaptureDirectory:
    '''muses-py code requires a number of files in a directory,
    these are essentially like hidden arguments to various muses-py
    functions.  If you want to be able to run muses-py code 
    (e.g., MusesTropomiForwardModel) then we need to save the directory
    when we capture runs, and extract it again to run the data again.
    This class handles this, wrapping everything up.'''
    def __init__(self):
        self.capture_directory = None
        self.runbase = None
        self.rundir = "."
        
    def save_directory(self, dirbase, vlidort_input):
        '''Capture information from the run directory so we can recreate the
        directory later. This is only needed by muses-py which uses a
        lot of files as "hidden" arguments to functions.  ReFRACtor
        doesn't need this.
        '''
        fh = io.BytesIO()
        dirbase = os.path.abspath(dirbase)
        self.runbase = os.path.basename(dirbase)
        # TODO This is probably too OMI specific
        osp_src_path = os.path.join(dirbase, "../OSP/OMI/")
        relpath = "./" + os.path.basename(dirbase)
        relpath2 = "./OSP/OMI"
        relpath3 = "./OSP/OMI/OMI_Solar"
        with tarfile.open(fileobj=fh, mode="x:bz2") as tar:
            for f in ("DateTime.asc", "Measurement_ID.asc", "Table.asc", "Table-final.asc",
                      "RamanInputs", "Input", vlidort_input):
                if(f is not None and os.path.exists(f"{dirbase}/{f}")):
                    tar.add(f"{dirbase}/{f}", f"{relpath}/{f}")
            for f in ("omi_rtm_driver", "ring", "ring_cli", 
                      "rayTable-NADIR.asc"):
                tar.add(f"{osp_src_path}/{f}", f"{relpath2}/{f}")
            tar.add(f"{osp_src_path}/OMI_Solar/omisol_v003_avg_nshi_backup.h5", f"{relpath2}/OMI_Solar/omisol_v003_avg_nshi_backup.h5")
        self.capture_directory = fh.getvalue()

    def extract_directory(self, path=".", change_to_dir = False,
                          osp_dir=None, gmao_dir=None):
        '''Extract a directory that has been previously saved.
        This gets extracted into the directory passed in the path. You can
        optionally change into the run directory.

        For pretty much everything below run_retrieval, the small OSP content
        we have stashed is sufficient to run. But for higher level functions,
        you need the full OSP directory. We don't carry this in this class,
        but if you supply a osp_dir we use that instead of the OSP we have
        stashed.'''
        if(self.capture_directory is None):
            raise RuntimeError("extract_directory can only be called if this object previously captured a directory")
        fh = io.BytesIO(self.capture_directory)
        with tarfile.open(fileobj=fh, mode="r:bz2") as tar:
            tar.extractall(path=path)
        if(osp_dir is not None):
            os.rename(path + "/OSP", path + "/OSP_not_used")
            os.symlink(osp_dir, path + "/OSP")
        if(gmao_dir is not None):
            os.symlink(gmao_dir, path+"/GMAO")
        self.rundir = os.path.abspath(path + "/" + self.runbase)
        if(change_to_dir):
            os.environ["MUSES_DEFAULT_RUN_DIR"] = self.rundir
            os.chdir(self.rundir)

__all__ = ["RefractorCaptureDirectory", "muses_py_call"]
