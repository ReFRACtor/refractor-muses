import os
import io
import tarfile

class RefractorCaptureDirectory:
    '''py-retrieve code requires a number of files in a directory,
    these are essentially like hidden arguments to various py-retrieve
    functions.  If you want to be able to run py-retrieve code 
    (e.g., MusesPyForwardModel) then we need to save the directory
    when we capture runs, and extract it again to run the data again.
    This class handles this, wrapping everything up.'''
    def __init__(self):
        self.capture_directory = None
    def save_directory(self, dirbase, vlidort_input):
        '''Capture information from the run directory so we can recreate the
        directory later. This is only needed by muses-py which uses a
        lot of files as "hidden" arguments to functions.  ReFRACtor
        doesn't need this.
        '''
        fh = io.BytesIO()
        dirbase = os.path.abspath(dirbase)
        # TODO This is probably too OMI specific
        osp_src_path = os.path.join(dirbase, "../OSP/OMI/")
        relpath = "./" + os.path.basename(dirbase)
        relpath2 = "./OSP/OMI"
        relpath3 = "./OSP/OMI/OMI_Solar"
        with tarfile.open(fileobj=fh, mode="x:bz2") as tar:
            for f in ("RamanInputs", "Input", vlidort_input):
                tar.add(f"{dirbase}/{f}", f"{relpath}/{f}")
            for f in ("omi_rtm_driver", "ring", "ring_cli", "vlidort_cli"):
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
        if(change_to_dir):
            runbase = os.path.basename(os.path.dirname(self.strategy_table))
            rundir = os.path.abspath(path + "/" + runbase)
            os.environ["MUSES_DEFAULT_RUN_DIR"] = rundir
            os.chdir(rundir)

__all__ = ["RefractorCaptureDirectory",]
            
