from . import muses_py as mpy
import shutil
import logging
import os
import subprocess

logger = logging.getLogger('refractor.muses')
class MusesRunDir:
    '''This provides a bit of support for copying a run directory
    from an amuse-me run to refractor_test_data, and for copying that
    data into a scratch area. This handles copying all of the input
    files also. This can then be used for running a full retrieval.
    Note that this is not at all necessary for ReFRACtor, only for doing
    a full py-retrieval call.
    '''
    def __init__(self, refractor_sounding_dir, osp_dir, gmao_dir,
                 path_prefix="."):
        '''Set up a run directory in the given path_prefix with the
        data saved in a sounding 1 save directory (e.g.,
        ~/muses/refractor_test_data/omi/sounding_1).

        This handles updating the paths in Measurement_ID to the data
        saved in the test/in directory'''
        sid = open(f"{refractor_sounding_dir}/sounding.txt").read().rstrip()
        self.run_dir = f"{path_prefix}/{sid}"
        subprocess.run(["mkdir","-p",self.run_dir])
        os.symlink(osp_dir, f"{path_prefix}/OSP")
        os.symlink(gmao_dir, f"{path_prefix}/GMAO")
        for f in ("Table", "DateTime"):
            shutil.copy(f"{refractor_sounding_dir}/{f}.asc",
                        f"{self.run_dir}/{f}.asc")
        _, d = mpy.read_all_tes(f"{refractor_sounding_dir}/Measurement_ID.asc")
        for k in ("OMI_filename", "OMI_Cloud_filename",
                  "TROPOMI_filename_BAND3",
                  "TROPOMI_IRR_filename",
                  "TROPOMI_Cloud_filename"):
            if k in d['preferences']:
                f = d['preferences'][k]
                freplace = os.path.abspath(f"{refractor_sounding_dir}/../{os.path.basename(f)}")
                d['preferences'][k] = freplace
        mpy.write_all_tes(d, f"{self.run_dir}/Measurement_ID.asc")

    def run_retrieval(self,
            vlidort_cli="~/muses/muses-vlidort/build/release/vlidort_cli"):
        '''Do a run of py_retrieve. Note this is a full run.

        Note OMI, but not TROPOMI now uses a separate vlidort_cli, which
        can be passed in.'''
        mpy.cli_options.vlidort_cli=vlidort_cli
        # Temporary, make sure libgfortran.so.4 is in path. See
        # https://jpl.slack.com/archives/CVBUUE5T5/p1664476320620079.
        # Note that currently omi uses muses-vlidort repository build, which
        # doesn't have this problem any longer. But tropomi does still
        old_run_dir = os.environ.get("MUSES_DEFAULT_RUN_DIR")
        old_ld_library_path = None
        if('CONDA_PREFIX' in os.environ):
            old_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
            os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib:{os.environ['LD_LIBRARY_PATH']}"
        try:
            from py_retrieve.cli import cli
            os.environ["MUSES_DEFAULT_RUN_DIR"] = os.path.abspath(self.run_dir)
            try:
                cli.main(["--targets", self.run_dir])
            except SystemExit as e:
                # cli.main always ends with throwing an exception. Sort of an odd
                # interface, but this is just the way it works. We just check
                # the exit status code.
                if(e.code != 0):
                    raise RuntimeError(f"py_retrieve run ended with exit status {e.code}")
        finally:
            if(old_run_dir):
                os.environ["MUSES_DEFAULT_RUN_DIR"] = old_run_dir
            else:
                del os.environ["MUSES_DEFAULT_RUN_DIR"]
            if(old_ld_library_path):
                os.environ["LD_LIBRARY_PATH"] = old_ld_library_path
        
    @classmethod
    def save_run_directory(cls, amuse_me_run_dir, refractor_sounding_dir):
        '''Copy data from the amuse_me run directory (e.g.,
        ~/muses/refractor-muses/muses_capture/output/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_11_23) to a sounding save directory
        (e.g., ~/muses/refractor_test_data/omi/sounding_1).
        
        We also copy any input files found in Measurement_ID.asc to the
        test in directory'''
        for f in ("Table", "Measurement_ID", "DateTime"):
            shutil.copy(f"{amuse_me_run_dir}/{f}.asc",
                        f"{refractor_sounding_dir}")
        _, d = mpy.read_all_tes(f"{amuse_me_run_dir}/Measurement_ID.asc")
        for k in ("OMI_filename", "OMI_Cloud_filename",
                  "TROPOMI_filename_BAND3",
                  "TROPOMI_IRR_filename",
                  "TROPOMI_Cloud_filename"):
            if k in d['preferences']:
                f = d['preferences'][k]
                fdest = os.path.abspath(f"{refractor_sounding_dir}/../{os.path.basename(f)}")
                if(not os.path.exists(fdest)):
                    logger.info(f"Copying {f} to {fdest}")
                    shutil.copy(f, fdest)
                else:
                    logger.info(f"{fdest} already exists")
                    
                    
