from __future__ import annotations
from .tes_file import TesFile
import shutil
from loguru import logger
import subprocess
import os
from pathlib import Path


class MusesRunDir:
    """This provides a bit of support for copying a run directory
    from an amuse-me run to refractor_test_data, and for copying that
    data into a scratch area. This handles copying all of the input
    files also. This can then be used for running a full retrieval.
    Note that this is not at all necessary for ReFRACtor, only for doing
    a full py-retrieval call.
    """

    def __init__(
        self,
        refractor_sounding_dir: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str],
        gmao_dir: str | os.PathLike[str],
        path_prefix: str | os.PathLike[str] = ".",
        skip_sym_link: bool = False,
        skip_obs_link: bool = False,
    ) -> None:
        """Set up a run directory in the given path_prefix with the
        data saved in a sounding 1 save directory (e.g.,
        ~/muses/refractor_test_data/omi/sounding_1).

        This handles updating the paths in Measurement_ID to the data
        saved in the test/in directory"""
        refractor_sounding_dir = Path(refractor_sounding_dir).absolute()
        path_prefix = Path(path_prefix).absolute()
        sid = open(refractor_sounding_dir / "sounding.txt").read().rstrip()
        self.run_dir = path_prefix / f"{sid}"
        subprocess.run(["mkdir", "-p", str(self.run_dir)])
        if not skip_sym_link:
            (path_prefix / "OSP").symlink_to(osp_dir)
            (path_prefix / "GMAO").symlink_to(gmao_dir)
        for f in ("Table", "DateTime"):
            shutil.copy(refractor_sounding_dir / f"{f}.asc", self.run_dir / f"{f}.asc")
        if not skip_obs_link:
            for f2 in refractor_sounding_dir.glob("*_obs.pkl"):
                (self.run_dir / f2.name).symlink_to(f2)
        for f in ("PRECONV_2STOKES", "rayTable-NADIR", "observationTable-NADIR"):
            if (refractor_sounding_dir / f"{f}.asc").exists():
                shutil.copy(
                    refractor_sounding_dir / f"{f}.asc", self.run_dir / f"{f}.asc"
                )
        d = TesFile(refractor_sounding_dir / "Measurement_ID.asc")
        dout = dict(d)
        for k in (
            "AIRS_filename",
            "OMI_filename",
            "OMI_Cloud_filename",
            "CRIS_filename",
            "TES_filename_L2",
            "TES_filename_L1B",
            "OCO2_filename",
            "OCO2_filename_l1b",
            "TROPOMI_filename_BAND3",
            "TROPOMI_filename_BAND7",
            "TROPOMI_filename_BAND8",
            "TROPOMI_IRR_filename",
            "TROPOMI_IRR_SIR_filename",
            "TROPOMI_Cloud_filename",
        ):
            if k in d:
                f2 = Path(d[k])
                # If this starts with a ".", assume we want a file in the sounding director.
                # otherwise we want the one in the input directory.
                if f2.parent == Path("."):
                    freplace = refractor_sounding_dir / f2.name
                else:
                    freplace = refractor_sounding_dir.parent / f2.name
                # Special handling for CRIS_filename, it uses the
                # string nasa_fsr normally found in the path to
                # know the type of file. Since we are mucking with the
                # path and removing the nasa_fsr directory this breaks
                # muses-py. We work around this by embedding this string
                # in the file name - this is enough to satisfy the
                # logic in muses-py for determining the file type.
                # refractor_test_data already has the file
                # available with this additional piece in the name -
                # we manually added a symbolic link with this name.
                if k == "CRIS_filename":
                    freplace = refractor_sounding_dir.parent / f"nasa_fsr_{f2.name}"
                dout[k] = str(freplace)
        TesFile.write(dout, str(self.run_dir / "Measurement_ID.asc"))

    def run_retrieval(
        self,
        debug: bool = False,
        plots: bool = False,
    ) -> None:
        """Do a run of py_retrieve. Note this is a full run."""
        from refractor.muses_py_fm import muses_py_call

        with muses_py_call(self.run_dir):
            from py_retrieve.cli import cli  # type: ignore

            try:
                arg = [
                    "--targets",
                    str(self.run_dir),
                    # "--vlidort-cli",
                    # str(vlidort_cli),
                ]
                if debug:
                    arg.append("--debug")
                if plots:
                    arg.append("--plots")
                cli.main(arg)
            except SystemExit as e:
                # cli.main always ends with throwing an exception. Sort of an odd
                # interface, but this is just the way it works. We just check
                # the exit status code.
                if e.code != 0:
                    raise RuntimeError(
                        f"py_retrieve run ended with exit status {e.code}"
                    )

    @classmethod
    def save_run_directory(
        cls,
        amuse_me_run_dir: str | os.PathLike[str],
        refractor_sounding_dir: str | os.PathLike[str],
    ) -> None:
        """Copy data from the amuse_me run directory (e.g.,
        ~/muses/refractor-muses/muses_capture/output/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_11_23) to a sounding save directory
        (e.g., ~/muses/refractor_test_data/omi/sounding_1).

        We also copy any input files found in Measurement_ID.asc to the
        test in directory"""
        refractor_sounding_dir = Path(refractor_sounding_dir).absolute()
        amuse_me_run_dir = Path(amuse_me_run_dir)
        for f in ("Table", "Measurement_ID", "DateTime"):
            shutil.copy(amuse_me_run_dir / f"{f}.asc", refractor_sounding_dir)
        d = TesFile(amuse_me_run_dir / "Measurement_ID.asc")
        for k in (
            "AIRS_filename",
            "OMI_filename",
            "OMI_Cloud_filename",
            "CRIS_filename",
            "TES_filename_L2",
            "TES_filename_L1B",
            "OCO2_filename",
            "OCO2_filename_l1b",
            "TROPOMI_filename_BAND3",
            "TROPOMI_filename_BAND7",
            "TROPOMI_filename_BAND8",
            "TROPOMI_IRR_filename",
            "TROPOMI_IRR_SIR_filename",
            "TROPOMI_Cloud_filename",
        ):
            if k in d:
                f2 = Path(d[k])
                fdest = refractor_sounding_dir.parent / f2.name
                if not fdest.exists():
                    logger.info(f"Copying {f2} to {fdest}")
                    shutil.copy(f2, fdest)
                else:
                    logger.info(f"{fdest} already exists")
