from __future__ import annotations
import os
import io
import tarfile
from pathlib import Path


class RefractorCaptureDirectory:
    """muses-py code requires a number of files in a directory,
    these are essentially like hidden arguments to various muses-py
    functions.  If you want to be able to run muses-py code
    (e.g., MusesTropomiForwardModel) then we need to save the directory
    when we capture runs, and extract it again to run the data again.
    This class handles this, wrapping everything up."""

    def __init__(self) -> None:
        self.capture_directory: None | bytes = None
        self.runbase: None | Path = None
        self.rundir = Path(".")

    def save_directory(
        self,
        dirbase: str | os.PathLike[str],
        vlidort_input: str | os.PathLike[str] | None,
    ) -> None:
        """Capture information from the run directory so we can recreate the
        directory later. This is only needed by muses-py which uses a
        lot of files as "hidden" arguments to functions.  ReFRACtor
        doesn't need this.
        """
        fh = io.BytesIO()
        dirbase = Path(dirbase).absolute()
        self.runbase = Path(dirbase.name)
        # TODO This is probably too OMI specific
        osp_src_path = dirbase.parent / "OSP/OMI/"
        relpath = Path("./" + dirbase.name)
        relpath2 = Path("./OSP/OMI")
        with tarfile.open(fileobj=fh, mode="x:bz2") as tar:
            for f in (
                "DateTime.asc",
                "Measurement_ID.asc",
                "Table.asc",
                "Table-final.asc",
                "RamanInputs",
                "Input",
                vlidort_input,
            ):
                if f is not None and os.path.exists(dirbase / f):
                    tar.add(dirbase / f, relpath / f)
            for f in ("omi_rtm_driver", "ring", "ring_cli", "rayTable-NADIR.asc"):
                tar.add(osp_src_path / f, relpath2 / f)
            tar.add(
                osp_src_path / "OMI_Solar/omisol_v003_avg_nshi_backup.h5",
                relpath2 / "OMI_Solar/omisol_v003_avg_nshi_backup.h5",
            )
        self.capture_directory = fh.getvalue()

    def extract_directory(
        self,
        path: str | os.PathLike[str] = ".",
        change_to_dir: bool = False,
        osp_dir: str | os.PathLike[str] | None = None,
        gmao_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """Extract a directory that has been previously saved.
        This gets extracted into the directory passed in the path. You can
        optionally change into the run directory.

        For pretty much everything below run_retrieval, the small OSP content
        we have stashed is sufficient to run. But for higher level functions,
        you need the full OSP directory. We don't carry this in this class,
        but if you supply a osp_dir we use that instead of the OSP we have
        stashed."""
        if self.runbase is None:
            raise RuntimeError("First need to save a directory before extracting it")
        path = Path(path).absolute()
        if self.capture_directory is None:
            raise RuntimeError(
                "extract_directory can only be called if this object previously captured a directory"
            )
        fh = io.BytesIO(self.capture_directory)
        with tarfile.open(fileobj=fh, mode="r:bz2") as tar:
            tar.extractall(path=path)
        if osp_dir is not None:
            (path / "OSP").rename(path / "OSP_not_used")
            (path / "OSP").symlink_to(osp_dir)
        if gmao_dir is not None:
            (path / "GMAO").symlink_to(gmao_dir)
        self.rundir = path / self.runbase
        if change_to_dir:
            os.environ["MUSES_DEFAULT_RUN_DIR"] = str(self.rundir)
            os.chdir(self.rundir)


__all__ = [
    "RefractorCaptureDirectory",
]
