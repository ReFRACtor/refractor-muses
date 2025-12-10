from __future__ import annotations
import os
import io
import tarfile
from pathlib import Path


class RefractorCaptureDirectory:
    """This captures the input needed in the run directory. This is pretty
    minimal, we can perhaps remove this. This was more important when we
    were more dependent on py-retrieve (which had lots of hidden files). But
    we removed all that.

    For now, leave this in place even though this is just three small text
    files. We can perhaps remove this, creating a tar files seems like a bit
    of overkill"""

    def __init__(self) -> None:
        self.capture_directory: None | bytes = None
        self.runbase: None | Path = None
        self.rundir = Path(".")

    def save_directory(
        self,
        dirbase: str | os.PathLike[str],
    ) -> None:
        """Capture information from the run directory so we can recreate the
        directory later. This is only needed by muses-py which uses a
        lot of files as "hidden" arguments to functions.  ReFRACtor
        doesn't need this.
        """
        fh = io.BytesIO()
        dirbase = Path(dirbase).absolute()
        self.runbase = Path(dirbase.name)
        relpath = Path("./" + dirbase.name)
        with tarfile.open(fileobj=fh, mode="x:bz2") as tar:
            for f in (
                "DateTime.asc",
                "Measurement_ID.asc",
                "Table.asc",
            ):
                if f is not None and os.path.exists(dirbase / f):
                    tar.add(dirbase / f, relpath / f)
        self.capture_directory = fh.getvalue()

    def extract_directory(
        self,
        path: str | os.PathLike[str] = ".",
        change_to_dir: bool = False,
    ) -> None:
        """Extract a directory that has been previously saved.
        This gets extracted into the directory passed in the path."""
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
        self.rundir = path / self.runbase
        if change_to_dir:
            os.environ["MUSES_DEFAULT_RUN_DIR"] = str(self.rundir)
            os.chdir(self.rundir)


__all__ = [
    "RefractorCaptureDirectory",
]
