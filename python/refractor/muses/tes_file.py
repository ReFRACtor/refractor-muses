from __future__ import annotations
import collections.abc
import re
import pandas as pd
import io
import os
from functools import lru_cache
import typing
from typing import Iterator, Any, Self
from loguru import logger

if typing.TYPE_CHECKING:
    from .input_file_helper import InputFilePath


class TesFile(collections.abc.Mapping):
    """There are a number of files that are in the "TES File"
    format. This is made up of a header with keyword/value pairs and a
    (possibly empty) table.

    There is muses-py code for reading this, but this is really pretty
    straight forward so we just implement this our self. We can turn
    around and use the muse-py version if we run into any issue.

    We present the various keyword/value pairs as dictionary like
    interface.

    The attribute "table" is either None or a pandas data frame with the
    table content
    """

    def __init__(
        self,
        fname: str | os.PathLike[str] | InputFilePath,
        use_mpy: bool = False,
    ) -> None:
        """Open the given file, and read the keyword/value pairs plus
        the (possibly empty) table.

        Note that you generally shouldn't call this initializer,
        rather use TesFile.create which adds caching, so opening the
        same file twice returns the same object.

        As a convenience for testing, you can specify use_mpy as True
        to use the old mpy code. This may go away at some point, but
        for now it is useful to test that we implement the reading
        correctly.
        """
        from .input_file_helper import InputFilePath

        self.file_name = InputFilePath.create_input_file_path(fname)
        # Kind of noisy, so we don't normally log this. But can be useful occasionally to turn
        # on
        if False:
            logger.debug(f"Reading file {self.file_name}")
        if use_mpy:
            from refractor.old_py_retrieve_wrapper import muses_py_read_all_tes

            d = muses_py_read_all_tes(str(fname))
            self.mpy_d = d
            self._d = d["preferences"]
            if d["numRows"] > 0:
                tbls = f"{d['labels1']}\n" + "\n".join(d["data"])
                self.table: pd.DataFrame | None = pd.read_table(
                    io.StringIO(tbls), sep=r"\s+", header=0
                )
            else:
                self.table = None
            return

        fdata = open(str(fname)).read()

        # Make sure we find the end of the header
        if not re.search(
            r"^\s*end_of_header.*$", fdata, flags=re.MULTILINE | re.IGNORECASE
        ):
            raise RuntimeError(f"Didn't find end_of_header in file {self.file_name}")

        # Split into header and table part. Note not all files have
        # table part, but should have a header part
        t2 = re.split(
            r"^\s*end_of_header.*$", fdata, flags=re.MULTILINE | re.IGNORECASE
        )
        # Turns out there was a sample
        # (Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_TATM_H2O_N2O_CH4_HDO_BAR_LAND.asc)
        # where there is a copy of the file from a previous version at the end
        # - so we actually have multiple end_of_header markers. Seems kind of
        # hokey, we should perhaps have commented out this old data. But for
        # backwards compatibility, just ignore this.
        # if(len(t2) not in (1,2)):
        if len(t2) < 1:
            raise RuntimeError(f"Trouble parsing file {self.file_name}")
        hdr = t2[0]
        tbl: str | None = t2[1] if len(t2) >= 2 else None

        # Strip comments out
        hdr = re.sub(r"//(.*)$", "", hdr, flags=re.MULTILINE)

        # Process each line in header and fill in keyword=value data
        self._d = {}
        for ln in re.split(r"\n", hdr):
            m = re.match(r"\s*(\S*)\s*=\s*(.*\S*)\s*", ln)
            if m:  # Just skip lines that don't have keyword=value form
                # Strip off any quotes
                self._d[m[1]] = re.sub(r'"', "", m[2]).strip()

        if tbl is not None and re.search(r"\S", tbl):
            # The table has 2 header lines, the actual header and the
            # units.  pandas can actually create a table like this,
            # using a multindex.  But this is more complicated than we
            # want, so just split out the second header and treat it
            # separately
            t = tbl.lstrip().splitlines()
            hdr = t[0]
            self.table_units = t[1].split()
            body = "\n".join(t[2:])
            # Determine number of rows. The file may have extra lines
            # at the bottom, so we make sure to only read the number
            # of rows that the file claims is there.
            self.table = pd.read_table(
                io.StringIO(f"{hdr}\n{body}"), sep=r"\s+", header=0, nrows=self.shape[0]
            )
        else:
            self.table = None

    @property
    def checked_table(self) -> pd.DataFrame:
        """We often read a file that should have a table in it. While we can check
        if table is None, we do this often enough that it is nice to have checked version
        of this."""
        if self.table is None:
            raise RuntimeError(
                f"Trouble reading file {self.file_name}, expected a table but didn't find it"
            )
        return self.table

    @property
    def shape(self) -> list[int]:
        """Return the shape of the table. Note this comes from the
        metadata 'Data_Size', not the actual table.
        """
        return [int(i) for i in self["Data_Size"].split("x")]

    def __getitem__(self, ky: str) -> str:
        return self._d[ky]

    def __len__(self) -> int:
        return len(self._d)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self._d.__iter__()

    @classmethod
    def write(cls, d: dict[str, Any], fout: str | os.PathLike[str]) -> None:
        with open(fout, "w") as fh:
            for k, v in d.items():
                print(f"{k} = {v}", file=fh)
            print(
                "End_of_Header  ****  End_of_Header  ****  End_of_Header  ****  End_of_Header",
                file=fh,
            )

    # _lru_cache_wrapper is not correctly typed, we get a spurious
    # error message (see
    # https://stackoverflow.com/questions/73517571/typevar-inference-broken-by-lru-cache-decorator)
    # This might eventually get fixed - the error is in functools but
    # at least as of python 3.11.10 this is still there. Just silence
    # the error to reduce noise in the output.
    @typing.no_type_check
    @classmethod
    @lru_cache(maxsize=50)
    def _create(cls, fname: str | os.PathLike[str] | InputFilePath) -> Self:
        """This creates a TesFile that reads the given file. Because we often
        open the same file multiple times in different contexts, this adds caching so
        open the same file a second time just returns the existing TesFile object.
        """
        return cls(fname, None)

    @classmethod
    def create(cls, fname: str | os.PathLike[str] | InputFilePath) -> Self:
        return cls._create(fname)


__all__ = [
    "TesFile",
]
