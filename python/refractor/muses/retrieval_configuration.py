from __future__ import annotations
import collections.abc
import os
import re
import copy
from .tes_file import TesFile
from pathlib import Path


class RetrievalConfiguration(collections.abc.MutableMapping):
    """There are a number of configuration parameters, e.g. directory
    for various outputs, run parameters like vlidort_nstokes etc.

    This class is little more than a dict which handles these
    values. Note that the "canonical" way to gets these values is to
    read a muses-py strategy table file.  However it can be useful for
    testing to just set these values, or for some special test to just
    override something in a strategy table file after reading it. So
    we separate this configuration from a strategy table file - while
    most of the time you'll read one of these the rest of the code
    doesn't make any assumption about this.

    I'm not sure how to best capture what values are expected, since
    the list seems to be a bit dynamic (e.g., if you are using ILS
    then things like "apodizationMethodObs" are needed - otherwise
    not). At least for now, we make no assumption in this class that
    any particular value is here, you simply get an error if a value
    is looked for and not found.

    Note that a strategy table file tends to use relative paths from
    wherever the file is located. We translate these to absolute paths
    so you don't need to assume that you are in the same directory as
    the strategy table file.

    We leave path/filename results as str. This is different then our
    normal convention of using Path, but because this dict may get
    passed to py-retrieve code we need to leave these as str
    (py-retrieve expected str, not Path objects).

    """

    def __init__(self, base_dir: str | Path = ".", osp_dir: str | Path | None = None):
        self._data = {}
        # These can be updated after the object is created, e.g. after
        # have a pickle file loaded. The relative paths in the our
        # data get converted based on these values.
        self.base_dir = Path(base_dir)
        self.osp_dir = osp_dir
        if self.osp_dir is not None:
            self.osp_dir = Path(self.osp_dir)

    def __getitem__(self, key):
        return self._abs_dir(self._data[key])

    def __setitem__(self, key, val):
        self._data[key] = val

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    @classmethod
    def create_from_strategy_file(
        cls, fname: str | Path, osp_dir: str | Path | None = None
    ):
        strategy_table_fname = Path(fname).absolute()
        strategy_table_dir = strategy_table_fname.parent
        res = cls(base_dir=strategy_table_dir, osp_dir=osp_dir)
        f = TesFile.create(strategy_table_fname)
        res._data = dict(f)

        # Start with default values, and then add anything we find in
        # the table Make sure to use os.path.join or pathlib.Path's /
        # operator so that if defaultStrategyTableFilename is already
        # an absolute path, we don't prepend the default directory.
        f = TesFile.create(
            Path(
                res["defaultStrategyTableDirectory"],
                res["defaultStrategyTableFilename"],
            )
        )
        d = dict(f)
        d.update(res._data)
        res._data = d
        # Add in cloud parameters.
        f = TesFile.create(res["CloudParameterFilename"])
        res._data.update(f)
        f = TesFile.create(res["allTESPressureLevelsFilename"])
        # This is the pressure levels that species information. This
        # is generally the initial pressure levels the forward model
        # is performed on, although these are distinct concepts. This
        # is really a column that might make sense to include in the
        # species information files, but is kept in this separate
        # file.
        res["pressure_species_input"] = list(f.table["Pressure"])

        # There really should be a liteDirectory included here, but
        # for some reason muses-py treats this differently as a hard
        # coded value - probably the general problem of always solving
        # problems locally rather than the best way.
        #
        # Go ahead and put into the data if it isn't there so we can treat this the
        # same everywhere.
        if "liteDirectory" not in res._data:
            res._data["liteDirectory"] = "../OSP/Lite/"

        # Similar for omiSolarReference
        if "omiSolarReference" not in res._data:
            res._data["omiSolarReference"] = (
                "../OSP/OMI/OMI_Solar/omisol_v003_avg_nshi_backup.h5"
            )

        # There is a table included in the strategy table file that
        # lists the required options. Note sure if this is complete,
        # but if we are missing one of these then muses-py marks this
        # as a failure
        f = TesFile.create(res["tableOptionsFilename"])
        for k in f.keys():
            if k not in res:
                raise RuntimeError(
                    f"Required option {k} is not found in the file {fname}"
                )

        # muses-py created some derived quantities. I think we can
        # skip this, we'll at least try that for now.
        return res

    def _abs_dir(self, v):
        """Convert values like ../OSP to the osp_dir passed in. Expand
        user ~ and environment variables. Convert relative paths to
        absolute paths.

        """
        # Skip if not something like a str
        if not isinstance(v, str):
            return v
        t = os.environ.get("strategy_table_dir")
        v = copy.copy(v)
        try:
            os.environ["strategy_table_dir"] = str(self.base_dir)
            v = os.path.expandvars(os.path.expanduser(v))
            m = re.match(r"^\.\./OSP/(.*)", v)
            if m and self.osp_dir:
                v = f"{self.osp_dir}/{m[1]}"
            if re.match(r"^\.\./", v) or re.match(r"^\./", v):
                v = os.path.normpath(f"{self.base_dir}/{v}")
            # Note, we leave path and filenames as str rather than
            # converting to Path.  This is different then our normal
            # convention, but the RetrievalConfiguration may get
            # passed to py-retrieve code which assumes str instead of
            # Path.
            return v
        finally:
            if t is not None:
                os.environ["strategy_table_dir"] = t
            else:
                del os.environ["strategy_table_dir"]


__all__ = [
    "RetrievalConfiguration",
]
