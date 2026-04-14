from __future__ import annotations
import collections.abc
import os
import re
import copy
from .input_file_helper import InputFileHelper
from pathlib import Path
import subprocess
import pyaml  # type:ignore
from typing import Any, Self, Iterator


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

    def __init__(
        self,
        base_dir: str | os.PathLike[str] = ".",
        ifile_hlp: InputFileHelper | None = None,
    ) -> None:
        self._data: dict[str, Any] = {}
        # These can be updated after the object is created, e.g. after
        # have a pickle file loaded. The relative paths in the our
        # data get converted based on these values.
        self.base_dir = Path(base_dir)
        if ifile_hlp is not None:
            self.input_file_helper = ifile_hlp
        else:
            self.input_file_helper = InputFileHelper()

    def __getitem__(self, key: str) -> Any:
        # Special handling for ElanorOSPDir, which is used in create_uip by old
        # muses-py code. Doesn't fully support the InputFilePath (such as
        # InputFilePathDelta)
        if key == "ElanorOSPDir":
            return self.input_file_helper.osp_dir.path_for_muses_py
        return self.abs_dir(self._data[key])

    def __setitem__(self, key: str, val: Any) -> None:
        self._data[key] = val

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def create_from_yaml(
        cls,
        fname: str | os.PathLike[str],
        ifile_hlp: InputFileHelper | None = None,
        output_directory: str | os.PathLike[str] | None = None,
    ) -> Self:
        fnm = Path(fname)
        res = cls(
            base_dir=fnm.parent,
            ifile_hlp=ifile_hlp,
        )
        res._data = res.input_file_helper.open_yaml(fnm)["retrieval_configuration"]
        # Put in the output directory. The run_dir is in principle different, but
        # generally the same. Set to the same value
        if output_directory is not None:
            res._data["output_directory"] = Path(output_directory)
        else:
            res._data["output_directory"] = fnm.parent
        res._data["run_dir"] = res._data["output_directory"]
        return res

    @classmethod
    def create_from_strategy_file(
        cls,
        fname: str | os.PathLike[str],
        ifile_hlp: InputFileHelper | None = None,
        output_directory: str | os.PathLike[str] | None = None
    ) -> Self:
        strategy_table_fname = Path(fname).absolute()
        strategy_table_dir = strategy_table_fname.parent
        res = cls(
            base_dir=strategy_table_dir,
            ifile_hlp=ifile_hlp,
        )
        f = res.input_file_helper.open_tes(strategy_table_fname)
        res._data = dict(f)
        res["GMAO_Directory"] = str(res.input_file_helper.gmao_dir)

        # Start with default values, and then add anything we find in
        # the table Make sure to use os.path.join or pathlib.Path's /
        # operator so that if defaultStrategyTableFilename is already
        # an absolute path, we don't prepend the default directory.
        if "defaultStrategyTableDirectory" in res:
            f = res.input_file_helper.open_tes(
                res["defaultStrategyTableDirectory"]
                / res["defaultStrategyTableFilename"]
            )
            d = dict(f)
            d.update(res._data)
            res._data = d
        # Add in cloud parameters.
        if "CloudParameterFilename" in res:
            f = res.input_file_helper.open_tes(res["CloudParameterFilename"])
            res._data.update(f)
        # Add in initial guess configuration
        if "initialGuessSetupDirectory" in res:
            f = res.input_file_helper.open_tes(
                res["initialGuessSetupDirectory"] / "L2_Setup_Control_Initial.asc"
            )
            res._data.update(f)
        # For some odd reason, Single_State_Directory is not relative path like
        # most others. No idea why this is different, but set up so we can handle
        # the same way
        if "Single_State_Directory" in res:
            res["Single_State_Directory"] = str(
                Path("../OSP", res["Single_State_Directory"])
            )
        if "allTESPressureLevelsFilename" in res:
            f = res.input_file_helper.open_tes(res["allTESPressureLevelsFilename"])
            # This is the pressure levels that species information uses. This
            # is generally the initial pressure levels the forward model
            # is performed on, although these are distinct concepts. This
            # is really a column that might make sense to include in the
            # species information files, but is kept in this separate
            # file.
            res["pressure_species_input"] = list(f.checked_table["Pressure"])

        # Make run dir available
        if output_directory is not None:
            res["run_dir"] = Path(output_directory)
        else:
            res["run_dir"] = strategy_table_dir

        # This is usually the same as run_dir, but have as a separate entry (and
        # could in principle be different. We want to get away from having a run_dir
        # except for old_py_retrieve_wrapper code, so it is good to separate these
        # two ideas.
        if output_directory is not None:
            res["output_directory"] = Path(output_directory)
        elif "outputDirectory" in res and "sessionID" in res:
            res["output_directory"] = res["outputDirectory"] / res["sessionID"]

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
        if "tableOptionsFilename" in res:
            f = res.input_file_helper.open_tes(
                res["tableOptionsFilename"],
            )
            for k in f.keys():
                if k not in res:
                    raise RuntimeError(
                        f"Required option {k} is not found in the file {fname}"
                    )

        # muses-py created some derived quantities. I think we can
        # skip this, we'll at least try that for now.
        return res

    def abs_dir(self, v: Any) -> Any:
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
            m2 = re.match(r"^\$OSP/(.*)", v)
            if m:
                v = self.input_file_helper.osp_dir / m[1]
            elif m2:
                v = self.input_file_helper.osp_dir / m2[1]
            elif re.match(r"^\.\./", v) or re.match(r"^\./", v):
                v = (self.base_dir / v).resolve()
            return v
        finally:
            if t is not None:
                os.environ["strategy_table_dir"] = t
            else:
                del os.environ["strategy_table_dir"]

    def to_yaml(self, fname: str | os.PathLike[str]) -> None:
        # There are things in retrieval_configuration that are just left over and
        # not used anywhere. Go through each key and check that it is actually used
        # before include. This might catch things that just happen to have a string
        # somewhere that matches (i.e., not read from retrieval config).
        #
        # We have a list of excluded items, found in the code base but not actually
        # coming from retrieval config
        exclude_key = (
            "GMAO_Directory",
            "TES_File_ID",
            "Data_Size",
            "comment",
            "sessionID",
            "outputDirectory",
            "ElanorOSPDir",
            "tableOptionsFilename",
            "allTESPressureLevelsFilename",
            "date",
            "defaultStrategyTableFilename",
            "maxOPD",
            "note",
            "output_directory",
            "run_dir",
        )
        data_save = {}
        search_loc = Path(os.path.dirname(__file__)).parent
        for k in sorted(self._data.keys()):
            if k not in exclude_key:
                res = subprocess.run(
                    f"cd {search_loc} && git grep {k}",
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    # Useful initially for finding the exclude_key list. Keep around,
                    # might be useful again
                    if False:
                        print(f"Doing {k}", flush=True)
                        print("------------------------------------", flush=True)
                        print(res.stdout, flush=True)
                    if isinstance(self._data[k], str):
                        data_save[k] = re.sub(r"\.\./OSP", "$OSP", self._data[k])
                    else:
                        data_save[k] = self._data[k]
        with open(fname, "w") as fh:
            print(pyaml.dump({"retrieval_configuration": data_save}, indent=4), file=fh)


class AdapterRetrievalConfiguration(collections.abc.Mapping):
    """For py-test code, we need to return a str instead of InputFilePath. This
    simple adapter does that."""

    def __init__(self, rconf: RetrievalConfiguration) -> None:
        self.rconf = rconf

    def __getitem__(self, key: str) -> Any:
        return str(self.rconf[key])

    def __iter__(self) -> Iterator[str]:
        return self.rconf.__iter__()

    def __len__(self) -> int:
        return len(self.rconf)


__all__ = ["RetrievalConfiguration", "AdapterRetrievalConfiguration"]
