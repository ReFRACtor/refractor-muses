from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
from refractor.muses import (
    osp_setup,
    order_species,
    InstrumentIdentifier,
    FilterIdentifier,
    RetrievalType,
    CurrentState,
)
from contextlib import contextmanager
import os
import numpy as np
from pathlib import Path


class StrategyTable:
    """This wraps the existing muses-py routines working with the
    strategy table into a python object."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
    ):
        """Read the given strategy table.  Note that the strategy
        table file tends to use a lot of relative paths. We either
        assume that the directory structure is set up, changing to the
        directory of table file name. Or if the osp_dir is supplied,
        we set up a temporary directory for reading this file (useful
        for example to read a file sitting in the refractor_test_data
        directory).

        """
        self.filename = Path(filename).absolute()
        self._table_step = -1
        self.osp_dir: Path | None = Path(osp_dir) if osp_dir is not None else None
        t = os.environ.get("strategy_table_dir")
        with self.chdir_run_dir():
            try:
                os.environ["strategy_table_dir"] = str(self.filename.parent)
                self.strategy_table_dict = mpy.table_read(str(self.filename))[
                    1
                ].__dict__
            finally:
                if t is not None:
                    os.environ["strategy_table_dir"] = t
                else:
                    del os.environ["strategy_table_dir"]

    @property
    def strategy_table_obj(self):
        return mpy.ObjectView(self.strategy_table_dict)

    @property
    def analysis_directory(self) -> Path:
        return self.filename.parent / self.strategy_table_dict["dirAnalysis"]

    @property
    def elanor_directory(self) -> Path:
        return self.filename.parent / self.strategy_table_dict["dirELANOR"]

    @property
    def step_directory(self) -> Path:
        return self.filename.parent / self.strategy_table_dict["stepDirectory"]

    @property
    def input_directory(self) -> Path:
        return self.filename.parent / self.strategy_table_dict["dirInput"]

    @property
    def pressure_fm(self):
        return self.strategy_table_dict["pressureFM"]

    @property
    def preferences(self) -> dict:
        """Preferences found in the strategy table"""
        return self.strategy_table_dict["preferences"]

    @property
    def do_not_update_list(self) -> list[str]:
        do_not_update = self.table_entry("donotupdate").lower()
        if do_not_update != "-":
            do_not_update = [x.upper() for x in do_not_update.split(",")]
        else:
            do_not_update = []
        return do_not_update

    @property
    def spectral_filename(self) -> Path:
        with self.chdir_run_dir():
            return self.abs_filename(
                mpy.table_get_spectral_filename(
                    self.strategy_table_dict, self.table_step
                )
            )

    def ils_method(self, instrument_name: InstrumentIdentifier):
        if str(instrument_name) == "OMI":
            res = self.preferences["ils_omi_xsection"].upper()
        elif str(instrument_name) == "TROPOMI":
            res = self.preferences["ils_tropomi_xsection"].upper()
        else:
            raise RuntimeError("instrument_name must be either 'OMI' or 'TROPOMI'")
        # NOAPPLY is alias of POSTCONV
        if res == "NOAPPLY":
            res = "POSTCONV"
        return res

    def abs_filename(self, filename: str | os.PathLike[str]) -> Path:
        """Translate a relative path found in the StrategyTable to a
        absolute path."""
        reldir = self.osp_dir if self.osp_dir is not None else self.filename.parent
        return Path(reldir, filename)

    @contextmanager
    def chdir_run_dir(self):
        """A number of muses-py routines assume they are in the directory
        that the strategy table lives in. This gives a nice way to ensure
        that is the case. Uses this as a context manager
        """
        # If we have an osp_dir, then set up a temporary directory with the OSP
        # set up.
        # TODO Would be nice to remove this. I think we'll need to look into
        # py-retrieval to do this, but this temporary directory is really an
        # kludge - it would be nice to handle relative paths in the strategy
        # table directly.
        if self.osp_dir is not None:
            with osp_setup(osp_dir=self.osp_dir):
                yield
        else:
            # Otherwise we assume that this is in a run directory
            curdir = os.getcwd()
            try:
                os.chdir(os.path.dirname(self.filename))
                yield
            finally:
                os.chdir(curdir)

    @property
    def table_step(self):
        return self._table_step

    @table_step.setter
    def table_step(self, v):
        with self.chdir_run_dir():
            self._table_step = v
            mpy.table_set_step(self.strategy_table_dict, self._table_step)

    @property
    def number_table_step(self):
        return self.strategy_table_dict["numRows"]

    def is_done(self):
        """Return True if we are at the end of the table."""
        return self._table_step >= self.number_table_step

    def next_step(self, current_state: CurrentState):
        """Go to next step. We take the CurrentState in so we can handle any
        conditional steps based on the state."""
        self._handle_bt(current_state)
        self.table_step = self.table_step + 1

    def _handle_bt(self, current_state: CurrentState):
        # We may introduce more complicated conditional steps, but at this
        # point the only thing that gets this treatment is BT steps.
        if (
            self.retrieval_type != RetrievalType("BT")
            or self.is_next_bt()
            or self.table_step not in current_state.brightness_temperature_data
        ):
            return
        species_igr = current_state.brightness_temperature_data[self.table_step][
            "species_igr"
        ]
        found = False
        available = ""
        step = self.table_step
        istep = step + 1
        while not self.is_done():
            self.table_step = istep
            if self.retrieval_type != RetrievalType("bt_ig_refine"):
                break
            relem = ",".join(self.retrieval_elements())
            if relem != species_igr:
                available = available + relem + "   "
                mpy.table_delete_row(self.strategy_table_dict, istep)
                istep = istep - 1
            else:
                found = True
            istep += 1

        if not found and species_igr is not None:
            raise RuntimeError(f"""Specified IG refinement not found (MUST be retrievalType BT_IG_Refine AND species listed in correct order).
   Expected retrieved species: {species_igr}
   Available from table:       {available}""")

        output_filename = f"{self.output_directory}/Table-final.asc"
        mpy.table_write(self.strategy_table_dict, output_filename)
        self.strategy_table_dict = mpy.table_read(output_filename)[1].__dict__
        self.table_step = step

    @property
    def step_name(self):
        return mpy.table_get_entry(
            self.strategy_table_dict, self.table_step, "stepName"
        )

    @property
    def max_num_iterations(self) -> int:
        return mpy.table_get_entry(
            self.strategy_table_dict, self.table_step, "maxNumIterations"
        )

    @property
    def output_directory(self) -> Path:
        return self.filename.parent / self.strategy_table_dict["outputDirectory"]

    @property
    def species_directory(self) -> Path:
        return self.abs_filename(self.preferences["speciesDirectory"])

    @property
    def error_species(self):
        return self.strategy_table_dict["errorSpecies"]

    @property
    def number_fm_levels(self):
        return int(self.preferences["num_FMLevels"])

    @property
    def error_map_type(self):
        return self.strategy_table_dict["errorMaptype"]

    @property
    def retrieval_type(self) -> RetrievalType:
        return RetrievalType(self.table_entry("retrievalType"))

    def is_next_bt(self):
        """This is a bit awkward, but it isn't clear exactly how we should get
        this. Used In RetrievalStrategyStepBT, we should replace this once working
        out what the interface to StrategyTable should be."""
        return self.table_entry("retrievalType", self.table_step + 1) == "BT"

    def retrieval_elements(self, stp=None):
        """This is the retrieval elements for the given step, defaulting to
        self.table_step if not specified.

        The data is returned ordered by order_species, because some of the
        muses-py code expects that."""
        # The muses-py kind of has an odd convention for an empty list here.
        # Use this convention, and just translate this to an empty list
        r = mpy.table_get_unpacked_entry(
            self.strategy_table_dict,
            stp if stp is not None else self.table_step,
            "retrievalElements",
        )
        r = mpy.flat_list(r)
        if r[0] in ("-", ""):
            return []
        return order_species(r)

    @property
    def retrieval_elements_all_step(self):
        """All the retrieval elements found in any of the steps."""
        # table_get_all_values only includes muses-py species list. So we can
        # just generate this by going through all the steps
        # return mpy.table_get_all_values(self.strategy_table_dict, 'retrievalElements')
        res = set()
        for i in range(self.number_table_step):
            res.update(set(self.retrieval_elements(i)))
        res.discard("")
        return order_species(list(res))

    def error_analysis_interferents(self, stp=None):
        """Interferent species/StateElement used in error analysis for the given
        step (defaults to self.table_step).

        The data is returned ordered by order_species, because some of the
        muses-py code expects that."""
        # The muses-py kind of has an odd convention for an empty list here.
        # Use this convention, and just translate this to an empty list
        r = mpy.table_get_unpacked_entry(
            self.strategy_table_dict,
            stp if stp is not None else self.table_step,
            "errorAnalysisInterferents",
        )
        r = mpy.flat_list(r)
        if r[0] in ("-", ""):
            return []
        return order_species(r)

    @property
    def error_analysis_interferents_all_step(self):
        """All the interferent species found in any of the steps."""
        # table_get_all_values only includes muses-py species list. So we can
        # just generate this by going through all the steps
        # return mpy.table_get_all_values(self.strategy_table_dict, 'errorAnalysisInterferents')
        res = set()
        for i in range(self.number_table_step):
            res.update(set(self.error_analysis_interferents(i)))
        res.discard("")
        return order_species(list(res))

    def spectral_window(
        self, instrument_name: InstrumentIdentifier, stp=None, all_step=False
    ):
        """This creates a rf.SpectralWindowRange for the given instrument and
        step (defaults to self.table_step). Note that a SpectralWindow has a number of
        microwindows associated with it - RefRACtor doesn't really distinguish this and
        just uses the whole SpectralWindow to choose which frequencies pass the SpectralWindow.

        This doesn't include bad sample masking, although that can be added to the
        rf.SpectralWindowRange returned."""
        # Not sure to handle this in a generic way. For some instruments we consider
        # different filters as different sensor_index and for others we don't. For
        # now just hardcode this, and we can perhaps look for a way to handle this in
        # the future
        if str(instrument_name) in ("OMI", "TROPOMI"):
            different_filter_different_sensor_index = True
        else:
            different_filter_different_sensor_index = False

        filter_list = self.filter_list(instrument_name)
        if not different_filter_different_sensor_index:
            filter_list = [
                None,
            ]
        mwall = [
            mw
            for mw in self.microwindows(stp=stp, all_step=all_step)
            if mw["instrument"] == str(instrument_name)
        ]
        nmw = []
        for flt in filter_list:
            mwlist = [mw for mw in mwall if mw["filter"] == str(flt) or flt is None]
            nmw.append(len(mwlist))
        mw_range = np.zeros((len(filter_list), max(nmw), 2))
        for i, flt in enumerate(filter_list):
            mwlist = [mw for mw in mwall if mw["filter"] == str(flt) or flt is None]
            for j, mw in enumerate(mwlist):
                mw_range[i, j, 0] = mw["start"]
                mw_range[i, j, 1] = mw["endd"]
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        return rf.SpectralWindowRange(mw_range)

    def filter_list(self, instrument_name: InstrumentIdentifier):
        """Not sure if the order matters here or not. We keep the same order this appears in
        table_new_mw_from_all_steps. We can revisit this if needed. I'm also not sure how important
        the filter list is."""
        return [
            FilterIdentifier(i)
            for i in dict.fromkeys(
                [
                    m["filter"]
                    for m in self.microwindows(all_step=True)
                    if m["instrument"] == str(instrument_name)
                ]
            )
        ]

    def filter_list_all(self):
        """Dict with all the instruments to the filter_list for that instrument."""
        res = {}
        for instrument_name in self.instrument_name(all_step=True):
            res[instrument_name] = self.filter_list(instrument_name)
        return res

    def spectral_window_all(self, stp=None, all_step=False):
        """dict of spectral_window for each instrument in the given step
        (self.table_step if not supplied), or all steps if all_step is True"""
        swin = {}
        for iname in self.instrument_name(stp=stp, all_step=all_step):
            swin[iname] = self.spectral_window(iname, stp=stp, all_step=all_step)
        return swin

    def instrument_name(self, stp=None, all_step=False):
        """The list of instruments for the given step, used self.table_step if not supplied.
        If all_step is True, then we return the list of instruments from all retrieval steps."""
        return [
            InstrumentIdentifier(i)
            for i in dict.fromkeys(
                [m["instrument"] for m in self.microwindows(stp=stp, all_step=all_step)]
            )
        ]

    def microwindows(self, stp=None, all_step=False):
        """Microwindows for the given step, used self.table_step if not supplied.
        If all_step is True, then we return table_new_mw_from_all_steps instead."""
        with self.chdir_run_dir():
            if all_step:
                res = mpy.table_new_mw_from_all_steps(self.strategy_table_dict)
            else:
                res = mpy.table_new_mw_from_step(
                    self.strategy_table_dict,
                    stp if stp is not None else self.table_step,
                )
        # This doesn't seem to actually be called in py-retrieve. Not sure if this is
        # important or not, we'll comment this out to make a note that we aren't doing this.
        # res = mpy.mw_combine_overlapping(res, self.apodization_threshold)
        return res

    def table_entry(self, nm, stp=None):
        return mpy.table_get_entry(
            self.strategy_table_dict, stp if stp is not None else self.table_step, nm
        )

    @property
    def apodization_threshold(self):
        res = self.preferences["apodizationWindowCombineThreshold"]
        return int(res.split()[0])


__all__ = [
    "StrategyTable",
]
