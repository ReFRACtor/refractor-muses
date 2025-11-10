from __future__ import annotations
from .tes_file import TesFile
from .filter_metadata import FilterMetadata, DictFilterMetadata
import refractor.framework as rf  # type: ignore
import copy
import numpy as np
import os
from pathlib import Path
from typing import Any, Self
import typing
from .identifier import (
    InstrumentIdentifier,
    FilterIdentifier,
    StateElementIdentifier,
    RetrievalType,
)
from .mpy import (
    mpy_table_get_spectral_filename,
    mpy_table_new_mw_from_step,
    mpy_radiance_get_indices,
)

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation


class MusesSpectralWindow(rf.SpectralWindow):
    """The refractor retrieval just uses normal SpectralWindow (e.g.,
    a SpectralWindowRange).  However there are places where it wants
    1) the data restricted to microwindows but including bad pixels
    (which are otherwise removed with the normal SpectralWindow) and
    2) the full data (referred to as full band).

    This class adds support for this. It wraps around an existing
    SpectralWindow and adds flags that can be set to
    "include_bad_data" or "full_band".

    In addition, the old muses-py code names each of the microwindows
    with a "filter" name. This is similar to, but not the same as the
    sensor index that refractor.framework uses. A particular sensor
    index value may have more than one filter assigned to it.

    The use of the filter name is fairly limited. It is used as metadata in
    the output files, the set of input data read by the observation
    (i.e., avoid reading the full L1B file and just read the data that
    will be used later in a retrieval), and as an index for other
    metadata.

    But we go ahead and include the filter name in the
    MusesSpectralWindow. If you have a new instrument that doesn't
    have filter names, this can be given the value of None.

    There is some additional metadata included in the
    microwindows. The metadata data has two flavors, the filter level
    metadata and microwindow level (in general a given filter has
    multiple microwindows). The filter level metadata is handled a
    separate class FilterMetadata which handles determining the
    metadata, see that class for a description of this. This separate
    class is used by function "muses_microwindows" to get the metadata
    to include.

    The microwindow metadata is filter name (already discussed), the
    RT, and the species list. Note we don't actually use the RT to
    control which radiative transfer code is used, this is just
    metadata passed to the UIP. We pass the RT and species list to the
    constructor.

    In all cases, the metadata can be specified as None. So new
    instruments don't need to make up data here, this metadata is only
    need for ForwardModels that use the old muses-py UIP structure. At
    least currently none of the metadata is used by refractor code,
    this really is only needed to generate the UIP.

    """

    def __init__(
        self,
        spec_win: rf.SpectralWindowRange | None,
        obs: MusesObservation | None,
        raman_ext: float = 3.01,
        instrument_name: InstrumentIdentifier | None = None,
        filter_metadata: FilterMetadata | None = None,
        filter_name: np.ndarray | None = None,
        rt: np.ndarray | None = None,
        species_list: np.ndarray | None = None,
    ) -> None:
        """Create a MusesSpectralWindow. The passed in spec_win should
        *not* have bad samples removed. We get the bad sample for the
        obs passed in and add it to spec_win.

        Right now we only work with a SpectralWindowRange using it's
        bad_sample_mask. We could extend this if needed - we really
        just need a SpectralWindow that we can create two versions - a
        with and without bad sample. But for right now, restrict
        ourselves to SpectralWindowRange.

        As a convenience spec_win and obs can be passed in as None -
        this is like always having a full_band. This is the same as
        having no SpectralWindow, but it is convenient to just always
        have a SpectralWindow so code doesn't need to have a special
        case.

        In addition to the normal spectral window, there is the one
        used in the RamanSioris calculation. This is a widened range,
        with the raman_ext added to each end. Note in the py-retrieve
        code the widening is hard coded to 3.01 nm.

        There is additional metadata that the muses-py microwindow
        structure has. I'm not sure how much of this is actually used,
        but for now we'll keep all this metadata as auxiliary
        information so we can create the microwindows from a
        MusesSpectralWindow. The data is split between a
        FilterMetadata and information in the table read for the
        filter_name, RT listed in the file (which doesn't actually
        control the RT we use, this is just metadata in the file) and
        the Species.

        """
        super().__init__()
        self.instrument_name = instrument_name
        if filter_metadata is None:
            self.filter_metadata: FilterMetadata = DictFilterMetadata(metadata={})
        else:
            self.filter_metadata = filter_metadata
        # Either take the values passed in, or fill in dummy values for this
        # metadata
        self.filter_name: np.ndarray | None = None
        self.rt: np.ndarray | None = None
        self.species_list: np.ndarray | None = None
        for v, sv in (
            (filter_name, "filter_name"),
            (rt, "rt"),
            (species_list, "species_list"),
        ):
            if v is None:
                if spec_win is not None:
                    setattr(
                        self,
                        sv,
                        np.full(
                            spec_win.range_array.value.shape,
                            None,
                            dtype=np.dtype(object),
                        ),
                    )
            else:
                setattr(self, sv, v)

        self.include_bad_sample = False
        self.full_band = False
        self.do_raman_ext = False
        self._raman_ext = raman_ext
        # Let bad samples pass through
        self._spec_win_with_bad_sample = spec_win
        if spec_win is not None:
            swin = copy.deepcopy(spec_win)
            if obs is not None:
                # Remove bad samples
                for i in range(obs.num_channels):
                    swin.bad_sample_mask(obs.bad_sample_mask(i), i)
            self._spec_win = swin
            d = spec_win.range_array.value
            draman_ext = np.zeros_like(d)
            for i in range(d.shape[0]):
                for j in range(d.shape[1]):
                    if d[i, j, 1] > d[i, j, 0]:
                        draman_ext[i, j, 0] = d[i, j, 0] - self._raman_ext
                        draman_ext[i, j, 1] = d[i, j, 1] + self._raman_ext
            draman_ext = rf.ArrayWithUnit_double_3(draman_ext, rf.Unit("nm"))
            self._spec_win_raman_ext = rf.SpectralWindowRange(draman_ext)
        else:
            self._spec_win = None
            self._spec_win_raman_ext = None

    def _v_number_spectrometer(self) -> int:
        return self._spec_win.number_spectrometer

    def _v_spectral_bound(self) -> rf.SpectralBound:
        return self._spec_win.spectral_bound

    def add_bad_sample_mask(self, obs: MusesObservation) -> None:
        """We have a bit of a chicken and an egg problem. We need the MusesSpectralWindow
        before we create the MusesObservation, but we need the MusesObservation to
        add the bad samples in. So we do this after the creation, when the obs is
        created."""
        if self._spec_win:
            for i in range(obs.num_channels):
                self._spec_win.bad_sample_mask(obs.bad_sample_mask(i), i)

    def desc(self) -> str:
        return "MusesSpectralWindow"

    def filter_name_list(self) -> list[tuple[FilterIdentifier, float]]:
        """This returns the list of the filter names in this MusesSpectralWindow, and the
        start wavenumber. This is needed because muses-py assumes the filter names are sorted
        by the start wavenumber."""
        res: list[tuple[FilterIdentifier, float]] = []
        if self.filter_name is None:
            return res
        d = self._spec_win.range_array.value
        for i in range(self.filter_name.shape[0]):
            for j in range(self.filter_name.shape[1]):
                res.append((FilterIdentifier(self.filter_name[i, j]), d[i, j, 0]))
        return res

    def grid_indexes(self, grid: rf.SpectralDomain, spec_index: int) -> list[int]:
        if self._spec_win is None or self.full_band:
            return list(range(grid.data.shape[0]))
        if self.do_raman_ext:
            return self._spec_win_raman_ext.grid_indexes(grid, spec_index)
        if self.include_bad_sample:
            if self._spec_win_with_bad_sample is not None:
                return self._spec_win_with_bad_sample.grid_indexes(grid, spec_index)
        return self._spec_win.grid_indexes(grid, spec_index)

    def muses_monochromatic(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """In certain places, muses-py uses a "monochromatic" list of
        points, along with a wavelength filter. This seems to serve
        much the same function as our high resolution grid in
        ReFRACtor, although this doesn't filter out bad points or
        anything like that.

        ReFRACtor doesn't directly use this, but it does get passed
        into the muses-py function calls such as getting the ILS
        information. So we go ahead and have this calculation here,
        much like we do the muses_microwindows down below.

        It is possible this can go away at some point, right now we
        only need this for muses-py calls, and if these get removed or
        replaced the need to for this function may go away.

        """
        mono_list = []
        mono_filter_list = []
        mono_list_length = []
        for w in self.muses_microwindows():
            mw_start = w["start"]
            mw_end = w["endd"]
            mw_monospacing = w["monoSpacing"]
            mw_monoextend = np.float64(w["monoextend"])
            mw_filter = w["filter"]
            mono_temp = np.arange(
                mw_start - mw_monoextend, mw_end + mw_monoextend, mw_monospacing
            )
            mono_list.append(mono_temp)
            mono_filter_list.extend(
                [
                    mw_filter,
                ]
                * len(mono_temp)
            )
            mono_list_length.append(len(mono_temp))
        mono_listv = np.concatenate(mono_list, axis=0)
        mono_filter_listv = np.array(mono_filter_list)
        return mono_listv, mono_filter_listv, mono_list_length

    def muses_microwindows(self) -> list[dict]:
        """Return the muses-py list of dict structure used as
        microwindows. This is used in a few places, e.g., for creating
        a UIP for forward models that depend on this.

        """
        res = []
        d = self._spec_win.range_array.value
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if d[i, j, 0] < d[i, j, 1]:
                    v = {
                        "THROW_AWAY_WINDOW_INDEX": -1,
                        "start": d[i, j, 0],
                        "endd": d[i, j, 1],
                        "instrument": str(self.instrument_name),
                        "RT": self.rt[i, j]
                        if self.rt is not None and self.rt[i, j] is not None
                        else "None",
                        "filter": self.filter_name[i, j]
                        if self.filter_name is not None
                        and self.filter_name[i, j] is not None
                        else "None",
                    }
                    v2 = self.filter_metadata.filter_metadata(
                        self.filter_name[i, j] if self.filter_name is not None else None
                    )
                    # Make a copy so we can update v2 without changing anything it might
                    # point to in self.filter_metadata
                    v2 = copy.deepcopy(v2)
                    # Prefer our species_list if found, but otherwise use the
                    # one in self.filter_metadata
                    if self.species_list is not None and (
                        self.species_list[i, j] is None or self.species_list[i, j] == ""
                    ):
                        if "speciesList" in v2:
                            v["speciesList"] = v2["speciesList"]
                        else:
                            v["speciesList"] = ""
                    else:
                        v["speciesList"] = (
                            self.species_list[i, j]
                            if self.species_list is not None
                            else ""
                        )
                    # Values in both v and v2 prefer the v one based on the rules for
                    # update.
                    v2.update(v)
                    # In a truly kludgy way, mw_augment_default in py-retrieve
                    # overrides these values, for AIRS only. Duplicate this
                    # functionality.
                    if str(self.instrument_name) == "AIRS":
                        v2["maxopd"] = 0
                        v2["spacing"] = 0
                        freqs = np.array(
                            [
                                649.62000,
                                706.13702,
                                847.13702,
                                1112.01500,
                                1524.35210,
                                2181.49390,
                                2183.30590,
                                2458.54390,
                            ]
                        )
                        ind = (
                            np.searchsorted(
                                np.array(freqs) - 0.1, v["endd"], side="right"
                            )
                            - 1
                        )
                        exts = [2, 3, 4, 5, 6, 7, 8, 9]
                        v2["monoextend"] = exts[ind]
                    res.append(v2)
        return res

    @classmethod
    def filter_list_dict_from_file(
        cls, spec_fname: str | os.PathLike[str]
    ) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """Return a dictionary going from instrument name to the list
        of filters for that given instrument.

        """
        fspec = TesFile.create(spec_fname)
        res = {}
        for iname in [
            InstrumentIdentifier(i)
            for i in dict.fromkeys(fspec.table["Instrument"].to_list())
        ]:
            res[iname] = [
                FilterIdentifier(i)
                for i in dict.fromkeys(
                    fspec.table[fspec.table["Instrument"] == str(iname)][
                        "Filter"
                    ].to_list()
                )
            ]
        return res

    @classmethod
    def create_dict_from_file(
        cls,
        spec_fname: str | os.PathLike[str],
        filter_list_dict: dict[InstrumentIdentifier, list[FilterIdentifier]]
        | None = None,
        filter_metadata: FilterMetadata | None = None,
    ) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Create a dict from instrument name to MusesSpectralWindow
        from the given microwindows file name. We also take an
        optional FilterMetadata which is used for additional metadata
        in the muses_microwindows function.

        """
        res = {}
        for iname in cls.filter_list_dict_from_file(spec_fname).keys():
            # TODO - Remove this. we should have AIRS and CRIS changed
            # to act like our other observation classes and have a different
            # sensor index for each filter, so we don't need
            # special handling here.
            # Temp, until we get this to work for AIRS and CRIS
            different_filter_different_sensor_index = True
            if str(iname) in ("AIRS", "CRIS", "TES"):
                different_filter_different_sensor_index = False
            res[iname] = cls.create_from_file(
                spec_fname,
                iname,
                filter_list_all=filter_list_dict[iname]
                if filter_list_dict is not None
                else None,
                filter_metadata=filter_metadata,
                different_filter_different_sensor_index=different_filter_different_sensor_index,
            )
        return res

    @classmethod
    def create_from_file(
        cls,
        spec_fname: str | os.PathLike[str],
        instrument_name: InstrumentIdentifier,
        filter_list_all: list[FilterIdentifier] | None = None,
        filter_metadata: FilterMetadata | None = None,
        different_filter_different_sensor_index: bool = True,
    ) -> Self:
        """Create a MusesSpectralWindow for the given instrument name
        from the given microwindow file name. We also take an optional
        FilterMetadata which is used for additional metadata in the
        muses_microwindows function.

        For some instruments we consider different filters as
        different sensor_index and for others we don't. The argument
        different_filter_different_sensor_index is used to control
        this.

        Note that while we in general don't require that spectral
        channels have a filter name, the file only works with filter
        names (that is how it identifies the microwindows). We need to
        know the full list of filter names that the MusesObservation
        has, so that we can properly create SpectralWindowRange with
        the right number of spectral channels include possibly empty
        ones. The filter_list_all generally comes from the
        MeasurementId.

        """
        fspec = TesFile.create(spec_fname)
        rowlist = fspec.table[fspec.table["Instrument"] == str(instrument_name)]

        flist = list(dict.fromkeys(rowlist["Filter"].to_list()))
        # I think it is ok to have this always True, but leave this knob in
        # place for now until we determine that this isn't needed.
        if not different_filter_different_sensor_index:
            flist = [
                None,
            ]
            nmw = [len(rowlist)]
        else:
            nmw = [len(rowlist[rowlist["Filter"] == flt]) for flt in flist]
        mw_range = np.zeros(
            (
                len(filter_list_all)
                if filter_list_all is not None
                and different_filter_different_sensor_index
                else len(flist),
                max(nmw),
                2,
            )
        )
        filter_name = np.full(
            (mw_range.shape[0], mw_range.shape[1]), None, dtype=np.dtype(object)
        )
        rt = np.full(filter_name.shape, None, dtype=np.dtype(object))
        species_list = np.full(filter_name.shape, None, dtype=np.dtype(object))
        for i, flt in enumerate(flist):
            if filter_list_all is not None and flt is not None:
                ind = filter_list_all.index(FilterIdentifier(flt))
            else:
                ind = i
            if flt is None:
                mwlist = rowlist
            else:
                mwlist = rowlist[rowlist["Filter"] == flt]
            for j, mw in enumerate(mwlist.iloc):
                mw_range[ind, j, 0] = mw["WindowStart"]
                mw_range[ind, j, 1] = mw["WindowEnd"]
                filter_name[ind, j] = mw["Filter"]
                rt[ind, j] = mw["RT"]
                species_list[ind, j] = mw["Species"]
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        return cls(
            spec_win=rf.SpectralWindowRange(mw_range),
            obs=None,
            instrument_name=instrument_name,
            filter_name=filter_name,
            rt=rt,
            species_list=species_list,
            filter_metadata=filter_metadata,
        )

    @classmethod
    def muses_microwindows_from_dict(
        cls, spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow]
    ) -> list[dict]:
        """Create the muses-py microwindows list of dict structure
        from a dict going from instrument name to MusesSpectralWindow"""
        res = []
        for iname, swin in spec_win_dict.items():
            res.extend(swin.muses_microwindows())
        return res

    @classmethod
    def muses_microwindows_fname(
        cls,
        viewing_mode: str,
        spectral_window_directory: str | os.PathLike[str],
        retrieval_elements: list[StateElementIdentifier],
        step_name: str,
        retrieval_type: RetrievalType,
        spec_file: str | os.PathLike[str] | None = None,
    ) -> Path:
        """Return the muses microwindows filename. Not clear that this belongs in
        MusesSpectralWindow, but this is at least a reasonable place for this. We may
        move this.

        The intent is that this returns the same thing as
        muses_microwindows_fname_from_muses_py, without using muses-py"""

        relem = StateElementIdentifier.sort_identifier(retrieval_elements)

        vmode = viewing_mode
        if vmode.lower() in ("nadir", "limb"):
            vmode = vmode.capitalize()

        filename = Path(spectral_window_directory) / f"Windows_{vmode}"

        # skip all the logic if spec_file is specified
        if spec_file is not None:
            return Path(f"{filename}_{spec_file}.asc")

        if retrieval_type in (RetrievalType("bt"), RetrievalType("forwardmodel")):
            filename = Path(f"{filename}_{step_name}")
        else:
            # Make species part of filename of elements that are atmospheric_species
            retpart = "_".join(str(t) for t in relem if t.is_atmospheric_species)
            # If no 'line' species found, make name from all species.
            if retpart == "":
                retpart = "_".join(str(t) for t in relem)
            filename = Path(f"{filename}_{retpart}")

        if retrieval_type in (RetrievalType("default"), RetrievalType("-")):
            pass
        elif retrieval_type == RetrievalType("fullfilter"):
            filename = Path(spectral_window_directory) / f"Windows_{vmode}_{step_name}"
        elif retrieval_type == RetrievalType("bt_ig_refine"):
            retpart = "_".join(str(t) for t in relem)
            filename = (
                Path(spectral_window_directory)
                / f"Windows_{vmode}_{retpart}_BT_IG_Refine"
            )
        elif retrieval_type == RetrievalType("joint") and "TROPOMI" in step_name:
            if "wide" in step_name:
                filename = Path(f"{filename}wide_{retrieval_type}")
            elif "Band_1_2_short" in step_name:
                filename = Path(f"{filename}Band_1_2_short_{retrieval_type}")
            elif "Band_1_2" in step_name:
                filename = Path(f"{filename}Band_1_2_{retrieval_type}")
            elif "Band_2" in step_name:
                filename = Path(f"{filename}Band_2_{retrieval_type}")
            else:
                filename = Path(f"{filename}_{retrieval_type}")
        else:
            filename = Path(f"{filename}_{retrieval_type}")

        return Path(f"{filename}.asc")

    @classmethod
    def muses_microwindows_fname_from_muses_py(
        cls,
        viewing_mode: str,
        spectral_window_directory: str | os.PathLike[str],
        retrieval_elements: list[StateElementIdentifier],
        step_name: str,
        retrieval_type: RetrievalType,
        spec_file: str | os.PathLike[str] | None = None,
    ) -> str:
        """For testing purposes, this calls the old mpy.table_get_spectral_filename to
        determine the microwindow file name use. This can be used to verify that
        we are finding the right name. This shouldn't be used for real code,
        instead use the SpectralWindowHandleSet."""
        # creates a dummy strategy_table dict with the values it expects to find
        stable: dict[str, Any] = {}
        stable["preferences"] = {
            "viewingMode": viewing_mode,
            "spectralWindowDirectory": str(spectral_window_directory),
        }
        t1 = [
            ",".join([str(i) for i in retrieval_elements])
            if len(retrieval_elements) > 0
            else "-",
            step_name,
            str(retrieval_type),
        ]
        t2 = ["retrievalElements", "stepName", "retrievalType"]
        if spec_file is not None:
            t1.append(str(spec_file))
            t2.append("specFile")
        stable["data"] = [
            " ".join(t1),
        ]
        stable["labels1"] = " ".join(t2)
        stable["numRows"] = 1
        stable["numColumns"] = len(t2)
        return mpy_table_get_spectral_filename(stable, 0)

    @classmethod
    def muses_microwindows_from_muses_py(
        cls,
        default_spectral_window_fname: str,
        viewing_mode: str,
        spectral_window_directory: str | os.PathLike[str],
        retrieval_elements: list[StateElementIdentifier],
        step_name: str,
        retrieval_type: RetrievalType,
        spec_file: str | os.PathLike[str] | None = None,
    ) -> list[dict[str, Any]]:
        """For testing purposes, this calls the old mpy.table_new_mw_from_step. This can
        be used to verify that the microwindows we generate are correct. This shouldn't
        be used for real code, instead use the SpectralWindowHandleSet."""
        # Wrap arguments into format expected by table_new_mw_from_step. This
        # creates a dummy strategy_table dict with the values it expects to find
        stable: dict[str, Any] = {}
        stable["preferences"] = {
            "defaultSpectralWindowsDefinitionFilename": default_spectral_window_fname,
            "viewingMode": viewing_mode,
            "spectralWindowDirectory": str(spectral_window_directory),
        }
        t1 = [
            ",".join([str(i) for i in retrieval_elements]),
            step_name,
            str(retrieval_type),
        ]
        t2 = ["retrievalElements", "stepName", "retrievalType"]
        if spec_file is not None:
            t1.append(str(spec_file))
            t2.append("specFile")
        stable["data"] = [
            " ".join(t1),
        ]
        stable["labels1"] = " ".join(t2)
        stable["numRows"] = 1
        stable["numColumns"] = len(t2)
        return mpy_table_new_mw_from_step(stable, 0)


class TesSpectralWindow(MusesSpectralWindow):
    """TES has a fairly complicated spectral window. In addition to
    checking for ranges (which it does with a tolerance), it also
    checks that a particular microwindow matches the filter. This
    appears to be TES only, but we need to match this because
    otherwise our observation doesn't match the frequencies used in
    the OSS forward model.

    This seems overly complicated, it seems like the same behavior
    could be done with the regular MusesSpectralWindow with the edges
    of the microwindows just set correctly. But to match the behavior
    of py-retrieve we want to do the same thing here.

    We could probably duplicate this behavior but 1) it is complicated
    2) it only applies to TES, and 3) we aren't really going to be
    doing a lot of TES retrievals, this is just for backwards
    compatibility.

    So we have a special adapter here to pull in the extra behavior. This is
    pretty much just a MusesSpectralWindow, except we have extra logic in the
    grid_indexes."""

    def __init__(self, swin: MusesSpectralWindow, obs: MusesObservation) -> None:
        super().__init__(
            swin._spec_win_with_bad_sample,
            obs,
            raman_ext=swin._raman_ext,
            instrument_name=swin.instrument_name,
            filter_metadata=swin.filter_metadata,
            filter_name=swin.filter_name,
            rt=swin.rt,
            species_list=swin.species_list,
        )
        self._obs = obs

    def desc(self) -> str:
        return "TesSpectralWindow"

    def grid_indexes(self, grid: rf.SpectralDomain, spec_index: int) -> list[int]:
        # We only have the extra logic for the unchanged ranges,
        # or include_bad_sample. So just return results otherwise
        if self._spec_win is None or self.full_band or self.do_raman_ext:
            return super().grid_indexes(grid, spec_index)
        # Determine the list of grid_indexes from py-retrieve. Note that
        # this includes bad_samples.
        muses_gindex = mpy_radiance_get_indices(
            self._obs.muses_py_dict["radianceStruct"], self.muses_microwindows()
        )
        if self.include_bad_sample:
            return [int(i) for i in muses_gindex]
        # Only include indices that aren't bad samples
        good_gindex = set(
            i for i in (self._obs.bad_sample_mask(spec_index) == False).nonzero()[0]
        )
        return [int(i) for i in muses_gindex if i in good_gindex]


__all__ = ["MusesSpectralWindow", "TesSpectralWindow"]
