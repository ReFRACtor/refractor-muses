from __future__ import annotations
from .mpy import (
    have_muses_py,
    mpy_update_uip,
    mpy_script_retrieval_ms,
    mpy_get_omi_radiance,
    mpy_get_tropomi_radiance,
    mpy_atmosphere_level,
    mpy_raylayer_nadir,
    mpy_pressure_sigma,
    mpy_oco2_get_wavelength,
    mpy_nir_match_wavelength_edges,
    mpy_make_uip_master,
    mpy_make_uip_airs,
    mpy_make_uip_cris,
    mpy_make_uip_tes,
    mpy_make_uip_omi,
    mpy_make_uip_tropomi,
    mpy_make_uip_oco2,
    mpy_make_maps,
)
from refractor.muses import (
    AttrDictAdapter,
    register_replacement_function_in_block,
    RefractorCaptureDirectory,
    InstrumentIdentifier,
    FilterIdentifier,
    FakeStateInfo,
    FakeRetrievalInfo,
    RetrievalConfiguration,
    MeasurementId,
    CurrentState,
    MusesObservation,
)
from .muses_py_call import muses_py_call
import refractor.framework as rf  # type: ignore
import os
from loguru import logger
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import logging
import numpy as np
import pickle
from collections import UserDict, defaultdict
import copy
from pathlib import Path
import math
import itertools
from typing import Any, Generator, Self, cast

if have_muses_py:

    class _FakeUipExecption(Exception):
        def __init__(
            self,
            uip: AttrDictAdapter,
            ret_info: AttrDictAdapter,
            retrieval_vec: AttrDictAdapter,
        ) -> None:
            self.uip = uip
            self.ret_info = ret_info
            self.retrieval_vec = retrieval_vec

    class _CaptureUip:
        """Note a complication. For CrIS-TROPOMI we have some steps that
        don't actually call levmar_nllsq_elanor. So we get the registered
        twice, once to run_retrieval and once to levmar_nllsq_elanor. We then
        count the number of run_retrieval calls, but once the total is reached
        we replace only the *next* levmar_nllsq_elanor call."""

        def __init__(self, func_count: int = 1):
            self.func_count = func_count

        def should_replace_function(
            self, func_name: str, parms: dict[str, Any]
        ) -> bool:
            if func_name == "run_retrieval":
                self.func_count -= 1
                print(f"In run_retrieval, func_count is {self.func_count}")
            if self.func_count <= 0 and func_name == "levmar_nllsq_elanor":
                return True
            return False

        def replace_function(self, func_name: str, parms: dict[str, Any]) -> Any:
            # The UIP passed in is *before* updating with xInit. We
            # want this after the update, so call that before passing
            # value back.
            o_x_vector = parms["xInit"]
            uip = parms["uip"]
            ret_info = parms["ret_info"]
            (uip, o_x_vector) = mpy_update_uip(uip, ret_info, o_x_vector)
            raise _FakeUipExecption(uip, ret_info, o_x_vector)


@contextmanager
def _all_output_disabled() -> Generator[None, None, None]:
    """Suppress stdout, stderr, and logging"""
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()):
            with redirect_stderr(io.StringIO()):
                yield
    finally:
        logging.disable(previous_level)


class RefractorCache(UserDict):
    """We ran into an issue where run_retrieval in muses-py tries to
    do a deepcopy of the final UIP and we had a failure because
    refractor_cache can't be deepcopied (some of the object can't be
    pickled, which is what deepcopy does).

    So we use a cache object that is just dict but returns an empty
    cache when we do a deepcopy.

    Even if we fix the pickle issue, this is probably what we want to
    do anyways - the cache really is just that. If we don't have an object
    already in the cache our code just recreates it, we just use the cache
    to be able to reuse objects between calls to ReFRACtor from muses-py.

    """

    def __deepcopy__(self, memo: Any) -> RefractorCache:
        return RefractorCache()


class RefractorUip:
    """The 'uip' is a central variable in muses-py. It is a python dict
    object, which contains all the input data need to generate the
    forward model. It is largely read only, but it is updated by the
    state vector (in update_uip).

    This is a light wrapper in ReFRACtor for working with the UIP.

    This class should be considered mostly deprecated. It creates a
    tight coupling in py-retrieve, essentially we have everything depending
    on everything else through the UIP.

    We have replaced this class in ReFRACtor, and don't use it. The one
    exception is in calling the old py-retrieve forward models.
    MusesForwardModelHandle handles this, creating the UIP if we have a
    forward model that needs this but otherwise not creating it. Perhaps
    at some point we will completely drop the py-retrieve forward models
    and this can go away. But for now, we provide support for this.

    We give a number of access routines to various pieces of the UIP we
    are interested in. This 1) gives a cleaner interface and 2) protects
    somewhat from changes to the uip (so changed names just need to be
    updated here).

    Note that the UIP doesn't include the basis matrix needed to map
    from the retrieval vector to the full model vector ('z' and 'x' in
    the notation of the TES paper “Tropospheric Emission Spectrometer:
    Retrieval Method and Error Analysis” (IEEE TRANSACTIONS ON
    GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY 2006). There
    really isn't another natural place to put this, so we stash this
    matrix into this class. Depending on the call chain, this may or
    may not be available, so code should check if this is None and somehow
    handle this (including throwing an exception. We store this
    as basis_matrix

    Note that although some thing need access to muses_py (e.g.,
    create_from_table), a lot of this functionality doesn't actually
    depend on muses-py. So if we have a pickled version of this object
    or the original uip, you can do things with it w/o muses-py. This
    can be useful for example for having pytest tests that don't depend
    on having muses-py available.

    Note that there are two microwindow indexes floating around. We have
    ii_mw which goes through all the instruments, so for step 7 in
    AIRS+OMI ii_mw goes through 12 values (only 10 and 11 are OMI).
    mw_index (also call fm_idx) is relative to a instrument,
    so if we are working with OMI the first microwindow has ii_mw = 10, but
    mw_index is 0 (UV1, with the second UV2).

    It isn't 100% clear what the right interface is here, so we may modify
    this class a bit in the future.

    """

    def __init__(
        self,
        uip: dict[str, Any] | AttrDictAdapter,
        basis_matrix: np.ndarray | None = None,
        state_info: dict[str, Any] | AttrDictAdapter | FakeStateInfo | None = None,
    ) -> None:
        """Constructor. This takes the uip structure (the muses-py dictionary)
        and a basis_matrix if available
        """
        # Depending on where this is called from, uip may be a dict or
        # an ObjectView. Just to make things simpler, we always store this
        # as a dict.
        if hasattr(uip, "as_dict"):
            self.uip = uip.as_dict(uip)
        else:
            self.uip = uip
        # Thought this would be useful to make sure UIP isn't changed
        # behind the scenes. But this turns out to break a bunch of
        # stuff. We could probably eventually sort this out, but this
        # seems like a lot or work for little gain. Instead, we can
        # just check by inspection if the UIP is changed in any
        # muses-py code self.uip = ConstantDict(self.uip)
        self.basis_matrix = basis_matrix
        self.capture_directory = RefractorCaptureDirectory()
        # Note the UIP is really just a reformatting of information found
        # in the state_info. We'd like to get away from using the UIP for
        # ReFRACtor, it is a needless complication. But we aren't ready to
        # abandon the uip because all the muses-py forward models depend on this.
        # We may have time at some point to figure out how to separate this out,
        # but for now just tack the state info on so we can use this in our
        # RefractorFmObjectCreator.
        self.state_info = state_info

    def __getstate__(self) -> dict[str, Any]:
        """Pickling grabs attributes, which includes properties.
        We don't actually want that, so just explicitly list what
        we want saved."""
        return {
            "uip": self.uip,
            "basis_matrix": self.basis_matrix,
            "capture_directory": self.capture_directory,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def uip_all(self, instrument_name: InstrumentIdentifier | str) -> dict[str, Any]:
        """Add in the stuff for the given instrument name. This is
        used in a number of places in muses-py calls."""
        # Depending on where this comes from, it may or may not have the
        # uip_all stuff included.
        # 'jacobians' happens
        # to be something not in the original uip, that gets added with
        # uip_all
        if "jacobians" in self.uip:
            return self.uip
        # This is similar to just d1 | d2, but the logic is a little different. In particular,
        # the left side of duplicates is chosen rather than the right side, and some
        # items are copied depending on the type.'''
        res = {}
        for k, v in itertools.chain(
            self.uip.items(), self.uip[f"uip_{instrument_name}"].items()
        ):
            if k not in res:
                res[k] = v.copy() if type(v) in (np.ndarray, list, dict) else v
        return res

    @property
    def run_dir(self) -> Path:
        """Return run_dir for capture_directory. Note this defaults to
        "." if RefractorCaptureDirectory hasn't changed this."""
        return Path(self.capture_directory.rundir)

    @run_dir.setter
    def run_dir(self, v: Path) -> None:
        self.capture_directory.rundir = v

    @property
    def refractor_cache(self) -> RefractorCache:
        """Return a simple dict we use for caching values. Note that
        by design this really is a cache, if this is missing or
        anything in it is then we just create on first use. Note this
        is the equivalent of a "mutable" in C++ - we allow things to
        get updated in the cache in places that should otherwise want
        the UIP to be held constant.
        """
        if "refractor_cache" not in self.uip or self.uip["refractor_cache"] is None:
            self.uip["refractor_cache"] = RefractorCache()
        return self.uip["refractor_cache"]

    @classmethod
    def create_from_table(
        cls,
        strategy_table: str,
        step: int = 1,
        capture_directory: bool = False,
        save_pickle_file: str | None = None,
        suppress_noisy_output: bool = True,
    ) -> Self:
        """This creates a UIP from a run directory (e.g., created
        by earlier steps of amuse-me).  The table is passed in that
        points to everything, usually this is called 'Table.asc' in
        the run directory (e.g. ~/output_py/omi/2016-04-14/setup-targets/Global_Survey/20160414_23_394_23/Table.asc).

        In addition to a uip, the muses-py code requires a number of
        files in a directory. To allow running the
        e.g. MusesTropomiForwardModell, we can also capture
        information form the directory the strategy_table is located
        at. This is only needed for muses-py code, the ReFRACtor
        forward model doesn't need this. You can set capture_directory
        to True if you intend on using the UIP to run muses-py code.

        Because it is common to do, you can optionally supply a pickle file
        name and we'll save the uip to the pickle file after creating it.

        Note I'm not exactly sure how to extract steps other than by
        doing a full run. What we currently do is run the retrieval
        until we get to the requested step, which can potentially be
        slow. So if you request a step other than 1, be aware that it
        might take a while to generate.  But for doing things like
        generating test data this should be fine, just pickle the
        object or otherwise save it for use later. We can probably
        work out a way to do this more directly if it becomes important.
        """
        strategy_table = os.path.abspath(strategy_table)
        # We would like to just call a function in muses-py to generate
        # the UIP. Unfortunately, one doesn't exist. Instead this is
        # created inline as the processing is set up.
        #
        # We could duplicate this functionality here, but then any
        # updates to the muses-py code wouldn't show up here.
        #
        # So instead, we do a trick. We pretend like we are doing
        # a retrieval, but once the call is made to levmar_nllsq_elanor
        # we intercept this, grab the uip, and then force a return
        # by throwing an exception. This is pretty evil, an exception
        # shouldn't be used for controlling execution. But this is a
        # special case, where breaking the normal rule is the right thing.
        #
        # A better long term solution it to get muses-py to add a function
        # call.
        with muses_py_call(os.path.dirname(strategy_table)):
            try:
                cfun = _CaptureUip(func_count=step)
                with register_replacement_function_in_block("run_retrieval", cfun):
                    with register_replacement_function_in_block(
                        "levmar_nllsq_elanor", cfun
                    ):
                        # This is pretty noisy, so suppress printing. We can revisit
                        # this if needed, but I think this is a good idea
                        if suppress_noisy_output:
                            with _all_output_disabled():
                                mpy_script_retrieval_ms(
                                    os.path.basename(strategy_table)
                                )
                        else:
                            mpy_script_retrieval_ms(os.path.basename(strategy_table))
            except _FakeUipExecption as e:
                res = cls(uip=e.uip, basis_matrix=e.ret_info["basis_matrix"])
        if capture_directory:
            res.tar_directory(strategy_table)
        if save_pickle_file is not None:
            pickle.dump(res, open(save_pickle_file, "wb"))
        return res

    def tar_directory(self, strategy_table: str) -> None:
        vlidort_input = None
        if "uip_OMI" in self.uip:
            vlidort_input = self.uip["uip_OMI"]["vlidort_input"]
        if "uip_TROPOMI" in self.uip:
            vlidort_input = self.uip["uip_TROPOMI"]["vlidort_input"]
        self.capture_directory.save_directory(
            os.path.dirname(strategy_table), vlidort_input
        )

    @property
    def step_directory(self) -> Path:
        return (self.run_dir / self.vlidort_input).parent.parent

    @property
    def current_state_x(self) -> np.ndarray:
        """Return the current guess. This is the same thing as retrieval_vec,
        update_uip sets this so we know this."""
        return self.uip["currentGuessList"]

    @property
    def current_state_x_fm(self) -> np.ndarray:
        """Return the current guess for the full state model (called fm_vec
        in some places) This is the same thing as retrieval_vec @ basis_matrix
        update_uip sets this so we know this."""
        return self.uip["currentGuessListFM"]

    @property
    def vlidort_input(self) -> str:
        if self.uip_omi:
            return self.uip_omi["vlidort_input"]
        elif self.uip_tropomi:
            return self.uip_tropomi["vlidort_input"]
        else:
            raise RuntimeError("Only support omi and tropomi")

    @property
    def vlidort_output(self) -> str:
        if self.uip_omi:
            return self.uip_omi["vlidort_output"]
        elif self.uip_tropomi:
            return self.uip_tropomi["vlidort_output"]
        else:
            raise RuntimeError("Only support omi and tropomi")

    @classmethod
    def load_uip(
        cls,
        save_pickle_file: str,
        path: str = ".",
        change_to_dir: bool = False,
        osp_dir: str | None = None,
        gmao_dir: str | None = None,
    ) -> RefractorUip:
        """This is the pair to create_from_table, it loads a RefractorUip
        from a pickle file, extracts the saved directory, and optionally
        changes to that directory."""
        uip = pickle.load(open(save_pickle_file, "rb"))
        # In testing, we may have already created the directory. If it there, skip
        # setting up
        if not uip.capture_directory.runbase.exists():
            uip.capture_directory.extract_directory(
                path=path,
                change_to_dir=change_to_dir,
                osp_dir=osp_dir,
                gmao_dir=gmao_dir,
                include_osp=True,
            )
        else:
            # Side effect of extract_directory is to set run_dir to absolute path. If
            # we don't do the extraction, we still need to set the run_dir.
            uip.run_dir = (
                uip.capture_directory.rundir / uip.capture_directory.runbase
            ).absolute()
        return uip

    def instrument_sub_basis_matrix(
        self,
        instrument_name: InstrumentIdentifier | str,
        use_full_state_vector: bool = True,
    ) -> np.ndarray:
        """Return the portion of the basis matrix that includes jacobians
        for the given instrument. This is what the various muses-py forward
        models return - only the subset of jacobians actually relevant for
        that instrument.
        """
        if not use_full_state_vector:
            if self.basis_matrix is None:
                raise RuntimeError("basis_matrix is None")
            return self.basis_matrix[
                :,
                [
                    t in list(self.state_vector_params(instrument_name))
                    for t in self.uip["speciesListFM"]
                ],
            ]
        bmatrix = np.eye(len(self.uip["speciesListFM"]))
        return bmatrix[
            :,
            [
                t in list(self.state_vector_params(instrument_name))
                for t in self.uip["speciesListFM"]
            ],
        ]

    @property
    def is_bt_retrieval(self) -> bool:
        """For BT retrievals, the species aren't set. This means we
        need to do special handling in some cases. Determine if we are
        doing a BT retrieval and return True if we are."""
        # Note the logic is a bit obscure here, but this matches what
        # fm_wrapper does. If the speciesListFM is ['',] then we just
        # "know" that this is a BT retrieval
        return len(self.uip["speciesListFM"]) == 0 or (
            len(self.uip["speciesListFM"]) == 1
            and self.uip["speciesListFM"]
            == [
                "",
            ]
        )

    def species_basis_matrix(self, species_name: list[str]) -> np.ndarray:
        """Muses does the retrieval on a subset of the full forward model
        grid. The mapping between the two sets is handled by the
        basis_matrix. We subset this for just this particular species_name
        (e.g, O3)."""
        t1 = np.array(self.uip["speciesList"]) == species_name
        t2 = np.array(self.uip["speciesListFM"]) == species_name
        if self.basis_matrix is None:
            raise RuntimeError("basis_matrix is None")
        return self.basis_matrix[t1, :][:, t2]

    def species_basis_matrix_calc(self, species_name: list[str]) -> np.ndarray:
        """Rather than return the basis matrix in ret_info, calculate
        this like get_species_information does in muses-py.

        Note that this is a bit circular, we use
        species_retrieval_level_subset which depends on self.basis_matrix
        (because we don't have this information available at this level of
        the processing tree).

        But go ahead and have this function, it is a nice documentation
        of how we would possibly move this calculation into refractor, and
        that our data is consistent."""
        # Note this is in Pa rather than hPa. make_maps expects this, so
        # it is consistent. But this is different than what refractor uses
        # elsewhere.
        plev = self.atmosphere_column("pressure")
        # +1 here is because make_maps is expecting 1 based levels rather
        # the 0 based we return from species_retrieval_level_subset.
        return mpy_make_maps(
            plev, self.species_retrieval_level_subset(species_name) + 1
        )["toState"]

    def species_retrieval_level_subset(self, species_name: list[str]) -> int:
        """This is the levels of the forward model grid that we do
        the retrieval on.

        It would be nice to get this directly, this is a value
        determined by get_species_information, which is called by
        script_retrieval_ms. But for now, we can indirectly back
        out this information by looking at the structure of the
        basis_matrix.

        Note that this is 0 based, although the py_retrieve function is
        in terms of 1 based.
        """
        i_levels = np.any(
            self.species_basis_matrix(species_name) == 1, axis=0
        ).nonzero()[0]
        return i_levels

    def atmosphere_column(self, species_name: str) -> np.ndarray:
        """Return the atmospheric column. Note that MUSES use
        a decreasing pressure order (to surface to TOA). This is
        the opposite of the normal ReFRACtor convention. This is
        handled by marking the pressure levels as
        PREFER_DECREASING_PRESSURE, so this difference is handled by
        the forward model. But be aware of the difference if you
        are looking at the data directly."""
        param_list = [n.lower() for n in self.uip["atmosphere_params"]]
        param_index = param_list.index(species_name.lower())
        return self.uip["atmosphere"][param_index, :]

    @property
    def uip_omi(self) -> dict[str, Any]:
        """Short cut to uip_OMI"""
        return cast(dict[str, Any], self.uip.get("uip_OMI"))

    @property
    def omi_params(self) -> dict[str, Any]:
        """Short cut for omiPars"""
        return cast(dict[str, Any], self.uip.get("omiPars"))

    @property
    def uip_tropomi(self) -> dict[str, Any]:
        """Short cut to uip_TROPOMI"""
        return cast(dict[str, Any], self.uip.get("uip_TROPOMI"))

    @property
    def tropomi_params(self) -> dict[str, Any]:
        """Short cut for tropomiPars"""
        return cast(dict[str, Any], self.uip.get("tropomiPars"))

    def frequency_list(self, instrument_name: InstrumentIdentifier | str) -> np.ndarray:
        return cast(np.ndarray, self.uip[f"uip_{instrument_name}"]["frequencyList"])

    @property
    def instrument_list(self) -> list[InstrumentIdentifier]:
        """List of all the radiance data we are generating, identifying
        which instrument fills in that particular index"""
        return [InstrumentIdentifier(i) for i in self.uip["instrumentList"]]

    @property
    def instrument(self) -> list[InstrumentIdentifier]:
        """List of instruments that are part of the UIP"""
        return [InstrumentIdentifier(i) for i in self.uip["instruments"]]

    def freq_index(self, instrument_name: InstrumentIdentifier | str) -> np.ndarray:
        """Return frequency index for given instrument"""
        if str(instrument_name) == "OMI":
            return self.uip_omi["freqIndex"]
        elif str(instrument_name) == "TROPOMI":
            return self.uip_tropomi["freqIndex"]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def freqfilter(
        self, instrument_name: InstrumentIdentifier | str, sensor_index: int
    ) -> np.ndarray:
        """freq_index is subsetted by the microwindow. This version gives
        the full set of indices for a particular sensor index"""
        if str(instrument_name) == "OMI":
            return self.uip_omi["frequencyfilterlist"] == str(
                self.filter_name(sensor_index)
            )
        elif str(instrument_name) == "TROPOMI":
            return self.uip_tropomi["frequencyfilterlist"] == str(
                self.filter_name(sensor_index)
            )
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def measured_radiance(
        self,
        instrument_name: InstrumentIdentifier | str,
        sensor_index: int = -1,
        full_freq: bool = False,
    ) -> dict[str, Any]:
        """Note muses-py handles the radiance data in pretty much the reverse
        way that ReFRACtor does.

        For a traditional ReFRACtor retrieval, we take the reflectance and
        multiple this with the solar model to give radiance that has units.
        We then compare this against the measured radiance in our cost
        function.

        muses-py on the other hand scales the measured radiance by the
        solar model. The cost function is then the difference between
        reflectance like values (so unitless).

        This means that the "omi_measured_radiance" here depends on the
        solar model, which in turn depends on the state vector.

        There is nothing wrong with this, but the ReFRACtor ForwardModel
        isn't currently set up to work this way. So we need to track the
        solar model state vector elements/jacobians separate from the
        ReFRACtor ForwardModel.

        The default version of this function returns the data subsetted by the
        microwindows. You can request instead the full set of values for a given
        sensor index.
        """
        # The py-retrieve radiances read from a pickle file. This is from a relative
        # path ./Input. So we need to be in the run directory for these to work
        # correctly
        curdir = os.getcwd()
        try:
            os.chdir(self.run_dir)
            if str(instrument_name) == "OMI":
                rad = mpy_get_omi_radiance(self.omi_params)
            elif str(instrument_name) == "TROPOMI":
                rad = mpy_get_tropomi_radiance(self.tropomi_params)
            else:
                raise RuntimeError(f"Invalid instrument_name {instrument_name}")
            if not full_freq:
                freqindex = self.freq_index(instrument_name)
            else:
                offset = [
                    i
                    for i in range(len(self.uip["microwindows_all"]))
                    if self.uip["microwindows_all"][i]["instrument"]
                    == str(instrument_name)
                ][0]
                freqindex = self.freqfilter(instrument_name, sensor_index + offset)
            return {
                "wavelength": rad["wavelength"][freqindex],
                "measured_radiance_field": rad["normalized_rad"][freqindex],
                "measured_nesr": rad["nesr"][freqindex],
                "normwav_jac": rad["normwav_jac"][freqindex],
                "odwav_jac": rad["odwav_jac"][freqindex],
                "odwav_slope_jac": rad["odwav_slope_jac"][freqindex],
            }
        finally:
            os.chdir(curdir)

    def nfreq_mw(
        self, mw_index: int, instrument_name: InstrumentIdentifier | str
    ) -> int:
        """Number of frequencies for microwindow."""
        if str(instrument_name) == "OMI":
            # It is a bit odd that mw_index get used twice here, but this
            # really is how this is set up. So although this looks odd, it
            # is correct
            startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw"][mw_index]
        elif str(instrument_name) == "TROPOMI":
            startmw_fm = self.uip_tropomi["microwindows"][mw_index]["startmw"][mw_index]
            endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw"][mw_index]

        return endmw_fm - startmw_fm + 1

    def atm_params(
        self,
        instrument_name: InstrumentIdentifier | str,
        set_pointing_angle_zero: bool = True,
    ) -> dict[str, Any]:
        uall = self.uip_all(instrument_name)
        # tropomi_fm and omi_fm set this to zero before calling raylayer_nadir.
        # I'm not sure if always want to do this or not. Note that uall
        # is a copy of uip, so no need to set this back.
        if set_pointing_angle_zero:
            uall["obs_table"]["pointing_angle"] = 0.0
        return mpy_atmosphere_level(uall)

    def ray_info(
        self,
        instrument_name: InstrumentIdentifier | str,
        set_pointing_angle_zero: bool = True,
        set_cloud_extinction_one: bool = False,
    ) -> dict[str, Any]:
        uall = self.uip_all(instrument_name)
        # tropomi_fm and omi_fm set this to zero before calling raylayer_nadir.
        # I'm not sure if always want to do this or not. Note that uall
        # is a copy of uip, so no need to set this back.
        if set_pointing_angle_zero:
            uall["obs_table"]["pointing_angle"] = 0.0
        if set_cloud_extinction_one:
            uall["cloud"]["extinction"][:] = 1.0
        return mpy_raylayer_nadir(
            AttrDictAdapter(uall), AttrDictAdapter(mpy_atmosphere_level(uall))
        )

    @property
    def omi_cloud_fraction(self) -> float:
        """Cloud fraction for OMI"""
        return self.omi_params["cloud_fraction"]

    @property
    def tropomi_cloud_fraction(self) -> float:
        """Cloud fraction for TROPOMI"""
        return self.tropomi_params["cloud_fraction"]

    @property
    def omi_obs_table(self) -> dict[str, Any] | None:
        """Short cut to omi_obs_table"""
        if self.uip_omi is not None:
            return self.uip_omi["omi_obs_table"]
        return None

    @property
    def tropomi_obs_table(self) -> dict[str, Any] | None:
        """Short cut to tropomi_obs_table"""
        if self.uip_tropomi:
            return self.uip_tropomi["tropomi_obs_table"]
        return None

    @property
    def number_micro_windows(self) -> int:
        """Total number of microwindows. This is like a channel_index,
        except muses-py can retrieve multiple instruments."""
        return len(self.uip["microwindows_all"])

    def instrument_name(self, ii_mw: int) -> InstrumentIdentifier:
        """Instrument name for the micro_window index ii_mw"""
        return InstrumentIdentifier(self.uip["microwindows_all"][ii_mw]["instrument"])

    def micro_windows(self, ii_mw: int) -> rf.ArrayWithUnit:
        """Return start and end of microwindow"""
        return rf.ArrayWithUnit(
            np.array(
                [
                    [
                        self.uip["microwindows_all"][ii_mw]["start"],
                        self.uip["microwindows_all"][ii_mw]["endd"],
                    ],
                ]
            ),
            "nm",
        )

    @property
    def jacobian_all(self) -> np.ndarray:
        """List of jacobians we are including in the state vector"""
        return self.uip["jacobians_all"]

    def state_vector_species_index(
        self, species_name: str, use_full_state_vector: bool = True
    ) -> tuple[int, int]:
        """Index and length for the location of the species_name in
        our state vector. We either do this for the retrieval state vector
        or the full state vector."""
        if self.is_bt_retrieval:
            # Special handling for BT retrieval. BTW, this is really just
            # sort of a "magic" logic in fm_wrapper, there is nothing that
            # indicates the length is 1 here except the hard coded logic
            # in fm_wrapper.
            pstart = 0
            plen = 1
        elif not use_full_state_vector:
            pstart = list(self.uip["speciesList"]).index(species_name)
            plen = list(self.uip["speciesList"]).count(species_name)
        else:
            pstart = list(self.uip["speciesListFM"]).index(species_name)
            plen = list(self.uip["speciesListFM"]).count(species_name)
        return pstart, plen

    def state_vector_params(
        self, instrument_name: InstrumentIdentifier | str
    ) -> list[str]:
        """List of parameter types to include in the state vector."""
        return self.uip[f"uip_{instrument_name}"]["jacobians"]

    def state_vector_names(
        self, instrument_name: InstrumentIdentifier | str
    ) -> list[str]:
        """Full list of the name for each state vector list item"""
        sv_list = []
        for jac_name in self.uip["speciesListFM"]:
            if jac_name in self.state_vector_params(instrument_name):
                sv_list.append(jac_name)
        return sv_list

    def state_vector_update_indexes(
        self, instrument_name: InstrumentIdentifier | str
    ) -> np.ndarray:
        """Indexes for this instrument's state vector element updates from the full update vector"""
        sv_extract_index = []
        for full_idx, jac_name in enumerate(self.uip["speciesListFM"]):
            if jac_name in self.state_vector_params(instrument_name):
                sv_extract_index.append(full_idx)

        return np.array(sv_extract_index)

    def species_lin_log_mapping(self, specie_name: str) -> str:
        output_map = None

        # JLL: figured we might as well go through this process of checking the UIP each call rather than doing any caching,
        # that way if the UIP changes we get the updated map type.
        for fm_spec, fm_map in zip(
            self.uip["speciesListFM"], self.uip["mapTypeListFM"]
        ):
            if fm_spec == specie_name and output_map is None:
                output_map = fm_map
            elif fm_spec == specie_name and output_map != fm_map:
                raise RuntimeError(
                    f"There were at least two different FM map types in the UIP for specie {specie_name}: {output_map} and {fm_map}"
                )

        if output_map is None:
            raise ValueError(
                f"Specie {specie_name} was not present in the FM species list, so could not find its map type"
            )
        else:
            return output_map

    def earth_sun_distance(self, instrument_name: InstrumentIdentifier | str) -> float:
        """Earth sun distance, in meters. Right now this is OMI specific"""
        # Same value for all the bands, so just grab the first one
        if str(instrument_name) == "OMI":
            if self.omi_obs_table is None:
                raise RuntimeError("omi_obs_table is None")
            return self.omi_obs_table["EarthSunDistance"][0]
        elif str(instrument_name) == "TROPOMI":
            if self.tropomi_obs_table is None:
                raise RuntimeError("omi_obs_table is None")
            return self.tropomi_obs_table["EarthSunDistance"][0]
        else:
            raise RuntimeError("Didn't find a observation table")

    def sample_grid(self, mw_index: int, ii_mw: int) -> rf.SpectralDomain:
        """This is the full set of samples. We only actually use a subset of
        these, but these are the values before the microwindow gets applied.

        Right now this is omi specific."""

        if self.ils_method(mw_index, self.instrument_name(ii_mw)) == "FASTCONV":
            ils_uip_info = self.ils_params(mw_index, self.instrument_name(ii_mw))

            return rf.SpectralDomain(ils_uip_info["central_wavelength"], rf.Unit("nm"))
        else:
            if self.instrument_name(ii_mw) == InstrumentIdentifier("OMI"):
                all_freq = self.uip_omi["fullbandfrequency"]
                filt_loc = np.array(self.uip_omi["frequencyfilterlist"])
            elif self.instrument_name(ii_mw) == InstrumentIdentifier("TROPOMI"):
                all_freq = self.uip_tropomi["tropomiInfo"]["Earth_Radiance"][
                    "Wavelength"
                ]
                filt_loc = np.array(
                    self.uip_tropomi["tropomiInfo"]["Earth_Radiance"][
                        "EarthWavelength_Filter"
                    ]
                )
            else:
                raise RuntimeError(f"Invalid instrument {self.instrument_name(ii_mw)}")
            return rf.SpectralDomain(
                all_freq[np.where(filt_loc == self.filter_name(ii_mw))], rf.Unit("nm")
            )

    def ils_params(
        self, mw_index: int, instrument_name: InstrumentIdentifier | str
    ) -> dict[str, Any]:
        """Returns ILS information for the given microwindow"""
        if str(instrument_name) == "OMI":
            return self.uip_omi["ils_%02d" % (mw_index + 1)]
        elif str(instrument_name) == "TROPOMI":
            # JLL: the TROPOMI UIP seems to use a different naming convention than the OMI UIP
            # (ils_mw_II, where II is the zero-based index - see end of make_uip_tropomi).
            return self.uip_tropomi["ils_mw_%02d" % (mw_index)]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def ils_method(
        self, mw_index: int, instrument_name: InstrumentIdentifier | str
    ) -> str:
        """Returns a string describing the ILS method configured by MUSES"""
        if str(instrument_name) == "OMI":
            return self.uip_omi["ils_omi_xsection"]
        elif str(instrument_name) == "TROPOMI":
            return self.uip_tropomi["ils_tropomi_xsection"]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def radiance_info(
        self, instrument_name: InstrumentIdentifier | str
    ) -> dict[str, Any]:
        """This is a bit convoluted. It comes from a python pickle file that
        gets created before the retrieval starts. So this is
        "control coupling". On the other hand, most of the UIP is sort of
        control coupling, so for now we'll just live with this.

        We 1) want to just directly evaluate this using ReFRACtor code or
        2) track down what exactly muses-py is doing to create this and
        do it directly.
        """
        input_directory = self.run_dir / "Input"
        if not os.path.exists(input_directory):
            raise RuntimeError(f"Input directory {input_directory} not found.")
        if str(instrument_name) == "OMI":
            fname = next(input_directory.glob("Radiance_OMI*.pkl"))
        elif str(instrument_name) == "TROPOMI":
            fname = next(input_directory.glob("Radiance_TROPOMI*.pkl"))
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return pickle.load(open(fname, "rb"))

    def mw_slice(
        self, filter_name: str, instrument_name: InstrumentIdentifier | str
    ) -> slice:
        """Variation of mw_slice that uses startmw and endmw. I think these are
        the same if we aren't doing an ILS, but different if we are. Should track
        this through, but for now just try this out"""
        startmw_fm = 0
        endmw_fm = 0
        if str(instrument_name) == "OMI":
            for mw_index in range(len(self.uip_omi["microwindows"])):
                if self.uip_omi["microwindows"][mw_index]["filter"] == filter_name:
                    startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw"][
                        mw_index
                    ]
                    endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw"][
                        mw_index
                    ]
        elif str(instrument_name) == "TROPOMI":
            for mw_index in range(len(self.uip_tropomi["microwindows"])):
                if self.uip_tropomi["microwindows"][mw_index]["filter"] == filter_name:
                    startmw_fm = self.uip_tropomi["microwindows"][mw_index]["startmw"][
                        mw_index
                    ]
                    endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw"][
                        mw_index
                    ]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return slice(startmw_fm, endmw_fm + 1)

    def mw_fm_slice(
        self, filter_name: str, instrument_name: InstrumentIdentifier | str
    ) -> slice:
        """This is the portion of the full microwindow frequencies that we are
        using in calculations such as RamanSioris. This is a bit
        bigger than the instrument_spectral_domain in
        RefractorObjectCreator, which is the range fitted in the
        retrieval. This has extra padding for things like the
        RamanSioris calculation"""
        startmw_fm = 0
        endmw_fm = 0
        if str(instrument_name) == "OMI":
            for mw_index in range(len(self.uip_omi["microwindows"])):
                if self.uip_omi["microwindows"][mw_index]["filter"] == filter_name:
                    startmw_fm = self.uip_omi["microwindows"][mw_index]["startmw_fm"][
                        mw_index
                    ]
                    endmw_fm = self.uip_omi["microwindows"][mw_index]["enddmw_fm"][
                        mw_index
                    ]
        elif str(instrument_name) == "TROPOMI":
            for mw_index in range(len(self.uip_tropomi["microwindows"])):
                if self.uip_tropomi["microwindows"][mw_index]["filter"] == filter_name:
                    startmw_fm = self.uip_tropomi["microwindows"][mw_index][
                        "startmw_fm"
                    ][mw_index]
                    endmw_fm = self.uip_tropomi["microwindows"][mw_index]["enddmw_fm"][
                        mw_index
                    ]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")
        return slice(startmw_fm, endmw_fm + 1)

    def full_band_frequency(
        self, instrument_name: InstrumentIdentifier | str
    ) -> np.ndarray:
        """This is the full frequency range for the instrument. I believe
        this is the same as the wavelengths found in the radiance pickle
        file (self.radiance_info), but this comes for a different source in
        the UIP object so we have this in case this is somehow different."""
        if str(instrument_name) == "OMI":
            return self.uip_omi["fullbandfrequency"]
        elif str(instrument_name) == "TROPOMI":
            return self.uip_tropomi["fullbandfrequency"]
        else:
            raise RuntimeError(f"Invalid instrument_name {instrument_name}")

    def rad_wavelength(
        self, filter_name: str, instrument_name: InstrumentIdentifier | str
    ) -> rf.SpectralDomain:
        """This is the wavelengths that the L1B data was measured at, truncated
        to fit our microwindow"""
        slc = self.mw_fm_slice(filter_name, instrument_name)
        rad_info = self.radiance_info(instrument_name)
        return rf.SpectralDomain(
            rad_info["Earth_Radiance"]["Wavelength"][slc], rf.Unit("nm")
        )

    def solar_irradiance(
        self, filter_name: str, instrument_name: InstrumentIdentifier | str
    ) -> rf.Spectrum:
        """This is currently just used for the Raman calculation of the
        RefractorRtfOmi class. This has been adjusted for the
        """
        slc = self.mw_fm_slice(filter_name, instrument_name)
        rad_info = self.radiance_info(instrument_name)

        # Note this looks wrong (why not use Solar_Radiance Wavelength here?),
        # but is actually correct. The solar data has already been interpolated
        # to the same wavelengths as  the Earth_Radiance, this happens in
        # daily_tropomi_irad for TROPOMI, and similarly for OMI. Not sure
        # why the original wavelengths are left in rad_info['Solar_Radiance'],
        # that is actually misleading.
        sol_domain = rf.SpectralDomain(
            rad_info["Earth_Radiance"]["Wavelength"][slc], rf.Unit("nm")
        )
        sol_range = rf.SpectralRange(
            rad_info["Solar_Radiance"]["AdjustedSolarRadiance"][slc],
            rf.Unit("ph / nm / s"),
        )
        return rf.Spectrum(sol_domain, sol_range)

    def filter_name(self, ii_mw: int) -> str:
        """The filter name (e.g., UV1 or UV2)"""
        return self.uip["microwindows_all"][ii_mw]["filter"]

    def channel_indexes(self, ii_mw: int) -> np.ndarray:
        """Determine the channel indexes that we are processing."""
        # You would think this would just be an argument, but it
        # isn't. We need to get the filter name from one place, and
        # use that to look up the channel index in another.
        if self.instrument_name(ii_mw) == InstrumentIdentifier("OMI"):
            if self.omi_obs_table is None:
                raise RuntimeError("omi_obs_table is None")
            return np.where(
                np.asarray(self.omi_obs_table["Filter_Band_Name"])
                == self.filter_name(ii_mw)
            )[0]
        if self.instrument_name(ii_mw) == InstrumentIdentifier("TROPOMI"):
            if self.tropomi_obs_table is None:
                raise RuntimeError("tropomi_obs_table is None")
            return np.where(
                np.asarray(self.tropomi_obs_table["Filter_Band_Name"])
                == self.filter_name(ii_mw)
            )[0]
        else:
            raise RuntimeError("Don't know how to find observation table")

    def _avg_obs(self, nm: str, filter_name: str) -> float:
        """Average values that match the self.channel_indexes.

        Not sure if this makes sense or not, but it is what py_retrieve
        does.

        Right now this is omi specific"""
        if self.omi_obs_table:
            cindex = np.where(
                np.asarray(self.omi_obs_table["Filter_Band_Name"]) == filter_name
            )[0]
            if len(cindex) == 0:
                raise RuntimeError(f"Bad filter name {filter_name}")
            return np.mean(np.asarray(self.omi_obs_table[nm])[cindex])
        if self.tropomi_obs_table:
            cindex = np.where(
                np.asarray(self.tropomi_obs_table["Filter_Band_Name"]) == filter_name
            )[0]
            if len(cindex) == 0:
                raise RuntimeError(f"Bad filter name {filter_name}")
            return np.mean(np.asarray(self.tropomi_obs_table[nm])[cindex])
        raise RuntimeError("Don't know how to find observation table")

    def observation_zenith(self, filter_name: str) -> float:
        """Observation zenith angle for the microwindow index filter_name"""
        return self._avg_obs("ViewingZenithAngle", filter_name)

    def observation_zenith_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Observation zenith angle for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.observation_zenith(filter_name)), "deg")

    def observation_azimuth(self, filter_name: str) -> float:
        """Observation azimuth angle for the microwindow index filter_name"""
        return self._avg_obs("ViewingAzimuthAngle", filter_name)

    def observation_azimuth_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Observation azimuth angle for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.observation_azimuth(filter_name)), "deg")

    def solar_azimuth(self, filter_name: str) -> float:
        """Solar azimuth angle for the microwindow index filter_name"""
        return self._avg_obs("SolarAzimuthAngle", filter_name)

    def solar_azimuth_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Solar azimuth angle for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.solar_azimuth(filter_name)), "deg")

    def solar_zenith(self, filter_name: str) -> float:
        """Solar zenith angle for the microwindow index filter_name"""
        return self._avg_obs("SolarZenithAngle", filter_name)

    def solar_zenith_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Solar zenith angle for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.solar_zenith(filter_name)), "deg")

    def relative_azimuth(self, filter_name: str) -> float:
        """Relative azimuth angle for the microwindow index filter_name"""
        return self._avg_obs("RelativeAzimuthAngle", filter_name)

    def relative_azimuth_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Relative azimuth angle for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.relative_azimuth(filter_name)), "deg")

    def latitude(self, filter_name: str) -> float:
        """Latitude for the microwindow index filter_name"""
        return self._avg_obs("Latitude", filter_name)

    def latitude_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Latitude for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.latitude(filter_name)), "deg")

    def longitude(self, filter_name: str) -> float:
        """Longitude for the microwindow index filter_name"""
        return self._avg_obs("Longitude", filter_name)

    def longitude_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Longitude for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.longitude(filter_name)), "deg")

    def surface_height(self, filter_name: str) -> float:
        """Surface height for the microwindow index filter_name"""
        return self._avg_obs("TerrainHeight", filter_name)

    def surface_height_with_unit(self, filter_name: str) -> rf.DoubleWithUnit:
        """Surface height for the microwindow index filter_name"""
        return rf.DoubleWithUnit(float(self.surface_height(filter_name)), "m")

    def across_track_indexes(
        self,
        filter_name: FilterIdentifier | str,
        instrument_name: InstrumentIdentifier | str,
    ) -> np.ndarray:
        """Across track indexes for the microwindow index ii_mw.

        Right now this is omi specific"""
        # Can't really average these to have anything that makes sense.
        # So for now we just pick the first one that matches
        if str(instrument_name) == "OMI":
            if self.omi_obs_table is None:
                raise RuntimeError("omi_obs_table is None")
            cindex = np.where(
                np.asarray(self.omi_obs_table["Filter_Band_Name"]) == str(filter_name)
            )[0]
            if len(cindex) == 0:
                raise RuntimeError(f"Bad filter name {filter_name}")
            return np.asarray(self.omi_obs_table["XTRACK"])[cindex]
        if str(instrument_name) == "TROPOMI":
            if self.tropomi_obs_table is None:
                raise RuntimeError("tropomi_obs_table is None")
            cindex = np.where(
                np.asarray(self.tropomi_obs_table["Filter_Band_Name"])
                == str(filter_name)
            )[0]
            if len(cindex) == 0:
                raise RuntimeError(f"Bad filter name {filter_name}")
            return np.asarray(self.tropomi_obs_table["XTRACK"])[cindex]
        raise RuntimeError("Don't know how to find observation table")

    def update_uip(self, i_retrieval_vec: np.ndarray) -> None:
        """This updates the underlying UIP with the new retrieval_vec,
        e.g., this is the muses-py equivalent up updating the
        StateVector in ReFRACtor.

        Note that this is the retrieval vector, not the full state vector.
        """
        # Fake the ret_info structure. update_uip only uses the basis
        # matrix
        ret_info = AttrDictAdapter({"basis_matrix": self.basis_matrix})

        # This is a copy of mpy.update_uip, modified slightly to ignore retrieval
        # elements we don't recognize.
        # Note I'd like to get away from the UIP, it is really just a shuffling
        # of our StateInfo. But at least for now, we need to call old muses-py code.

        o_uip = AttrDictAdapter(self.uip)

        o_retrieval_vec = copy.deepcopy(
            i_retrieval_vec
        )  # Make a deep copy of i_retrieval_vec in case we need to modify it later in this function.

        # MAP THE RETRIEVAL VECTOR TO THE FULL STATE VECTOR
        # replace things in this in specific cases, e.g. exp()
        fm_vec = i_retrieval_vec @ ret_info.basis_matrix

        num_map = ret_info.basis_matrix.shape[1]  # Get the 2nd dimension of ((163,471).
        update_arr = np.zeros(shape=(num_map), dtype=int)

        for ii in range(num_map):
            if np.sum(ret_info.basis_matrix[:, ii]) != 0.0:
                update_arr[ii] = 1

        # POPULATE THE UIP

        # AT_LINE 69 Optimization/update_uip.pro update_uip

        # We need to set a flag if len(uip.jacobians_all) is 9 or more because it contains these jacobian species:
        # ['O3',
        # 'OMICLOUDFRACTION',
        # 'OMISURFACEALBEDOUV1', 'OMISURFACEALBEDOUV2', 'OMISURFACEALBEDOSLOPEUV2',
        # 'OMINRADWAVUV1', 'OMINRADWAVUV2',
        # 'OMIODWAVUV1', 'OMIODWAVUV2']

        for ii in range(len(o_uip.jacobians_all)):
            specie = o_uip.jacobians_all[ii]

            ind_ret = np.where(np.asarray(o_uip.speciesList) == specie)[0]
            ind = ind_ret
            ind_fm = np.where(np.asarray(o_uip.speciesListFM) == specie)[0]
            mapType = o_uip.mapTypeListFM[ind_fm[0]]

            if "TSUR" == specie:
                o_uip.surface_temperature = fm_vec[ind_fm][0]
            elif "PSUR" == specie:
                #  Susan Kulawik, 12/2021
                o_uip.atmosphere[0, 0] = fm_vec[ind_fm]
                o_uip.atmosphere[0, :] = mpy_pressure_sigma(
                    o_uip.atmosphere[0, 0], len(o_uip.atmosphere[0, :]), "surface"
                )
            elif "EMIS" == specie:
                for jj in range(len(ind_fm)):
                    my_ind = ind_fm[0] + jj
                    if update_arr[my_ind] == 1:
                        o_uip.emissivity["value"][jj] = fm_vec[my_ind]
            elif "CALSCALE" == specie:
                raise RuntimeError("Needs update... look at EMIS")
            elif "CALOFFSET" == specie:
                raise RuntimeError("Needs update... look at EMIS")
            elif "PTGANG" == specie:
                o_uip.obs_table["pointing_angle"] = fm_vec[ind_fm][0]
            elif "RESSCALE" == specie:
                o_uip.res_scale = fm_vec[ind_fm][0]
            elif "CLOUDEXT" == specie:
                # If we didn't include AIRS or CRIS, this might be the wrong
                # size. Skip updating in this case.
                if o_uip.cloud["extinction"].shape[0] == 2 and len(ind_fm) > 2:
                    pass
                else:
                    for jj in range(len(ind_fm)):
                        my_ind = ind_fm[0] + jj
                        if update_arr[my_ind] == 1:
                            fm_vec[my_ind] = math.exp(fm_vec[my_ind])
                            if fm_vec[my_ind] > 20:
                                fm_vec[my_ind] = 20
                            o_uip.cloud["extinction"][jj] = fm_vec[my_ind]

            elif "PCLOUD" == specie:
                fm_vec[ind_fm] = np.exp(fm_vec[ind_fm][0])
                if fm_vec[ind_fm] < 50:
                    fm_vec[ind_fm] = 50
                o_uip.cloud["pressure"] = fm_vec[ind_fm][0]
            elif "OMICLOUDFRACTION" == specie:
                o_uip.omiPars["cloud_fraction"] = fm_vec[ind_fm][0]
            elif "OMISURFACEALBEDOUV1" == specie:
                o_uip.omiPars["surface_albedo_uv1"] = fm_vec[ind_fm][0]
            elif "OMISURFACEALBEDOUV2" == specie:
                o_uip.omiPars["surface_albedo_uv2"] = fm_vec[ind_fm][0]
            elif "OMISURFACEALBEDOSLOPEUV2" == specie:
                o_uip.omiPars["surface_albedo_slope_uv2"] = fm_vec[ind_fm][0]
            elif "OMINRADWAVUV1" == specie:
                o_uip.omiPars["nradwav_uv1"] = fm_vec[ind_fm][0]
            elif "OMINRADWAVUV2" == specie:
                o_uip.omiPars["nradwav_uv2"] = fm_vec[ind_fm][0]
            elif "OMIODWAVUV1" == specie:
                o_uip.omiPars["odwav_uv1"] = fm_vec[ind_fm][0]
            elif "OMIODWAVUV2" == specie:
                o_uip.omiPars["odwav_uv2"] = fm_vec[ind_fm][0]
            elif "OMIODWAVSLOPEUV1" == specie:
                o_uip.omiPars["odwav_slope_uv1"] = fm_vec[ind_fm][0]
            elif "OMIODWAVSLOPEUV2" == specie:
                o_uip.omiPars["odwav_slope_uv2"] = fm_vec[ind_fm][0]
            elif "OMIRINGSFUV1" == specie:
                o_uip.omiPars["ring_sf_uv1"] = fm_vec[ind_fm][0]
            elif "OMIRINGSFUV2" == specie:
                o_uip.omiPars["ring_sf_uv2"] = fm_vec[ind_fm][0]

            ## EM 04-2021 Adding TROPOMI parameters
            elif "TROPOMICLOUDFRACTION" == specie:
                o_uip.tropomiPars["cloud_fraction"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOBAND1" == specie:
                o_uip.tropomiPars["surface_albedo_BAND1"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOBAND2" == specie:
                o_uip.tropomiPars["surface_albedo_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOBAND3" == specie:
                o_uip.tropomiPars["surface_albedo_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOBAND7" == specie:
                o_uip.tropomiPars["surface_albedo_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOBAND3TIGHT" == specie:
                o_uip.tropomiPars["surface_albedo_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOSLOPEBAND2" == specie:
                o_uip.tropomiPars["surface_albedo_slope_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOSLOPEBAND3" == specie:
                o_uip.tropomiPars["surface_albedo_slope_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOSLOPEBAND7" == specie:
                o_uip.tropomiPars["surface_albedo_slope_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOSLOPEBAND3TIGHT" == specie:
                o_uip.tropomiPars["surface_albedo_slope_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMISURFACEALBEDOSLOPEORDER2BAND2" == specie:
                o_uip.tropomiPars["surface_albedo_slope_order2_BAND2"] = fm_vec[ind_fm][
                    0
                ]
            elif "TROPOMISURFACEALBEDOSLOPEORDER2BAND3" == specie:
                o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"] = fm_vec[ind_fm][
                    0
                ]
            elif "TROPOMISURFACEALBEDOSLOPEORDER2BAND7" == specie:
                o_uip.tropomiPars["surface_albedo_slope_order2_BAND7"] = fm_vec[ind_fm][
                    0
                ]
            elif "TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT" == specie:
                o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"] = fm_vec[ind_fm][
                    0
                ]
            elif "TROPOMISOLARSHIFTBAND1" == specie:
                o_uip.tropomiPars["solarshift_BAND1"] = fm_vec[ind_fm][0]
            elif "TROPOMISOLARSHIFTBAND2" == specie:
                o_uip.tropomiPars["solarshift_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMISOLARSHIFTBAND3" == specie:
                o_uip.tropomiPars["solarshift_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMISOLARSHIFTBAND7" == specie:
                o_uip.tropomiPars["solarshift_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADIANCESHIFTBAND1" == specie:
                o_uip.tropomiPars["radianceshift_BAND1"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADIANCESHIFTBAND2" == specie:
                o_uip.tropomiPars["radianceshift_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADIANCESHIFTBAND3" == specie:
                o_uip.tropomiPars["radianceshift_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADIANCESHIFTBAND7" == specie:
                o_uip.tropomiPars["radianceshift_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADSQUEEZEBAND1" == specie:
                o_uip.tropomiPars["radsqueeze_BAND1"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADSQUEEZEBAND2" == specie:
                o_uip.tropomiPars["radsqueeze_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADSQUEEZEBAND3" == specie:
                o_uip.tropomiPars["radsqueeze_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRADSQUEEZEBAND7" == specie:
                o_uip.tropomiPars["radsqueeze_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRINGSFBAND1" == specie:
                o_uip.tropomiPars["ring_sf_BAND1"] = fm_vec[ind_fm][0]
            elif "TROPOMIRINGSFBAND2" == specie:
                o_uip.tropomiPars["ring_sf_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRINGSFBAND3" == specie:
                o_uip.tropomiPars["ring_sf_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRINGSFBAND7" == specie:
                o_uip.tropomiPars["ring_sf_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO0BAND2" == specie:
                o_uip.tropomiPars["resscale_O0_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO1BAND2" == specie:
                o_uip.tropomiPars["resscale_O1_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO2BAND2" == specie:
                o_uip.tropomiPars["resscale_O2_BAND2"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO0BAND3" == specie:
                o_uip.tropomiPars["resscale_O0_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO1BAND3" == specie:
                o_uip.tropomiPars["resscale_O1_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO2BAND3" == specie:
                o_uip.tropomiPars["resscale_O2_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMITEMPSHIFTBAND3" == specie:
                o_uip.tropomiPars["temp_shift_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO0BAND7" == specie:
                o_uip.tropomiPars["resscale_O0_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO1BAND7" == specie:
                o_uip.tropomiPars["resscale_O1_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMIRESSCALEO2BAND7" == specie:
                o_uip.tropomiPars["resscale_O2_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMITEMPSHIFTBAND7" == specie:
                o_uip.tropomiPars["temp_shift_BAND7"] = fm_vec[ind_fm][0]
            elif "TROPOMITEMPSHIFTBAND3TIGHT" == specie:
                o_uip.tropomiPars["temp_shift_BAND3"] = fm_vec[ind_fm][0]
            elif "TROPOMICLOUDSURFACEALBEDO" == specie:
                o_uip.tropomiPars["cloud_Surface_Albedo"] = fm_vec[ind_fm][0]

            # OCO
            elif specie == "NIRAEROD" or specie == "AEROD":
                # if linear, OD cannot be less than 0.

                if mapType == "log":
                    o_uip.nirPars["aerod"] = np.exp(fm_vec[ind_fm])
                else:
                    indx = np.where(fm_vec[ind_fm] < 0.0001)[0]
                    if len(indx) > 0:
                        fm_vec[ind_fm[indx]] = 0.0001

                    indx = np.where(i_retrieval_vec[ind_ret] < 0.00001)[0]
                    if len(indx) > 0:
                        o_retrieval_vec[ind_ret[indx]] = 0.00001

                    o_uip.nirPars["aerod"] = fm_vec[ind_fm]

            elif specie == "NIRAERP" or specie == "AERP":
                # check range on fm_vec and retrieval_vec
                indx = np.where(fm_vec[ind_fm] < 0.02)[0]
                if len(indx) > 0:
                    fm_vec[ind_fm[indx]] = 0.02
                indx = np.where(fm_vec[ind_fm] > 0.98)[0]
                if len(indx) > 0:
                    fm_vec[ind_fm[indx]] = 0.98

                indx = np.where(i_retrieval_vec[ind_ret] < 0.02)[0]
                if len(indx) > 0:
                    o_retrieval_vec[ind_ret[indx]] = 0.02
                indx = np.where(i_retrieval_vec[ind_ret] > 0.98)[0]
                if len(indx) > 0:
                    o_retrieval_vec[ind_ret[indx]] = 0.98

                # old = uip.nirPars.aerp
                o_uip.nirPars["aerp"] = fm_vec[ind_fm]

            elif specie == "NIRAERW" or specie == "AERW":
                # old = uip.nirPars.aerw
                indx = np.where(fm_vec[ind_fm] < 0)[0]
                if len(indx) > 0:
                    fm_vec[ind_fm[indx]] = 0.01
                indx = np.where(fm_vec[ind_fm] > 0.2)[0]
                if len(indx) > 0:
                    fm_vec[ind_fm[indx]] = 0.20

                indx = np.where(i_retrieval_vec[ind_ret] < 0)[0]
                if len(indx) > 0:
                    o_retrieval_vec[ind_ret[indx]] = 0.01
                indx = np.where(i_retrieval_vec[ind_ret] > 0.2)[0]
                if len(indx) > 0:
                    o_retrieval_vec[ind_ret[indx]] = 0.20

                o_uip.nirPars["aerw"] = fm_vec[ind_fm]

            elif specie == "NIRALBBRDF" or specie == "ALBBRDF":
                mult = 1.0
                if o_uip.nirPars["albtype"] == 1:
                    mult = 0.07  # convert species BRDF to uip lambertian
                # npoly = N_ELEMENTS(fs_vec[indfs])/3
                # ;uip.nirPars.alb = 0
                # ;uip.nirPars.alb[0:npoly-1,*] = fs_vec[indfs] * mult
                o_uip.nirPars["albpl"] = fm_vec[ind_fm] * mult
            elif specie == "NIRALBLAMB" or specie == "ALBLAMB":
                mult = 1.0
                if o_uip.nirPars["albtype"] == 2:
                    mult = 1 / 0.07  # convert species lambertian to uip BRDF
                # npoly = N_ELEMENTS(fs_vec[indfs])/3
                # uip.nirPars.alb = 0
                # uip.nirPars.alb[0:npoly-1,*] = fs_vec[indfs] * mult
                o_uip.nirPars["albpl"] = fm_vec[ind_fm] * mult
            elif specie == "NIRALBCM" or specie == "ALBCM":
                # cox-munk albedo multiplier (ocean).  Does not convert with other types.
                if o_uip.nirPars["albtype"] != 3:
                    assert "Type mismatch for ALBCM"
                o_uip.nirPars["albpl"] = fm_vec[ind_fm]
            elif (
                specie == "NIRALBBRDFPL"
                or specie == "ALBBRDFPL"
                or specie == "NIRALBLAMBPL"
                or specie == "ALBLAMBPL"
            ):
                mult = 1.0
                if o_uip.nirPars["albtype"] == 2 and (
                    specie == "NIRALBLAMBPL" or specie == "ALBLAMBPL"
                ):
                    mult = 1 / 0.07  # convert species lambertian to uip BRDF
                if o_uip.nirPars["albtype"] == 1 and (
                    specie == "NIRALBBRDFPL" or specie == "ALBBRDFPL"
                ):
                    mult = 0.07  # convert species lambertian to uip BRDF
                if mapType == "linearpca":
                    constraint = o_uip.constraintVectorListFM[ind_fm]
                    map = ret_info.basis_matrix[ind_ret, :]
                    map = map[:, ind_fm]
                    fm_vec[ind_fm] = constraint + map.T @ o_retrieval_vec[ind]
                elif mapType == "logpca":
                    constraint = o_uip.constraintVectorListFM[ind_fm]
                    map = ret_info.basis_matrix[ind_ret, :]
                    map = map[:, ind_fm]
                    fm_vec[ind_fm] = np.exp(
                        np.log(constraint) + map.T @ o_retrieval_vec[ind]
                    )
                o_uip.nirPars["albpl"] = fm_vec[ind_fm] * mult
            elif specie == "NIRDISP" or specie == "DISP":
                # only fill in retrieved values; leave higher order as is
                npoly = int(len(fm_vec[ind_fm]) / 3)
                for iq in range(0, 3):
                    o_uip.nirPars["disp"][iq, 0:npoly] = fm_vec[
                        ind_fm[0] + npoly * iq : ind_fm[0] + npoly * (iq + 1)
                    ]

                # update wavelength if dispersion is retrieved
                # apply to uip.nirPars.alblambplwave beginning and end frequencies
                # apply to uip.uip_OCO2.wavelength and uip.uip_OCO2.frequencylist

                # dispersion applies to OBSERVED wavelength
                # therefore update OBSERVED wavelength in UIP
                ff = mpy_oco2_get_wavelength(
                    o_uip.nirPars["disp"], o_uip.uip_OCO2["sample_indexes"]
                )
                o_uip.uip_OCO2["wavelength"] = ff
                o_uip.uip_OCO2["frequencylist"] = ff
                # update albplwave, if this is used, according to wavelength ranges of bands.
                # IF uip.nirPars.albplwave[0] GE 0 THEN BEGIN
                #    result = get_nir_albedo_piecewise(fltarr(2,3)+1e-5, wavefull = uip.uip_OCO2.wavelength)
                #    uip.nirPars.albplwave = [result.wavelength1,result.wavelength2,result.wavelength3]
                # ENDIF

                # match wavelength edges
                o_uip.nirPars["albplwave"] = mpy_nir_match_wavelength_edges(
                    o_uip.uip_OCO2["wavelength"], o_uip.nirPars["albplwave"]
                )

            elif specie == "NIREOF" or specie == "EOF":
                # this is right way round
                o_uip.nirPars["eof"][:, :] = fm_vec[ind_fm].reshape(3, 3)
            elif specie == "NIRCLOUD3DOFFSET" or specie == "CLOUD3DOFFSET":
                # not sure if the right way round
                o_uip.nirPars["cloud3doffset"][:] = fm_vec[ind_fm]  # .reshape(3,2)
            elif specie == "NIRCLOUD3DSLOPE" or specie == "CLOUD3DSLOPE":
                # not sure if the right way round
                o_uip.nirPars["cloud3dslope"][:] = fm_vec[ind_fm]
            elif specie == "NIRFLUOR" or specie == "FLUOR":
                o_uip.nirPars["fluor"] = fm_vec[ind_fm]
            elif specie == "NIRWIND" or specie == "WIND":
                o_uip.nirPars["wind"] = fm_vec[ind_fm]

            #  line species
            else:
                param_pos = np.where(o_uip.atmosphere_params == specie)[0]

                if len(param_pos) == 0:
                    # This is something not in the UIP, so just ignore for now
                    continue
                    # raise RuntimeError(f'Retrieval parameter label not found {specie}')

                if isinstance(mapType, rf.StateMapping):
                    fm_vec[ind_fm] = mapType.mapped_state(
                        rf.ArrayAd_double_1(fm_vec[ind_fm])
                    ).value
                    o_uip.atmosphere[param_pos[0], :] = fm_vec[ind_fm]
                elif mapType == "linearscale":
                    initial = o_uip.constraintVectorListFM[ind_fm]
                    o_uip.atmosphere[param_pos[0], :] = (
                        initial + o_retrieval_vec[ind[0]]
                    )
                    fm_vec[ind_fm] = o_uip.atmosphere[param_pos[0], :]
                elif mapType == "linearpca":
                    initial = o_uip.constraintVectorListFM[ind_fm]
                    map = ret_info.basis_matrix[ind_ret, :]
                    map = map[:, ind_fm]
                    o_uip.atmosphere[param_pos[0], :] = (
                        initial + map.T @ o_retrieval_vec[ind]
                    )
                    fm_vec[ind_fm] = o_uip.atmosphere[param_pos[0], :]
                elif mapType == "logscale":
                    initial = o_uip.constraintVectorListFM[ind_fm]
                    o_uip.atmosphere[param_pos[0], :] = (
                        initial * o_retrieval_vec[ind[0]]
                    )
                    fm_vec[ind_fm] = o_uip.atmosphere[param_pos[0], :]
                elif mapType == "logpca":
                    initial = o_uip.constraintVectorListFM[ind_fm]
                    map = ret_info.basis_matrix[ind_ret, :]
                    map = map[:, ind_fm]
                    o_uip.atmosphere[param_pos[0], :] = np.exp(
                        np.log(initial) + map.T @ o_retrieval_vec[ind]
                    )
                    fm_vec[ind_fm] = o_uip.atmosphere[param_pos[0], :]
                elif mapType == "log":
                    fm_vec[ind_fm] = np.exp(fm_vec[ind_fm])
                    o_uip.atmosphere[param_pos[0], :] = fm_vec[ind_fm]
                elif mapType == "linear":
                    value = fm_vec[
                        ind_fm
                    ]  # Note that in Python, the slice does not include the end point.

                    # this is for negative VMRs.  Small VMRs also cause
                    # issues (get large Jacobians for very small VMRs)
                    threshold = -999.0
                    if specie == "HCN":
                        threshold = 1e-12

                    if specie == "NH3":
                        threshold = 5e-11

                    if specie == "ACET":
                        threshold = 1e-12

                    # Don't do this correction for PAN.
                    # Starting from v1.14 small and negative PAN is handled in fm_oss.pro
                    # if specie == 'PAN':
                    #     threshold = 1e-12

                    ind_neg = np.where(value < threshold)[0]
                    if len(ind_neg) > 0:
                        value[ind_neg] = threshold

                    o_uip.atmosphere[param_pos[0], :] = value
                    fm_vec[ind_fm] = value

                    # update retrieval vector
                    value = o_retrieval_vec[ind]
                    ind_neg = np.where(value < threshold)[0]
                    if len(ind_neg) > 0:
                        value[ind_neg] = threshold

                    o_retrieval_vec[ind] = (
                        value  # Use o_retrieval_vec so it is more evident that we are returning it.
                    )
                else:
                    raise RuntimeError(f"Maptype not found: {mapType}")
                # end else portion of if len(ind1) == 0 and specie != 'TATM':
            # end long else portion of 'TSUR' == specie:
        # end for ii in range(len(uip.jacobians_all)):

        # AT_LINE 256 Optimization/update_uip.pro update_uip

        # update uip with retrieval and fm vectors
        o_uip.currentGuessList = o_retrieval_vec
        o_uip.currentGuessListFM = fm_vec

        self.uip = o_uip.as_dict(o_uip)

    @classmethod
    def create_uip_from_refractor_objects(
        cls,
        obs_list: list[MusesObservation],
        cstate: CurrentState,
        rconf: MeasurementId | RetrievalConfiguration,
        pointing_angle: rf.DoubleWithUnit | None = None,
    ) -> RefractorUip:
        """Create a RefractorUIP from the higher level refractor.muses objects.

        This takes a list of observations. This can either have one observation
        to create a UIP for one instrument, or multiple to create joint UIP for
        multiple instruments (e.g., AIRS and OMI). py-retrieve created one
        UIP for joint retrievals, however for ReFRACtor we create 2 UIPS, one
        for each instrument. This has some duplication, and 2 UIPS that need to
        be updated. But it make for cleaner logic at the MusesForwardModel level
        where we don't need to worry about creating a joint UIP.

        The pointing angle can be passed in, to use this instead
        of the pointing angle found in the state_info. This is
        used by the IRK calculation.
        """
        logger.debug(
            f"Creating rf_uip for {[str(obs.instrument_name) for obs in obs_list]}"
        )
        # Special case for CurrentStateUip, we just return a copy of UIP. This is useful
        # for unit testing where we get the UIP from another source but what to pretend
        # that we are doing normal processing.
        if hasattr(cstate, "rf_uip"):
            logger.info("Copying uip from cstate rather than creating")
            return copy.deepcopy(cstate.rf_uip)
        mwin = []
        for obs in obs_list:
            mwin.extend(obs.spectral_window.muses_microwindows())
        # Dummy strategy table, with the information needed by
        # RefractorUip.create_uip
        fake_table = {
            "preferences": rconf,
            "vlidort_dir": str(cstate.step_directory / "vlidort") + "/",
            "numRows": cstate.strategy_step.step_number,
            "numColumns": 1,
            "step": cstate.strategy_step.step_number,
            "labels1": "retrievalType",
            "data": [cstate.retrieval_type.lower()] * cstate.strategy_step.step_number,
        }
        fake_state_info = FakeStateInfo(cstate, obs_list=obs_list)
        # fake_retrieval_info = FakeRetrievalInfo(cstate, use_state_mapping=True)
        fake_retrieval_info = FakeRetrievalInfo(cstate)
        if cstate.use_systematic:
            rinfo: AttrDictAdapter | FakeRetrievalInfo = (
                fake_retrieval_info.retrieval_info_systematic
            )
        else:
            rinfo = fake_retrieval_info

        o_xxx = {
            "AIRS": None,
            "TES": None,
            "CRIS": None,
            "OMI": None,
            "TROPOMI": None,
            "OCO2": None,
        }
        for obs in obs_list:
            iname = obs.instrument_name
            if str(iname) in o_xxx:
                if hasattr(obs, "muses_py_dict"):
                    o_xxx[str(iname)] = obs.muses_py_dict
        rf_uip = RefractorUip.create_uip(
            fake_state_info,  # type: ignore[arg-type]
            fake_table,
            mwin,
            rinfo,  # type: ignore[arg-type]
            o_xxx["AIRS"],
            o_xxx["TES"],
            o_xxx["CRIS"],
            o_xxx["OMI"],
            o_xxx["TROPOMI"],
            o_xxx["OCO2"],
            None,
            pointing_angle=pointing_angle,
        )
        rf_uip.run_dir = rconf["run_dir"]
        return rf_uip

    @classmethod
    def create_uip(
        cls,
        i_stateInfo: dict[str, Any] | AttrDictAdapter | FakeStateInfo,
        i_strategy_table: dict[str, Any],
        i_windows: list[Any],
        i_retrievalInfo: dict[str, Any] | AttrDictAdapter | FakeRetrievalInfo,
        i_airs: dict[str, Any] | None,
        i_tes: dict[str, Any] | None,
        i_cris: dict[str, Any] | None,
        i_omi: dict[str, Any] | None,
        i_tropomi: dict[str, Any] | None,
        i_oco2: dict[str, Any] | None,
        only_create_instrument: InstrumentIdentifier | None = None,
        pointing_angle: rf.DoubleWithUnit | None = None,
    ) -> RefractorUip:
        """We duplicate what mpy.run_retrieval does to make the uip.

        To help reduce coupling, you can give the instrument you want, we'll create
        only that UIP. Default is None for everything found in the instrument windows.
        You can supply the point angle to use for the boresight as a DoubleWithUnit.
        This can be used instead of angles found in i_stateInfo. This is used
        by the IRK calculation.
        """
        i_windows = copy.deepcopy(i_windows)
        # Filter to only include the desired instrument
        if only_create_instrument is not None:
            i_windows = [
                w for w in i_windows if w["instrument"] == str(only_create_instrument)
            ]
        # Bit if a kludge here, but we adjust the windows for the CRIS instrument
        if i_cris is not None:
            for win in i_windows:
                if win["instrument"] == "CRIS":  # EM - Necessary for joint retrievals
                    con1 = i_cris["FREQUENCY"] >= win["start"]
                    con2 = i_cris["FREQUENCY"] <= win["endd"]

                    tempind = np.where(np.logical_and(con1 == True, con2 == True))[0]

                    MAXOPD = np.unique(i_cris["MAXOPD"][tempind])
                    SPACING = np.unique(i_cris["SPACING"][tempind])

                    if len(MAXOPD) > 1 or len(SPACING) > 1:
                        raise RuntimeError(
                            "ERROR!!! Microwindowds across CrIS filter bands leading to spacing and OPD does not uniform in this MW!"
                        )

                    win["maxopd"] = np.float32(MAXOPD[0])
                    win["spacing"] = np.float32(SPACING[0])
                    win["monoextend"] = np.float32(SPACING[0]) * 4.0

        # Temp, we are sorting out the interface of i_retrievalInfo
        if hasattr(i_retrievalInfo, "retrieval_info_obj"):
            retrieval_info = copy.deepcopy(i_retrievalInfo.retrieval_info_obj)
        else:
            retrieval_info = copy.deepcopy(i_retrievalInfo)
            if isinstance(retrieval_info, dict):
                retrieval_info = AttrDictAdapter(retrieval_info)
        # Temp, and also with i_stateInfo
        if hasattr(i_stateInfo, "state_info_obj"):
            i_state = copy.deepcopy(i_stateInfo.state_info_obj)
        else:
            i_state = copy.deepcopy(i_stateInfo)
            if isinstance(i_state, dict):
                i_state = AttrDictAdapter(i_state)
        # Temp, and also with i_table
        if hasattr(i_strategy_table, "strategy_table_dict"):
            i_table = copy.deepcopy(i_strategy_table.strategy_table_dict)
        else:
            i_table = copy.deepcopy(i_strategy_table)

        jacobian_speciesNames = retrieval_info.species[0 : retrieval_info.n_species]

        # If requested, replace the pointing angle that is used. Note this
        # really is mixed units down below, TES is in radians while others are
        # in degrees. We have already copied i_state, so we don't need to worry
        # about moving these back at the end of this function.
        if pointing_angle:
            i_state.current["cris"]["scanAng"] = pointing_angle.convert("deg").value
            i_state.current["airs"]["scanAng"] = pointing_angle.convert("deg").value
            i_state.current["tes"]["boresightNadirRadians"] = pointing_angle.convert(
                "rad"
            ).value
            i_state.current["omi"]["vza_uv2"] = pointing_angle.convert("deg").value
            i_state.current["tropomi"]["vza_BAND1"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND2"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND3"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND4"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND5"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND6"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND7"] = pointing_angle.convert(
                "deg"
            ).value
            i_state.current["tropomi"]["vza_BAND8"] = pointing_angle.convert(
                "deg"
            ).value
            # TODO Not sure what the OCO-2 equivalent is here, we'll leave that off

        uip = mpy_make_uip_master(
            i_state,
            i_state.current,
            i_table,
            i_windows,
            jacobian_speciesNames,
            i_cloudIndex=0,
            i_modifyCloudFreq=True,
        )
        if i_omi is not None and hasattr(i_stateInfo, "omi_params"):
            uip["omiPars"] = i_stateInfo.omi_params(i_omi)

        # run_forward_model doesn't have mapType, not really sure why. It
        # just puts an empty list here. Similarly no n_totalParameters.
        if hasattr(retrieval_info, "mapType") and retrieval_info.n_totalParameters > 0:
            uip["jacobiansLinear"] = [
                retrieval_info.species[i]
                for i in range(len(retrieval_info.mapType))
                if retrieval_info.mapType[i] == "linear"
                and retrieval_info.species[i] not in ("EMIS", "TSUR", "TATM")
            ]
            uip["speciesList"] = copy.deepcopy(
                retrieval_info.speciesList[0 : retrieval_info.n_totalParameters]
            )
            uip["speciesListFM"] = copy.deepcopy(
                retrieval_info.speciesListFM[0 : retrieval_info.n_totalParametersFM]
            )
            uip["mapTypeListFM"] = copy.deepcopy(
                retrieval_info.mapTypeListFM[0 : retrieval_info.n_totalParametersFM]
            )
            uip["initialGuessListFM"] = copy.deepcopy(
                retrieval_info.initialGuessListFM[
                    0 : retrieval_info.n_totalParametersFM
                ]
            )
            uip["constraintVectorListFM"] = copy.deepcopy(
                retrieval_info.constraintVectorListFM[
                    0 : retrieval_info.n_totalParametersFM
                ]
            )  # only needed for PCA map type.
        else:
            uip["jacobiansLinear"] = [""]
            uip["speciesList"] = copy.deepcopy(retrieval_info.speciesList)
            uip["speciesListFM"] = copy.deepcopy(retrieval_info.speciesListFM)
            uip["mapTypeListFM"] = copy.deepcopy(
                retrieval_info.mapTypeListFM[0 : retrieval_info.n_totalParametersFM]
            )
            uip["initialGuessListFM"] = copy.deepcopy(
                retrieval_info.initialGuessListFM[
                    0 : retrieval_info.n_totalParametersFM
                ]
            )
            uip["constraintVectorListFM"] = copy.deepcopy(
                retrieval_info.constraintVectorListFM[
                    0 : retrieval_info.n_totalParametersFM
                ]
            )  # only needed for PCA map type.
        uip["microwindows_all"] = i_windows
        # Basis matrix if available, this isn't in run_forward_model.
        if (
            hasattr(retrieval_info, "mapToState")
            and retrieval_info.n_totalParameters > 0
        ):
            mmm = retrieval_info.n_totalParameters
            nnn = retrieval_info.n_totalParametersFM
            basis_matrix = retrieval_info.mapToState[0:mmm, 0:nnn]
        else:
            basis_matrix = None
        rf_uip = cls(uip, basis_matrix, state_info=i_stateInfo)

        # Group windows by instrument
        inst_to_window = defaultdict(list)
        for w in i_windows:
            inst_to_window[w["instrument"]].append(w)
        if "AIRS" in inst_to_window:
            if i_airs is None:
                raise RuntimeError("Need to supply i_airs")
            uip["uip_AIRS"] = mpy_make_uip_airs(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["AIRS"],
                uip["jacobians_all"],
                # OSS and uip code doesn't handle empty species list. We run
                # into that with the BT step. So we add a simple H2O species, even
                # though we don't actually use the resulting jacobian. But put in
                # so the code is happy
                uip["speciesListFM"]
                if len(uip["speciesListFM"]) > 0
                else [
                    "H2O",
                ],
                None,
                i_airs["radiance"],
                i_modifyCloudFreq=True,
            )

        if "CRIS" in inst_to_window:
            if i_cris is None:
                raise RuntimeError("Need to supply i_cris")
            uip["uip_CRIS"] = mpy_make_uip_cris(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["CRIS"],
                uip["jacobians_all"],
                # OSS and uip code doesn't handle empty species list. We run
                # into that with the BT step. So we add a simple H2O species, even
                # though we don't actually use the resulting jacobian. But put in
                # so the code is happy
                uip["speciesListFM"]
                if len(uip["speciesListFM"]) > 0
                else [
                    "H2O",
                ],
                i_cris["radianceStruct".upper()],
                i_modifyCloudFreq=True,
            )

        if "TES" in inst_to_window:
            if i_tes is None:
                raise RuntimeError("Need to supply i_tes")
            uip["uip_TES"] = mpy_make_uip_tes(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["TES"],
                i_tes["radianceStruct"],
                "",
                # OSS and uip code doesn't handle empty species list. We run
                # into that with the BT step. So we add a simple H2O species, even
                # though we don't actually use the resulting jacobian. But put in
                # so the code is happy
                uip["jacobians_all"]
                if len(uip["jacobians_all"]) > 0
                else [
                    "H2O",
                ],
            )
        if "OMI" in inst_to_window:
            uip["uip_OMI"] = mpy_make_uip_omi(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["OMI"],
                uip["jacobians_all"],
                uip,
                i_omi,
            )
        if "TROPOMI" in inst_to_window:
            uip["uip_TROPOMI"] = mpy_make_uip_tropomi(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["TROPOMI"],
                uip["jacobians_all"],
                uip,
                i_tropomi,
            )
        if "OCO2" in inst_to_window:
            # Catch us not setting pointing angle for OCO-2
            if pointing_angle is not None:
                raise RuntimeError(
                    "Don't currently support setting the pointing_angle for OCO-2, we just need to update the code to do that if needed."
                )
            uip["uip_OCO2"] = mpy_make_uip_oco2(
                i_state,
                i_state.current,
                i_table,
                inst_to_window["OCO2"],
                uip["jacobians_all"],
                uip,
                i_oco2,
            )

        # Correct surface pointing angle. Not sure why this needs to be
        # done, but this matches what run_retrieval does. Note OCO-2 has
        # this same logic, but commented out. OCO-2 isn't currently working,
        # so not sure if it should have this or not.
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "TES"):
            if f"uip_{k}" in uip:
                uip[f"uip_{k}"]["obs_table"]["pointing_angle_surface"] = (
                    rf_uip.ray_info(k, set_pointing_angle_zero=False)[
                        "ray_angle_surface"
                    ]
                )

        # Make jacobians entry only have unique element.
        #
        # Note that starting with python 3.7 dict preserves insertion order
        # (guaranteed, 3.6 actually did this also but it was just an
        # implementation detail rather than guaranteed),
        # so list(dict.fromkeys(v)) will have a list of unique elements in the
        # order that the first of each item appears
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if f"uip_{k}" in uip:
                uip[f"uip_{k}"]["jacobians"] = np.array(
                    list(dict.fromkeys(uip[f"uip_{k}"]["jacobians"]))
                )

        # Create instrument list
        uip["instrumentList"] = []
        for k in ("AIRS", "CRIS", "OMI", "TROPOMI", "OCO2"):
            if f"uip_{k}" in uip:
                uip["instrumentList"].extend(
                    [
                        k,
                    ]
                    * len(uip[f"uip_{k}"]["frequencyList"])
                )

        # Add extra pieces to the microwindows.
        for w in i_windows:
            for k in ("enddmw_fm", "enddmw", "startmw_fm", "startmw"):
                if k not in w:
                    w[k] = 0

        if basis_matrix is not None and retrieval_info.n_totalParameters > 0:
            xig = retrieval_info.initialGuessList[0 : retrieval_info.n_totalParameters]
            rf_uip.update_uip(xig)
        else:
            uip["currentGuessList"] = retrieval_info.initialGuessList
            uip["currentGuessListFM"] = retrieval_info.initialGuessListFM

        return rf_uip


__all__ = ["RefractorUip", "AttrDictAdapter"]
