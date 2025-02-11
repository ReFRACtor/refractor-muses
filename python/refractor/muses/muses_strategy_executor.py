from __future__ import annotations
from .misc import osp_setup
from contextlib import contextmanager
from .refractor_capture_directory import muses_py_call
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .retrieval_info import RetrievalInfo
from .error_analysis import ErrorAnalysis
from .order_species import order_species
from .current_state import CurrentStateStateInfo
from .qa_data_handle import QaDataHandleSet
from .muses_strategy import (
    MusesStrategyOldStrategyTable,
    MusesStrategy,
    CurrentStrategyStep,
)
from .observation_handle import ObservationHandleSet
from .refractor_uip import RefractorUip
from .identifier import StateElementIdentifier
import refractor.framework as rf  # type: ignore
import abc
import copy
import os
from loguru import logger
import time
import functools
import numpy as np
import numpy.testing as npt
from pathlib import Path
import typing
from typing import Callable

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .muses_observation import MusesObservation, MeasurementId
    from .cost_function import CostFunction
    from .retrieval_configuration import RetrievalConfiguration
    from .state_info import StateInfo
    from .cost_function_creator import CostFunctionCreator
    from .current_state import CurrentState
    from .identifier import InstrumentIdentifier, FilterIdentifier


def log_timing(f):
    """Decorator to log the timing of a function."""

    @functools.wraps(f)
    def log_tm(*args, **kwargs):
        start_date = time.strftime("%c")
        start_time = time.time()
        res = f(*args, **kwargs)
        stop_date = time.strftime("%c")
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        elapsed_time_seconds = stop_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60.0
        logger.info("\n---")
        logger.info(f"start_date {start_date}")
        logger.info(f"stop_date {stop_date}")
        logger.info(f"elapsed_time {elapsed_time}")
        logger.info(f"elapsed_time_seconds {elapsed_time_seconds}")
        logger.info(f"elapsed_time_minutes {elapsed_time_minutes}")
        return res

    return log_tm


def struct_compare(s1, s2, skip_list=None, verbose=False):
    if skip_list is None:
        skip_list = []
    for k in s1.keys():
        if k in skip_list:
            if verbose:
                print(f"Skipping {k}")
            continue
        if verbose:
            print(k)
        if isinstance(s1[k], np.ndarray) and np.can_cast(s1[k], np.float64):
            npt.assert_allclose(s1[k], s2[k])
        elif isinstance(s1[k], np.ndarray):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]


def array_compare(s1, s2, skip_list=None, verbose=False):
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        struct_compare(s1[i], s2[i], skip_list=skip_list, verbose=verbose)


class MusesStrategyExecutor(object, metaclass=abc.ABCMeta):
    """This is the base class for executing a strategy.

    Note that there a refractor.framework class StrategyExecutor. This
    class has a similar intention as that older StrategyExecutor
    class, however this really is a complete rewrite of this for the
    way py-retrieve does this. It is possible that these classes might
    get merged at some point, but for now it is better to think of
    these as completely separate classes that just happen to have
    similar names.

    The canonical way of determining the strategy is to read the old
    strategy table ("Table.asc") that amuse-me populates.

    This base class provides an abstract interface so we can have
    different implementations of executing a strategy.

    It isn't clear how much flexibility we actually need here, a lot
    of the configuration/customization happens at a lower level of
    processing (e.g., ForwardModelHandleSet), but we'll go ahead a set
    up the inheritance structure since this is pretty cheap. If we
    only end up with one implementation that's fine.

    """

    pass


class MusesStrategyExecutorRetrievalStrategyStep(MusesStrategyExecutor):
    """Much of the time our strategy is going to depend on having
    a RetrievalStrategyStepSet to get the RetrievalStrategyStep based
    off a retrieval type name. This adds that functionality."""

    def __init__(
        self,
        retrieval_config: RetrievalConfiguration,
        run_dir: Path,
        state_info: StateInfo,
        cost_function_creator: CostFunctionCreator,
        observation_handle_set: ObservationHandleSet | None = None,
        retrieval_strategy_step_set: RetrievalStrategyStepSet | None = None,
        qa_data_handle_set: QaDataHandleSet | None = None,
        vlidort_cli: Path | None = None,
        **kwargs,
    ):
        self.retrieval_config = retrieval_config
        self.retrieval_info: RetrievalInfo | None = None
        self.vlidort_cli = vlidort_cli
        self.run_dir = run_dir
        self.state_info = state_info
        self.cost_function_creator = cost_function_creator
        self.kwargs = kwargs
        if observation_handle_set is None:
            self.observation_handle_set = copy.deepcopy(
                ObservationHandleSet.default_handle_set()
            )
        else:
            self.observation_handle_set = observation_handle_set
        if retrieval_strategy_step_set is None:
            self._retrieval_strategy_step_set = copy.deepcopy(
                RetrievalStrategyStepSet.default_handle_set()
            )
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set
        if qa_data_handle_set is None:
            self._qa_data_handle_set = copy.deepcopy(
                QaDataHandleSet.default_handle_set()
            )
        else:
            self._qa_data_handle_set = qa_data_handle_set
        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId):
        """Have updated the target we are processing."""
        self.measurement_id = measurement_id
        self.qa_data_handle_set.notify_update_target(self.measurement_id)

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        raise NotImplementedError

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        raise NotImplementedError()

    @property
    def qa_data_handle_set(self):
        """The QaDataHandleSet to use to get the QA flag filename."""
        return self._qa_data_handle_set

    @property
    def retrieval_strategy_step_set(self):
        """The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep."""
        return self._retrieval_strategy_step_set

    def rf_uip_func_cost_function(
        self, do_systematic, jacobian_speciesIn
    ) -> Callable[[InstrumentIdentifier | None], RefractorUip]:
        return functools.partial(
            self._rf_uip_func,
            do_systematic=do_systematic,
            jacobian_speciesIn=jacobian_speciesIn,
        )

    def rf_uip_irk(self, obs: MusesObservation, pointing_angle: rf.DoubleWithUnit):
        return self._rf_uip_func(
            obs.instrument_name,
            obs_list=[
                obs,
            ],
            pointing_angle=pointing_angle,
        )

    def _rf_uip_func(
        self,
        instrument: InstrumentIdentifier,
        obs_list: list[MusesObservation] | None = None,
        do_systematic=False,
        jacobian_speciesIn=None,
        pointing_angle: rf.DoubleWithUnit | None = None,
    ):
        """To reduce coupling, you can give the instrument name to
        use.

        You can also pass in the observation list. Normally this
        function can just create this for you using our
        observation_handle_set. But the AIRS IRK creates a fake TES
        observation, which we want to be able to pass in.

        Also the pointing angle can be passed in, to use this instead
        of the pointing angle found in the state_info. Again, this is
        used by the IRK calculation.

        """
        logger.debug(f"Creating rf_uip for {instrument}")
        if self.retrieval_info is None:
            raise RuntimeError("Need to have retrieval_info defined")
        if do_systematic:
            rinfo = self.retrieval_info.retrieval_info_systematic()
        else:
            rinfo = self.retrieval_info
        cstep = self.current_strategy_step
        if obs_list is None:
            obs_list = []
            for iname in cstep.instrument_name:
                if instrument is None or iname == instrument:
                    obs = self.observation_handle_set.observation(
                        iname, None, cstep.spectral_window_dict[iname], None
                    )
                    obs_list.append(obs)
        mwin = []
        for obs in obs_list:
            mwin.extend(obs.spectral_window.muses_microwindows())
        # Dummy strategy table, with the information needed by
        # RefractorUip.create_uip
        fake_table = {
            "preferences": self.retrieval_config,
            "vlidort_dir": str(
                self.run_dir
                / f"Step{self.current_strategy_step.step_number:02d}_{self.current_strategy_step.step_name}/vlidort/"
            ),
            "numRows": cstep.step_number,
            "numColumns": 1,
            "step": cstep.step_number,
            "labels1": "retrievalType",
            "data": [cstep.retrieval_type.lower()] * cstep.step_number,
        }
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
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            return RefractorUip.create_uip(
                self.state_info,
                fake_table,
                mwin,
                rinfo,
                o_xxx["AIRS"],
                o_xxx["TES"],
                o_xxx["CRIS"],
                o_xxx["OMI"],
                o_xxx["TROPOMI"],
                o_xxx["OCO2"],
                jacobian_speciesIn=jacobian_speciesIn,
                only_create_instrument=instrument,
                pointing_angle=pointing_angle,
            )

    def current_state(
        self, do_systematic=False, jacobian_speciesIn=None
    ) -> CurrentState:
        return CurrentStateStateInfo(
            self.state_info,
            self.retrieval_info,
            self.run_dir
            / f"Step{self.current_strategy_step.step_number:02d}_{self.current_strategy_step.step_name}",
            do_systematic=do_systematic,
            retrieval_state_element_override=jacobian_speciesIn,
        )

    def create_forward_model(self):
        """Create a forward model for the current step."""
        if len(self.current_strategy_step.instrument_name) != 1:
            raise RuntimeError(
                "create_forward_model can only work with one instrument, we don't have handling for multiple."
            )
        iname = self.current_strategy_step.instrument_name[0]
        obs = self.observation_handle_set.observation(
            iname, None, self.current_strategy_step.spectral_window_dict[iname], None
        )
        fm_sv = rf.StateVector()
        return self.rs.forward_model_handle_set.forward_model(
            iname,
            self.current_state(),
            obs,
            fm_sv,
            self.rf_uip_func_cost_function(False, None),
        )

    def create_cost_function(
        self,
        do_systematic=False,
        include_bad_sample=False,
        fix_apriori_size=False,
        jacobian_speciesIn=None,
    ) -> CostFunction:
        """Create a CostFunction, for use either in retrieval or just
        for running the forward model (the CostFunction is a little
        overkill for just a forward model run, but it has all the
        pieces needed so no reason not to just generate everything).

        If do_systematic is True, then we use the systematic species list.

        """
        # TODO Would probably be good to remove include_bad_sample, it
        # isn't clear that we ever want to run the forward model for
        # bad samples. But right now the existing py-retrieve code
        # requires this is a few places.a
        return self.cost_function_creator.cost_function(
            self.current_strategy_step.instrument_name,
            self.current_state(
                do_systematic=do_systematic, jacobian_speciesIn=jacobian_speciesIn
            ),
            self.current_strategy_step.spectral_window_dict,
            self.rf_uip_func_cost_function(do_systematic, jacobian_speciesIn),
            include_bad_sample=include_bad_sample,
            **self.kwargs,
        )


class MusesStrategyExecutorOldStrategyTable(MusesStrategyExecutorRetrievalStrategyStep):
    """Placeholder that wraps the muses-py strategy table up, so we
    can get the infrastructure in place before all the pieces are
    ready

    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        rs: RetrievalStrategy,
        osp_dir=None,
        retrieval_strategy_step_set=None,
        spectral_window_handle_set=None,
        qa_data_handle_set=None,
    ):
        super().__init__(
            rs.retrieval_config,
            rs.run_dir,
            rs.state_info,
            rs._cost_function_creator,
            observation_handle_set=rs.observation_handle_set,
            retrieval_strategy_step_set=retrieval_strategy_step_set,
            qa_data_handle_set=qa_data_handle_set,
            **rs.keyword_arguments,
        )
        self.strategy: MusesStrategy = MusesStrategyOldStrategyTable(
            filename,
            osp_dir=osp_dir,
            spectral_window_handle_set=spectral_window_handle_set,
        )
        self.osp_dir: Path | None = Path(osp_dir) if osp_dir is not None else None
        self.rs = rs
        self.retrieval_info: RetrievalInfo | None = None
        self.measurement_id: MeasurementId | None = None

    @property
    def spectral_window_handle_set(self):
        """The SpectralWindowHandleSet to use for getting the MusesSpectralWindow."""
        return self.strategy.spectral_window_handle_set

    def notify_update_target(self, measurement_id: MeasurementId):
        super().notify_update_target(measurement_id)
        self.strategy.notify_update_target(measurement_id)

    def notify_update(self, location: str, **kwargs):
        self.rs.notify_update(location, **kwargs)

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        return self.strategy.filter_list_dict

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        return self.strategy.current_strategy_step()

    def restart(self) -> None:
        """Set step to the first one."""
        self.strategy.restart()

    def set_step(self, step_number: int) -> None:
        """Go to the given step. This is used by RetrievalStrategy.load_state_info
        where we jump to a given step number."""
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            self._restart_and_error_analysis()
            while self.current_strategy_step.step_number < step_number:
                self.next_step()

    def _restart_and_error_analysis(self) -> None:
        '''Restart and recreate error analysis. Put together just for convenience,
        so we can use with "set_step"'''
        # List of state elements we need covariance from. This is all the elements
        # we will retrieve, plus any interferents that get added in. This list
        # is unique elements, sorted by the order_species sorting
        covariance_state_element_name = [
            StateElementIdentifier(i)
            for i in order_species(
                list(
                    set([str(i) for i in self.strategy.retrieval_elements])
                    | set([str(i) for i in self.strategy.error_analysis_interferents])
                )
            )
        ]

        self.restart()
        # TODO
        # Note clear why, but we get slightly different results if we
        # update the original state_info. May want to track this down,
        # but as a work around we just copy this. This is just needed
        # to get the mapping type, I don't think anything else is
        # needed. We should be able to pull that out from the full
        # initial guess update at some point, so we don't need to do
        # the full initial guess
        sinfo = copy.deepcopy(self.state_info)
        for sname in covariance_state_element_name:
            selem = sinfo.state_element(sname)
            selem.update_initial_guess(self.current_strategy_step)
        self.error_analysis = ErrorAnalysis(sinfo, covariance_state_element_name)

    def next_step(self) -> None:
        """Advance to the next step"""
        self.strategy.next_step(self.current_state())

    def is_done(self) -> bool:
        """Return true if we are done, otherwise false."""
        return self.strategy.is_done()

    @property
    def instrument_name_all_step(self):
        return self.strategy.instrument_name

    @property
    def error_analysis_interferents_all_step(self):
        return self.strategy.error_analysis_interferents

    @property
    def retrieval_elements_all_step(self):
        return self.strategy.retrieval_elements

    def get_initial_guess(self):
        """Set retrieval_info, errorInitial and errorCurrent for the current step."""
        # Temp, we'll want to get this update done automatically. But do this
        # to figure out issue
        self.retrieval_info = RetrievalInfo(
            self.error_analysis,
            Path(self.retrieval_config["speciesDirectory"]),
            self.current_strategy_step,
            self.state_info,
        )

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self.retrieval_info.n_totalParameters
        logger.info(
            f"Step: {self.current_strategy_step.step_number}, Total Parameters: {nparm}"
        )

        if nparm > 0:
            xig = self.retrieval_info.initial_guess_list[0:nparm]
            self.state_info.update_state(
                self.retrieval_info,
                xig,
                [],
                self.retrieval_config,
                self.current_strategy_step.step_number,
            )

    def number_steps_left(self, retrieval_element_name: str):
        """This returns the number of retrieval steps left that
        contain a given retrieval element name. This is an odd seeming
        function, but is used by RetrievalL2Output to name files. So
        for example we have Products_L2-O3-0.nc for the last step that
        retrieves O3, Products_L2-O3-1.nc for the previous step
        retrieving O3, etc.

        I'm not sure if this is something we can calculate in general
        for a StrategyExecutor (what if some decision is added if a
        future step is run or not?)  If this occurs, we can perhaps
        come up with a different naming convention.  Right now, this
        function is *only* used in RetrievalL2Output, so we can update
        this if needed.

        """
        step_number_start = self.current_strategy_step.step_number
        state_start = self.current_state()
        res = 0
        self.next_step()
        while not self.is_done():
            if retrieval_element_name in self.current_strategy_step.retrieval_elements:
                res += 1
            self.next_step()
        self.strategy.set_step(step_number_start, state_start)
        return res

    def run_step(self):
        """Run a the current step."""
        self.state_info.copy_current_initial()
        logger.info("\n---")
        logger.info(
            f"Step: {self.current_strategy_step.step_number}, Step Name: {self.current_strategy_step.step_name}"
        )
        logger.info("\n---")
        self.get_initial_guess()
        self.notify_update("done get_initial_guess")
        logger.info(
            f"Step: {self.current_strategy_step.step_number}, Retrieval Type {self.current_strategy_step.retrieval_type}"
        )
        self.retrieval_strategy_step_set.retrieval_step(
            self.current_strategy_step.retrieval_type,
            self.rs,
            **self.current_strategy_step.retrieval_step_parameters,
            **self.kwargs,
        )
        self.notify_update("done retrieval_step")
        self.state_info.next_state_to_current()
        self.notify_update("done next_state_to_current")
        logger.info(f"Done with step {self.current_strategy_step.step_number}")

    @log_timing
    def execute_retrieval(self, stop_at_step=None):
        """Run through all the steps, i.e., do a full retrieval.

        Note for various testing purposes, you can have the retrieval
        stop at the given step. This can be useful for looking at
        problems with an individual step, or to run a simulation at a
        particular step.
        """
        # Currently the initialization etc. code assumes we are in the run directory.
        # Hopefully we can remove this in the future, but for now we need this
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            self.execute_retrieval_body(stop_at_step=stop_at_step)

    def execute_retrieval_body(self, stop_at_step=None):
        """Run through all the steps, i.e., do a full retrieval.

        Note for various testing purposes, you can have the retrieval
        stop at the given step. This can be useful for looking at
        problems with an individual step, or to run a simulation at a
        particular step.
        """
        if self.measurement_id is None:
            raise RuntimeError(
                "Need to call notify_update_target before calling this function"
            )
        with self.chdir_run_dir():
            self.state_info.init_state(
                self.measurement_id,
                self.observation_handle_set,
                self.retrieval_elements_all_step,
                self.error_analysis_interferents_all_step,
                self.instrument_name_all_step,
                self.run_dir,
                osp_dir=self.osp_dir,
            )

        self._restart_and_error_analysis()
        self.notify_update("initial set up done")

        # Note the original muses-py ran through all the initial guess
        # steps at the beginning to make sure there weren't any
        # issues. I think we can remove this, it isn't particularly
        # important to fail early and it seems a waste of time to go
        # through this twice.
        #
        # However, the output actually changes if we don't run
        # this. This is bad, our initial guess shouldn't modify future
        # running. We should track this down when we start working on
        # the initial guess/state info portion. But for now, leave
        # this in place until we understand this
        self.restart()
        while not self.is_done():
            self.get_initial_guess()
            self.next_step()
        # Not sure that this is needed or used anywhere, but for now
        # go ahead and this this until we know for sure it doesn't
        # matter.
        self.state_info.copy_current_initialInitial()
        self.notify_update("starting retrieval steps")
        self.restart()
        while not self.is_done():
            if (
                stop_at_step is not None
                and stop_at_step == self.current_strategy_step.step_number
            ):
                return
            self.notify_update("starting run_step")
            self.run_step()
            self.next_step()

    def continue_retrieval(self, stop_after_step=None) -> None:
        """After saving a pickled step, you can continue the processing starting
        at that step to diagnose a problem."""
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            while not self.is_done():
                self.notify_update("starting run_step")
                self.run_step()
                if (
                    stop_after_step is not None
                    and stop_after_step == self.current_strategy_step.step_number
                ):
                    return
                self.next_step()

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
                os.chdir(self.run_dir)
                yield
            finally:
                os.chdir(curdir)


__all__ = [
    "MusesStrategyExecutor",
    "MusesStrategyExecutorRetrievalStrategyStep",
    "MusesStrategyExecutorOldStrategyTable",
]
