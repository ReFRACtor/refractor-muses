from __future__ import annotations
from .misc import osp_setup
from contextlib import contextmanager
from .refractor_capture_directory import muses_py_call
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .error_analysis import ErrorAnalysis
from .order_species import order_species
from .current_state import CurrentState
from .current_state_state_info import CurrentStateStateInfo
from .qa_data_handle import QaDataHandleSet
from .muses_strategy import (
    MusesStrategy,
    CurrentStrategyStep,
)
from .observation_handle import ObservationHandleSet
from .refractor_uip import RefractorUip
from .identifier import StateElementIdentifier, ProcessLocation
from .muses_strategy import MusesStrategyHandleSet
from .spectral_window_handle import SpectralWindowHandleSet
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
from .retrieval_strategy_step import RetrievalStepCaptureObserver
import refractor.framework as rf  # type: ignore
import abc
import copy
import os
import pickle
from loguru import logger
import time
import functools
import numpy as np
import numpy.testing as npt
from pathlib import Path
import typing
from typing import Callable, Generator, Any

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .muses_observation import MusesObservation, MeasurementId
    from .cost_function import CostFunction
    from .retrieval_configuration import RetrievalConfiguration
    from .cost_function_creator import CostFunctionCreator
    from .identifier import InstrumentIdentifier, FilterIdentifier
    from .state_info import StateElementHandleSet


@contextmanager
def log_timing() -> Generator[None, None, None]:
    start_date = time.strftime("%c")
    start_time = time.time()
    yield
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


def struct_compare(
    s1: dict, s2: dict, skip_list: None | list[str] = None, verbose: bool = False
) -> None:
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


def array_compare(
    s1: list[dict],
    s2: list[dict],
    skip_list: None | list[str] = None,
    verbose: bool = False,
) -> None:
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
        rs: RetrievalStrategy,
        observation_handle_set: ObservationHandleSet | None = None,
        muses_strategy_handle_set: MusesStrategyHandleSet | None = None,
        retrieval_strategy_step_set: RetrievalStrategyStepSet | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        qa_data_handle_set: QaDataHandleSet | None = None,
        vlidort_cli: Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.rs = rs
        self.current_state = CurrentStateStateInfo()
        self.vlidort_cli = vlidort_cli
        self.kwargs = copy.copy(kwargs)
        if vlidort_cli is not None:
            self.kwargs["vlidort_cli"] = vlidort_cli
        if observation_handle_set is None:
            self.observation_handle_set = copy.deepcopy(
                ObservationHandleSet.default_handle_set()
            )
        else:
            self.observation_handle_set = observation_handle_set
        if muses_strategy_handle_set is None:
            self._muses_strategy_handle_set = copy.deepcopy(
                MusesStrategyHandleSet.default_handle_set()
            )
        else:
            self._muses_strategy_handle_set = muses_strategy_handle_set
        if retrieval_strategy_step_set is None:
            self._retrieval_strategy_step_set = copy.deepcopy(
                RetrievalStrategyStepSet.default_handle_set()
            )
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set

        if spectral_window_handle_set is None:
            self._spectral_window_handle_set = copy.deepcopy(
                SpectralWindowHandleSet.default_handle_set()
            )
        else:
            self._spectral_window_handle_set = spectral_window_handle_set

        if qa_data_handle_set is None:
            self._qa_data_handle_set = copy.deepcopy(
                QaDataHandleSet.default_handle_set()
            )
        else:
            self._qa_data_handle_set = qa_data_handle_set

        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Have updated the target we are processing."""
        self.measurement_id = measurement_id
        self.qa_data_handle_set.notify_update_target(self.measurement_id)

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        return self.rs.retrieval_config

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self.current_state.state_element_handle_set

    @property
    def run_dir(self) -> Path:
        return self.rs.run_dir

    @property
    def cost_function_creator(self) -> CostFunctionCreator:
        return self.rs.cost_function_creator

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        raise NotImplementedError

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        raise NotImplementedError()

    @property
    def qa_data_handle_set(self) -> QaDataHandleSet:
        """The QaDataHandleSet to use to get the QA flag filename."""
        return self._qa_data_handle_set

    @property
    def muses_strategy_handle_set(self) -> MusesStrategyHandleSet:
        """The MusesStrategyHandleSet used for getting MusesStrategy"""
        return self._muses_strategy_handle_set

    @property
    def retrieval_strategy_step_set(self) -> RetrievalStrategyStepSet:
        """The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep."""
        return self._retrieval_strategy_step_set

    @property
    def spectral_window_handle_set(self) -> SpectralWindowHandleSet:
        """The SpectralWindowHandleSet to use for getting MusesSpectralWindow."""
        return self._spectral_window_handle_set

    def rf_uip_func_cost_function(
        self,
        do_systematic: bool,
        jacobian_speciesIn: list[StateElementIdentifier] | None,
    ) -> Callable[[InstrumentIdentifier | None], RefractorUip]:
        return functools.partial(
            self._rf_uip_func,
            do_systematic=do_systematic,
            jacobian_speciesIn=jacobian_speciesIn,
        )

    def rf_uip_irk(
        self, obs: MusesObservation, pointing_angle: rf.DoubleWithUnit
    ) -> RefractorUip:
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
        do_systematic: bool = False,
        jacobian_speciesIn: None | list[StateElementIdentifier] = None,
        pointing_angle: rf.DoubleWithUnit | None = None,
    ) -> RefractorUip:
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
                / f"Step{self.current_strategy_step.strategy_step.step_number:02d}_{self.current_strategy_step.strategy_step.step_name}/vlidort/"
            ),
            "numRows": cstep.strategy_step.step_number,
            "numColumns": 1,
            "step": cstep.strategy_step.step_number,
            "labels1": "retrievalType",
            "data": [cstep.retrieval_type.lower()] * cstep.strategy_step.step_number,
        }
        fake_state_info = FakeStateInfo(self.current_state, obs_list=obs_list)
        fake_retrieval_info = FakeRetrievalInfo(self.current_state)
        if do_systematic:
            rinfo = fake_retrieval_info.retrieval_info_systematic
        else:
            rinfo = fake_retrieval_info

        # Maybe an update happens in the UIP, that doesn't get propogated back to
        # state_info? See if we can figure out what is changed here
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
                fake_state_info,
                fake_table,
                mwin,
                rinfo,
                o_xxx["AIRS"],
                o_xxx["TES"],
                o_xxx["CRIS"],
                o_xxx["OMI"],
                o_xxx["TROPOMI"],
                o_xxx["OCO2"],
                jacobian_speciesIn=[str(i) for i in jacobian_speciesIn]
                if jacobian_speciesIn is not None
                else None,
                only_create_instrument=instrument,
                pointing_angle=pointing_angle,
            )

    def create_forward_model(self) -> rf.ForwardModel:
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
            self.current_state,
            obs,
            fm_sv,
            self.rf_uip_func_cost_function(False, None),
        )

    def create_cost_function(
        self,
        do_systematic: bool = False,
        include_bad_sample: bool = False,
        use_empty_apriori: bool = False,
        jacobian_speciesIn: None | list[StateElementIdentifier] = None,
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
        # requires this is a few places.
        cstate: CurrentState = self.current_state
        if do_systematic or jacobian_speciesIn is not None:
            cstate = self.current_state.current_state_override(
                do_systematic, retrieval_state_element_override=jacobian_speciesIn
            )
        return self.cost_function_creator.cost_function(
            self.current_strategy_step.instrument_name,
            cstate,
            self.current_strategy_step.spectral_window_dict,
            self.rf_uip_func_cost_function(do_systematic, jacobian_speciesIn),
            include_bad_sample=include_bad_sample,
            use_empty_apriori=use_empty_apriori,
            **self.kwargs,
        )


class MusesStrategyExecutorMusesStrategy(MusesStrategyExecutorRetrievalStrategyStep):
    """This is a strategy executor that uses a MusesStrategy to determine the
    strategy.

    It isn't clear if we will ever need a different strategy executor, having different
    MusesStrategy may be all the flexibility we need. But we go ahead and set up
    the infrastructure here since it is fairly cheap to do so, just in case we need
    a different implementation in the future.
    """

    def __init__(
        self,
        rs: RetrievalStrategy,
        osp_dir: None | str | os.PathLike[str] = None,
        muses_strategy_handle_set: None | MusesStrategyHandleSet = None,
        retrieval_strategy_step_set: None | RetrievalStrategyStepSet = None,
        spectral_window_handle_set: None | SpectralWindowHandleSet = None,
        qa_data_handle_set: None | QaDataHandleSet = None,
    ) -> None:
        super().__init__(
            rs,
            observation_handle_set=rs.observation_handle_set,
            muses_strategy_handle_set=muses_strategy_handle_set,
            retrieval_strategy_step_set=retrieval_strategy_step_set,
            spectral_window_handle_set=spectral_window_handle_set,
            qa_data_handle_set=qa_data_handle_set,
            **rs.keyword_arguments,
        )
        self._strategy: MusesStrategy | None = None
        self.osp_dir: Path | None = Path(osp_dir) if osp_dir is not None else None
        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        super().notify_update_target(measurement_id)
        self._strategy = self.muses_strategy_handle_set.muses_strategy(
            measurement_id,
            osp_dir=self.osp_dir,
            spectral_window_handle_set=self.spectral_window_handle_set,
        )
        self.strategy.notify_update_target(measurement_id)
        # Only do notify_update_target if we already have the filter_list_dict filled in.
        # If we don't just skip this
        if len(measurement_id.filter_list_dict) > 0:
            self.current_state.notify_update_target(
                measurement_id,
                self.retrieval_config,
                self.strategy,
                self.observation_handle_set,
            )

    @property
    def strategy(self) -> MusesStrategy:
        if self._strategy is None:
            raise RuntimeError("Call update_target before this function")
        return self._strategy

    def notify_update(self, location: str | ProcessLocation, **kwargs: Any) -> None:
        self.rs.notify_update(location, **kwargs)

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        return self.strategy.filter_list_dict

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        cstep = self.strategy.current_strategy_step()
        if cstep is None:
            raise RuntimeError("current_strategy_step called after the last step.")
        return cstep

    def restart(self) -> None:
        """Set step to the first one."""
        self.strategy.restart()
        self.current_state.notify_start_retrieval(
            self.current_strategy_step, self.retrieval_config
        )

    def notify_new_step(self, skip_initial_guess_update: bool = False) -> None:
        """Called to notify other object that we are on a new retrieval step."""
        self.current_state.notify_new_step(
            self.current_strategy_step,
            self.error_analysis,
            self.retrieval_config,
            skip_initial_guess_update=skip_initial_guess_update,
        )

    def set_step(self, step_number: int) -> None:
        """Go to the given step. This is used by RetrievalStrategy.load_state_info
        where we jump to a given step number."""
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            self._restart_and_error_analysis()
            while self.current_strategy_step.strategy_step.step_number < step_number:
                self.notify_new_step(skip_initial_guess_update=True)
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
        self.error_analysis = ErrorAnalysis(
            self.current_state,
            self.current_strategy_step,
            covariance_state_element_name,
        )

    def next_step(self) -> None:
        """Advance to the next step."""
        self.strategy.next_step(self.current_state)
        cstep = self.strategy.current_strategy_step()
        if cstep is not None:
            logger.info(str(cstep.strategy_step))

    def is_done(self) -> bool:
        """Return true if we are done, otherwise false."""
        return self.strategy.is_done()

    @property
    def instrument_name_all_step(self) -> list[InstrumentIdentifier]:
        return self.strategy.instrument_name

    @property
    def error_analysis_interferents_all_step(self) -> list[StateElementIdentifier]:
        return self.strategy.error_analysis_interferents

    @property
    def retrieval_elements_all_step(self) -> list[StateElementIdentifier]:
        return self.strategy.retrieval_elements

    def run_step(self) -> None:
        """Run a the current step."""
        logger.info("\n---")
        logger.info(str(self.current_strategy_step.strategy_step))
        logger.info("\n---")
        logger.info(
            f"Step: {self.current_strategy_step.strategy_step.step_number}, Retrieval Type {self.current_strategy_step.retrieval_type}"
        )
        self.retrieval_strategy_step_set.retrieval_step(
            self.current_strategy_step.retrieval_type,
            self.rs,
            **self.current_strategy_step.retrieval_step_parameters,
            **self.kwargs,
        )
        self.notify_update(ProcessLocation("done retrieval_step"))
        logger.info(f"Done with {str(self.current_strategy_step.strategy_step)}")

    def execute_retrieval(self, stop_at_step: None | int = None) -> None:
        """Run through all the steps, i.e., do a full retrieval.

        Note for various testing purposes, you can have the retrieval
        stop at the given step. This can be useful for looking at
        problems with an individual step, or to run a simulation at a
        particular step.
        """
        with log_timing():
            # Currently the initialization etc. code assumes we are in the run directory.
            # Hopefully we can remove this in the future, but for now we need this
            with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
                self.execute_retrieval_body(stop_at_step=stop_at_step)

    def execute_retrieval_body(self, stop_at_step: None | int = None) -> None:
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
        self._restart_and_error_analysis()
        self.notify_update(ProcessLocation("initial set up done"))

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
            self.notify_new_step(skip_initial_guess_update=True)
            self.next_step()
        # Not sure that this is needed or used anywhere, but for now
        # go ahead and this this until we know for sure it doesn't
        # matter.
        # self.current_state._state_info.copy_current_initialInitial()
        self.notify_update(ProcessLocation("starting retrieval steps"))
        self.restart()
        while not self.is_done():
            if (
                stop_at_step is not None
                and stop_at_step == self.current_strategy_step.strategy_step.step_number
            ):
                return
            self.notify_update(ProcessLocation("starting run_step"))
            self.notify_new_step()
            self.notify_update(ProcessLocation("notify_new_step done"))
            self.run_step()
            self.next_step()
        self.notify_update("retrieval done")

    def continue_retrieval(self, stop_after_step: None | int = None) -> None:
        """After saving a pickled step, you can continue the processing starting
        at that step to diagnose a problem."""
        with muses_py_call(self.run_dir, vlidort_cli=self.vlidort_cli):
            while not self.is_done():
                self.notify_new_step()
                self.notify_update(ProcessLocation("starting run_step"))
                self.run_step()
                if (
                    stop_after_step is not None
                    and stop_after_step
                    == self.current_strategy_step.strategy_step.step_number
                ):
                    return
                self.next_step()
            self.notify_update("retrieval done")

    @contextmanager
    def chdir_run_dir(self) -> Generator[None, None, None]:
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

    def save_state_info(self, save_pickle_file: str | os.PathLike[str]) -> None:
        """Dump a pickled version of the StateInfo object.
        We may play with this, but currently we gzip this.
        We would like to have something a bit more stable, not tied to the
        object structure. But for now, just do a straight json dump of the object"""

        # Doesn't work yet. Not overly important, looks like a bug in
        # jsonpickle Just use normal pickle for now, we want to change
        # what gets saved anyways
        #
        # with gzip.GzipFile(save_pickle_file, "wb") as fh:
        #    fh.write(jsonpickle.encode(self.state_info).encode('utf-8'))
        pickle.dump(
            self.current_state._current_state_old._state_info,
            open(save_pickle_file, "wb"),
        )

    def load_state_info(
        self,
        state_info_pickle_file: str | os.PathLike[str],
        step_number: int,
        ret_state_file: str | os.PathLike[str] | None = None,
    ) -> None:
        """This pairs with save_state_info. Instead of pickling the
        entire RetrievalStrategy, we just save the state. We then
        set up to process the given target_filename with the given
        state, jumping to the given retrieval step_number.

        Note for some tests in addition to the StateInfo we want the
        results saved by RetrievalStepCaptureObserver (e.g., we want
        to test the output writing). You can optionally pass in the
        json file for this and we will also pass that information to
        the RetrievalStrategyStep.

        """
        # self._state_info = jsonpickle.decode(
        #    gzip.open(state_info_pickle_file, "rb").read())
        hset = self.state_element_handle_set
        self.current_state._current_state_old._state_info = pickle.load(
            open(state_info_pickle_file, "rb")
        )
        self.current_state.state_element_handle_set = hset
        if (
            self.current_state._current_state_old._state_info.retrieval_config
            is not None
        ):
            self.current_state._current_state_old._state_info.retrieval_config.base_dir = self.run_dir
            self.current_state._current_state_old._state_info.retrieval_config.osp_dir = (
                Path(self.osp_dir) if self.osp_dir is not None else None
            )
        self.set_step(step_number)
        if ret_state_file is not None:
            t = RetrievalStepCaptureObserver.load_retrieval_state(ret_state_file)
            self.kwargs["ret_state"] = t


__all__ = [
    "MusesStrategyExecutor",
    "MusesStrategyExecutorRetrievalStrategyStep",
    "MusesStrategyExecutorMusesStrategy",
]
