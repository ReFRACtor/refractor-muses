from __future__ import annotations
from . import muses_py as mpy  # type: ignore
from .current_state import (
    CurrentState,
    CurrentStateStateInfoOld,
    SoundingMetadata,
    PropagatedQA,
)
from .identifier import StateElementIdentifier
from .state_element_old_wrapper import StateElementOldWrapperHandle
from .state_info import StateElementHandleSet
import numpy as np
from pathlib import Path
from copy import copy
import typing
from typing import cast

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import CurrentStrategyStep
    from .error_analysis import ErrorAnalysis
    from .muses_strategy import MusesStrategy
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
    from .state_info import StateElement


class CurrentStateStateInfo(CurrentState):
    """Implementation of CurrentState that uses our StateInfo."""

    # TODO - We use the CurrentStateStateInfoOld as scaffolding as we
    # work out this code. This should go away.
    def __init__(
        self,
    ) -> None:
        from .state_info import StateInfo
        super().__init__()
        self._state_info = StateInfo()
        # Temp, clumsy but this will go away
        for p in sorted(self._state_info.state_element_handle_set.handle_set.keys(), reverse=True):
            for h in self._state_info.state_element_handle_set.handle_set[p]:
                if hasattr(h, "_current_state_old"):
                    self._current_state_old = h._current_state_old
        self.retrieval_state_element_override: None | list[StateElementIdentifier] = (
            None
        )
        self.do_systematic = False
        self._step_directory: None | Path = None
        self._retrieval_info: None | RetrievalInfo = None

    def current_state_override(
        self,
        do_systematic: bool,
        retrieval_state_element_override: None | list[StateElementIdentifier],
    ) -> CurrentState:
        res = copy(self)
        res._current_state_old = cast(
            CurrentStateStateInfoOld,
            res._current_state_old.current_state_override(
                do_systematic, retrieval_state_element_override
            ),
        )
        res.retrieval_state_element_override = retrieval_state_element_override
        res.do_systematic = do_systematic
        res.clear_cache()
        return res

    @property
    def initial_guess(self) -> np.ndarray:
        """Initial guess"""
        return self._current_state_old.initial_guess

    @property
    def apriori_cov(self) -> np.ndarray:
        """Apriori Covariance"""
        return self._current_state_old.apriori_cov

    @property
    def sqrt_constraint(self) -> np.ndarray:
        """Sqrt matrix from covariance"""
        if self.do_systematic:
            return np.eye(len(self.initial_guess))
        else:
            return (mpy.sqrt_matrix(self.apriori_cov)).transpose()

    @property
    def apriori(self) -> np.ndarray:
        """Apriori value"""
        return self._current_state_old.apriori

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to full model
        vector.  We don't always have this, so we return None if there
        isn't a basis matrix.

        """
        if self.do_systematic:
            return None
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return self.retrieval_info.basis_matrix

    @property
    def step_directory(self) -> Path:
        if self._step_directory is None:
            raise RuntimeError("Set step directory first")
        return self._step_directory

    @step_directory.setter
    def step_directory(self, val: Path) -> None:
        self._step_directory = val
        self._current_state_old.step_directory = val

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self._current_state_old.propagated_qa

    @property
    def brightness_temperature_data(self) -> dict:
        return self._current_state_old.brightness_temperature_data

    def update_state(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> None:
        """Update the state info"""
        self._current_state_old.update_state(
            retrieval_info, results_list, do_not_update, retrieval_config, step
        )
        self.retrieval_info = retrieval_info

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        self._current_state_old.update_full_state_element(
            state_element_id, current, apriori, step_initial, retrieval_initial, true
        )

    @property
    def retrieval_info(self) -> RetrievalInfo:
        if self._retrieval_info is None:
            raise RuntimeError("Need to set self._retrieval_info")
        return self._retrieval_info

    @retrieval_info.setter
    def retrieval_info(self, val: RetrievalInfo) -> None:
        self._retrieval_info = val
        self._current_state_old.retrieval_info = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element(self) -> list[StateElementIdentifier]:
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self.do_systematic:
            return [
                StateElementIdentifier(i) for i in self.retrieval_info.species_names_sys
            ]
        return [StateElementIdentifier(i) for i in self.retrieval_info.species_names]

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that make up the full state, generally a
        larger list than retrieval_state_element"""
        return self._current_state_old.full_state_element_id

    # TODO Perhaps this can go away. Only used in FakeStateInfo
    @property
    def full_state_element_on_levels_id(self) -> list[StateElementIdentifier]:
        """Subset of full_state_element_id for species that are on levels, so things like
        H2O"""
        return self._current_state_old.full_state_element_on_levels_id

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the forward model
        state vector for a particular state element name (state
        elements not being retrieved don't get listed here)

        """
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_id in self.retrieval_state_element:
                if self.do_systematic:
                    plen = self.retrieval_info.species_list_sys.count(
                        str(state_element_id)
                    )
                else:
                    plen = self.retrieval_info.species_list_fm.count(
                        str(state_element_id)
                    )

                # As a convention, if plen is 0 py-retrieve pads this
                # to 1, although the state vector isn't actually used
                # - it does get set. I think this is to avoid having a
                # 0 size state vector. We should perhaps clean this up
                # as some point, there isn't anything wrong with a
                # zero size state vector (although this might have
                # been a problem with IDL). But for now, use the
                # py-retrieve convention. This can generally only
                # happen if we have retrieval_state_element_override
                # set, i.e., we are doing RetrievalStrategyStepBT.
                if plen == 0:
                    plen = 1
                self._fm_sv_loc[state_element_id] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        """Return the sounding metadata. It isn't clear if this really
        belongs in CurrentState or not, but there isn't another
        obvious place for this so for now we'll have this here.

        Perhaps this can migrate to the MuseObservation or MeasurementId if we decide
        that makes more sense.
        """
        return self._current_state_old.sounding_metadata

    def full_state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        """Return the StateElement for the state_element_id. I'm not sure if we want to
        have this exposed or not, but there is a bit of useful information we have in
        each StateElement (such as the sa_cross_covariance). We can have this exposed for
        now, and revisit it if we end up deciding this is too much coupling. There are
        only a few spots that use full_state_element vs something like full_state_value,
        so we will just need to revisit those few spots if this becomes an issue."""
        return self._state_info[state_element_id]

    def full_state_element_old(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement:
        """Return the StateElement for the state_element_id. I'm not sure if we want to
        have this exposed or not, but there is a bit of useful information we have in
        each StateElement (such as the sa_cross_covariance). We can have this exposed for
        now, and revisit it if we end up deciding this is too much coupling. There are
        only a few spots that use full_state_element vs something like full_state_value,
        so we will just need to revisit those few spots if this becomes an issue."""
        return self._current_state_old.full_state_element_old(state_element_id)
    
    def full_state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        """Return the spectral domain (as nm) for the given state_element_id, or None if
        there isn't an associated frequency for the given state_element_id"""
        selem = self._current_state_old.full_state_element_old(state_element_id)
        return selem.spectral_domain_wavelength

    def full_state_value(self, state_element_id: StateElementIdentifier) -> np.ndarray:
        """Return the full state value for the given state element
        name.  Just as a convention we always return a np.ndarray, so if
        there is only one value put that in a length 1 np.ndarray.

        """
        return self._state_info[state_element_id].value

    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray:
        """Return the initial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        return self._state_info[state_element_id].step_initial_value

    def full_state_value_str(self, state_element_id: StateElementIdentifier) -> str:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like full_state_value, but we
        return a str instead.
        """
        return self._state_info[state_element_id].value_str

    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        """Return the true value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        If we don't have a true value, return None
        """
        return self._state_info[state_element_id].true_value

    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray:
        """Return the initialInitial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        return self._state_info[state_element_id].retrieval_initial_value

    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray:
        """Return the apriori value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        return self._state_info[state_element_id].apriori

    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray:
        """Return the covariance of the apriori value of the given state element identification."""
        return self._state_info[state_element_id].apriori_cov

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Like fm_sv_loc, but for the retrieval state vactor (rather than the
        forward model state vector. If we don't have a basis_matrix, these are the
        same. With a basis_matrix, the total length of the fm_sv_loc is the
        basis_matrix column size, and retrieval_vector_loc is the smaller basis_matrix
        row size."""
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for state_element_id in self.retrieval_state_element:
                if self.do_systematic:
                    plen = self.retrieval_info.species_list_sys.count(
                        str(state_element_id)
                    )
                else:
                    plen = self.retrieval_info.species_list.count(str(state_element_id))

                # As a convention, if plen is 0 py-retrieve pads this
                # to 1, although the state vector isn't actually used
                # - it does get set. I think this is to avoid having a
                # 0 size state vector. We should perhaps clean this up
                # as some point, there isn't anything wrong with a
                # zero size state vector (although this might have
                # been a problem with IDL). But for now, use the
                # py-retrieve convention. This can generally only
                # happen if we have retrieval_state_element_override
                # set, i.e., we are doing RetrievalStrategyStepBT.
                if plen == 0:
                    plen = 1
                self._retrieval_sv_loc[state_element_id] = (
                    self._retrieval_state_vector_size,
                    plen,
                )
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    def pressure_list(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the retrieval state vector
        levels (generally smaller than the pressure_list_fm).
        """
        return self._current_state_old.pressure_list(state_element_id)

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> np.ndarray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the forward model state
        vector levels (generally larger than the pressure_list).
        """
        return self._current_state_old.pressure_list_fm(state_element_id)

    # Some of these arguments may get put into the class, but for now have explicit
    # passing of these
    def get_initial_guess(
        self,
        current_strategy_step: CurrentStrategyStep,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Set retrieval_info, errorInitial and errorCurrent for the current step."""
        self._current_state_old.get_initial_guess(
            current_strategy_step, error_analysis, retrieval_config
        )
        self.retrieval_info = self._current_state_old.retrieval_info

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Have updated the target we are processing."""
        self._current_state_old.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self._state_info.state_element_handle_set

    @state_element_handle_set.setter
    def state_element_handle_set(self, val) -> None:
        self._state_info.state_element_handle_set = val
    
    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        """Called when MusesStrategy is starting a new step.

        The logic for when to update the initial guess in the state info table is kind of
        complicated and confusing. For now we duplicate this behavior, in some cases we do
        this and in others we don't. We can hopefully sort this out, the logic should be
        straight forward"""
        self._current_state_old.notify_new_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )
        self._retrieval_info = self._current_state_old.retrieval_info
        self._step_directory = self._current_state_old.step_directory

    def restart(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Called when muses_strategy_executor has restarted"""
        self._current_state_old.restart(current_strategy_step, retrieval_config)
        self._step_directory = self._current_state_old.step_directory

# Right now, only fall back to old py-retrieve code

StateElementHandleSet.add_default_handle(
    StateElementOldWrapperHandle(), priority_order=-1
)

__all__ = [
    "CurrentStateStateInfo",
]
