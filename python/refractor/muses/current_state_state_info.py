from __future__ import annotations
from . import muses_py as mpy  # type: ignore
from .current_state import (
    CurrentState,
    CurrentStateStateInfoOld,
)
from .identifier import StateElementIdentifier
from .state_element_old_wrapper import StateElementOldWrapperHandle
from .state_info import StateElementHandleSet
import numpy as np
import scipy # type: ignore
import refractor.framework as rf # type: ignore
import numpy.testing as npt
from pathlib import Path
from copy import copy
import typing
from typing import cast

if typing.TYPE_CHECKING:
    from .current_state import PropagatedQA, SoundingMetadata
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import CurrentStrategyStep
    from .error_analysis import ErrorAnalysis
    from .muses_strategy import MusesStrategy
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
    from .state_info import StateElement


# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray


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
        for p in sorted(
            self._state_info.state_element_handle_set.handle_set.keys(), reverse=True
        ):
            for h in self._state_info.state_element_handle_set.handle_set[p]:
                if hasattr(h, "_current_state_old"):
                    self._current_state_old = h._current_state_old
        self.retrieval_state_element_override: None | list[StateElementIdentifier] = (
            None
        )
        self.do_systematic = False
        self._step_directory: None | Path = None

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
    def initial_guess(self) -> RetrievalGridArray:
        """Initial guess"""
        # TODO Remove current_state_old
        # TODO
        # By convention, muses-py returns a length 1 array even if we don't
        # have any retrieval_state_element_id. I think this was just to avoid
        # zero size arrays in IDL. But python is fine with this. For now conform
        # to muses-py since it is expected in various places. We should clean this
        # up at some point by tracking down where this gets used and handling empty
        # arrays - it is cleaner than having a "special rule". But for now, conform
        # to the convention
        # if len(self.retrieval_state_element_id) == 0:
        #    res = np.zeros((1,))
        # else:
        #    res =  np.concatenate(
        #        [
        #            self._state_info[sid].step_initial_value
        #            for sid in self.retrieval_state_element_id
        #        ]
        #    )
        res2 = self._current_state_old.initial_guess
        # npt.assert_allclose(res, res2)
        return res2

    @property
    def initial_guess_fm(self) -> ForwardModelGridArray:
        """Initial guess on forward mode"""
        # TODO Remove current_state_old
        # TODO
        # By convention, muses-py returns a length 1 array even if we don't
        # have any retrieval_state_element_id. I think this was just to avoid
        # zero size arrays in IDL. But python is fine with this. For now conform
        # to muses-py since it is expected in various places. We should clean this
        # up at some point by tracking down where this gets used and handling empty
        # arrays - it is cleaner than having a "special rule". But for now, conform
        # to the convention
        # if len(self.retrieval_state_element_id) == 0:
        #    res = np.zeros((1,))
        # else:
        #    res = np.concatenate(
        #        [
        #            self._state_info[sid].step_initial_value_fm
        #            for sid in self.retrieval_state_element_id
        #        ]
        #    )
        res2 = self._current_state_old.initial_guess_fm
        # npt.assert_allclose(res, res2)
        return res2

    @property
    def apriori_cov(self) -> RetrievalGridArray:
        """Apriori Covariance"""
        # TODO Remove current_state_old
        return self._current_state_old.apriori_cov

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        """Sqrt matrix from covariance"""
        if self.do_systematic:
            return np.eye(len(self.initial_guess))
        else:
            return (mpy.sqrt_matrix(self.apriori_cov)).transpose()

    @property
    def apriori(self) -> RetrievalGridArray:
        """Apriori value"""
        # TODO
        # By convention, muses-py returns a length 1 array even if we don't
        # have any retrieval_state_element_id. I think this was just to avoid
        # zero size arrays in IDL. But python is fine with this. For now conform
        # to muses-py since it is expected in various places. We should clean this
        # up at some point by tracking down where this gets used and handling empty
        # arrays - it is cleaner than having a "special rule". But for now, conform
        # to the convention
        if len(self.retrieval_state_element_id) == 0:
            res = np.zeros((1,))
        else:
            res = np.concatenate(
                [
                    self._state_info[sid].apriori_value
                    for sid in self.retrieval_state_element_id
                ]
            )
        return res

    @property
    def apriori_fm(self) -> ForwardModelGridArray:
        """Apriori value on forward model"""
        # TODO Remove current_state_old
        if len(self.retrieval_state_element_id) == 0:
            # Oddly muses-py doesn't use the normal convention of returning [0],
            # but instead returns []
            res = np.zeros((0,))
        else:
            res = np.concatenate(
                [
                    self._state_info[sid].apriori_value_fm
                    for sid in self.retrieval_state_element_id
                ]
            )
        res2 = self._current_state_old.apriori_fm
        # TODO Fix this
        # npt.assert_allclose(res, res2)
        return res2

    @property
    def true_value(self) -> RetrievalGridArray:
        """True value"""
        # Note muses_py always has a true value vector, even if we don't have
        # a true value (so full_state_true_value is None). It just puts zeros
        # in for any missing data.
        res = np.zeros(
            (
                len(
                    self.apriori,
                )
            )
        )
        for sid in self.retrieval_state_element_id:
            tvalue = self._state_info[sid].true_value
            if tvalue is not None:
                ps, pl = self.retrieval_sv_loc[sid]
                res[ps : ps + pl] = tvalue
        return res

    @property
    def true_value_fm(self) -> ForwardModelGridArray:
        """Apriori value"""
        # TODO Remove current_state_old
        # Note muses_py always has a true value vector, even if we don't have
        # a true value (so full_state_true_value is None). It just puts zeros
        # in for any missing data.
        res = np.zeros(
            (
                len(
                    self.apriori_fm,
                )
            )
        )
        # for sid in self.retrieval_state_element_id:
        #    tvalue = self._state_info[sid].true_value_fm
        #    if(tvalue is not None):
        #        ps, pl = self.fm_sv_loc[sid]
        #        res[ps:ps+pl] = tvalue
        res2 = self._current_state_old.true_value_fm
        # npt.assert_allclose(res, res2)
        return res2

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to full model
        vector.  We don't always have this, so we return None if there
        isn't a basis matrix.

        """
        if self.do_systematic:
            return None
        blist = [self._state_info[sid].basis_matrix
                 for sid in self.retrieval_state_element_id]
        blist = [i for i in blist if i is not None]
        if(len(blist) == 0):
            return None
        res = scipy.linalg.block_diag(*blist)
        return res

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from basis matrix"""
        if self.do_systematic:
            return None
        mlist = [self._state_info[sid].map_to_parameter_matrix
                 for sid in self.retrieval_state_element_id]
        mlist = [i for i in mlist if i is not None]
        if(len(mlist) == 0):
            return None
        res = scipy.linalg.block_diag(*mlist)
        return res

    @property
    def step_directory(self) -> Path:
        if self._step_directory is None:
            raise RuntimeError("Set step directory first")
        return self._step_directory

    @step_directory.setter
    def step_directory(self, val: Path) -> None:
        # TODO Remove current_state_old. Does this even get called? Can possibly remove
        self._step_directory = val
        self._current_state_old.step_directory = val

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self._state_info.propagated_qa

    @property
    def brightness_temperature_data(self) -> dict:
        # TODO Remove current_state_old
        # Right now, need the old brightness_temperature_data.
        # We can probably straighten this out later
        return self._current_state_old.state_info.brightness_temperature_data

    @property
    def updated_fm_flag(self) -> ForwardModelGridArray:
        """This is array of boolean flag indicating which parts of the forward
        model state vector got updated when we last called update_state. A 1 means
        it was updated, a 0 means it wasn't. This is used in the ErrorAnalysis."""
        # TODO Remove current_state_old
        return self._current_state_old.updated_fm_flag

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> None:
        """Update the state info"""
        # TODO Remove current_state_old
        self._current_state_old.update_state(
            results_list, do_not_update, retrieval_config, step
        )

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
        # TODO Remove current_state_old
        self._current_state_old.update_full_state_element(
            state_element_id, current, apriori, step_initial, retrieval_initial, true
        )
        
    def clear_cache(self) -> None:
        super().clear_cache()
        # TODO Remove current_state_old
        self._current_state_old.clear_cache()

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        # TODO Update to remove current_state_old
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        if self.do_systematic:
            return self._current_state_old.retrieval_state_element_id
        return self._current_state_old.retrieval_state_element_id

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that are in the systematic list (used by the
        ErrorAnalysis)."""
        # TODO Remove current_state_old
        return self._current_state_old.systematic_state_element_id

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        """Return list of state elements that make up the full state, generally a
        larger list than retrieval_state_element_id"""
        # TODO Remove current_state_old
        return self._current_state_old.full_state_element_id

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Dict that gives the starting location in the forward model
        state vector for a particular state element name (state
        elements not being retrieved don't get listed here)

        """
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for sid in self.retrieval_state_element_id:
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
                if(self.do_systematic):
                    plen = self._state_info[sid].sys_sv_length
                else:
                    plen = self._state_info[sid].forward_model_sv_length
                if plen == 0:
                    plen = 1
                self._fm_sv_loc[sid] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        if True:
            assert self._fm_sv_loc == self._current_state_old.fm_sv_loc
        return self._fm_sv_loc

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        """Return the sounding metadata. It isn't clear if this really
        belongs in CurrentState or not, but there isn't another
        obvious place for this so for now we'll have this here.

        Perhaps this can migrate to the MuseObservation or MeasurementId if we decide
        that makes more sense.
        """
        # TODO Remove current_state_old
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
        sd = self._state_info[state_element_id].spectral_domain
        if(sd is None):
            return None
        return sd.convert_wave(rf.Unit("nm"))

    def full_state_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the full state value for the given state element
        name.  Just as a convention we always return a np.ndarray, so if
        there is only one value put that in a length 1 np.ndarray.

        """
        res = self._state_info[state_element_id].value_fm
        return res

    def full_state_step_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the initial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        return self._state_info[state_element_id].step_initial_value_fm

    def full_state_value_str(self, state_element_id: StateElementIdentifier) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like full_state_value, but we
        return a str instead.
        """
        return self._state_info[state_element_id].value_str

    def full_state_true_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """Return the true value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.

        If we don't have a true value, return None
        """
        res = self._state_info[state_element_id].true_value_fm
        return res

    def full_state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the initialInitial value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        res = self._state_info[state_element_id].retrieval_initial_value_fm
        return res

    def full_state_apriori_value(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the apriori value of the given state element identification.
        Just as a convention we always return a np.array, so if
        there is only one value put that in a length 1 np.array.
        """
        return self._state_info[state_element_id].apriori_value_fm

    def full_state_apriori_covariance(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray:
        """Return the covariance of the apriori value of the given state element identification."""
        res = self._state_info[state_element_id].apriori_cov_fm
        return res

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        """Like fm_sv_loc, but for the retrieval state vactor (rather than the
        forward model state vector. If we don't have a basis_matrix, these are the
        same. With a basis_matrix, the total length of the fm_sv_loc is the
        basis_matrix column size, and retrieval_vector_loc is the smaller basis_matrix
        row size."""
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for sid in self.retrieval_state_element_id:
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
                plen = self._state_info[sid].retrieval_sv_length
                if plen == 0:
                    plen = 1
                self._retrieval_sv_loc[sid] = (
                    self._retrieval_state_vector_size,
                    plen,
                )
                self._retrieval_state_vector_size += plen
        if True:
            assert self._retrieval_sv_loc == self._current_state_old.retrieval_sv_loc
        return self._retrieval_sv_loc

    def map_type(self, state_element_id: StateElementIdentifier) -> str:
        """For ReFRACtor we use a general rf.StateMapping, which can mostly
        replace the map type py-retrieve uses. However there are some places
        where old code depends on the map type strings (for example, writing
        metadata to an output file). It isn't clear what we will need to do if
        we have a more general mapping type like a scale retrieval or something like
        that. But for now, supply the old map type. The string will be something
        like "log" or "linear" """
        return self._state_info[state_element_id].map_type

    def pressure_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the retrieval state vector
        levels (generally smaller than the pressure_list_fm).
        """
        return self._state_info[state_element_id].pressure_list

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the forward model state
        vector levels (generally larger than the pressure_list).
        """
        return self._state_info[state_element_id].pressure_list_fm

    def altitude_list(
        self, state_element_id: StateElementIdentifier
    ) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels.  This is for the retrieval state vector
        levels (generally smaller than the altitude_list_fm).
        """
        return self._state_info[state_element_id].altitude_list

    def altitude_list_fm(
        self, state_element_id: StateElementIdentifier
    ) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels.  This is for the forward model state
        vector levels (generally larger than the pressure_list).
        """
        return self._state_info[state_element_id].altitude_list_fm

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Have updated the target we are processing."""
        # TODO Remove current_state_old
        self._current_state_old.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )
        self._state_info.notify_update_target(measurement_id)
        self.clear_cache()

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self._state_info.state_element_handle_set

    @state_element_handle_set.setter
    def state_element_handle_set(self, val: StateElementHandleSet) -> None:
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
        # TODO Remove current_state_old
        self._current_state_old.notify_new_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )
        self._step_directory = self._current_state_old.step_directory
        self.clear_cache()

    def restart(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Called when muses_strategy_executor has restarted"""
        # TODO Remove current_state_old
        self._current_state_old.restart(current_strategy_step, retrieval_config)
        self._step_directory = self._current_state_old.step_directory


# Right now, only fall back to old py-retrieve code

StateElementHandleSet.add_default_handle(
    StateElementOldWrapperHandle(), priority_order=-1
)

__all__ = [
    "CurrentStateStateInfo",
]
