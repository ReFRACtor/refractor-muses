from __future__ import annotations
from .state_info import StateElement, StateElementHandle
import refractor.framework as rf  # type: ignore
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .identifier import StateElementIdentifier
    from .current_state import CurrentStateStateInfoOld
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration

# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray


class StateElementOldWrapper(StateElement):
    """This wraps around a CurrentStateStateInfoOld and presents the
    data like a StateElement. This is used for backwards testing against
    our StateInfoOld that wraps around the old py-retrieve state info code.
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        current_state_old: CurrentStateStateInfoOld,
        is_first
    ) -> None:
        super().__init__(state_element_id)
        self._current_state_old = current_state_old
        # Bit klunky, but way to push notifications down a bit so
        # it looks like we are just notifying StateElement, even though
        # we forward that to self._current_state_old
        self.is_first = is_first

    def sa_cross_covariance(self, selem2: StateElement) -> np.ndarray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        selem_old = self._current_state_old.full_state_element(self.state_element_id)
        selem2_old = self._current_state_old.full_state_element(selem2.state_element_id)
        return selem_old.sa_cross_covariance(selem2_old)

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return self._current_state_old.full_state_element_old(self.state_element_id).spectral_domain

    @property
    def retrieval_slice(self) -> slice | None:
        if self.state_element_id in self._current_state_old.retrieval_sv_loc:
            ps, pl = self._current_state_old.retrieval_sv_loc[self.state_element_id]
            return slice(ps, ps+pl)
        return None

    @property
    def fm_slice(self) -> slice | None:
        if self.state_element_id in self._current_state_old.fm_sv_loc:
            ps, pl = self._current_state_old.fm_sv_loc[self.state_element_id]
            return slice(ps, ps+pl)
        return None
    
    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to forward model
        vector. Would be nice to replace this with a general
        rf.StateMapping, but for now this is assumed in a lot of
        muses-py code."""
        bmatrix = self._current_state_old.basis_matrix
        s1 = self.retrieval_slice
        s2 = self.fm_slice
        if bmatrix is None or s1 is None or s2 is None:
            return None
        return bmatrix[s1, s2]

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from the basis matrix, going from
        the forward model vector the retrieval vector."""
        mmatrix = self._current_state_old.map_to_parameter_matrix
        s1 = self.fm_slice
        s2 = self.retrieval_slice
        if mmatrix is None or s1 is None or s2 is None:
            return None
        return mmatrix[s1, s2]

    @property
    def retrieval_sv_length(self) -> int:
        if self.state_element_id not in self._current_state_old.retrieval_sv_loc:
            return 0
        return self._current_state_old.retrieval_sv_loc[self.state_element_id][1]

    @property
    def sys_sv_length(self) -> int:
        cstate = self._current_state_old.current_state_override(
            do_systematic=True,
            retrieval_state_element_override=None)
        if self.state_element_id not in cstate.fm_sv_loc:
            return 0
        return cstate.fm_sv_loc[self.state_element_id][1]
    
    @property
    def forward_model_sv_length(self) -> int:
        if self.state_element_id not in self._current_state_old.fm_sv_loc:
            return 0
        return self._current_state_old.fm_sv_loc[self.state_element_id][1]

    @property
    def map_type(self) -> str:
        """For ReFRACtor we use a general rf.StateMapping, which can mostly
        replace the map type py-retrieve uses. However there are some places
        where old code depends on the map type strings (for example, writing
        metadata to an output file). It isn't clear what we will need to do if
        we have a more general mapping type like a scale retrieval or something like
        that. But for now, supply the old map type. The string will be something
        like "log" or "linear" """
        return self._current_state_old.map_type(self.state_element_id)
    
    @property
    def altitude_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        res = self._current_state_old.altitude_list(self.state_element_id)
        # Kind of obscure, but species are the only items with a pressumre list > 2
        if(res is None or len(res) < 3):
            return None
        return res

    @property
    def altitude_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        res = self._current_state_old.altitude_list_fm(self.state_element_id)
        # Kind of obscure, but species are the only items with a pressumre list > 2
        if(res is None or len(res) < 3):
            return None
        return res
    
    @property
    def pressure_list(self) -> RetrievalGridArray| None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        res = self._current_state_old.pressure_list(self.state_element_id)
        # Kind of obscure, but species are the only items with a pressumre list > 2
        if(res is None or len(res) < 3):
            return None
        return res


    @property
    def pressure_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        res = self._current_state_old.pressure_list_fm(self.state_element_id)
        # Kind of obscure, but species are the only items with a pressumre list > 2
        if(res is None or len(res) < 3):
            return None
        return res
    
    @property
    def value(self) -> RetrievalGridArray:
        """Current value of StateElement"""
        raise NotImplementedError()

    @property
    def value_fm(self) -> ForwardModelGridArray:
        """Current value of StateElement"""
        return self._current_state_old.full_state_value(self.state_element_id)

    @property
    def value_str(self) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like value, but we
        return a str instead. For most StateElement, this returns "None"
        instead which indicates we don't have a str value.

        It isn't clear that this is the best interface, on the other hand these
        str values don't have an obvious other place to go. And these value are
        at least in the spirit of other StateElement. So at least for now we
        will support this, possibly reworking this in the future.
        """
        return self._current_state_old.full_state_value_str(self.state_element_id)

    @property
    def apriori_value(self) -> RetrievalGridArray:
        """Apriori value of StateElement"""
        s = self.retrieval_slice
        if(s is not None):
            return self._current_state_old.apriori[s]
        raise RuntimeError("apriori only present for stuff in state vector")

    @property
    def apriori_value_fm(self) -> ForwardModelGridArray:
        """Apriori value of StateElement"""
        if self.state_element_id not in self._current_state_old.retrieval_state_element_id:
            return self._current_state_old.full_state_apriori_value(self.state_element_id)
        res = self._current_state_old.retrieval_info.species_constraint(str(self.state_element_id))
        if isinstance(res, float):
            return np.array([res,])
        return res

    @property
    def apriori_cov(self) -> RetrievalGridArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    @property
    def apriori_cov_fm(self) -> ForwardModelGridArray:
        """Apriori Covariance"""
        return self._current_state_old.full_state_apriori_covariance(
            self.state_element_id
        )

    @property
    def retrieval_initial_value(self) -> RetrievalGridArray:
        raise NotImplementedError()

    @property
    def retrieval_initial_value_fm(self) -> ForwardModelGridArray:
        """Value StateElement had at the start of the retrieval."""
        # TODO It is not clear why this isn't directly calculated from step_initial_value,
        # but it is different. For now, we use the existing value. We will want to sort this
        # out, this function may end up going away.
        return self._current_state_old.full_state_retrieval_initial_value(
            self.state_element_id
        )

    @property
    def step_initial_value(self) -> RetrievalGridArray:
        s = self.retrieval_slice
        if(s is not None):
            return self._current_state_old.initial_guess[s]
        raise RuntimeError("step_initial_value only present for stuff in state vector")

    @property
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        """Value StateElement had at the start of the retrieval step."""
        # TODO It is not clear why this isn't directly calculated from step_initial_value,
        # but it is different. For now, we use the existing value. We will want to sort this
        # out, this function may end up going away.
        return self._current_state_old.full_state_step_initial_value(
            self.state_element_id
        )

    @property
    def true_value_fm(self) -> ForwardModelGridArray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        # TODO It is not clear why this isn't directly calculated from step_initial_value,
        # but it is different. For now, we use the existing value. We will want to sort this
        # out, this function may end up going away.
        s = self.fm_slice
        if(s is not None):
            return self._current_state_old.true_value_fm[s]
        return self._current_state_old.full_state_true_value(self.state_element_id)

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        """Update the value of the StateElement. This function updates
        each of the various values passed in.  A value of 'None' (the
        default) means skip updating that part of the StateElement.
        """
        self._current_state_old.update_full_state_element(
            self.state_element_id,
            current,
            apriori,
            step_initial,
            retrieval_initial,
            true_value,
        )

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        if(self.is_first):
            self._current_state_old.notify_new_step(
                current_strategy_step,
                error_analysis,
                retrieval_config,
                skip_initial_guess_update,
            )
            
    def restart(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if(self.is_first):
            self._current_state_old.restart(current_strategy_step, retrieval_config)

class StateElementOldWrapperHandle(StateElementHandle):
    def __init__(self) -> None:
        from .current_state import CurrentStateStateInfoOld
        self._current_state_old = CurrentStateStateInfoOld(None)
        self.is_first = True

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        self._current_state_old.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )
        self.is_first = True

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        r = StateElementOldWrapper(state_element_id, self._current_state_old, self.is_first)
        self.is_first = False
        return r


__all__ = ["StateElementOldWrapper", "StateElementOldWrapperHandle"]
