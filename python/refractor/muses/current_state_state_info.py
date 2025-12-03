from __future__ import annotations
from .current_state import CurrentState
from .retrieval_array import (
    RetrievalGridArray,
    FullGridArray,
    FullGridMappedArray,
    FullGridMappedArrayFromRetGrid,
    RetrievalGrid2dArray,
    FullGrid2dArray,
)
from .identifier import StateElementIdentifier
from .state_element import StateElementHandleSet, StateElement
import numpy as np
import scipy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy.testing as npt
import weakref
from pathlib import Path
from copy import copy
from loguru import logger
import typing
from typing import cast

if typing.TYPE_CHECKING:
    from .current_state import PropagatedQA
    from .identifier import RetrievalType, StrategyStepIdentifier
    from .sounding_metadata import SoundingMetadata
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import CurrentStrategyStep
    from .muses_strategy import MusesStrategy
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
    from .state_info import StateElement
    from .cross_state_element import CrossStateElementHandleSet
    from .cost_function import CostFunction


class CostFunctionStateElementNotify(rf.ObserverMaxAPosterioriSqrtConstraint):
    """Small adapter class to map framework observer to notify StateElement
    when parameters change"""

    def __init__(self, selem: StateElement, slc: slice):
        super().__init__()
        self.slc = slc
        # We want this object to be kept alive, but not force selem to
        # be kept alive. This set of references does that
        self.selem = weakref.ref(selem)
        selem._cost_function_notify_helper = self

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        s = self.selem()
        if s is not None:
            s.notify_parameter_update(mstand.parameters[self.slc])


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
        self._step_directory: None | Path = None
        self._strategy_step: None | StrategyStepIdentifier = None
        self._retrieval_type: None | RetrievalType = None

        # When doing initial development, we matched against old state info stuff
        # This is normally not used, but have in place in case we need to diagnose
        # some problem
        self._current_state_old = self._state_info._current_state_old
        # Temp, while until we move previous_aposteriori_cov_fm to StateElements
        self._covariance_state_element_name: list[StateElementIdentifier] = []

    def current_state_override(
        self,
        do_systematic: bool,
        retrieval_state_element_override: None | list[StateElementIdentifier],
    ) -> CurrentState:
        res = copy(self)
        res.retrieval_state_element_override = retrieval_state_element_override
        res.do_systematic = do_systematic
        res.clear_cache()
        return res

    def match_old(self) -> None:
        """A kludge to handle current_state_override with our old state info stuff.
        Temporarily, ensure that the two are in sync"""
        cstate = self._current_state_old
        if cstate is None:
            return
        if (
            self.retrieval_state_element_override
            != cstate.retrieval_state_element_override
            or self.do_systematic != cstate.do_systematic
        ):
            cstate.do_systematic = self.do_systematic
            cstate.retrieval_state_element_override = (
                self.retrieval_state_element_override
            )
            cstate.clear_cache()

    @property
    def initial_guess(self) -> RetrievalGridArray:
        if len(self.retrieval_state_element_id) == 0:
            res = np.zeros((0,))  # np.concatenate doesn't work with zero size
        else:
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].step_initial_ret
                    for sid in self.retrieval_state_element_id
                ]
            )
        if (
            CurrentState.check_old_state_element_value
            and self._current_state_old is not None
        ):
            res2 = self._current_state_old.initial_guess
            npt.assert_allclose(res, res2, 1e-12)
        return res.view(RetrievalGridArray)

    @property
    def initial_guess_full(self) -> FullGridArray:
        if len(self.retrieval_state_element_id) == 0:
            res = np.zeros((0,))  # np.concatenate doesn't work with zero size
        else:
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].step_initial_full
                    for sid in self.retrieval_state_element_id
                ]
            )
        if (
            CurrentState.check_old_state_element_value
            and self._current_state_old is not None
        ):
            res2 = self._current_state_old.initial_guess_full
            npt.assert_allclose(res, res2, 1e-12)
        return res.view(FullGridArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        if self.do_systematic:
            return np.zeros((1, 1)).view(RetrievalGrid2dArray)
        self.match_old()
        # Note that we handle the cross terms after this. In
        # particular, the constraint_matrix for H2OxH2O here will be
        # without considering HDO being available. This gets changed
        # in the next block with the cross terms.
        blist = [
            self._state_info[sid].constraint_matrix
            for sid in self.retrieval_state_element_id
        ]
        res = scipy.linalg.block_diag(*blist)
        # Handle cross terms, and any updates because of cross terms (e.g., H2OxH2O)
        for i, selem1_sid in enumerate(self.retrieval_state_element_id):
            for selem2_sid in self.retrieval_state_element_id[i + 1 :]:
                r1 = self.retrieval_sv_slice(selem1_sid)
                r2 = self.retrieval_sv_slice(selem2_sid)
                m_s1xs1, m_s1xs2, m_s2xs2 = self._state_info.cross_constraint_matrix(
                    selem1_sid, selem2_sid
                )
                if m_s1xs1 is not None:
                    res[r1, r1] = m_s1xs1
                if m_s1xs2 is not None:
                    res[r1, r2] = m_s1xs2
                    res[r2, r1] = m_s1xs2.transpose()
                if m_s2xs2 is not None:
                    res[r2, r2] = m_s2xs2
        if (
            CurrentState.check_old_state_element_value
            and self._current_state_old is not None
        ):
            res2 = self._current_state_old.constraint_matrix
            npt.assert_allclose(res, res2, atol=1e-12)
        return res.view(RetrievalGrid2dArray)

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        if self.do_systematic:
            return np.eye(len(self.initial_guess)).view(RetrievalGridArray)
        else:
            return (
                (self.sqrt_matrix(self.constraint_matrix))
                .transpose()
                .view(RetrievalGridArray)
            )

    def constraint_vector(self, fix_negative: bool = True) -> RetrievalGridArray:
        if len(self.retrieval_state_element_id) == 0:
            # np.concatenate doesn't work with zero size, so handle degenerate case
            res = np.zeros((0,)).view(RetrievalGridArray)
        else:
            self.match_old()
            resv: list[np.ndarray] = []
            for sid in self.retrieval_state_element_id:
                selem = self._state_info[sid]
                v = np.array(selem.constraint_vector_ret)
                if (
                    fix_negative
                    and selem.should_fix_negative
                    and v.min() < 0
                    and v.max() > 0
                ):
                    logger.info(
                        f"Fixing negative mapping for constraint vector for {sid}"
                    )
                    v[v < 0] = v[v > 0].min()
                resv.append(v)
            res = np.concatenate(resv).view(RetrievalGridArray)
        if (
            CurrentState.check_old_state_element_value
            and self._current_state_old is not None
        ):
            res2 = self._current_state_old.constraint_vector(fix_negative=fix_negative)
            npt.assert_allclose(res, res2)
        return res

    @property
    def constraint_vector_full(self) -> FullGridArray:
        if len(self.retrieval_state_element_id) == 0:
            # np.concatenate doesn't work with zero size, so handle degenerate case
            res = np.zeros((0,)).view(FullGridArray)
        else:
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].constraint_vector_full
                    for sid in self.retrieval_state_element_id
                ]
            ).view(FullGridArray)
        if (
            CurrentState.check_old_state_element_value
            and self._current_state_old is not None
        ):
            try:
                res2 = self._current_state_old.constraint_vector_full
                # Need to fix
                npt.assert_allclose(res, res2)
            except KeyError:
                pass
        return res

    @property
    def true_value(self) -> RetrievalGridArray:
        # Note muses_py always has a true value vector, even if we don't have
        # a true value (so state_true_value is None). It just puts zeros
        # in for any missing data.
        res = np.zeros(
            (
                len(
                    self.constraint_vector(),
                )
            )
        )
        for sid in self.retrieval_state_element_id:
            self.match_old()
            tvalue = self._state_info[sid].true_value_ret
            if tvalue is not None:
                res[self.retrieval_sv_slice(sid)] = tvalue
        return res.view(RetrievalGridArray)

    @property
    def true_value_full(self) -> FullGridArray:
        # Note muses_py always has a true value vector, even if we don't have
        # a true value (so state_true_value is None). It just puts zeros
        # in for any missing data.
        res = np.zeros(
            (
                len(
                    self.initial_guess_full,
                )
            )
        )
        self.match_old()
        for sid in self.retrieval_state_element_id:
            tvalue = self._state_info[sid].true_value_full
            if tvalue is not None:
                res[self.fm_sv_slice(sid)] = tvalue
        return res.view(FullGridArray)

    @property
    def basis_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        self.match_old()
        blist = [
            self._state_info[sid].basis_matrix
            for sid in self.retrieval_state_element_id
        ]
        blist = [i for i in blist if i is not None]
        if len(blist) == 0:
            return None
        res = scipy.linalg.block_diag(*blist)
        return res

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        self.match_old()
        mlist = [
            self._state_info[sid].map_to_parameter_matrix
            for sid in self.retrieval_state_element_id
        ]
        mlist = [i for i in mlist if i is not None]
        if len(mlist) == 0:
            return None
        res = scipy.linalg.block_diag(*mlist)
        return res

    @property
    def step_directory(self) -> Path:
        if self._step_directory is None:
            raise RuntimeError("Set step directory first")
        return self._step_directory

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        """Similar to step_directory, step_number is used by RefractorUip. Supply that."""
        if self._strategy_step is None:
            raise RuntimeError("Set strategy step identifier first")
        return self._strategy_step

    @property
    def retrieval_type(self) -> RetrievalType:
        """Similar to step_directory, retrieval_type is used by RefractorUip. Supply that."""
        if self._retrieval_type is None:
            raise RuntimeError("Set retrieval type first")
        return self._retrieval_type

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self._state_info.propagated_qa

    def brightness_temperature_data(self, step: int) -> dict[str, float | None] | None:
        if step in self._state_info.brightness_temperature_data:
            return self._state_info.brightness_temperature_data[step]
        return None

    def set_brightness_temperature_data(
        self, step: int, val: dict[str, float | None]
    ) -> None:
        if self.record is not None:
            self.record.record("set_brightness_temperature_data", step, val)
        self._state_info.brightness_temperature_data[step] = val

    @property
    def updated_fm_flag(self) -> FullGridArray:
        self.match_old()
        return np.concatenate(
            [
                self._state_info[sid].updated_fm_flag
                for sid in self.retrieval_state_element_id
            ]
        ).view(FullGridArray)

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        value_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        next_constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        next_step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        if self.record is not None:
            self.record.record(
                "update_full_state_element",
                state_element_id,
                value_fm,
                constraint_vector_fm,
                next_constraint_vector_fm,
                step_initial_fm,
                next_step_initial_fm,
                retrieval_initial_fm,
                true_value_fm,
            )
        self.match_old()
        self._state_info[state_element_id].update_state_element(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            next_constraint_vector_fm=next_constraint_vector_fm,
            step_initial_fm=step_initial_fm,
            next_step_initial_fm=next_step_initial_fm,
            retrieval_initial_fm=retrieval_initial_fm,
            true_value_fm=true_value_fm,
        )
        if self._current_state_old is not None:
            self._current_state_old.state_element_old(state_element_id).update_state(
                current=value_fm,
                apriori=constraint_vector_fm
                if constraint_vector_fm is not None
                else next_constraint_vector_fm,
                initial=step_initial_fm
                if step_initial_fm is not None
                else next_step_initial_fm,
                initial_initial=retrieval_initial_fm,
                true=true_value_fm,
            )

    def clear_cache(self) -> None:
        super().clear_cache()
        self._updated_fm_flag = None

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        res = self._sys_element_id if self.do_systematic else self._retrieval_element_id
        return res

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        res = self._sys_element_id
        return res

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        return list(self._state_info.keys())

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            self.match_old()
            for sid in self.retrieval_state_element_id:
                if self.do_systematic:
                    plen = self._state_info[sid].sys_sv_length
                else:
                    plen = self._state_info[sid].forward_model_sv_length
                self._fm_sv_loc[sid] = (self._fm_state_vector_size, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        return self._state_info.sounding_metadata

    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        self.match_old()
        sid = (
            StateElementIdentifier(state_element_id)
            if isinstance(state_element_id, str)
            else state_element_id
        )
        return self._state_info[sid]

    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        sd = self.state_element(state_element_id).spectral_domain
        if sd is None:
            return None
        return sd.convert_wave(rf.Unit("nm"))

    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        return self.state_element(state_element_id).value_fm

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        return self.state_element(state_element_id).step_initial_fm

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        return self.state_element(state_element_id).true_value_fm

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        return self.state_element(state_element_id).retrieval_initial_fm

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        return self.state_element(state_element_id).constraint_vector_fm

    def state_constraint_vector_fmprime(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArrayFromRetGrid:
        return self.state_element(state_element_id).constraint_vector_fmprime

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        return self.state_element(state_element_id).apriori_cov_fm

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        self.match_old()
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for sid in self.retrieval_state_element_id:
                plen = self._state_info[sid].step_initial_ret.shape[0]
                self._retrieval_sv_loc[sid] = (
                    self._retrieval_state_vector_size,
                    plen,
                )
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    def pressure_list(
        self, state_element_id: StateElementIdentifier | str
    ) -> RetrievalGridArray | None:
        return self.state_element(state_element_id).pressure_list

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        return self.state_element(state_element_id).pressure_list_fm

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self._state_info.state_element_handle_set

    @state_element_handle_set.setter
    def state_element_handle_set(self, val: StateElementHandleSet) -> None:
        self._state_info.state_element_handle_set = val

    @property
    def cross_state_element_handle_set(self) -> CrossStateElementHandleSet:
        return self._state_info.cross_state_element_handle_set

    @cross_state_element_handle_set.setter
    def cross_state_element_handle_set(self, val: CrossStateElementHandleSet) -> None:
        self._state_info.cross_state_element_handle_set = val

    def notify_cost_function(self, cfunc: CostFunction) -> None:
        # Attach StateElement to get notified when current state changes.
        #
        # A note on the different lifetimes. For the CostFunction, if we
        # have a UIP than the UIP state observer is *required*. If we pickle
        # and reload the cost function, it should have the UIP and observer.
        # So we use "add_observer_and_keep_reference". However, for the
        # StateElement these are outside of the CostFunction. You can
        # have a CostFunction without any StateElement, if we pickle and reload
        # we don't want to pull all the StateElement along. So for this
        # we use "add_observer" which uses weak pointers - we notify if the
        # object is still there but don't carry around it lifetime and if the
        # object is deleted then we just don't notify it.
        for sid in self.retrieval_state_element_id:
            cfunc.max_a_posteriori.add_observer(
                CostFunctionStateElementNotify(
                    self.state_element(sid),
                    cast(slice, self.retrieval_sv_slice(sid)),
                )
            )

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self._state_info.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )
        self._covariance_state_element_name = StateElementIdentifier.sort_identifier(
            list(
                set(strategy.retrieval_elements)
                | set(strategy.error_analysis_interferents)
            )
        )
        self.clear_cache()

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if self.record is not None:
            self.record.notify_start_retrieval()
        if current_strategy_step is not None:
            self.match_old()
            self._retrieval_element_id = current_strategy_step.retrieval_elements
            self._sys_element_id = current_strategy_step.error_analysis_interferents
            self._step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            self._strategy_step = current_strategy_step.strategy_step
            self._retrieval_type = current_strategy_step.retrieval_type
            # Temp, until we move all this into the StateElements
            self.setup_previous_aposteriori_cov_fm(
                self._covariance_state_element_name, current_strategy_step
            )
            self._state_info.notify_start_retrieval(
                current_strategy_step, retrieval_config
            )
            self.clear_cache()

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        # current_strategy_step being None means we are past the last step in our
        # MusesStrategy, so we just skip doing anything
        if current_strategy_step is not None:
            if self.record is not None:
                self.record.notify_start_step()
            self.match_old()
            self._retrieval_element_id = current_strategy_step.retrieval_elements
            self._sys_element_id = current_strategy_step.error_analysis_interferents
            self._step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            self._strategy_step = current_strategy_step.strategy_step
            self._retrieval_type = current_strategy_step.retrieval_type
            self._state_info.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )
            self.clear_cache()

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        if self.record is not None:
            self.record.record("notify_step_solution", xsol)
        self.match_old()
        self._state_info.notify_step_solution(xsol, self)


__all__ = [
    "CurrentStateStateInfo",
    "CostFunctionStateElementNotify",
]
