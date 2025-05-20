from __future__ import annotations
from . import muses_py as mpy  # type: ignore
from .current_state import (
    CurrentState,
    RetrievalGridArray,
    FullGridArray,
    FullGridMappedArray,
    RetrievalGrid2dArray,
    FullGrid2dArray
)
from .identifier import StateElementIdentifier
from .state_element_old_wrapper import StateElementOldWrapperHandle
from .state_info import StateElementHandleSet
import numpy as np
import scipy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy.testing as npt
from pathlib import Path
from copy import copy
import typing

if typing.TYPE_CHECKING:
    from .current_state import PropagatedQA, SoundingMetadata
    from .retrieval_configuration import RetrievalConfiguration
    from .muses_strategy import CurrentStrategyStep
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
        # Temp
        self._current_state_old = self._state_info._current_state_old
        self.retrieval_state_element_override: None | list[StateElementIdentifier] = (
            None
        )
        self.do_systematic = False
        self._step_directory: None | Path = None
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
        cstate = self._state_info._current_state_old
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
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].step_initial_ret
                    for sid in self.retrieval_state_element_id
                ]
            )
        if True:
            res2 = self._current_state_old.initial_guess
            # Short term, see if this is only difference
            #return res2.view(RetrievalGridArray)
            # Need to fix
            npt.assert_allclose(res, res2, 1e-12)
        return res.view(RetrievalGridArray)

    @property
    def initial_guess_full(self) -> FullGridArray:
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
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].step_initial_full
                    for sid in self.retrieval_state_element_id
                ]
            )
        if True:
            res2 = self._current_state_old.initial_guess_full
            # Short term, see if this is only difference
            #return res2.view(FullGridArray)
            # Need to fix
            npt.assert_allclose(res, res2, 1e-12)
        return res.view(FullGridArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        if self.do_systematic:
            return np.zeros((1, 1)).view(RetrievalGrid2dArray)
        self.match_old()
        blist = [
            self._state_info[sid].constraint_matrix
            for sid in self.retrieval_state_element_id
        ]
        res = scipy.linalg.block_diag(*blist)
        for i, selem1_sid in enumerate(self.retrieval_state_element_id):
            for selem2_sid in self.retrieval_state_element_id[i + 1 :]:
                m = self._state_info[selem1_sid].constraint_cross_covariance(
                    self._state_info[selem2_sid]
                )
                if m is not None:
                    r1 = self.retrieval_sv_slice(selem1_sid)
                    r2 = self.retrieval_sv_slice(selem2_sid)
                    res[r1, r2] = m
                    res[r2, r1] = np.transpose(m)
        # TODO Remove current_state_old
        if True:
            res2 = self._current_state_old.constraint_matrix
            # Short term, see if this is only difference
            return res2.view(RetrievalGrid2dArray)
            # Need to fix
            npt.assert_allclose(res, res2)
        return res.view(RetrievalGrid2dArray)

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        if self.do_systematic:
            return np.eye(len(self.initial_guess)).view(RetrievalGridArray)
        else:
            return (mpy.sqrt_matrix(self.constraint_matrix)).transpose().view(RetrievalGridArray)

    @property
    def constraint_vector(self) -> RetrievalGridArray:
        # TODO
        # By convention, muses-py returns a length 1 array even if we don't
        # have any retrieval_state_element_id. I think this was just to avoid
        # zero size arrays in IDL. But python is fine with this. For now conform
        # to muses-py since it is expected in various places. We should clean this
        # up at some point by tracking down where this gets used and handling empty
        # arrays - it is cleaner than having a "special rule". But for now, conform
        # to the convention
        if len(self.retrieval_state_element_id) == 0:
            res = np.zeros((1,)).view(RetrievalGridArray)
        else:
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].constraint_vector_ret
                    for sid in self.retrieval_state_element_id
                ]
            )
        if True:
            res2 = self._current_state_old.constraint_vector
            # Short term, see if this is only difference
            return res2.view(RetrievalGridArray)
            # Need to fix
            npt.assert_allclose(res, res2)
        return res.view(RetrievalGridArray)

    @property
    def constraint_vector_full(self) -> FullGridArray:
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
            self.match_old()
            res = np.concatenate(
                [
                    self._state_info[sid].constraint_vector_full
                    for sid in self.retrieval_state_element_id
                ]
            )
        if True:
            try:
                res2 = self._current_state_old.constraint_vector_full
                # Short term, see if this is only difference
                return res2.view(FullGridArray)
                # Need to fix
                npt.assert_allclose(res, res2)
            except KeyError:
                pass
        return res.view(FullGridArray)

    @property
    def true_value(self) -> RetrievalGridArray:
        # Note muses_py always has a true value vector, even if we don't have
        # a true value (so state_true_value is None). It just puts zeros
        # in for any missing data.
        res = np.zeros(
            (
                len(
                    self.constraint_vector,
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
    def propagated_qa(self) -> PropagatedQA:
        return self._state_info.propagated_qa

    @property
    def brightness_temperature_data(self) -> dict:
        return self._state_info.brightness_temperature_data

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
        current_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        self.match_old()
        self._state_info[state_element_id].update_state_element(
            current_fm, constraint_vector_fm, step_initial_fm, retrieval_initial_fm, true_value_fm
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
                if self.do_systematic:
                    plen = self._state_info[sid].sys_sv_length
                else:
                    plen = self._state_info[sid].forward_model_sv_length
                if plen == 0:
                    plen = 1
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
        sid = StateElementIdentifier(state_element_id) if isinstance(state_element_id, str)  else state_element_id
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
        res = self.state_element(state_element_id).value_fm
        return res.view(FullGridMappedArray)

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        res = self.state_element(state_element_id).step_initial_fm
        return res.view(FullGridMappedArray)

    def state_value_str(
        self, state_element_id: StateElementIdentifier | str
    ) -> str | None:
        res = self.state_element(state_element_id).value_str
        return res

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        res = self.state_element(state_element_id).true_value_fm
        if(res is None):
            return None
        return res.view(FullGridMappedArray)

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        res = self.state_element(state_element_id).retrieval_initial_fm
        return res.view(FullGridMappedArray)

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        res = self.state_element(state_element_id).constraint_vector_fm
        return res.view(FullGridMappedArray)

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        res = self.state_element(state_element_id).apriori_cov_fm
        return res.view(FullGrid2dArray)

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        self.match_old()
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
                plen = self._state_info[sid].step_initial_ret.shape[0]
                if plen == 0:
                    plen = 1
                self._retrieval_sv_loc[sid] = (
                    self._retrieval_state_vector_size,
                    plen,
                )
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    def pressure_list(
        self, state_element_id: StateElementIdentifier | str
    ) -> RetrievalGridArray | None:
        res = self.state_element(state_element_id).pressure_list
        if(res is None):
            return None
        return res.view(RetrievalGridArray)

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        res = self.state_element(state_element_id).pressure_list_fm
        if(res is None):
            return None
        return res.view(FullGridMappedArray)

    def altitude_list(
        self, state_element_id: StateElementIdentifier | str
    ) -> RetrievalGridArray | None:
        res = self.state_element(state_element_id).altitude_list
        if(res is None):
            return None
        return res.view(RetrievalGridArray)

    def altitude_list_fm(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        res = self.state_element(state_element_id).altitude_list_fm
        if(res is None):
            return None
        return res.view(FullGridMappedArray)

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self._state_info.state_element_handle_set

    @state_element_handle_set.setter
    def state_element_handle_set(self, val: StateElementHandleSet) -> None:
        self._state_info.state_element_handle_set = val

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
        if current_strategy_step is not None:
            self.match_old()
            self._retrieval_element_id = current_strategy_step.retrieval_elements
            self._sys_element_id = current_strategy_step.error_analysis_interferents
            self._step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
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
            self.match_old()
            self._retrieval_element_id = current_strategy_step.retrieval_elements
            self._sys_element_id = current_strategy_step.error_analysis_interferents
            self._step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            self._state_info.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )
            self.clear_cache()

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        self.match_old()
        for selem in self._state_info.values():
            selem.notify_step_solution(
                xsol, self.retrieval_sv_slice(selem.state_element_id)
            )


# Right now, only fall back to old py-retrieve code
h_old = StateElementOldWrapperHandle()
StateElementHandleSet.add_default_handle(h_old, priority_order=-1)
__all__ = [
    "CurrentStateStateInfo",
]
