from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
import numpy as np
from pathlib import Path
from copy import copy
import os
import typing
from refractor.muses import (
    StateElementIdentifier,
    RetrievalConfiguration,
    MeasurementId,
    CurrentStrategyStep,
    MusesStrategy,
    ObservationHandleSet,
    StateElement,
    StateElementHandleSet,
    CurrentState,
    RetrievalGridArray,
    FullGridArray,
    FullGridMappedArray,
    FullGrid2dArray,
    RetrievalGrid2dArray,
    SoundingMetadata,
    PropagatedQA,
)
from .retrieval_info import RetrievalInfo

if typing.TYPE_CHECKING:
    from refractor.old_py_retrieve_wrapper import (  # type: ignore
        StateInfoOld,
        StateElementOld,
        StateElementHandleSetOld,
    )


class CurrentStateStateInfoOld(CurrentState):
    """Implementation of CurrentState that uses our StateInfoOld. This is
    the way the actual full retrieval works.

    This class uses the old py-retrieve code we wrapped, and was used for
    initial development, and testing our implementation of CurrentStateStateInfo.
    Unless you are testing old code, you don't actually want to use this class,
    but instead use CurrentStateStateInfo.
    """

    def __init__(
        self,
        state_info: StateInfoOld | None,
        retrieval_info: RetrievalInfo | None = None,
        step_directory: str | os.PathLike[str] | None = None,
    ) -> None:
        """The retrieval_state_element_override is an odd argument, it
        overrides the retrieval_state_element_id in RetrievalInfo with a
        different set. It isn't clear why this is handled this way -
        why doesn't RetrievalInfo just figure out the right
        retrieval_state_element_id list?  But for now, do it the same way
        as py-retrieve. This seems to only be used in the
        RetrievalStrategyStepBT - I'm guessing this was a kludge put
        in to support this retrieval step.

        In addition, the CurrentState can also be used when we are
        calculating the "systematic" jacobian. This create a
        StateVector with a different set of state elements.  This
        isn't used to do a retrieval, but rather to just calculate a
        jacobian.  If do_systematic is set to True, we use this values
        instead.
        """

        super().__init__()
        self._state_info = state_info
        self.retrieval_state_element_override: None | list[StateElementIdentifier] = (
            None
        )
        self.do_systematic = False
        self._step_directory = (
            Path(step_directory) if step_directory is not None else None
        )
        self._retrieval_info: RetrievalInfo | None = None

    @property
    def state_info(self) -> StateInfoOld:
        if self._state_info is None:
            from refractor.old_py_retrieve_wrapper import StateInfoOld  # type: ignore

            self._state_info = StateInfoOld()
        return self._state_info

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

    @property
    def initial_guess(self) -> RetrievalGridArray:
        # Not sure about systematic handling here. I think this is all
        # zeros, not sure if that is right or not.
        if self._retrieval_info is None:
            raise RuntimeError("_retrieval_info is None")
        if self.do_systematic:
            return copy(
                self._retrieval_info.retrieval_info_systematic().initialGuessList
            ).view(RetrievalGridArray)
        else:
            return copy(self._retrieval_info.initial_guess_list).view(
                RetrievalGridArray
            )

    @property
    def initial_guess_full(self) -> FullGridArray:
        if self._retrieval_info is None:
            raise RuntimeError("_retrieval_info is None")
        if self.do_systematic:
            return copy(
                self._retrieval_info.retrieval_info_systematic().initialGuessListFM
            ).view(FullGridArray)
        else:
            return copy(self._retrieval_info.initial_guess_list_fm).view(FullGridArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((1, 1)).view(RetrievalGrid2dArray)
        else:
            if self._retrieval_info is None:
                raise RuntimeError("_retrieval_info is None")
            return copy(self._retrieval_info.constraint_matrix).view(
                RetrievalGrid2dArray
            )

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.eye(len(self.initial_guess)).view(RetrievalGridArray)
        else:
            return (
                (mpy.sqrt_matrix(self.constraint_matrix))
                .transpose()
                .view(RetrievalGridArray)
            )

    def constraint_vector(self, fix_negative: bool = True) -> RetrievalGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((len(self.initial_guess),)).view(RetrievalGridArray)
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return copy(self.retrieval_info.constraint_vector).view(RetrievalGridArray)

    @property
    def constraint_vector_full(self) -> FullGridArray:
        # Not sure about systematic handling here.
        if self.do_systematic:
            return np.zeros((len(self.initial_guess_full),)).view(FullGridArray)
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return copy(self.retrieval_info.retrieval_dict["constraintVectorFM"]).view(
                FullGridArray
            )

    @property
    def true_value(self) -> RetrievalGridArray:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return copy(self.retrieval_info.true_value).view(RetrievalGridArray)

    @property
    def true_value_full(self) -> FullGridArray:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return copy(self.retrieval_info.true_value_fm).view(FullGridArray)

    @property
    def basis_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return self.retrieval_info.basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        if self.do_systematic:
            return None
        else:
            if self.retrieval_info is None:
                raise RuntimeError("retrieval_info is None")
            return self.retrieval_info.map_to_parameter_matrix

    @property
    def step_directory(self) -> Path:
        if self._step_directory is None:
            raise RuntimeError("Set step directory first")
        return self._step_directory

    @step_directory.setter
    def step_directory(self, val: Path) -> None:
        self._step_directory = val

    @property
    def propagated_qa(self) -> PropagatedQA:
        return self.state_info.propagated_qa

    @property
    def brightness_temperature_data(self) -> dict:
        return self.state_info.brightness_temperature_data

    @property
    def updated_fm_flag(self) -> FullGridArray:
        return self.retrieval_info.retrieval_info_obj.doUpdateFM

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> None:
        self.state_info.update_state(
            self.retrieval_info, results_list, do_not_update, retrieval_config, step
        )

    def update_full_state_element(
        self,
        state_element_id: StateElementIdentifier,
        current_fm: FullGridMappedArray | None = None,
        constraint_vector_fm: FullGridMappedArray | None = None,
        next_constraint_vector_fm: FullGridMappedArray | None = None,
        step_initial_fm: FullGridMappedArray | None = None,
        next_step_initial_fm: FullGridMappedArray | None = None,
        retrieval_initial_fm: FullGridMappedArray | None = None,
        true_value_fm: FullGridMappedArray | None = None,
    ) -> None:
        selem = self.state_element_old(state_element_id)
        selem.update_state(
            current=current_fm,
            apriori=constraint_vector_fm
            if constraint_vector_fm is not None
            else next_constraint_vector_fm,
            initial=step_initial_fm if step_initial_fm else next_step_initial_fm,
            initial_initial=retrieval_initial_fm,
            true=true_value_fm,
        )

    @property
    def retrieval_info(self) -> RetrievalInfo:
        if self._retrieval_info is None:
            raise RuntimeError("Need to set self._retrieval_info")
        return self._retrieval_info

    @retrieval_info.setter
    def retrieval_info(self, val: RetrievalInfo) -> None:
        self._retrieval_info = val
        # Clear cache, we need to regenerate these after update
        self.clear_cache()

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
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
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        if self.retrieval_state_element_override is not None:
            return self.retrieval_state_element_override
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        return [
            StateElementIdentifier(i) for i in self.retrieval_info.species_names_sys
        ]

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        return [i.name for i in self.state_info.state_element_list()]

    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
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
        return self.state_info.sounding_metadata()

    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        raise NotImplementedError()

    def state_element_old(
        self,
        state_element_id: StateElementIdentifier | str,
        step: str = "current",
        other_name: bool = True,
    ) -> StateElementOld:
        sid = state_element_id
        # The old code uses CLOUDEXT and cloudEffExt and EMIS and emissivity as almost
        # synonyms. The code does different things for each value, which really doesn't
        # make sense - but was how the code was set up. We want to collapse this to just
        # CLOUDEXT and EMIS, which we do here. Depending on what we are doing with this,
        # we either want to use the other name or not when we go to the old code.
        if other_name and str(sid) == "CLOUDEXT":
            sid = StateElementIdentifier("cloudEffExt")
        if other_name and str(sid) == "EMIS":
            sid = StateElementIdentifier("emissivity")
        return self.state_info.state_element(sid, step)

    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        selem = self.state_element_old(state_element_id)
        return selem.spectral_domain_wavelength

    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        selem = self.state_element_old(state_element_id)
        return copy(selem.value).view(FullGridMappedArray)

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        selem = self.state_element_old(state_element_id, step="initial")
        return copy(selem.value).view(FullGridMappedArray)

    def state_value_str(
        self, state_element_id: StateElementIdentifier | str
    ) -> str | None:
        """We no longer use value_str in our StateElement, it was an awkward way
        to handle poltype used by things like NH3. However we need access
        to the old state element data so we provide that here.
        """
        selem = self.state_element_old(state_element_id)
        if not hasattr(selem, "value_str"):
            return None
        return selem.value_str

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        selem = self.state_element_old(state_element_id, step="true")
        return copy(selem.value).view(FullGridMappedArray)

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        selem = self.state_element_old(state_element_id, step="initialInitial")
        return copy(selem.value).view(FullGridMappedArray)

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        selem = self.state_element_old(state_element_id, other_name=True)
        return copy(selem.apriori_value).view(FullGridMappedArray)

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        selem = self.state_element_old(state_element_id)
        return copy(selem.sa_covariance).view(FullGrid2dArray)

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self.retrieval_info is None:
            raise RuntimeError("retrieval_info is None")
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for state_element_id in self.retrieval_state_element_id:
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

    def state_mapping(
        self, state_element_id: StateElementIdentifier | str
    ) -> rf.StateMapping:
        selem = self.state_element_old(state_element_id, other_name=False)
        mtype = selem.map_type
        if mtype == "linear":
            return rf.StateMappingLinear()
        elif mtype == "log":
            return rf.StateMappingLog()
        else:
            raise RuntimeError(f"Don't recognize mtype {mtype}")

    def pressure_list(
        self, state_element_id: StateElementIdentifier | str
    ) -> RetrievalGridArray | None:
        selem = self.state_element_old(state_element_id, other_name=False)
        if hasattr(selem, "pressureList"):
            return selem.pressureList.view(RetrievalGridArray)
        return None

    def pressure_list_fm(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        selem = self.state_element_old(state_element_id, other_name=False)
        if hasattr(selem, "pressureListFM"):
            return selem.pressureListFM.view(FullGridMappedArray)
        return None

    # Some of these arguments may get put into the class, but for now have explicit
    # passing of these
    def get_initial_guess(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        """Set retrieval_info, errorInitial and errorCurrent for the current step."""
        # Chicken and egg problem with circular dependency, so we just import this
        # when we need it here
        from .retrieval_info import RetrievalInfo

        for selem_id in current_strategy_step.retrieval_elements:
            selem = self.state_element_old(selem_id, other_name=False)
            selem.update_initial_guess(current_strategy_step)

        self._retrieval_info = RetrievalInfo(
            Path(retrieval_config["speciesDirectory"]),
            current_strategy_step,
            self,
        )

        # Isn't really clear why RetrievalInfo is different, but for
        # now this update is needed. Without this we get different results.

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self._retrieval_info.n_totalParameters
        if nparm > 0:
            xig = self._retrieval_info.initial_guess_list[0:nparm]
            self.update_state(
                xig,
                [],
                retrieval_config,
                current_strategy_step.strategy_step.step_number,
            )
        self.clear_cache()

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self.state_info.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set
        )
        # According to Susan, historically the initial guess stuff was translated
        # to and from the retrieval grid. This was done so that paired retrievals
        # are in sync, so if we retrieve H2O with O3 held fixed and then O3 we
        # don't want the O3 to jump a bunch as it goes to the retrieval grid. She
        # said this is less important now where a lot of stuff is retrieved at the
        # same time. But muses-py cycled through all the strategy table, which
        # had the side effect of calling get_initial_guess() and taking values to
        # and from the retrieval grid (so FullGridMappedArrayFromRetGrid). For
        # now, we duplicate this here.
        curdir = os.getcwd()
        try:
            os.chdir(retrieval_config["run_dir"])
            cstep = strategy.current_strategy_step()
            if(cstep is None):
                raise RuntimeError("Called with strategy is_done() is True")
            cstepnum = cstep.strategy_step.step_number
            strategy.restart()
            while not strategy.is_done():
                self.notify_start_step(
                    strategy.current_strategy_step(),
                    retrieval_config,
                    skip_initial_guess_update=True,
                )
                strategy.next_step(self)
            strategy.set_step(cstepnum, self)
        finally:
            os.chdir(curdir)

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        raise NotImplementedError()

    @property
    def state_element_handle_set_old(self) -> StateElementHandleSetOld:
        return self.state_info.state_element_handle_set

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        # current_strategy_step being None means we are past the last step in our
        # MusesStrategy, so we just skip doing anything
        if current_strategy_step is not None:
            self.step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            if not skip_initial_guess_update:
                self.state_info.next_state_to_current()
                self.state_info.copy_current_initial()
            # Doesn't seem right that we update initial *before* doing get_initial_guess,
            # but that seems to be what happens
            self.get_initial_guess(current_strategy_step, retrieval_config)
            # Save some data needed by notify_step_solution
            self._retrieval_config = retrieval_config
            self._do_not_update = current_strategy_step.retrieval_elements_not_updated
            self._step = current_strategy_step.strategy_step.step_number

    def notify_step_solution(self, xsol: RetrievalGridArray) -> None:
        """Called with MusesStrategy has found a solution for the current step."""
        self.state_info.update_state(
            self.retrieval_info,
            xsol,
            self._do_not_update,
            self._retrieval_config,
            self._step,
        )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if current_strategy_step is not None:
            self.step_directory = (
                retrieval_config["run_dir"]
                / f"Step{current_strategy_step.strategy_step.step_number:02d}_{current_strategy_step.strategy_step.step_name}"
            )
            self.state_info.restart()
            self.state_info.copy_current_initialInitial()
            self.state_info.copy_current_initial()


__all__ = [
    "CurrentStateStateInfoOld",
]
