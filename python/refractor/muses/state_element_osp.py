from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .osp_reader import OspCovarianceMatrixReader, OspSpeciesReader
from .state_element import StateElementImplementation, StateElement, StateElementHandle
from .current_state import FullGridMappedArray, RetrievalGrid2dArray, FullGrid2dArray
from .identifier import StateElementIdentifier, RetrievalType
from loguru import logger
from pathlib import Path
import typing
from typing import cast, Self

if typing.TYPE_CHECKING:
    from .state_element_old_wrapper import (
        StateElementOldWrapper,
        StateElementOldWrapperHandle,
    )
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import SoundingMetadata

class StateElementOspFile(StateElementImplementation):
    """This implementation of StateElement gets the apriori/initial guess as a hard coded
    value, and the constraint_matrix and apriori_cov_fm from OSP files. This seems a
    bit convoluted to me - why not just have all the values given in the python configuration
    file? But this is the way muses-py works, and at the very least we need to implementation
    for backwards testing.  We may replace this StateElement, there doesn't seem to be any
    good reason to spread everything across multiple files.

    In some cases, we have the species in the covariance species_directory but not the
    covariance_directory. You can optionally request that we just use the constraint
    matrix as the apriori_cov_fm.

    Also, muses-py has the bad habit of not actually having files and/or valid files for
    all the state elements. For example, for AIRS OMI TATM points to a nonexistent constraint
    matrix file.  muses-py doesn't run into problems because it only reads the files when things
    are used in a retrieval.

    We need to duplicate this functionality, so things are read on first use rather than in the
    constructor. This gives the bad behavior that trying to look at constraint_matrix for the
    wrong StateElement may suddenly give you an error. Unfortunately, we can't avoid this - this
    is how the OSP files are set up and how muses-py works. 
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        constraint_vector_fm: FullGridMappedArray | None,
        latitude: float,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
        copy_on_first_use: bool = False
    ):
        if(constraint_vector_fm is not None):
            value_fm = constraint_vector_fm.copy()
            constraint_vector_fm = constraint_vector_fm.copy()
        else:
            value_fm = None
            constraint_vector_fm = None
            
        self.osp_species_reader = OspSpeciesReader.read_dir(species_directory)
        self.cov_is_constraint = cov_is_constraint
        if(not self.cov_is_constraint):
            self.osp_cov_reader = OspCovarianceMatrixReader.read_dir(covariance_directory)
            self.latitude=latitude
        self.retrieval_type = RetrievalType("default")
        # This is to support testing. We currently have a way of populate StateInfoOld when
        # we restart a step, but not StateInfo. Longer term we will fix this, but short term
        # just propagate any values in selem_wrapper to this class
        if selem_wrapper is not None:
            value_fm = selem_wrapper.value_fm
        super().__init__(
            state_element_id,
            value_fm,
            constraint_vector_fm,
            apriori_cov_fm=None,
            constraint_matrix=None,
            state_mapping=None,
            selem_wrapper=selem_wrapper,
            copy_on_first_use=copy_on_first_use
        )
        if selem_wrapper is not None:
            self._step_initial_fm = selem_wrapper.value_fm

    def _fill_in_constraint(self):
        if(self._constraint_matrix is not None):
            return
        self._constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            self.state_element_id, self.retrieval_type
        ).view(RetrievalGrid2dArray)

    def _fill_in_state_mapping(self):
        if(self._state_mapping is not None):
            return
        t = self.osp_species_reader.read_file(
            self.state_element_id, self.retrieval_type
        )
        self._map_type = t["mapType"].lower()
        if self._map_type == "linear":
            self._state_mapping = rf.StateMappingLinear()
        elif map_type == "log":
            self._state_mapping = rf.StateMappingLog()
        else:
            raise RuntimeError(f"Don't recognize map_type {map_type}")

    def _fill_in_apriori(self):
        if(self._apriori_cov_fm is not None):
            return
        if self.cov_is_constraint:
            self._fill_in_constraint()
            self._apriori_cov_fm = self.constraint_matrix.view(FullGrid2dArray)
        else:
            self._fill_in_state_mapping()
            self._apriori_cov_fm = self._sold.apriori_cov_fm
            self._apriori_cov_fm = self.osp_cov_reader.read_cov(self.state_element_id, self._map_type, self.latitude).view(FullGrid2dArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        self._fill_in_constraint()
        res = self._constraint_matrix
        if self._sold is not None:
            res2 = self._sold.constraint_matrix
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def apriori_cov_fm(self) -> FullGrid2dArray:
        self._fill_in_apriori()
        res = self._apriori_cov_fm
        if self._sold is not None:
            try:
                res2 = self._sold.apriori_cov_fm
            except AssertionError:
                res2 = None
            if res2 is not None:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res
    
    @property
    def state_mapping(self) -> rf.StateMapping:
        self._fill_in_state_mapping()
        return self._state_mapping
    
    @classmethod
    def create_from_handle(
        cls,
        state_element_id: StateElementIdentifier,
        constraint_vector_fm: FullGridMappedArray | None,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
        copy_on_first_use:bool = False
    ) -> Self | None:
        """Create object from the set of parameter the StateElementOspFileHandle supplies.

        We don't actually use all the arguments, but they are there for other classes
        """
        res = cls(
            state_element_id,
            constraint_vector_fm,
            sounding_metadata.latitude.value,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
            copy_on_first_use=copy_on_first_use
        )
        return res

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        super().notify_start_step(
            current_strategy_step,
            retrieval_config,
            skip_initial_guess_update,
        )
        self.retrieval_type = current_strategy_step.retrieval_type
        # Most of the time this will just return the same value, but there might be
        # certain steps with a different constraint matrix. So we empty the cache here
        # Note the reader does caching, so reading this multiple times isn't as
        # inefficient as it might seem.
        self._constraint_matrix = None
        self._map_type = None
        self._state_mapping = None
        
    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        # TODO Add handling for pressure_list
        return None

class StateElementOspFileHandle(StateElementHandle):
    def __init__(
        self,
        sid: StateElementIdentifier,
        constraint_vector_fm: FullGridMappedArray,
        hold: StateElementOldWrapperHandle | None = None,
        cls: type[StateElementOspFile] = StateElementOspFile,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obj_cls = cls
        self.sid = sid
        self.hold = hold
        self.constraint_vector_fm = constraint_vector_fm
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = cov_is_constraint

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config
        self.strategy = strategy
        self.observation_handle_set = observation_handle_set
        self.sounding_metadata = sounding_metadata

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        from .state_element_old_wrapper import StateElementOldWrapper

        if state_element_id != self.sid:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = cast(
                StateElementOldWrapper, self.hold.state_element(state_element_id)
            )
        else:
            sold = None
        res = self.obj_cls.create_from_handle(
            state_element_id,
            self.constraint_vector_fm,
            self.measurement_id,
            self.retrieval_config,
            self.strategy,
            self.observation_handle_set,
            self.sounding_metadata,
            sold,
            self.cov_is_constraint,
        )
        if res is not None:
            logger.debug(f"Creating {self.obj_cls.__name__} for {state_element_id}")
        return res

__all__ = [
    "StateElementOspFileHandle",
    "StateElementOspFile",
]    
