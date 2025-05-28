from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .osp_reader import OspCovarianceMatrixReader, OspSpeciesReader
from .state_element import StateElementImplementation, StateElement, StateElementHandle
from .current_state import (
    FullGridMappedArray,
    RetrievalGrid2dArray,
    FullGrid2dArray,
    RetrievalGridArray
)
from .identifier import StateElementIdentifier, RetrievalType
from loguru import logger
from pathlib import Path
import numpy as np
import typing
from typing import Self, Any

if typing.TYPE_CHECKING:
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
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        latitude: float,
        surface_type: str,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: Any | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
    ):
        if value_fm is not None:
            value_fm = value_fm.copy()
        else:
            value_fm = None
        if constraint_vector_fm is not None:
            constraint_vector_fm = constraint_vector_fm.copy()
        else:
            constraint_vector_fm = None
        self._map_type: str | None = None
        self.osp_species_reader = OspSpeciesReader.read_dir(species_directory)
        self.cov_is_constraint = cov_is_constraint
        if not self.cov_is_constraint:
            self.osp_cov_reader = OspCovarianceMatrixReader.read_dir(
                covariance_directory
            )
        self.latitude = latitude
        self.surface_type = surface_type
        self.poltype = poltype
        self.poltype_used_constraint = poltype_used_constraint
        self.retrieval_type = RetrievalType("default")
        self._pressure_level = pressure_list_fm
        # This is to support testing. We currently have a way of populate StateInfoOld when
        # we restart a step, but not StateInfo. Longer term we will fix this, but short term
        # just propagate any values in selem_wrapper to this class
        if selem_wrapper is not None:
            if selem_wrapper.value_str is None:
                value_fm = selem_wrapper.value_fm
        super().__init__(
            state_element_id,
            value_fm,
            constraint_vector_fm,
            apriori_cov_fm=None,
            constraint_matrix=None,
            state_mapping=None,
            selem_wrapper=selem_wrapper,
        )
        if selem_wrapper is not None:
            if selem_wrapper.value_str is None:
                self._step_initial_fm = selem_wrapper.value_fm
        if self.poltype is not None:
            self._metadata["poltype"] = self.poltype

    def _fill_in_constraint(self) -> None:
        if self._constraint_matrix is not None:
            return
        self._constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            self.state_element_id,
            self.retrieval_type,
            self.basis_matrix.shape[0],
            poltype=self.poltype if self.poltype_used_constraint else None,
        ).view(RetrievalGrid2dArray)
        # TODO Short term we cast this to float32, just so can match the old muses-py code.
        # We'll change this shortly, but we want to do this one step at a time so we know
        # exactly what is making things change. Note the float32 is only needed for the cross
        # terms, which get calculated in a different spot in the old code.
        if self.pressure_list_fm is not None:
            self._constraint_matrix = (
                self._constraint_matrix.astype(np.float32)
                .astype(np.float64)
                .view(RetrievalGrid2dArray)
            )

    def _fill_in_state_mapping(self) -> None:
        if self._state_mapping is not None:
            return
        t = self.osp_species_reader.read_file(
            self.state_element_id, self.retrieval_type
        )
        if "retrievalLevels" in t:
            self._retrieval_levels: list[int] | None = [
                int(i) for i in t["retrievalLevels"].split(",")
            ]
        else:
            self._retrieval_levels = None
        self._map_type = t["mapType"].lower()
        if self._map_type == "linear":
            self._state_mapping = rf.StateMappingLinear()
        elif self._map_type == "log":
            self._state_mapping = rf.StateMappingLog()
        else:
            raise RuntimeError(f"Don't recognize map_type {self._map_type}")
        if self._retrieval_levels is None or len(self._retrieval_levels) < 2:
            self._pressure_list_fm = None
        else:
            if self._pressure_level is None:
                raise RuntimeError("pressure_level should not be None")
            self._pressure_list_fm = self._pressure_level.copy()

    def _fill_in_state_mapping_retrieval_to_fm(self) -> None:
        if self._state_mapping_retrieval_to_fm is not None:
            return
        if self._sold is not None:
            # We'll want to create this shortly
            self._state_mapping_retrieval_to_fm = (
                self._sold.state_mapping_retrieval_to_fm
            )
        else:
            self._state_mapping_retrieval_to_fm = rf.StateMappingLinear()

    def _fill_in_apriori(self) -> None:
        if self._apriori_cov_fm is not None:
            return
        if self.cov_is_constraint:
            self._fill_in_constraint()
            self._apriori_cov_fm = self.constraint_matrix.view(FullGrid2dArray)
        else:
            self._fill_in_state_mapping()
            assert self._map_type is not None
            cov_matrix = self.osp_cov_reader.read_cov(
                self.state_element_id,
                self._map_type,
                self.latitude,
                poltype=self.poltype,
            )
            if self._retrieval_levels is None or len(self._retrieval_levels) < 2:
                self._apriori_cov_fm = cov_matrix.original_cov.view(FullGrid2dArray)
                # TODO Temp, we have things like TSUR that are hard coded to using a diagonal matrix.
                # Just punt for now.
                if self._sold is not None:
                    self._apriori_cov_fm = self._sold.apriori_cov_fm
            else:
                if self._pressure_level is None:
                    raise RuntimeError("pressure_level should not be None")
                self._apriori_cov_fm = cov_matrix.interpolated_covariance(
                    self._pressure_level
                ).view(FullGrid2dArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        self._fill_in_constraint()
        res = self._constraint_matrix
        if res is None:
            raise RuntimeError("This can't happen")
        # Skip for H2O and HDO, we have moved cross term handling out so this is
        # different than the old data
        if self.state_element_id not in (
            StateElementIdentifier("H2O"),
            StateElementIdentifier("HDO"),
        ):
            self._check_result(res, "constraint_matrix")
        return res

    @property
    def apriori_cov_fm(self) -> FullGrid2dArray:
        self._fill_in_apriori()
        res = self._apriori_cov_fm
        if res is None:
            raise RuntimeError("This can't happen")
        self._check_result(res, "apriori_cov_fm")
        return res

    @property
    def state_mapping(self) -> rf.StateMapping:
        self._fill_in_state_mapping()
        return self._state_mapping

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        self._fill_in_state_mapping_retrieval_to_fm()
        return self._state_mapping_retrieval_to_fm

    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise). This is the levels that
        the apriori_cov_fm are on."""
        self._fill_in_state_mapping()
        res = self._pressure_list_fm
        self._check_result(res, "pressure_list_fm")
        return res

    @classmethod
    def create_from_handle(
        cls,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray | None,
        constraint_vector_fm: FullGridMappedArray | None,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        selem_wrapper: Any | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
    ) -> Self | None:
        """Create object from the set of parameter the StateElementOspFileHandle supplies.

        We don't actually use all the arguments, but they are there for other classes
        """
        res = cls(
            state_element_id,
            pressure_list_fm,
            value_fm,
            constraint_vector_fm,
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
            poltype=poltype,
            poltype_used_constraint=poltype_used_constraint,
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
        # We need to get the initial guess stuff in place, but for now just
        # grab from old data
        if (
            self._sold is not None
            and current_strategy_step.strategy_step.step_number == 0
        ):
            if self._sold.value_str is None:
                self._step_initial_fm = self._sold.step_initial_fm.copy()
                assert self._step_initial_fm is not None
                self._value_fm = self._step_initial_fm.copy()
        self.retrieval_type = current_strategy_step.retrieval_type
        if(self.state_element_id in current_strategy_step.retrieval_elements):
            t = np.array(self._constraint_vector_fm.to_ret(self.state_mapping_retrieval_to_fm, self.state_mapping))
            # I'm not sure if all types get this correct or not, but I think so.
            if t.min() < 0 and t.max() > 0:
                logger.info(f"Fixing negative mapping for constraint vector for {self.state_element_id}")
                t[t<0] = t[t>0].min()
            self._constraint_vector_fm = t.view(RetrievalGridArray).to_fm(self.state_mapping_retrieval_to_fm, self.state_mapping)
            
        # Most of the time this will just return the same value, but there might be
        # certain steps with a different constraint matrix. So we empty the cache here
        # Note the reader does caching, so reading this multiple times isn't as
        # inefficient as it might seem.
        self._constraint_matrix = None
        self._map_type = None
        self._state_mapping = None
        self._state_mapping_retrieval_to_fm = None


class StateElementOspFileHandle(StateElementHandle):
    def __init__(
        self,
        sid: StateElementIdentifier,
        value_fm: FullGridMappedArray,
        constraint_vector_fm: FullGridMappedArray,
        hold: Any | None = None,
        cls: type[StateElementOspFile] = StateElementOspFile,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obj_cls = cls
        self.sid = sid
        self.hold = hold
        self.value_fm = value_fm
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
        if state_element_id != self.sid:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = self.hold.state_element(state_element_id)
        else:
            sold = None
        # Determining pressure is spread across a number of muses-py functions. We'll need
        # to track all this down, but short term just get this from the old data
        if self.hold is not None:
            p = self.hold.state_element(StateElementIdentifier("pressure"))
            assert p is not None
            pressure_list_fm = p.value_fm.copy()
        else:
            pressure_list_fm = None
        res = self.obj_cls.create_from_handle(
            state_element_id,
            pressure_list_fm,
            self.value_fm,
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
