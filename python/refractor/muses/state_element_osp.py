from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .osp_reader import (
    OspCovarianceMatrixReader,
    OspSpeciesReader,
    OspDiagonalUncertainityReader,
)
from .state_element import StateElementWithCreate
from .retrieval_array import FullGridMappedArray, RetrievalGrid2dArray, FullGrid2dArray
from .identifier import StateElementIdentifier, RetrievalType
from pathlib import Path
import numpy as np
import typing
from typing import Self, Any, NamedTuple


if typing.TYPE_CHECKING:
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
    from .input_file_helper import InputFileHelper
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .sounding_metadata import SoundingMetadata
    from .state_info import StateInfo


# Return type for _setup_create
class OspSetupReturn(NamedTuple):
    value_fm: FullGridMappedArray
    constraint_vector_fm: FullGridMappedArray | None = None
    sid: StateElementIdentifier | None = None
    create_kwargs: dict[str, Any] = {}


class StateElementOspFile(StateElementWithCreate):
    """This implementation of StateElement gets the apriori/initial
    guess as a hard coded value, and the constraint_matrix and
    apriori_cov_fm from OSP files. This seems a bit convoluted to me -
    why not just have all the values given in the python configuration
    file? But this is the way muses-py works, and at the very least we
    need to implementation for backwards testing.  We may replace this
    StateElement, there doesn't seem to be any good reason to spread
    everything across multiple files.

    In some cases, we have the species in the covariance
    species_directory but not the covariance_directory. You can
    optionally request that we just use the constraint matrix as the
    apriori_cov_fm.

    Also, muses-py has the bad habit of not actually having files
    and/or valid files for all the state elements. For example, for
    AIRS OMI TATM points to a nonexistent constraint matrix file.
    muses-py doesn't run into problems because it only reads the files
    when things are used in a retrieval.

    We need to duplicate this functionality, so things are read on
    first use rather than in the constructor. This gives the bad
    behavior that trying to look at constraint_matrix for the wrong
    StateElement may suddenly give you an error. Unfortunately, we
    can't avoid this - this is how the OSP files are set up and how
    muses-py works.

    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray | None,
        value_fm: FullGridMappedArray,
        constraint_vector_fm: FullGridMappedArray,
        latitude: float,
        surface_type: str,
        ifile_hlp: InputFileHelper,
        species_directory: Path,
        covariance_directory: Path,
        spectral_domain: rf.SpectralDomain | None = None,
        selem_wrapper: Any | None = None,
        cov_is_constraint: bool = False,
        poltype: str | None = None,
        poltype_used_constraint: bool = True,
        diag_cov: bool = False,
        diag_directory: Path | None = None,
        metadata: dict[str, Any] | None = None,
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
        self._surf_type: str | None = None
        self.ifile_hlp = ifile_hlp
        self.osp_species_reader: OspSpeciesReader = OspSpeciesReader.read_dir(
            species_directory, ifile_hlp
        )
        self.cov_is_constraint = cov_is_constraint
        self.diag_cov = diag_cov
        if not self.cov_is_constraint:
            self.osp_cov_reader = OspCovarianceMatrixReader.read_dir(
                covariance_directory, ifile_hlp
            )
        if self.diag_cov:
            self.osp_diag_reader = OspDiagonalUncertainityReader.read_dir(
                diag_directory
                if diag_directory is not None
                else covariance_directory / "DiagonalUncertainty",
                ifile_hlp,
            )
        self.latitude = latitude
        self.surface_type = surface_type
        self.poltype = poltype
        self.poltype_used_constraint = poltype_used_constraint
        self.retrieval_type = RetrievalType("default")
        if pressure_list_fm is not None:
            self._pressure_level: FullGridMappedArray | None = pressure_list_fm.astype(
                np.float64, copy=True
            ).view(FullGridMappedArray)
        else:
            self._pressure_level = None
        self._pressure_species_input = np.array(
            [
                0.0,
            ]
        )  # Filled in with notify_start_step.
        super().__init__(
            state_element_id,
            value_fm,
            constraint_vector_fm,
            apriori_cov_fm=None,
            constraint_matrix=None,
            state_mapping=None,
            selem_wrapper=selem_wrapper,
            spectral_domain=spectral_domain,
        )
        if self.poltype is not None:
            self._metadata["poltype"] = self.poltype
        if metadata is not None:
            self._metadata.update(metadata)

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
        self._retrieval_levels = self.osp_species_reader.retrieval_levels(
            self.state_element_id, self.retrieval_type
        )
        self._map_type = self.osp_species_reader.map_type(
            self.state_element_id, self.retrieval_type
        )
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
        self._fill_in_state_mapping()
        if self._retrieval_levels is not None:
            ind = self.tes_levels(self._retrieval_levels, self._pressure_species_input)
            self._state_mapping_retrieval_to_fm = (
                rf.StateMappingBasisMatrix.from_x_subset(self.pressure_list_fm, ind)
            )
        else:
            self._state_mapping_retrieval_to_fm = rf.StateMappingLinear()

    def _fill_in_apriori(self) -> None:
        if self._apriori_cov_fm is not None:
            return
        if self.cov_is_constraint:
            self._fill_in_constraint()
            self._apriori_cov_fm = self.constraint_matrix.copy().view(FullGrid2dArray)
        else:
            self._fill_in_state_mapping()
            assert self._map_type is not None
            if self.diag_cov:
                self._apriori_cov_fm = self.osp_diag_reader.read_cov(
                    self.state_element_id, self.surface_type, self.latitude
                ).view(FullGrid2dArray)
            else:
                cov_matrix = self.osp_cov_reader.read_cov(
                    self.state_element_id,
                    self._map_type,
                    self.latitude,
                    poltype=self.poltype,
                )
                if self._retrieval_levels is None or len(self._retrieval_levels) < 2:
                    self._apriori_cov_fm = cov_matrix.original_cov.view(FullGrid2dArray)
                else:
                    if self._pressure_level is None:
                        raise RuntimeError("pressure_level should not be None")
                    self._apriori_cov_fm = cov_matrix.interpolated_covariance(
                        self._pressure_level
                    ).view(FullGrid2dArray)

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        self._fill_in_constraint()
        assert self._constraint_matrix is not None
        res = self._constraint_matrix.view()
        res.flags.writeable = False
        # Skip for H2O and HDO, we have moved cross term handling out so this is
        # different than the old data. Also EMIS and CLOUDEXT have different handling,
        # and can be slightly larger than check_result, so skip for them
        if self.state_element_id not in (
            StateElementIdentifier("H2O"),
            StateElementIdentifier("HDO"),
            StateElementIdentifier("EMIS"),
            StateElementIdentifier("CLOUDEXT"),
        ):
            self._check_result(res, "constraint_matrix")
        return res

    @property
    def apriori_cov_fm(self) -> FullGrid2dArray:
        self._fill_in_apriori()
        assert self._apriori_cov_fm is not None
        res = self._apriori_cov_fm.view()
        res.flags.writeable = False
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

    def need_retrieval_initial_fm_from_cycle(self) -> bool:
        if self._need_retrieval_initial_fm_from_cycle is None:
            # muses-py is bad about not having OSP files for elements
            # that it doesn't actually retrieve. We assume if this
            # happens that we don't retrieve the element, and don't
            # need to cycle.
            try:
                self._need_retrieval_initial_fm_from_cycle = (
                    self.pressure_list_fm is not None
                    or self.spectral_domain is not None
                )
            except (FileNotFoundError, RuntimeError):
                self._need_retrieval_initial_fm_from_cycle = False
        return self._need_retrieval_initial_fm_from_cycle

    @property
    def pressure_list_fm(self) -> FullGridMappedArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise). This is the levels that
        the apriori_cov_fm are on."""
        self._fill_in_state_mapping()
        if self._pressure_list_fm is not None:
            res: FullGridMappedArray | None = self._pressure_list_fm.view()
            assert res is not None
            res.flags.writeable = False
        else:
            res = None
        self._check_result(res, "pressure_list_fm")
        return res

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        **kwargs: Any,
        # Mark as kwarg only, to make mypy happy when down stream
        # classes ignore arguments
        # List of arguments
        # pressure_list_fm: FullGridMappedArray,
        # sid: StateElementIdentifier,
        # retrieval_config: RetrievalConfiguration,
        # sounding_metadata: SoundingMetadata,
        # measurement_id: MeasurementId,
        # strategy: MusesStrategy,
        # observation_handle_set: ObservationHandleSet,
        # state_info: StateInfo,
        # selem_wrapper: Any,
        # **kwargs: Any,
    ) -> OspSetupReturn | None:
        """Return the StateElementIdentifier, initial value_fm, constraint_vector_fm
        (if different, can be None if we don't have a separate constraint_vector_fm value),
        any key word arguments that should be passed to the object constructor, and if
        we need to modify it the StateElementIdentifier (useful for example for a single
        StateElementIdentifier which we then don't need to pass to the create function)"

        If for some reason we can't actually construct this cls, return None
        """
        return None

    @classmethod
    def create(
        cls,
        sid: StateElementIdentifier | None = None,
        measurement_id: MeasurementId | None = None,
        retrieval_config: RetrievalConfiguration | None = None,
        strategy: MusesStrategy | None = None,
        observation_handle_set: ObservationHandleSet | None = None,
        sounding_metadata: SoundingMetadata | None = None,
        state_info: StateInfo | None = None,
        selem_wrapper: Any | None = None,
        **extra_kwargs: Any,
    ) -> Self | None:
        if retrieval_config is None or sounding_metadata is None:
            raise RuntimeError("Need retrieval_config and sounding_metadata")
        # Get pressure from StateInfo, creating if needed. This is
        # probably StateElementFromGmaoPressure (the current default),
        # but could conceivably be something else. Note that the
        # pressure needs to create itself without needing this create
        # function, so we don't get an infinite recursion

        if sid is not None and sid == StateElementIdentifier("pressure"):
            # Avoid infinite recursion, if we are called for "pressure" we
            # can't handle that in this creator.
            return None
        if state_info is None:
            return None
        p = state_info[StateElementIdentifier("pressure")]
        pressure_list_fm = p.value_fm.copy()
        sret = cls._setup_create(
            pressure_list_fm=pressure_list_fm,
            sid=sid,
            retrieval_config=retrieval_config,
            sounding_metadata=sounding_metadata,
            measurement_id=measurement_id,
            strategy=strategy,
            observation_handle_set=observation_handle_set,
            state_info=state_info,
            selem_wrapper=selem_wrapper,
            **extra_kwargs,
        )
        if sret is None:
            return None
        sid2 = sret.sid if sret.sid is not None else sid
        if sid2 is None:
            raise RuntimeError(
                "Either you need to pass in the StateElementIdentifier, or this class needs to fill that in. Neither of these happened"
            )
        res = cls(
            sid2,
            # I think this logic is sufficient to only have pressure for
            # state elements on pressure levels. We can rework this if needed.
            pressure_list_fm
            if sret.value_fm.shape[0] == pressure_list_fm.shape[0]
            else None,
            sret.value_fm,
            sret.constraint_vector_fm
            if sret.constraint_vector_fm is not None
            else sret.value_fm,
            sounding_metadata.latitude.value,
            sounding_metadata.surface_type,
            retrieval_config.input_file_helper,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            **sret.create_kwargs,
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
        self._pressure_species_input = np.array(
            retrieval_config["pressure_species_input"]
        ).astype(np.float64, copy=True)
        self._constraint_matrix = None
        self._map_type = None
        self._state_mapping = None
        self._state_mapping_retrieval_to_fm = None


class StateElementOspFileFixedValue(StateElementOspFile):
    """There are some StateElements that just have a fixed initial value.
    This class support that."""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        initial_value: FullGridMappedArray,
        create_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        return OspSetupReturn(
            value_fm=initial_value,
            create_kwargs=create_kwargs if create_kwargs is not None else {},
        )


__all__ = [
    "StateElementOspFile",
    "StateElementOspFileFixedValue",
    "OspSetupReturn",
]
