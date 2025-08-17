from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .retrieval_array import FullGridMappedArray
from pathlib import Path
import numpy as np
import h5py  # type: ignore
from loguru import logger
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .sounding_metadata import SoundingMetadata
    from .state_info import StateInfo


class StateElementFromClimatology(StateElementOspFile):
    """State element listed in Species_List_From_Climatology in L2_Setup_Control_Initial.asc
    control file"""

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
        if retrieval_config is None:
            raise RuntimeError("Need retrieval_config")
        # Check if the state element is in the list of Species_List_From_Single, if not
        # we don't process it
        #
        # Note this is really just a sanity check, we only create handles down below
        # for the state elements in this list. The L2_Setup_Control_Initial.asc doesn't
        # really control this, and it doesn't like it really controlled muses-py old
        # initial guess stuff. But we should at least notice if there is an inconsistency
        # here. Perhaps this can go away, it isn't really clear why we can't just do
        # this in our python configuration vs. a separate control file.
        if sid is not None and sid not in [
            StateElementIdentifier(i)
            for i in retrieval_config["Species_List_From_Climatology"].split(",")
        ]:
            return None
        return super(StateElementFromClimatology, cls).create(
            sid,
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            sounding_metadata,
            state_info,
            selem_wrapper,
            **extra_kwargs,
        )

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        clim_dir = Path(
            retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        )
        value_fm, _ = cls.read_climatology_2022(
            sid, pressure_list_fm, False, clim_dir, sounding_metadata
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid, pressure_list_fm, True, clim_dir, sounding_metadata
        )
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
        )

    @classmethod
    def read_climatology_2022(
        cls,
        sid: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray,
        is_constraint: bool,
        climate_dir: Path,
        sounding_metadata: SoundingMetadata,
        linear_interp: bool = True,
    ) -> tuple[FullGridMappedArray, str | None]:
        """This is a copy of mpy.read_climatology_2022, modified a bit"""
        logger.debug(f"Reading climatology for: {sid}")
        if is_constraint:
            filename = climate_dir / f"climatology_{sid}_prior.nc"
            if not filename.exists():
                filename = climate_dir / f"climatology_{sid}.nc"
        else:
            filename = climate_dir / f"climatology_{sid}.nc"
        f = h5py.File(filename, "r")
        pressure = f["pressure"][:]
        if "vmr" in f or "type_index" in f:
            # 1 = January
            month_index = sounding_metadata.month - 1
            idx = []
            longitude = sounding_metadata.longitude.value
            for v, vname in (
                (sounding_metadata.hour, "hour"),
                (sounding_metadata.latitude.value, "latitude"),
                (longitude + 360 if longitude < 0 else longitude, "longitude"),
            ):
                ind = np.where(
                    (f[f"{vname}_min"][:] <= v) & (v < f[f"{vname}_max"][:])
                )[0]
                if ind.shape[0] != 1:
                    raise RuntimeError(
                        f"Did not find 1 match for {vname} in {filename}"
                    )
                idx.append(ind[0])
            hour_index, latitude_index, longitude_index = idx

        # Two forms of the file, depending on the species type
        type_name = None
        if "vmr" in f:
            vmr = f["vmr"][month_index, latitude_index, longitude_index, hour_index, :]
        elif "type_index" in f:
            tindex = f["type_index"][
                month_index, latitude_index, longitude_index, hour_index
            ]
            if len(f["type_vmr"].shape) == 2:
                vmr = f["type_vmr"][tindex, :]
            else:
                vmr = f["type_vmr"][month_index, tindex, :]
            # Convert array of int8 type to a string, stripping off trailing '\0'
            type_name = "".join(chr(i) for i in f["type_name"][tindex]).rstrip("\0")
        else:
            raise RuntimeError(f"Didn't find vmr or type_name in file {filename}")

        if "yearly_multiplier" in f:
            ind = (np.where(f["year"][:] == sounding_metadata.year))[0]
            if len(ind) < 1:
                raise RuntimeError(
                    f"Didn't find year {sounding_metadata.year} in file {filename}"
                )
            if ind[0] >= f["yearly_multiplier"].shape[0]:
                logger.warning(
                    f"{sid}: yearly_multiplier not available for {sounding_metadata.year}."
                )
                ind[0] = f["yearly_multiplier"].shape[0] - 1
                logger.warning(
                    f"{sid}: using yearly_multiplier for {f['year'][ind[0]]} instead"
                )
            yearly_multiplier = f["yearly_multiplier"][ind[0]]
            if yearly_multiplier < 0:
                raise RuntimeError(
                    f"yearly_multiplier value is fill for {sounding_metadata.year} in file {filename}. Need to update file"
                )
            vmr = vmr * yearly_multiplier

        if linear_interp:
            my_map = mpy.make_interpolation_matrix_susan(pressure, pressure_list_fm)
            vmr = np.exp(np.matmul(my_map, np.log(vmr)))
        else:
            vmr = mpy.supplier_shift_profile(vmr, pressure, pressure_list_fm)
        return vmr.view(FullGridMappedArray), type_name


class StateElementFromClimatologyCh3oh(StateElementFromClimatology):
    """Specialization for CH3OH"""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        clim_dir = Path(
            retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        )
        value_fm, poltype = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            False,
            clim_dir,
            sounding_metadata,
            linear_interp=False,
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            linear_interp=False,
        )
        create_kwargs = {}
        if poltype is not None:
            create_kwargs["poltype"] = poltype
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            create_kwargs=create_kwargs,
        )


class StateElementFromClimatologyCo(StateElementFromClimatology):
    """Specialization for CO"""

    def retrieval_initial_fm_from_cycle_update_constraint(self) -> bool:
        # For some state elements, the constraint vector is also
        # updated
        return True


class StateElementFromClimatologyHdo(StateElementFromClimatology):
    """Specialization for HDO. It is fraction of H2O rather than independent value."""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        state_info: StateInfo,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        clim_dir = Path(
            retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        )
        value_fm, _ = cls.read_climatology_2022(
            sid, pressure_list_fm, False, clim_dir, sounding_metadata
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid, pressure_list_fm, True, clim_dir, sounding_metadata
        )
        value_fm = (
            value_fm.view(np.ndarray)
            * state_info[StateElementIdentifier("H2O")].value_fm.view(np.ndarray)
        ).view(FullGridMappedArray)
        constraint_vector_fm = (
            constraint_vector_fm.view(np.ndarray)
            * state_info[StateElementIdentifier("H2O")].constraint_vector_fm.view(
                np.ndarray
            )
        ).view(FullGridMappedArray)
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
        )


for sid in [
    "CO2",
    "HNO3",
    "CFC12",
    "CCL4",
    "CFC22",
    "N2O",
    "O3",
    "CH4",
    "SF6",
    "C2H4",
    # "PAN",
    "HCN",
    "CFC11",
]:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(
            StateElementIdentifier(sid),
            StateElementFromClimatology,
            include_old_state_info=True,
        ),
        priority_order=0,
    )

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("HDO"),
        StateElementFromClimatologyHdo,
        include_old_state_info=True,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("CO"),
        StateElementFromClimatologyCo,
        include_old_state_info=True,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("CH3OH"),
        StateElementFromClimatologyCh3oh,
        include_old_state_info=True,
    ),
    priority_order=0,
)

__all__ = [
    "StateElementFromClimatology",
    "StateElementFromClimatologyHdo",
    "StateElementFromClimatologyCo",
    "StateElementFromClimatologyCh3oh",
]
