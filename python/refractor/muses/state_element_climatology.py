from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier, InstrumentIdentifier
from .observation_handle import mpy_radiance_from_observation_list
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .retrieval_array import FullGridMappedArray
from .tes_file import TesFile
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
        #
        # Note that muses-py tacks on NH3 and HCOOH, even though it is not listed
        # in Species_List_From_Climatology, so add that on
        if (
            sid is not None
            and sid
            not in [
                StateElementIdentifier(i)
                for i in retrieval_config["Species_List_From_Climatology"].split(",")
            ]
            and sid
            not in (
                StateElementIdentifier("NH3"),
                StateElementIdentifier("HCOOH"),
                StateElementIdentifier("CH3OH"),
            )
        ):
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
        ind_type: None | str = None,
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
            # convert type_name to string
            # Convert array of int8 type to a string, stripping off trailing '\0'
            type_name_list = []
            for tindex in range(f["type_name"].shape[0]):
                type_name_list.append(
                    "".join(chr(i) for i in f["type_name"][tindex]).rstrip("\0")
                )
            type_name = ind_type
            if type_name is None:
                raise RuntimeError(
                    f"Need to supply type name to index to for file {filename}"
                )
            try:
                tindex = type_name_list.index(type_name)
            except ValueError:
                raise RuntimeError(f"Didn't find type {type_name} in file {filename}")
            vmr = f["type_vmr"][tindex, :]

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
            # For types HCOOH, NH3 and CH3OH shifting the vmr rather
            # than cutting off the lower part of profile. Not sure
            # exactly how this was determined, this is just what
            # muses-py does in read_climatology_2022.py. I assume
            # somebody did an analysis to determine we want to do
            # that. We have the logic for this in the various
            # StateElements, so we just do what "linear_interp" tells
            # us to do here.
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
        value_fm, _ = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            False,
            clim_dir,
            sounding_metadata,
            linear_interp=False,
        )
        constraint_vector_fm, poltype = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            linear_interp=False,
        )
        # In some cases, the CH3OH isn't read from the climatology, although we still
        # use that for determining the polytype. We instead get CH3OH from the
        # State_AtmProfiles.asc file
        # This seems to only be used in the TES retrieval
        # TODO - Is this actually intended? Or was it just some accident? Not clear why
        # we would use the climatology for the constraint_vector, but the State_AtmProfiles.asc
        # for the initial value. But this is what py-retrieve is doing.
        if StateElementIdentifier("CH3OH") not in [
            StateElementIdentifier(i)
            for i in retrieval_config["Species_List_From_Climatology"].split(",")
        ]:
            fatm = TesFile(
                Path(retrieval_config["Single_State_Directory"])
                / "State_AtmProfiles.asc"
            )
            if fatm.table is None:
                return None
            value_fm = np.array(fatm.table["CH3OH"]).view(FullGridMappedArray)
            pressure = fatm.table["Pressure"]
            value_fm = np.exp(
                mpy.my_interpolate(
                    np.log(value_fm), np.log(pressure), np.log(pressure_list_fm)
                )
            )
            value_fm = value_fm[(value_fm.shape[0] - pressure_list_fm.shape[0]) :].view(
                FullGridMappedArray
            )
            # TODO - Is this actually intended?
            # Even if we get the value_fm from the fatm file, we still get the
            # constraint_vector_fm from the climatology. Not sure if this was actually
            # intended, but this is what happens. This seems to only apply to the TES
            # retrieval, so this might not be overly important to sort out.
            # constraint_vector_fm = value_fm
        create_kwargs = {}
        if poltype is not None:
            create_kwargs["poltype"] = poltype
        r = OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            create_kwargs=create_kwargs,
        )
        return r


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


class StateElementFromClimatologyNh3(StateElementFromClimatology):
    """NH3 initial guess."""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        state_info: StateInfo,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        if sid is not None and sid in (
            StateElementIdentifier("TSUR"),
            StateElementIdentifier("TATM"),
        ):
            # Avoid infinite recursion
            return None
        if strategy is None:
            return None
        # We only use this if NH3 is in the error analysis
        # interferents or retrieval elements. Not sure of the exact
        # reason for this, but this is the logic used in muses-py in
        # states_initial_update.py and we duplicate this here.
        #
        # TODO Is the actually the right logic? Why should the initial
        # guess depend on it being an interferent or in the retrieval?
        if (
            StateElementIdentifier("NH3") not in strategy.error_analysis_interferents
            and StateElementIdentifier("NH3") not in strategy.retrieval_elements
        ):
            return None
        # Only have handling for CRIS, TES and AIRS.
        if (
            InstrumentIdentifier("CRIS") not in strategy.instrument_name
            and InstrumentIdentifier("TES") not in strategy.instrument_name
            and InstrumentIdentifier("AIRS") not in strategy.instrument_name
        ):
            return None
        surface_type = sounding_metadata.surface_type
        tsur = state_info[StateElementIdentifier("TSUR")].constraint_vector_fm[0]
        tatm0 = state_info[StateElementIdentifier("TATM")].constraint_vector_fm[0]
        nh3type: str | None = None
        for ins, sfunc in [
            (InstrumentIdentifier("CRIS"), mpy.supplier_nh3_type_cris),
            (InstrumentIdentifier("TES"), mpy.supplier_nh3_type_tes),
            (InstrumentIdentifier("AIRS"), mpy.supplier_nh3_type_airs),
        ]:
            if ins in strategy.instrument_name:
                olist = [
                    observation_handle_set.observation(ins, None, None, None),
                ]
                rad = mpy_radiance_from_observation_list(olist, full_band=True)
                nh3type = sfunc(rad, tsur, tatm0, surface_type)
                break
        if nh3type is None:
            nh3type = "MOD"
        clim_dir = Path(
            retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        )
        # TODO Check if this is actually correct
        # Oddly the initial value comes from the prior file (so is_constraint is
        # True). Not sure if this is what was intended, but it is what muses_py
        # does in states_initial_update.py. We duplicate that here, but should
        # determine at some point if this is actually correct. Why even have the
        # non prior climatology if we aren't using it?
        value_fm, poltype = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            ind_type=nh3type,
            linear_interp=False,
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            ind_type=nh3type,
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


class StateElementFromClimatologyHcooh(StateElementFromClimatology):
    """HCOOH initial guess."""

    @classmethod
    # type: ignore[override]
    def _setup_create(
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        # We only use this if HCOOH is in the error analysis
        # interferents or retrieval elements. Not sure of the exact
        # reason for this, but this is the logic used in muses-py in
        # states_initial_update.py and we duplicate this here.
        #
        # TODO Is the actually the right logic? Why should the initial
        # guess depend on it being an interferent or in the retrieval?
        if (
            StateElementIdentifier("HCOOH") not in strategy.error_analysis_interferents
            and StateElementIdentifier("HCOOH") not in strategy.retrieval_elements
        ):
            return None
        # Only have handling for CRIS, TES and AIRS.
        if (
            InstrumentIdentifier("CRIS") not in strategy.instrument_name
            and InstrumentIdentifier("TES") not in strategy.instrument_name
            and InstrumentIdentifier("AIRS") not in strategy.instrument_name
        ):
            return None
        hcoohtype: str | None = None
        for ins in [
            InstrumentIdentifier("CRIS"),
            InstrumentIdentifier("TES"),
            InstrumentIdentifier("AIRS"),
        ]:
            if ins in strategy.instrument_name:
                olist = [
                    observation_handle_set.observation(ins, None, None, None),
                ]
                rad = mpy_radiance_from_observation_list(olist, full_band=True)
                hcoohtype = mpy.supplier_hcooh_type(rad)
                break
        if hcoohtype is None:
            hcoohtype = "MOD"
        clim_dir = Path(
            retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        )
        # TODO Check if this is actually correct
        # Oddly the initial value comes from the prior file (so is_constraint is
        # True). Not sure if this is what was intended, but it is what muses_py
        # does in states_initial_update.py. We duplicate that here, but should
        # determine at some point if this is actually correct. Why even have the
        # non prior climatology if we aren't using it?
        value_fm, poltype = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            ind_type=hcoohtype,
            linear_interp=False,
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            ind_type=hcoohtype,
            linear_interp=False,
        )
        create_kwargs: dict[str, Any] = {}
        if poltype is not None:
            create_kwargs["poltype"] = poltype
        # Muses-py just "knows" that we don't use the poltype in the constraint file name
        # for HCOOH. Probably something historic, this seems to only be used for TES,
        # and is probably the oldest code, perhaps before the poltype was used to change
        # the constraint
        create_kwargs["poltype_used_constraint"] = False
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            create_kwargs=create_kwargs,
        )


for sid in [
    "CO",
    "CO2",
    "HNO3",
    "CFC12",
    "CCL4",
    "CFC22",
    "NO2",
    "N2O",
    "O3",
    "CH4",
    "SF6",
    "C2H4",
    "PAN",
    "HCN",
    "CFC11",
]:
    StateElementHandleSet.add_default_handle(
        StateElementWithCreateHandle(
            StateElementIdentifier(sid),
            StateElementFromClimatology,
            include_old_state_info=False,
        ),
        priority_order=0,
    )

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("HDO"),
        StateElementFromClimatologyHdo,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("CH3OH"),
        StateElementFromClimatologyCh3oh,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("NH3"),
        StateElementFromClimatologyNh3,
        include_old_state_info=False,
    ),
    priority_order=1,
)

# See state_element_single for fall back for NH3

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("HCOOH"),
        StateElementFromClimatologyHcooh,
        include_old_state_info=False,
    ),
    priority_order=1,
)

# See state_element_single for fall back for HCOOH

__all__ = [
    "StateElementFromClimatology",
    "StateElementFromClimatologyHdo",
    "StateElementFromClimatologyCh3oh",
    "StateElementFromClimatologyNh3",
    "StateElementFromClimatologyHcooh",
]
