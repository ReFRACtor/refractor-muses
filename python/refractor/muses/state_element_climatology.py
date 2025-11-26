from __future__ import annotations
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
import scipy
import math
import h5py  # type: ignore
from loguru import logger
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
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
    def _setup_create(  # type: ignore[override]
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
            my_map = cls.make_interpolation_matrix_susan(pressure, pressure_list_fm)
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
            vmr = cls.supplier_shift_profile(vmr, pressure, pressure_list_fm)
        return vmr.view(FullGridMappedArray), type_name

    @classmethod
    def make_interpolation_matrix_susan(
        cls, xFrom: np.ndarray, xTo: np.ndarray
    ) -> np.ndarray:
        num_xFrom = xFrom.size
        num_xTo = xTo.size

        my_matrix = np.zeros(
            shape=(num_xTo, num_xFrom), dtype=np.float64
        )  # shape=(63,87)

        small = np.max(xFrom) / 10000

        for ii in range(0, num_xTo):
            # Find bracketing locations.
            distance = xTo[ii] - xFrom
            absDistance = np.absolute(xTo[ii] - xFrom)

            sub1 = np.where((distance + small) >= 0)[0]
            index1 = sub1[np.argmin(absDistance[sub1])]
            sub2 = np.where((small - distance) >= 0)[0]
            index2 = sub2[np.argmin(absDistance[sub2])]

            a = xFrom[index1]
            b = xFrom[index2]

            if math.isclose(b - a, 0.0, rel_tol=1e-6):
                my_matrix[ii, index1] = 1
            else:
                my_matrix[ii, index1] = (b - xTo[ii]) / (b - a)
                my_matrix[ii, index2] = (xTo[ii] - a) / (b - a)

        return my_matrix

    @classmethod
    def bt(cls, frequency: float, rad: float) -> float:
        """converts from radiance (W/cm2/cm-1/sr) to BT (erg/sec/cm2/cm-1/sr)"""
        planck = 6.626176e-27
        clight = 2.99792458e10
        boltz = 1.380662e-16
        radcn1 = 2.0 * planck * clight * clight * 1.0e-07
        radcn2 = planck * clight / boltz
        return radcn2 * frequency / math.log(1 + (radcn1 * frequency**3 / rad))

    @classmethod
    def bt_vec(cls, frequency: np.ndarray, rad: np.ndarray) -> np.ndarray:
        """converts from radiance (W/cm2/cm-1/sr) to BT (erg/sec/cm2/cm-1/sr)"""
        planck = 6.626176e-27
        clight = 2.99792458e10
        boltz = 1.380662e-16
        radcn1 = 2.0 * planck * clight * clight * 1.0e-07
        radcn2 = planck * clight / boltz
        return np.array(
            [
                radcn2
                * frequency[i]
                / math.log(1 + (radcn1 * frequency[i] ** 3 / rad[i]))
                for i in range(frequency.shape[0])
            ]
        )

    @classmethod
    def supplier_shift_profile(
        cls, profileIn: np.ndarray, pressureIn: np.ndarray, pressureOut: np.ndarray
    ) -> np.ndarray:
        # NH3, CH3OH shift profile rather than cutting it off at the
        # surface.  if pressure > 1000 mb, then interpolate the profile.
        # if pressure < 1000 mb, then add p to all levels except TOA,
        # where p + surface pressure = 1000

        if np.max(pressureOut) >= 990.0:
            log_profileIn = np.log(profileIn)
            log_pressureIn = np.log(pressureIn)
            log_pressureOut = np.log(pressureOut)
        else:
            # pretend surface pressure is at 1000 mb, and interpolate profile to pressures
            pp = 1000 - np.max(pressureOut)
            pressureTemp = pressureOut + pp

            # keep TOA (top of atmosphere)
            pressureTemp[len(pressureTemp) - 1] = pressureOut[len(pressureOut) - 1]

            log_profileIn = np.log(profileIn)
            log_pressureIn = np.log(pressureIn)
            log_pressureOut = np.log(pressureTemp)
        x = scipy.interpolate.interp1d(
            log_pressureIn, log_profileIn, fill_value="extrapolate"
        )(log_pressureOut)
        return np.exp(x)


class StateElementFromClimatologyCh3oh(StateElementFromClimatology):
    """Specialization for CH3OH"""

    @classmethod
    def _setup_create(  # type: ignore[override]
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
                cls.my_interpolate(
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

    @classmethod
    def my_interpolate(
        cls, initialY: np.ndarray, initialX: np.ndarray, finalX: np.ndarray
    ) -> np.ndarray:
        # TODO This seems overly complicated. Worth going through this at some point and
        # possibly cleaning this up, I'm not sure this is much more than just a
        # scipy.interpolate.
        if len(initialY) == 1:
            o_finalY = finalX
            o_finalY[:] = initialY[0]
            return o_finalY
        N0 = len(initialX)
        N = len(finalX)

        o_finalY = np.zeros(finalX.shape)

        # If the initialX and finalX exactly agree, then transfer
        # values from initialY to finalY.
        ii = 0
        while ii <= len(initialY) - 1:
            nn = np.where(initialX[ii] == finalX)[0]
            if len(nn) >= 0:
                o_finalY[nn] = initialY[ii]
            ii = ii + 1

        # note:  start, stop are range of original data
        # startf, stopf are range of final data
        start_range = -1
        stop_range = -1

        for ii in range(0, N0):
            # get first index of a nonzero value in this segment
            if (initialY[ii] != 0 and np.abs(initialY[ii] + 999) > 0.1) and (
                start_range == -1
            ):
                start_range = ii

            # We see if this is the last nonzero value in this segment
            # (or the last data value and non-zero)
            # if so, we calculate range to iterpolate to, then interpolate.
            if (start_range != -1) and (
                (ii == N0 - 1) or np.abs(initialY[ii] + 999) < 0.1
            ):
                stop_range = ii - 1
                if initialY[ii] != 0 or np.abs(initialY[ii] + 999) < 0.1:
                    stop_range = ii

                # calculate the indices of the final values to interpolate to,
                # (all final x values inside the initial x range)
                # (the initial x range is increased by 1%)
                stopf = -1
                startf = -1
                for jj in range(0, N):
                    if (finalX[jj] - initialX[start_range] * 0.99) * (
                        finalX[jj] - initialX[stop_range] * 1.01
                    ) <= 0:
                        if startf == -1:
                            startf = jj
                        stopf = jj

                    if (finalX[jj] - initialX[start_range] * 1.01) * (
                        finalX[jj] - initialX[stop_range] * 0.99
                    ) <= 0:
                        if startf == -1:
                            startf = jj
                        stopf = jj
                # interpolate using IDL INTERPOL function and 'good' ranges.
                if (stopf >= startf) and (stopf != -1) and (start_range != stop_range):
                    o_finalY = scipy.interpolate.interp1d(
                        initialX[start_range : stop_range + 1],
                        initialY[start_range : stop_range + 1],
                        fill_value="extrapolate",
                    )(finalX[startf : stopf + 1])
                    # PYTHON_NOTE: We have to reduce the size of o_finalY here.
                    o_finalY = o_finalY[startf : stopf + 1]
                    start_range = -1
                    stop_range = -1

        return o_finalY


class StateElementFromClimatologyHdo(StateElementFromClimatology):
    """Specialization for HDO. It is fraction of H2O rather than independent value."""

    @classmethod
    def _setup_create(  # type: ignore[override]
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
    def _setup_create(  # type: ignore[override]
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
            (InstrumentIdentifier("CRIS"), cls.supplier_nh3_type_cris),
            (InstrumentIdentifier("TES"), cls.supplier_nh3_type_tes),
            (InstrumentIdentifier("AIRS"), cls.supplier_nh3_type_airs),
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

    @classmethod
    def supplier_nh3_type_airs(
        cls, radiance: dict[str, Any], TSUR: float, TATM0: float, surfaceType: str
    ) -> str:
        # This implements the logic described in the DFM by Mark Shephard et
        # al., to select "unpolluted", "moderate", or "polluted" NH3 case, and
        # choose the corresponding profile and constraint matrix.  Once these
        # are selected, they are shifted so the values are the same near the
        # surface no matter what pressure the surface is at.
        # update to version 9 of DFM 3/17/2010
        #
        # THis version works for CrIS data: JUne 2018
        # Karen Cady-Pereira

        freq = np.asarray(radiance["frequency"])
        rad = np.asarray(radiance["radiance"])

        indb = np.argmin(np.abs(freq - 981.186))  # used for v4, v5 and v6
        indn = np.argmin(np.abs(freq - 967.5))

        bt_bkgd = float(np.mean(cls.bt(freq[indb], rad[indb])))
        bt_nh3 = float(np.mean(cls.bt(freq[indn], rad[indn])))

        snr = -0.991 + 12.454 * (bt_bkgd - bt_nh3)
        if bt_bkgd < 278.0:
            snr = 0.0

        if snr <= -6.0:
            ind = 2
        if snr > -6.0 and snr <= -2.0:
            ind = 1
        if snr > -2.0 and snr <= 2.0:
            ind = 0
        if snr > 2.0 and snr <= 3.0:
            ind = 1
        if snr > 3.0:
            ind = 2
        if surfaceType.upper() == "OCEAN" and snr < -6.0:
            ind = 1  # v5

        if TSUR < 278:
            ind = 0

        # ind = 0: unpolluted
        # ind = 1: moderate
        # ind = 2: polluted

        nh3types = ["CLN", "MOD", "ENH"]
        o_nh3type = nh3types[ind]
        return o_nh3type

    @classmethod
    def supplier_nh3_type_cris(
        cls, radiance: dict[str, Any], TSUR: float, TATM0: float, surfaceType: str
    ) -> str:
        # This implements the logic described in the DFM by Mark Shephard et
        # al., to select "unpolluted", "moderate", or "polluted" NH3 case, and
        # choose the corresponding profile and constraint matrix.  Once these
        # are selected, they are shifted so the values are the same near the
        # surface no matter what pressure the surface is at.
        # update to version 9 of DFM 3/17/2010

        # This version works for CrIS data: June 2018
        # Karen Cady-Pereira

        freq = radiance["frequency"]
        rad = radiance["radiance"]

        indb = np.where(freq == 981.25)[0]
        indn = np.where(freq == 967.5)[0]

        bt_bkgd = float(np.mean(cls.bt_vec(freq[indb], rad[indb])))
        bt_nh3 = float(np.mean(cls.bt_vec(freq[indn], rad[indn])))

        snr = -0.991 + 12.454 * (bt_bkgd - bt_nh3)

        if bt_bkgd < 278.0:
            snr = 0.0

        # update... I had snr = 2.9 and it was unassigned
        if snr <= -6.0:
            ind = 2

        if snr > -6.0 and snr <= -2.0:
            ind = 1

        if snr > -2.0 and snr <= 2.0:
            ind = 0

        if snr > 2.0 and snr <= 3.0:
            ind = 1

        if snr > 3.0:
            ind = 2

        if TSUR < 278:
            ind = 0

        # ind = 0: unpolluted
        # ind = 1: moderate
        # ind = 2: polluted

        nh3types = ["CLN", "MOD", "ENH"]
        o_nh3type = nh3types[ind]
        return o_nh3type

    @classmethod
    def supplier_nh3_type_tes(
        cls, radiance: dict[str, Any], TSUR: float, TATM0: float, surfaceType: str
    ) -> str:
        # IDL_LEGACY_NOTE: This function supplier_nh3_type_cris is the same as Supplier_NH3_type_tes in Supplier_NH3_type_tes.pro file.

        # ; This implements the logic described in the DFM by Mark Shephard et
        # ; al., to select "unpolluted","moderate", or "polluted" NH3 case, and
        # ; choose the corresponding profile and constraint matrix.  Once these
        # ; are selected, they are shifted so the values are the same near the
        # ; surface no matter what pressure the surface is at.
        # ; update to version 9 of DFM 3/17/2010

        # ; This modifies the nominal values already put in for NH3 in the
        # ; "retrieval" structure

        TC = TSUR - TATM0

        freq = radiance["frequency"]
        rad = radiance["radiance"]
        nesr = radiance["NESR"]

        indb = np.where((freq > 968.35) * (freq < 968.49))[0]
        indn = np.where((freq > 967.27) * (freq < 967.41))[0]

        bt_bkgd = np.mean(cls.bt_vec(freq[indb], rad[indb]))
        bt_nh3 = np.mean(cls.bt_vec(freq[indn], rad[indn]))

        # calculate dBT/dR analytically using planck function (from Karen)
        x = (
            1.1911e-12
            * np.mean(freq[indn])
            * np.mean(freq[indn])
            * np.mean(freq[indn])
            / np.mean(rad[indn])
        )
        dBTdR = (
            1.4388
            * np.mean(freq[indn])
            * x
            / (np.log(x + 1.0) * np.log(x + 1.0) * (x + 1.0) * np.mean(rad[indn]))
        )
        # NEdT = NESR * dBT / dR
        NEdT = np.mean(nesr[indn]) / np.sqrt(3) * dBTdR
        snr = (bt_bkgd - bt_nh3) / NEdT

        # [unpolluted, moderate, polluted]
        a = [0.001, 0.225, 0.762]
        b = [0.116, -0.126, 0.270]

        nn = len(a)
        d = np.zeros(nn, dtype=np.float64)
        for ii in range(0, nn):
            x = (snr + TC / a[ii] - b[ii]) / (a[ii] + 1.0 / a[ii])
            y = a[ii] * x + b[ii]
            d[ii] = np.sqrt((x - TC) * (x - TC) + (y - snr) * (y - snr))

        ind = np.where(d == np.min(d))[0][0]

        # special cases in table 2
        # Condition	Type
        # SNR < 1.0	Unpolluted
        # -3.<SNR<3. AND -2<TC<2	Unpolluted << update 1/2012
        # 3.<SNR<5. AND 0.<TC<4.	Moderate
        # 0.<SNR<3. AND 2.<TC<4.	Moderate

        if np.abs(snr) < 1.0:
            ind = 0
        if (np.abs(snr) < 3.0) and (np.abs(TC) < 2):  # R12.1
            ind = 0
        if (np.abs(snr) > 3.0) and (np.abs(snr) < 5) and (TC > 0) and (TC < 4):
            ind = 1
        if (np.abs(snr) > 0.0) and (np.abs(snr) < 3.0) and (TC > 2) and (TC < 4):
            ind = 1
        if surfaceType.upper() == "OCEAN":  # R12.2
            ind = 0
        if TSUR < 278:
            ind = 0

        # ind = 0: unpolluted
        # ind = 1: moderate
        # ind = 2: polluted

        nh3types = ["CLN", "MOD", "ENH"]
        o_nh3type = nh3types[ind]
        return o_nh3type


class StateElementFromClimatologyHcooh(StateElementFromClimatology):
    """HCOOH initial guess."""

    @classmethod
    def _setup_create(  # type: ignore[override]
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
                hcoohtype = cls.supplier_hcooh_type(rad)
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

    @classmethod
    def supplier_hcooh_type(cls, radiance: dict[str, Any]) -> str:
        # Here is a brief outline of the steps required to select the
        # a priori profile:
        # 1. Calculate the brightness temperature mean over the HCOOH
        #    window (1108.88,1109.06): BT_LINE
        # 2. Repeat for the clear window (1105.04,1105.16): BT_CLEAR.
        # 3. Calculate the difference: BT_DIFF=BT_CLEAR-BT_LINE, where
        #    BT_DIFF is the estimated HCOOH signal.
        # 4. Estimate the SNR: SNR = -0.1398+2.6301*BT_DIFF.
        # 5. If -0.6<SNR<0.8, a priori is clean, otherwise it is
        #    enhanced.

        freq = np.asarray(radiance["frequency"])
        rad = np.asarray(radiance["radiance"])

        indb = []
        for i in range(0, len(freq)):
            if freq[i] >= 1108.87 and freq[i] <= 1109.07:
                indb.append(i)

        indc = []
        for i in range(0, len(freq)):
            if freq[i] >= 1105.03 and freq[i] <= 1105.17:
                indc.append(i)

        # To follow the convention of IDL, if the where clauses
        # returns no elements, we set variables indb and indc to the
        # last index.  Because of the next 4 lines, the length of both
        # indb and indc will never be zero size.
        if len(indb) == 0:
            indb = [-1]

        if len(indc) == 0:
            indc = [-1]

        # Because we don't want the function bt() to divide by zero,
        # we need to check to see if any values in rad array
        # containing zero.
        num_zeros_in_rad = 0
        for ii, b_index in enumerate(indb):
            if math.isclose(rad[b_index], 0.0, rel_tol=1e-6):
                num_zeros_in_rad = num_zeros_in_rad + 1

        if num_zeros_in_rad > 0:
            bt_line = 0
        else:
            bt_line = np.mean(cls.bt_vec(freq[indb], rad[indb]))

        if len(indc) == 0:
            bt_clear = 0
        else:
            # Because we don't want the function bt() to divide by
            # zero, we need to check to see if any values in rad array
            # containing zero.
            num_zeros_in_rad = 0
            for ii, c_index in enumerate(indc):
                if math.isclose(rad[c_index], 0.0, rel_tol=1e-6):
                    num_zeros_in_rad = num_zeros_in_rad + 1

            if num_zeros_in_rad > 0:
                bt_clear = 0
            else:
                bt_clear = np.mean(cls.bt_vec(freq[indc], rad[indc]))

        SNR = -0.1398 + 2.6301 * (bt_line - bt_clear)

        o_type = "ENH"
        if SNR > -0.6 and SNR < 0.8:
            o_type = "CLN"

        return o_type


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
