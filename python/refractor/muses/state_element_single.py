from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier, InstrumentIdentifier
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .priority_handle_set import NoHandleFound
from .state_element_osp import StateElementOspFileFixedValue
from .retrieval_array import FullGridMappedArray
import numpy as np
import scipy
from typing import Any, Self
import typing

if typing.TYPE_CHECKING:
    from .observation_handle import ObservationHandleSet
    from .muses_observation import MeasurementId
    from .muses_strategy import MusesStrategy
    from .retrieval_configuration import RetrievalConfiguration
    from .sounding_metadata import SoundingMetadata
    from .state_info import StateInfo
    from .tes_file import TesFile


class StateElementFromSingle(StateElementOspFile):
    """State element listed in Species_List_From_Single in L2_Setup_Control_Initial.asc
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
            for i in retrieval_config["Species_List_From_Single"].split(",")
        ]:
            return None
        fcloud = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_Cloud_IR.asc"
        )
        fatm = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_AtmProfiles.asc"
        )
        fcal = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_CalibrationData.asc"
        )
        return super(StateElementFromSingle, cls).create(
            sid,
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            sounding_metadata,
            state_info,
            selem_wrapper,
            fcloud=fcloud,
            fatm=fatm,
            fcal=fcal,
            **extra_kwargs,
        )

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        sid: StateElementIdentifier,
        pressure_list_fm: FullGridMappedArray,
        fatm: TesFile,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        value_fm = np.array(fatm.checked_table[str(sid)]).view(FullGridMappedArray)
        pressure = fatm.checked_table["Pressure"]
        value_fm = np.exp(
            cls.my_interpolate(
                np.log(value_fm), np.log(pressure), np.log(pressure_list_fm)
            )
        )
        value_fm = value_fm[(value_fm.shape[0] - pressure_list_fm.shape[0]) :].view(
            FullGridMappedArray
        )
        return OspSetupReturn(value_fm)

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


class StateElementFromCalibration(StateElementFromSingle):
    """State element read from calibration file"""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        fcal: TesFile,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        value_fm = np.array(
            [
                float(fcal["CalibrationScale"]),
            ]
        ).view(FullGridMappedArray)
        spectral_domain = rf.SpectralDomain(
            np.array(fcal.checked_table["Frequency"]), rf.Unit("nm")
        )
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        create_kwargs = {
            "spectral_domain": spectral_domain,
        }
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("calibrationScale"),
            create_kwargs=create_kwargs,
        )


class StateElementPcloud(StateElementFromSingle):
    """State element for PCLOUD."""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        fcloud: TesFile,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        value_fm = np.array(
            [
                float(fcloud["CloudPressure"]),
            ]
        ).view(FullGridMappedArray)
        # There are a handful of state element that muses-py just "knows" get
        # the apriori covariance from a different diagonal uncertainty file
        # (see get_prior_covariance.py in muses-py, about line 100)
        kwargs = {"diag_cov": True}
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("PCLOUD"),
            create_kwargs=kwargs,
        )


class StateElementPtgAng(StateElementOspFile):
    """State element for PTGANG."""

    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        measurement_id: MeasurementId,
        observation_handle_set: ObservationHandleSet,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        # Only tes retrievals have a nonzero ptgang
        value_fm = np.array([0.0]).view(FullGridMappedArray)
        if InstrumentIdentifier("TES") in measurement_id.filter_list_dict:
            try:
                otes = observation_handle_set.observation(
                    InstrumentIdentifier("TES"), None, None, None
                )
                value_fm = np.array(
                    [
                        otes.boresight_angle.value,
                    ]
                ).view(FullGridMappedArray)
            except NoHandleFound:
                # Just use default value, even though TES was found in the filter list
                pass
        return OspSetupReturn(
            value_fm=value_fm,
        )


StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("PCLOUD"),
        StateElementPcloud,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("calibrationScale"),
        StateElementFromCalibration,
        include_old_state_info=False,
    ),
    priority_order=-10,
)

# These values are just fixed hardcoded  values.
StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("calibrationScale"),
        StateElementOspFileFixedValue,
        initial_value=np.array(
            [
                0.0,
            ]
            * 25
        ).astype(FullGridMappedArray),
        create_kwargs={
            "spectral_domain": rf.SpectralDomain(
                np.array(
                    [
                        0.0,
                    ]
                    * 25
                ),
                rf.Unit("nm"),
            )
        },
        include_old_state_info=False,
    )
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("calibrationOffset"),
        StateElementOspFileFixedValue,
        initial_value=np.array(
            [
                0.0,
            ]
            * 300
        ).astype(FullGridMappedArray),
        include_old_state_info=False,
    )
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("residualScale"),
        StateElementOspFileFixedValue,
        initial_value=np.array(
            [
                0.0,
            ]
            * 40
        ).astype(FullGridMappedArray),
        include_old_state_info=False,
    )
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("scalePressure"),
        StateElementOspFileFixedValue,
        initial_value=np.array(
            [
                0.1,
            ]
        ).astype(FullGridMappedArray),
        include_old_state_info=False,
    )
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("PTGANG"),
        StateElementPtgAng,
        include_old_state_info=False,
    )
)


# Note, although NH3 and HCOOH are listed in Species_List_From_Single, there is
# separate logic in states_initial_update.py that overrides these in some cases.
# So we have two handles for these species

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("SO2"),
        StateElementFromSingle,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("NH3"),
        StateElementFromSingle,
        include_old_state_info=False,
    ),
    priority_order=0,
)

# See state_element_climatology for NH3 that may override

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("OCS"),
        StateElementFromSingle,
        include_old_state_info=False,
    ),
    priority_order=0,
)

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("HCOOH"),
        StateElementFromSingle,
        include_old_state_info=False,
    ),
    priority_order=0,
)

# See state_element_climatology for HCOOH that may override

StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("N2"),
        StateElementFromSingle,
        include_old_state_info=False,
    ),
    priority_order=0,
)


__all__ = [
    "StateElementPcloud",
    "StateElementFromSingle",
    "StateElementFromCalibration",
]
