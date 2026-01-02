from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .get_emis_uwis import UwisCamelOptions, get_emis_dispatcher
from .state_element import (
    StateElementWithCreateHandle,
    StateElementHandleSet,
)
from .retrieval_array import (
    FullGridMappedArray,
    RetrievalGridArray,
    RetrievalGrid2dArray,
)
from .state_element_osp import StateElementOspFile, OspSetupReturn
from .identifier import StateElementIdentifier, RetrievalType
import numpy as np
from pathlib import Path
import scipy
from loguru import logger
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_strategy import CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .sounding_metadata import SoundingMetadata
    from .input_file_helper import InputFileHelper


class StateElementFreqShared(StateElementOspFile):
    """Handful of functions common to StateElementEmis and StateElementCloudExt"""

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
        super().__init__(
            state_element_id,
            pressure_list_fm,
            value_fm,
            constraint_vector_fm,
            latitude,
            surface_type,
            ifile_hlp,
            species_directory,
            covariance_directory,
            spectral_domain=spectral_domain,
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
            poltype=poltype,
            poltype_used_constraint=poltype_used_constraint,
            diag_cov=diag_cov,
            diag_directory=diag_directory,
            metadata=metadata,
        )
        # A few things get handled differently for the BT IG refine step, so track
        # that
        self.is_bt_ig_refine = False

    def _fill_in_constraint(self) -> None:
        if self._constraint_matrix is not None:
            return
        self._constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            self.state_element_id,
            self.retrieval_type,
            self.basis_matrix.shape[0],
            poltype=self.poltype if self.poltype_used_constraint else None,
            pressure_list=self.pressure_list,
            pressure_list_fm=self.pressure_list_fm,
            map_to_parameter_matrix=self.map_to_parameter_matrix,
        ).view(RetrievalGrid2dArray)

    def _fill_in_apriori(self) -> None:
        if self._apriori_cov_fm is not None:
            return
        super()._fill_in_apriori()
        # TODO - This is done to match the existing data. We'll want to remove this
        # at some point, but it will update the expected results. Go ahead and
        # hold this fixed for now, so we can make just this one change and know it
        # is why the output changes. We round this to 32 bit, and then back to 64 bit float
        assert self._apriori_cov_fm is not None
        self._apriori_cov_fm = self._apriori_cov_fm.astype(np.float32).astype(
            np.float64
        )

    def _fill_in_state_mapping(self) -> None:
        super()._fill_in_state_mapping()
        # TODO Fix this
        # Note very confusingly this is actually the spectral domain wavelength
        # instead of pressure. We should fix this naming at some point, but for
        # now match what py-retrieve expects.
        if self._retrieved_this_step:
            assert self.spectral_domain is not None
            self._pressure_list_fm = self.spectral_domain.data.view(FullGridMappedArray)
        else:
            self._pressure_list_fm = None

    # TODO This seems to be doing something pretty similar to rf.SpectralWindow.
    # we should replace this at some point, but for now leave this here since I'm
    # not sure about all the special cases in this function
    def mw_frequency_needed(
        self,
        i_mw: list[dict[str, Any]],
        frequency: np.ndarray,
        i_stepType: RetrievalType | None = None,
        i_freqMode: str | None = None,
    ) -> np.ndarray:
        # Determine which of frequency array needed to cover mw windows
        # return array with 1 if needed 0 if not.  For EMIS and cloud
        # retrieval pars.  Returns an array the same size as the input
        # frequency array with 1 if needed, 0 if not

        if np.count_nonzero(frequency == 0) > 0:
            raise RuntimeError("Frequency array contains zeroes")

        # Populate specifies whether each frequency is needed.  It is the same
        # size as frequency and 1 if needed, 0 if not.
        o_populate = np.zeros(frequency.shape)

        nn = len(i_mw)

        # Loop through each micro window.
        for jj in range(nn):
            # if monoextend not set can use 1.44 (maximum)
            # get frequencies inside windows
            indv = []
            for gg in range(frequency.shape[0]):
                if (
                    frequency[gg] > i_mw[jj]["start"]
                    and frequency[gg] < i_mw[jj]["endd"]
                ):
                    indv.append(gg)

            if len(indv) > 0:
                # window spans more than one frequency
                ind = np.asarray(indv)
                o_populate[ind] = 1
                o_populate[ind - 1] = 1
                if max(ind) < len(o_populate) - 1:
                    o_populate[ind + 1] = 1
            else:
                # whole window (start to end) between 2 EMIS frequencies
                ind = np.where(
                    (frequency[0:-1] - i_mw[jj]["start"])
                    * (frequency[1:] - i_mw[jj]["start"])
                    < 0
                )[0]
                if ind.size == 0:
                    o_populate[0] = 1
                    o_populate[-1] = 1
                else:
                    o_populate[ind] = 1
                    o_populate[ind + 1] = 1

        temp_populate = []
        for ii in range(len(o_populate)):
            temp_populate.append(o_populate[ii])

        # if special type, e.g. bracket, select appropriate frequencies
        # i_stepType = 'bt_ig_refine'; # FOR_DEVELOPER_TESTING_FLAG
        # i_freqMode = '100';          # FOR_DEVELOPER_TESTING_FLAG
        if i_stepType is not None:
            if i_stepType == RetrievalType("bt_ig_refine") and i_freqMode is not None:
                # This is to pick out min and max only.
                if i_freqMode.lower() == "bracket":
                    # It is possible that the where clause next will
                    # produce zero elements.  So we first check before
                    # applying the np.amin()
                    not_zero_indices = np.where(o_populate != 0)[0]
                    if len(not_zero_indices) == 0:
                        logger.warning(
                            "Where clause 'np.where(o_populate != 0)' returned 0 elements"
                        )
                    else:
                        # Get the smallest of the indices where the value is not 0.
                        ind1 = np.asarray(np.amin(np.where(o_populate != 0)[0]))
                        # Get the largest  of the indices where the value is not 0.
                        ind2 = np.asarray(np.amax(np.where(o_populate != 0)[0]))

                        o_populate[:] = 0  # Set all elements to 0 first.
                        o_populate[ind1] = 1
                        o_populate[ind2] = 1

                # To check if i_freqMode is a number, we go ahead and
                # perform a conversion.  If successful, it is a number.
                # this is to pick out every hundred points
                try:
                    converted_number = float(i_freqMode)
                    numberFlag = True
                except ValueError:
                    numberFlag = False
                    converted_number = 0.0
                if numberFlag:
                    ind = (np.where(o_populate != 0))[0]
                    last = frequency[
                        ind[0]
                    ]  # The first number where the o_populate value is not 0.
                    for ii in range(1, len(ind)):
                        if (frequency[ind[ii]] - last) < converted_number:
                            o_populate[ind[ii]] = 0
                        else:
                            last = frequency[ind[ii]]

                    indv = []
                    for gg in range(len(o_populate)):
                        if o_populate[gg] != 0:
                            indv.append(gg)
                    o_populate[indv] = 1

        # Value of populate can be > 1.  Set all values > 1 to 1
        ind = (np.where(o_populate > 1))[0]
        n = len(ind)
        if n > 0:
            o_populate[np.asarray(ind)] = 1

        return o_populate


class StateElementEmis(StateElementFreqShared):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        f = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_Emissivity_IR.asc",
        )
        # Despite the name frequency, this is actually wavelength. Also, we don't actually
        # read the Emissivity column. I'm guessing this was an older way to get the
        # initial guess that got replaced
        spectral_domain = rf.SpectralDomain(f.checked_table["Frequency"], rf.Unit("nm"))
        # Use get_emis_uwis to get the emissivity. This matches what
        # script_retrieval_setup_ms does.
        emis_type = retrieval_config["TIR_EMIS_Source"]
        uwis_data = get_emis_dispatcher(
            emis_type,
            sounding_metadata.latitude.value,
            sounding_metadata.longitude.value,
            sounding_metadata.surface_altitude.value,
            1 if sounding_metadata.is_ocean else 0,
            sounding_metadata.year,
            sounding_metadata.month,
            spectral_domain.data,
            retrieval_config.input_file_helper,
            retrieval_config.get("CAMEL_Coef_Directory"),
            retrieval_config.get("CAMEL_Lab_Directory"),
        )
        value_fm = uwis_data["instr_emis"].view(FullGridMappedArray)
        # Other things we may need
        # native_emis = uwis_data['native_emis']
        # native_emis_wavenumber = uwis_data['native_wavenumber']
        camel_distance = uwis_data["dist_to_tgt"]
        prior_source = UwisCamelOptions.emis_source_citation(emis_type)
        create_kwargs = {
            "spectral_domain": spectral_domain,
            "metadata": {
                "camel_distance": camel_distance,
                "prior_source": prior_source,
            },
        }
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("EMIS"),
            create_kwargs=create_kwargs,
        )

    def _fill_in_state_mapping_retrieval_to_fm(self) -> None:
        if self._state_mapping_retrieval_to_fm is not None:
            return
        self._fill_in_state_mapping()
        assert self.spectral_domain is not None
        # TODO See about doing this directly
        wflag = self.mw_frequency_needed(
            self.microwindows,
            self.spectral_domain.data,
        )
        ind = np.array([i for i in np.nonzero(wflag)[0]])
        self._state_mapping_retrieval_to_fm = (
            rf.StateMappingBasisMatrix.from_x_subset_exclude_gap(
                self.spectral_domain.data.view(FullGridMappedArray),
                ind,
                log_interp=False,
                gap_threshold=50.0,
            )
        )
        # Line 1240 state_element_old for emis
        # Line 1366 state_element_old for cloud

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        # We've already called notify_parameter_update, so no need to update
        # self._value here
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None

        if retrieval_slice is not None:
            # Note we really are replacing value_fm as a FullGridMappedArray with
            # the results from the RetrievalGridArray solution. This is what we want
            # to do after a retrieval step. We have the "fprime" here just to make
            # that explicit that this is what we are intending on doing here. So
            # to_fmprime returns a FullGridMappedArrayFromRetGrid, but we then use
            # that as a FullGridMappedArray.

            res = (
                xsol[retrieval_slice]
                .view(RetrievalGridArray)
                .to_fmprime(self.state_mapping_retrieval_to_fm, self.state_mapping)
                .view(FullGridMappedArray)
            )
            if self._value_fm is None:
                raise RuntimeError("self._value_fm can't be none")
            self._value_fm[self.fm_update_flag] = res[self.fm_update_flag]
            if not self._initial_guess_not_updated:
                self._next_step_initial_fm = self._value_fm.copy()

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
        # Grab microwindow information, needed in later calculations
        self.microwindows: list[dict] = []
        for swin in current_strategy_step.spectral_window_dict.values():
            self.microwindows.extend(swin.muses_microwindows())
        # Filter out the UV windows, this just isn't wanted in
        # mw_frequency_needed
        self.microwindows = [mw for mw in self.microwindows if "UV" not in mw["filter"]]
        if self._retrieved_this_step:
            # muses-py maps value_fm to fmprime when we are doing a retrieval step
            self._value_fm[self.fm_update_flag] = self._value_fm.to_fmprime(
                self.state_mapping_retrieval_to_fm, self.state_mapping
            )[self.fm_update_flag]

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        res = np.empty(self._value_fm.shape, dtype=bool).view(FullGridMappedArray)
        # This actually seems wrong to me, I don't think we aren't updating all the
        # elements of value_fm. However, this is what muses-py currently returns, so
        # match that.
        res[:] = True
        self._check_result(res, "updated_fm_flag")
        return res


class StateElementNativeEmis(StateElementFreqShared):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        f = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_Emissivity_IR.asc",
        )
        # Despite the name frequency, this is actually wavelength. Also, we don't actually
        # read the Emissivity column. I'm guessing this was an older way to get the
        # initial guess that got replaced
        spectral_domain_in = rf.SpectralDomain(
            f.checked_table["Frequency"], rf.Unit("nm")
        )
        # Despite the name frequency, this is actually wavelength. Also, we don't actually
        # read the Emissivity column. I'm guessing this was an older way to get the
        # initial guess that got replaced
        # Use get_emis_uwis to get the emissivity. This matches what
        # script_retrieval_setup_ms does.
        emis_type = retrieval_config["TIR_EMIS_Source"]
        uwis_data = get_emis_dispatcher(
            emis_type,
            sounding_metadata.latitude.value,
            sounding_metadata.longitude.value,
            sounding_metadata.surface_altitude.value,
            1 if sounding_metadata.is_ocean else 0,
            sounding_metadata.year,
            sounding_metadata.month,
            spectral_domain_in.data,
            retrieval_config.input_file_helper,
            retrieval_config.get("CAMEL_Coef_Directory"),
            retrieval_config.get("CAMEL_Lab_Directory"),
        )
        spectral_domain = rf.SpectralDomain(
            uwis_data["native_wavenumber"], rf.Unit("nm")
        )
        value_fm = uwis_data["native_emis"].view(FullGridMappedArray)
        create_kwargs = {
            "spectral_domain": spectral_domain,
        }
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("native_emissivity"),
            create_kwargs=create_kwargs,
        )


class StateElementCloudExt(StateElementFreqShared):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        retrieval_config: RetrievalConfiguration,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        f = retrieval_config.input_file_helper.open_tes(
            retrieval_config["Single_State_Directory"] / "State_Cloud_IR.asc",
        )
        # Despite the name frequency, this is actually wavelength.
        spectral_domain = rf.SpectralDomain(
            f.checked_table["Frequencies"], rf.Unit("nm")
        )
        value_fm = np.array(f.checked_table["verticalEXT"]).view(FullGridMappedArray)
        create_kwargs = {"spectral_domain": spectral_domain}
        return OspSetupReturn(
            value_fm=value_fm,
            sid=StateElementIdentifier("CLOUDEXT"),
            create_kwargs=create_kwargs,
        )

    def _fill_in_state_mapping_retrieval_to_fm(self) -> None:
        if self._state_mapping_retrieval_to_fm is not None:
            return
        self._fill_in_state_mapping()
        assert self.spectral_domain is not None
        wflag = self.mw_frequency_needed(
            self.microwindows,
            self.spectral_domain.data,
            self.retrieval_type,
            self.freq_mode,
        )
        ind = np.array([i for i in np.nonzero(wflag)[0]])
        self._state_mapping_retrieval_to_fm = rf.StateMappingBasisMatrix.from_x_subset(
            self.spectral_domain.data.view(FullGridMappedArray), ind, log_interp=False
        )
        # Line 1240 state_element_old for emis
        # Line 1366 state_element_old for cloud

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
        # Grab microwindow information, needed in later calculations
        self.microwindows: list[dict] = []
        for swin in current_strategy_step.spectral_window_dict.values():
            self.microwindows.extend(swin.muses_microwindows())
        # Filter out the UV windows, this just isn't wanted in
        # mw_frequency_needed
        self.microwindows = [mw for mw in self.microwindows if "UV" not in mw["filter"]]
        self.freq_mode = retrieval_config["CLOUDEXT_IGR_Min_Freq_Spacing"]
        self.update_ave = retrieval_config["CLOUDEXT_IGR_Average"].lower()
        self.max_ave = float(retrieval_config["CLOUDEXT_IGR_Max"])
        self.reset_ave = float(retrieval_config["CLOUDEXT_Reset_Value"])
        self.is_bt_ig_refine = current_strategy_step.retrieval_type == RetrievalType(
            "bt_ig_refine"
        )
        if self.is_bt_ig_refine:
            self._value_fm[self._value_fm >= self.max_ave] = self.reset_ave

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_fm = None
        if retrieval_slice is None:
            return

        # Note we really are replacing value_fm as a FullGridMappedArray with
        # the results from the RetrievalGridArray solution. This is what we want
        # to do after a retrieval step. We have the "fprime" here just to make
        # that explicit that this is what we are intending on doing here. Sp
        # to_fmprime returns a FullGridMappedArrayFromRetGrid, but we then use
        # that as a FullGridMappedArray.

        res = (
            xsol[retrieval_slice]
            .view(RetrievalGridArray)
            .to_fmprime(self.state_mapping_retrieval_to_fm, self.state_mapping)
            .view(FullGridMappedArray)
        )
        # Set of values actually updated
        ind = np.array([i for i in np.nonzero(self.fm_update_flag)[0]])
        if self._value_fm is None:
            raise RuntimeError("value_fm can't be none")
        if self.spectral_domain is None:
            raise RuntimeError("spectral_domain can't be none")
        varr = self._value_fm
        # Special handling for bt_ig_refine step, we use average value rather than
        # copying the full set over
        if self.is_bt_ig_refine:
            if ind.size < 4:
                if ind.size > 0:
                    ave = np.exp(
                        np.sum(np.log(res[self.fm_update_flag]))
                        / len(res[self.fm_update_flag])
                    )
            else:
                ind0 = ind[1 : ind.size - 1]  # Exclude end points
                ave = np.exp(np.sum(np.log(res[ind0])) / len(res[ind0]))
            varr[:] = ave
            if self.update_ave == "no":
                if ind.size > 0:
                    varr[self.fm_update_flag] = res[self.fm_update_flag]
                    varr[ind.min(), ind.max() + 1] = np.exp(
                        scipy.interpolate.interp1d(
                            self.spectral_domain.data[self.fm_update_flag],
                            np.log(res[self.fm_update_flag]),
                        )(self.spectral_domain.data[ind.min(), ind.max() + 1])
                    )
            else:
                varr[:] = ave
            varr[varr >= self.max_ave] = self.reset_ave
        else:
            # For other steps, just handle like normal. Update data not held fixed
            varr[self.fm_update_flag] = res[self.fm_update_flag]
        if not self._initial_guess_not_updated:
            self._next_step_initial_fm = self._value_fm.copy()

    @property
    def updated_fm_flag(self) -> FullGridMappedArray:
        # bt_ig_refine is handled differently
        if not self.is_bt_ig_refine:
            return super().updated_fm_flag
        assert self.spectral_domain is not None
        wflag = self.mw_frequency_needed(
            self.microwindows,
            self.spectral_domain.data,
            self.retrieval_type,
            self.freq_mode,
        )
        ind = np.array([i for i in np.nonzero(wflag)[0]])
        res = np.zeros(wflag.shape)
        res[ind[0] : ind[1] + 1] = 1
        self._check_result(res, "updated_fm_flag")
        return res.view(FullGridMappedArray)


StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(StateElementIdentifier("EMIS"), StateElementEmis),
    priority_order=0,
)
StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("native_emissivity"), StateElementNativeEmis
    ),
    priority_order=0,
)
StateElementHandleSet.add_default_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("CLOUDEXT"), StateElementCloudExt
    ),
    priority_order=0,
)

__all__ = [
    "StateElementEmis",
    "StateElementNativeEmis",
    "StateElementCloudExt",
]
