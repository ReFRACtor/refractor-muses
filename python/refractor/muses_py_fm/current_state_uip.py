from __future__ import annotations
import numpy as np
from pathlib import Path
from copy import copy
import re
from .refractor_uip import AttrDictAdapter, RefractorUip
from refractor.muses import (
    RetrievalGridArray,
    FullGridArray,
    FullGridMappedArray,
    RetrievalGrid2dArray,
    FullGrid2dArray,
    StateElementIdentifier,
    CurrentState,
    StateElement,
    SoundingMetadata,
    StrategyStepIdentifier,
    RetrievalType,
)


class CurrentStateUip(CurrentState):
    """Implementation of CurrentState that uses a RefractorUip"""

    def __init__(self, rf_uip: RefractorUip, ret_info: dict | None = None):
        """Get the CurrentState from a RefractorUip and ret_info. Note
        that this is just for backwards testing, we don't use the UIP
        in our current processing but rather something like
        CurrentStateStateInfo.

        The RefractorUip doesn't have everything we need, specifically
        we don't have the apriori and sqrt_constraint. We can get this
        from a ret_info, if available.  For testing we don't always
        have ret_info. This is fine if we don't actually need the
        apriori and sqrt_constraint. We still need a value for this,
        so if ret_info is None we return arrays of all zeros of the
        right size.

        """
        super().__init__()
        self.rf_uip = rf_uip
        self._initial_guess = rf_uip.current_state_x
        self._basis_matrix = rf_uip.basis_matrix
        self.ret_info = ret_info

    @property
    def step_directory(self) -> Path:
        """Return the step directory. This is a bit odd, but it is
        needed by MusesOpticalDepthFile. Since the current state
        depends on the step we are using, it isn't ridiculous to have
        this here. However if we find a better home for or better
        still remove the need for this that would be good.

        """
        return self.rf_uip.step_directory

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        """Similar to step_directory, step_number is used by RefractorUip. Supply that."""
        t = re.match(r"Step(\d+)_(.*)", self.step_directory.name)
        if t is None:
            raise RuntimeError(f"Don't recognize {self.step_directory.name}")
        return StrategyStepIdentifier(int(t[1]), t[2])

    @property
    def retrieval_type(self) -> RetrievalType:
        """Similar to step_directory, retrieval_type is used by RefractorUip. Supply that."""
        # I don't think this actually matters. This only seems to get checked if it
        # is BT, and it doesn't seem to do anything. So set to default, we can revisit
        # this if needed. This class is only used for unit tests, so it doesn't matter
        # if the BT case isn't handled correctly, since we don't use that in unit tests
        return RetrievalType("default")

    @property
    def initial_guess(self) -> RetrievalGridArray:
        return copy(self._initial_guess).view(RetrievalGridArray)

    @property
    def initial_guess_full(self) -> FullGridArray:
        """Return the initial guess for the forward model grid.  This
        isn't independent, it is directly calculated from the
        initial_guess and basis_matrix. But convenient to supply this
        (mostly as a help in unit testing).

        """
        raise NotImplementedError()

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        # Don't think we need this. We can calculate something frm
        # sqrt_constraint if needed, but for now just leave
        # unimplemented
        raise NotImplementedError()

    @property
    def sqrt_constraint(self) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["sqrt_constraint"].view(RetrievalGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.eye(len(self.initial_guess)).view(RetrievalGridArray)

    def constraint_vector(self, fix_negative: bool = True) -> RetrievalGridArray:
        if self.ret_info:
            return self.ret_info["const_vec"].view(RetrievalGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.zeros((len(self.initial_guess),)).view(RetrievalGridArray)

    @property
    def constraint_vector_full(self) -> FullGridArray:
        if self.ret_info:
            return self.ret_info["const_vec"].view(FullGridArray)
        else:
            # Dummy value, of the right size. Useful when we need
            # this, but don't actually care about the value (e.g., we
            # are running the forward model only in the CostFunction).
            #
            # This is entirely a matter of convenience, we could
            # instead just duplicate the stitching together part of
            # our CostFunction and skip this. But for now this seems
            # like the easiest thing thing to do. We can revisit this
            # decision in the future if needed - it is never great to
            # have fake data but in this case seemed the easiest path
            # forward. Since this function is only used for backwards
            # testing, the slightly klunky design doesn't seem like
            # much of a problem.
            return np.zeros((len(self.initial_guess),)).view(FullGridArray)

    @property
    def basis_matrix(self) -> np.ndarray | None:
        return self._basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        raise NotImplementedError()

    # We don't have the other gas species working yet. Short term,
    # just have a different implementation of fm_sv_loc. We should
    # sort this out at some point.
    @property
    def fm_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._fm_sv_loc is None:
            self._fm_sv_loc = {}
            self._fm_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(str(species_name))
                self._fm_sv_loc[species_name] = (pstart, plen)
                self._fm_state_vector_size += plen
        return self._fm_sv_loc

    @property
    def retrieval_sv_loc(self) -> dict[StateElementIdentifier, tuple[int, int]]:
        if self._retrieval_sv_loc is None:
            self._retrieval_sv_loc = {}
            self._retrieval_state_vector_size = 0
            for species_name in self.retrieval_state_element_id:
                pstart, plen = self.rf_uip.state_vector_species_index(
                    str(species_name), use_full_state_vector=False
                )
                self._retrieval_sv_loc[species_name] = (pstart, plen)
                self._retrieval_state_vector_size += plen
        return self._retrieval_sv_loc

    @property
    def retrieval_state_element_id(self) -> list[StateElementIdentifier]:
        return [StateElementIdentifier(i) for i in self.rf_uip.jacobian_all]

    @property
    def systematic_state_element_id(self) -> list[StateElementIdentifier]:
        raise NotImplementedError()

    @property
    def full_state_element_id(self) -> list[StateElementIdentifier]:
        # I think we could come up with something here if needed, but for now
        # just punt on this
        raise NotImplementedError()

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        raise NotImplementedError()

    def state_element(
        self, state_element_id: StateElementIdentifier | str
    ) -> StateElement:
        raise NotImplementedError()

    def state_spectral_domain_wavelength(
        self, state_element_id: StateElementIdentifier | str
    ) -> np.ndarray | None:
        raise NotImplementedError()

    def state_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        # We've extracted this logic out from update_uip
        o_uip = AttrDictAdapter(self.rf_uip.uip)
        res = None
        if str(state_element_id) == "TSUR":
            res = np.array(
                [
                    o_uip.surface_temperature,
                ]
            )
        elif str(state_element_id) == "PSUR":
            res = np.array([o_uip.atmosphere[0, 0]])
        elif str(state_element_id) == "EMIS":
            res = np.array(o_uip.emissivity["value"])
        elif str(state_element_id) == "PTGANG":
            res = np.array([o_uip.obs_table["pointing_angle"]])
        elif str(state_element_id) == "RESSCALE":
            res = np.array([o_uip.res_scale])
        elif str(state_element_id) == "CLOUDEXT":
            res = np.array(o_uip.cloud["extinction"])
        elif str(state_element_id) == "PCLOUD":
            res = np.array([o_uip.cloud["pressure"]])
        elif str(state_element_id) == "OMICLOUDFRACTION":
            res = np.array([o_uip.omiPars["cloud_fraction"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV1":
            res = np.array([o_uip.omiPars["surface_albedo_uv1"]])
        elif str(state_element_id) == "OMISURFACEALBEDOUV2":
            res = np.array([o_uip.omiPars["surface_albedo_uv2"]])
        elif str(state_element_id) == "OMISURFACEALBEDOSLOPEUV2":
            res = np.array([o_uip.omiPars["surface_albedo_slope_uv2"]])
        elif str(state_element_id) == "OMINRADWAVUV1":
            res = np.array([o_uip.omiPars["nradwav_uv1"]])
        elif str(state_element_id) == "OMINRADWAVUV2":
            res = np.array([o_uip.omiPars["nradwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVUV1":
            res = np.array([o_uip.omiPars["odwav_uv1"]])
        elif str(state_element_id) == "OMIODWAVUV2":
            res = np.array([o_uip.omiPars["odwav_uv2"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV1":
            res = np.array([o_uip.omiPars["odwav_slope_uv1"]])
        elif str(state_element_id) == "OMIODWAVSLOPEUV2":
            res = np.array([o_uip.omiPars["odwav_slope_uv2"]])
        elif str(state_element_id) == "OMIRINGSFUV1":
            res = np.array([o_uip.omiPars["ring_sf_uv1"]])
        elif str(state_element_id) == "OMIRINGSFUV2":
            res = np.array([o_uip.omiPars["ring_sf_uv2"]])
        elif str(state_element_id) == "TROPOMICLOUDFRACTION":
            res = np.array([o_uip.tropomiPars["cloud_fraction"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND1":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND1"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND2":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND2"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND7":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND7"]])
        elif str(state_element_id) == "TROPOMISURFACEALBEDOSLOPEORDER2BAND3TIGHT":
            res = np.array([o_uip.tropomiPars["surface_albedo_slope_order2_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND1":
            res = np.array([o_uip.tropomiPars["solarshift_BAND1"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND2":
            res = np.array([o_uip.tropomiPars["solarshift_BAND2"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND3":
            res = np.array([o_uip.tropomiPars["solarshift_BAND3"]])
        elif str(state_element_id) == "TROPOMISOLARSHIFTBAND7":
            res = np.array([o_uip.tropomiPars["solarshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND1":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND2":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND3":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADIANCESHIFTBAND7":
            res = np.array([o_uip.tropomiPars["radianceshift_BAND7"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND1":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND1"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND2":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND2"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND3":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND3"]])
        elif str(state_element_id) == "TROPOMIRADSQUEEZEBAND7":
            res = np.array([o_uip.tropomiPars["radsqueeze_BAND7"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND1":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND1"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND2":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND2"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND3":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND3"]])
        elif str(state_element_id) == "TROPOMIRINGSFBAND7":
            res = np.array([o_uip.tropomiPars["ring_sf_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND2":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND2"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND3":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND3"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO0BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O0_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO1BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O1_BAND7"]])
        elif str(state_element_id) == "TROPOMIRESSCALEO2BAND7":
            res = np.array([o_uip.tropomiPars["resscale_O2_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND7":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND7"]])
        elif str(state_element_id) == "TROPOMITEMPSHIFTBAND3TIGHT":
            res = np.array([o_uip.tropomiPars["temp_shift_BAND3"]])
        elif str(state_element_id) == "TROPOMICLOUDSURFACEALBEDO":
            res = np.array([o_uip.tropomiPars["cloud_Surface_Albedo"]])
        if res is not None:
            return res.view(FullGridMappedArray)
        # Check if it is a column
        try:
            return self.rf_uip.atmosphere_column(str(state_element_id)).view(
                FullGridMappedArray
            )
        except ValueError:
            pass
        raise RuntimeError(f"Don't recognize {state_element_id}")

    def state_step_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_true_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray | None:
        raise NotImplementedError()

    def state_retrieval_initial_value(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_constraint_vector(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGridMappedArray:
        raise NotImplementedError()

    def state_apriori_covariance(
        self, state_element_id: StateElementIdentifier | str
    ) -> FullGrid2dArray:
        raise NotImplementedError()


__all__ = [
    "CurrentStateUip",
]
