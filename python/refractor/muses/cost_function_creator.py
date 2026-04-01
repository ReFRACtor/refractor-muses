from __future__ import annotations
from .cost_function import CostFunction
from .creator_dict import CreatorDict
from .creator_handle import CreatorHandleWithContext, CreatorHandleWithContextSet
from .current_state import CurrentState
from .retrieval_array import RetrievalGridArray
from .forward_model_combine import ForwardModelCombine
import refractor.framework as rf  # type: ignore
from loguru import logger
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_spectral_window import MusesSpectralWindow
    from .muses_observation import MusesObservation
    from refractor.muses_py_fm import RefractorUip
    from .identifier import InstrumentIdentifier
    from .muses_strategy_context import MusesStrategyContext


class CostFunctionHandle(CreatorHandleWithContext):
    """This creates the set of ForwardModel and Observation and then
    uses those to create the CostFunction.

    This uses CreatorHandleSet for creating the ForwardModel and
    Observation, see that class for a discussion on using this.
    """

    def __init__(self) -> None:
        super().__init__(add_as_context_observer=True)

    def notify_update_strategy_context(
        self, strategy_context: MusesStrategyContext
    ) -> None:
        # This doesn't normally happen, but we have some unit
        # tests in old_py_retrieve that don't initialize the
        # strategy_context. Put handling in here, just to support those
        # old unit tests. Otherwise, we don't need to do anything special
        # here
        if self._strategy_context is None:
            logger.debug(
                f"Call to {self.__class__.__name__}::notify_update_strategy_context"
            )
            self._strategy_context = strategy_context

    def forward_model(
        self,
        creator_dict: CreatorDict,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
        obs_list: list[MusesObservation] | None = None,
        **kwargs: Any,
    ) -> ForwardModelCombine:
        """This is like the cost_function call, but instead of a full CostFunction we
        just return a ForwardModelCombine to get the forward model and observation data."""
        original_use_systematic = current_state.use_systematic
        try:
            current_state.use_systematic = use_systematic
            _, fmlist, obslist, fm_sv, _, _, _ = self._forward_model(
                creator_dict,
                instrument_name_list,
                current_state,
                spec_win_dict,
                use_systematic=use_systematic,
                include_bad_sample=include_bad_sample,
                obs_list=obs_list,
                **kwargs,
            )
            return ForwardModelCombine(fmlist, obslist, fm_sv)
        finally:
            current_state.use_systematic = original_use_systematic

    def cost_function(
        self,
        creator_dict: CreatorDict,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        obs_list: list[MusesObservation] | None = None,
        **kwargs: Any,
    ) -> CostFunction:
        """Return cost function for the RetrievalStrategy. This also
        attaches all the StateElement in CurrentState to be notified
        when the cost function parameters are updated.

        This takes the list of instrument names that make up this
        particular retrieval step, the current state information (see
        CurrentState class for description), and a dict going from the
        instrument name to the MusesSpectralWindow. Note that the
        MusesSpectralWindow should *not* exclude bad pixels at this
        point. We generally need the Observation that we create as
        part of this function to determine bad pixels, so we add that
        to the passed in MusesSpectralWindow. As a convenience, None
        can be passed in which uses the full band. Normally you don't
        want this, but it can be convenient for testing.

        It can also be useful in some testing scenarios to have the
        Observation created by some other method, so you can
        optionally pass in the obs_list to use in place of what the
        class would normally create. This isn't something you would
        normally use for "real", this is just to support testing.
        """

        args = self._forward_model(
            creator_dict,
            instrument_name_list,
            current_state,
            spec_win_dict,
            obs_list=obs_list,
            **kwargs,
        )
        cfunc = CostFunction(*args)
        current_state.notify_cost_function(cfunc)

        cfunc.parameters = current_state.initial_guess
        return cfunc

    def _forward_model(
        self,
        creator_dict: CreatorDict,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
        obs_list: list[MusesObservation] | None = None,
        **kwargs: Any,
    ) -> tuple[
        list[InstrumentIdentifier],
        list[rf.ForwardModel],
        list[rf.Observation],
        rf.StateVector,
        RetrievalGridArray,
        RetrievalGridArray,
        rf.StateMapping,
    ]:
        self.obs_list = []
        fm_sv = current_state.setup_fm_state_vector()
        if obs_list is not None:
            self.obs_list = obs_list
        else:
            for instrument_name in instrument_name_list:
                obs = creator_dict[rf.Observation].observation(
                    instrument_name,
                    current_state,
                    (
                        spec_win_dict[instrument_name]
                        if spec_win_dict is not None
                        else None
                    ),
                    fm_sv,
                    use_systematic=use_systematic,
                    **kwargs,
                )
                # TODO Would probably be good to remove
                # include_bad_sample, it isn't clear that we ever want
                # to run the forward model for bad samples. But right
                # now the existing py-retrieve code requires this is a
                # few places.
                if include_bad_sample:
                    obs.spectral_window.include_bad_sample = include_bad_sample
                self.obs_list.append(obs)

        self.fm_list = []
        for i, instrument_name in enumerate(instrument_name_list):
            fm = creator_dict[rf.ForwardModel].forward_model(
                instrument_name,
                current_state,
                self.obs_list[i],
                fm_sv,
                use_systematic=use_systematic,
                **kwargs,
            )
            self.fm_list.append(fm)
        smap = current_state.state_mapping_retrieval_to_fm
        retrieval_sv_apriori = current_state.constraint_vector(fix_negative=True)
        retrieval_sv_sqrt_constraint = current_state.sqrt_constraint.transpose()
        # Initialize jacobians
        fm_sv.update_state(fm_sv.state)
        return (
            instrument_name_list,
            self.fm_list,
            self.obs_list,
            fm_sv,
            retrieval_sv_apriori,
            retrieval_sv_sqrt_constraint,
            smap,
        )

    def cost_function_from_uip(
        self,
        creator_dict: CreatorDict,
        rf_uip: RefractorUip,
        obs_list: list[MusesObservation] | None,
        ret_info: dict | None,
        **kwargs: Any,
    ) -> CostFunction:
        """Create a cost function from a RefractorUip and a
        ret_info. Note that this is really just for backwards testing,
        we are trying to get away from using the RefractorUip because
        it ties stuff too tightly together.

        As a convenience, ret_info can be passed as None. It is useful
        to use our CostFunction to calculate the
        fm_wrapper/run_forward_model function because it has all the
        logic in place for stitching the different ForwardModel
        together. However we don't actually have all the data we need
        to calculate the Observation, nor do we have access to the
        apriori and sqrt_constraint.  However we don't actually need
        that to just calculate the ForwardModel data. So it can be
        useful to create all the pieces and just have dummy data for
        the missing parts.

        This is entirely a matter of convenience, we could instead
        just duplicate the stitching together part of our CostFunction
        and skip this. But for now this seems like the easiest thing
        thing to do. We can revisit this decision in the future if
        needed - it is never great to have fake data but in this case
        seemed the easiest path forward. Since this function is only
        used for backwards testing, the slightly klunky design doesn't
        seem like much of a problem.
        """
        from refractor.muses_py_fm import CurrentStateUip

        cstate = CurrentStateUip(rf_uip, ret_info)
        return self.cost_function(
            creator_dict,
            rf_uip.instrument,  # type:ignore[arg-type]
            cstate,  # type:ignore[arg-type]
            None,
            obs_list=obs_list,
            **kwargs,
        )


class CostFunctionHandleSet(CreatorHandleWithContextSet):
    """This takes a CurrentStrategyStep and maps that to a dict. The
    dict in turn maps a instrument name to the MusesSpectralWindow to
    use for that instrument.

    """

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("_dispatch", strategy_context)

    def forward_model(
        self,
        creator_dict: CreatorDict,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
        obs_list: list[MusesObservation] | None = None,
        **kwargs: Any,
    ) -> ForwardModelCombine:
        return self.handle(
            "forward_model",
            creator_dict,
            instrument_name_list,
            current_state,
            spec_win_dict,
            use_systematic,
            include_bad_sample,
            obs_list,
            **kwargs,
        )

    def cost_function(
        self,
        creator_dict: CreatorDict,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        obs_list: list[MusesObservation] | None = None,
        **kwargs: Any,
    ) -> CostFunction:
        return self.handle(
            "cost_function",
            creator_dict,
            instrument_name_list,
            current_state,
            spec_win_dict,
            obs_list,
            **kwargs,
        )

    def cost_function_from_uip(
        self,
        creator_dict: CreatorDict,
        rf_uip: RefractorUip,
        obs_list: list[MusesObservation] | None,
        ret_info: dict | None,
        **kwargs: Any,
    ) -> CostFunction:
        return self.handle(
            "cost_function_from_uip",
            creator_dict,
            rf_uip,
            obs_list,
            ret_info,
            **kwargs,
        )


CostFunctionHandleSet.add_default_handle(CostFunctionHandle())
# Register creator set
CreatorDict.register(CostFunction, CostFunctionHandleSet)

# Old name we used for this. We can perhaps clean this up
CostFunctionCreator = CostFunctionHandle

__all__ = ["CostFunctionCreator", "CostFunctionHandle", "CostFunctionHandleSet"]
