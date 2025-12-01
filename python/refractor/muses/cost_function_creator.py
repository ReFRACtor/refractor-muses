from __future__ import annotations
from .cost_function import CostFunction
from .current_state import CurrentState
from .forward_model_handle import ForwardModelHandleSet
from .observation_handle import ObservationHandleSet
from .retrieval_array import RetrievalGridArray
import refractor.framework as rf  # type: ignore
import copy
from loguru import logger
from collections.abc import Callable
import typing
from typing import Any
import numpy as np

if typing.TYPE_CHECKING:
    from .muses_spectral_window import MusesSpectralWindow
    from .muses_observation import MusesObservation, MeasurementId
    from refractor.muses_py_fm import RefractorUip
    from .retrieval_strategy import RetrievalStrategy
    from .identifier import InstrumentIdentifier


class CostFunctionCreator:
    """This creates the set of ForwardModel and Observation and then
    uses those to create the CostFunction.

    This uses CreatorHandleSet for creating the ForwardModel and
    Observation, see that class for a discussion on using this.
    """

    def __init__(self, rs: RetrievalStrategy | None = None) -> None:
        self.forward_model_handle_set = copy.deepcopy(
            ForwardModelHandleSet.default_handle_set()
        )
        self.observation_handle_set = copy.deepcopy(
            ObservationHandleSet.default_handle_set()
        )
        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Set up for processing a target.

        Note we separate this out from the cost_function creator
        because we want to allow internal caching based on the
        sounding - e.g., read the input file only once. Ignoring
        performance, functionally this is just like an extra argument
        passed to cost_function.

        We take measure_id, which is a MeasurementId.
        """
        logger.debug(f"Call to {self.__class__.__name__}::notify_update_target")
        self.measurement_id = measurement_id
        self.forward_model_handle_set.notify_update_target(self.measurement_id)
        self.observation_handle_set.notify_update_target(self.measurement_id)

    def cost_function(
        self,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        include_bad_sample: bool = False,
        obs_list: list[MusesObservation] | None = None,
        use_empty_apriori: bool = False,
        **kwargs: Any,
    ) -> CostFunction:
        """Return cost function for the RetrievalStrategy. This also
        attaches all the StateElement in CurrentState to be notified
        when the cost function parameters are updated.

        You can optional leave off the augmented/apriori piece. This is
        useful if you are just running a forward model. The muses-py
        code requires this in some cases, because it hasn't calculated
        an apriori (special handling for retrieval steps "bt" and
        "forward_model").

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

        The muses-py versions of the ForwardModel depend on a
        RefractorUip structure. To support these older ForwardModel,
        and function is passed in that can be called to return the
        RefractorUip to use. We have this as a function, so we can
        avoid creating the RefractorUip if we don't need it. The
        RefractorUip shouldn't be used otherwise, it only has
        information that was needed by the old ForwardModel, and it is
        a convoluted way to pass the information around. Basically the
        uip + muses-py code is a old structural programming way to
        have a ForwardModel object.  We also don't just generate the
        UIP internally when needed because it depends on other
        muses-py structures that we don't have access to (by design,
        to reduce coupling in our code).  If you know you aren't using
        a muses-py ForwardModel, it is fine to just pass this as
        None. It is possible that this argument will go away if we
        move far enough away from the old muses-py code - however for
        the near future we want to be able maintain the ability to run
        the old code to test against and diagnose any issues with
        ReFRACtor.

        It can also be useful in some testing scenarios to have the
        Observation created by some other method, so you can
        optionally pass in the obs_list to use in place of what the
        class would normally create. This isn't something you would
        normally use for "real", this is just to support testing.
        """

        self.forward_model_handle_set.notify_start_cost_function()

        args = self._forward_model(
            instrument_name_list,
            current_state,
            spec_win_dict,
            rf_uip_func,
            include_bad_sample=include_bad_sample,
            obs_list=obs_list,
            use_empty_apriori=use_empty_apriori,
            **kwargs,
        )
        cfunc = CostFunction(*args)
        current_state.notify_cost_function(cfunc, use_empty_apriori)

        # TODO
        # If we are using use_empty_apriori, then our initial guess is just a
        # set of zeros. This is really kind of arcane, but muses-py
        # has special handling in the BT strategy step and systematic jacobians.
        # Conform to that for now,
        # although it would be nice to remove this special handling at some point
        # when we have all the larger stuff sorted out.
        if use_empty_apriori:
            cfunc.parameters = np.zeros((cfunc.fm_sv.observer_claimed_size,))
        else:
            cfunc.parameters = current_state.initial_guess
        return cfunc

    def _forward_model(
        self,
        instrument_name_list: list[InstrumentIdentifier],
        current_state: CurrentState,
        spec_win_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        include_bad_sample: bool = False,
        obs_list: list[MusesObservation] | None = None,
        use_empty_apriori: bool = False,
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
        fm_sv = rf.StateVector()
        if obs_list is not None:
            self.obs_list = obs_list
        else:
            for instrument_name in instrument_name_list:
                obs = self.observation_handle_set.observation(
                    instrument_name,
                    current_state,
                    (
                        spec_win_dict[instrument_name]
                        if spec_win_dict is not None
                        else None
                    ),
                    fm_sv,
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
            fm = self.forward_model_handle_set.forward_model(
                instrument_name,
                current_state,
                self.obs_list[i],
                fm_sv,
                rf_uip_func,
                **kwargs,
            )
            self.fm_list.append(fm)
        fm_sv.observer_claimed_size = current_state.fm_state_vector_size
        # Leave off the apriori part if requested
        if use_empty_apriori:
            # TODO
            # Note by convention muses-py using a length 1 array here. I think
            # we can eventually relax that, but right now that is assumed in a few
            # places. This was probably to avoid zero size arrays, but there isn't
            # any actually problem in python with those. But for now, fit muses-py
            # convention
            retrieval_sv_apriori = np.zeros((current_state.fm_state_vector_size,)).view(
                RetrievalGridArray
            )
            retrieval_sv_sqrt_constraint = np.zeros(
                (current_state.fm_state_vector_size, current_state.fm_state_vector_size)
            ).view(RetrievalGridArray)
            smap = rf.StateMappingLinear()
        else:
            smap = current_state.state_mapping_retrieval_to_fm
            retrieval_sv_apriori = current_state.constraint_vector(fix_negative=True)
            retrieval_sv_sqrt_constraint = current_state.sqrt_constraint.transpose()
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
            rf_uip.instrument,  # type:ignore[arg-type]
            cstate,  # type:ignore[arg-type]
            None,
            rf_uip_func=lambda instrument_name: rf_uip,
            obs_list=obs_list,
            **kwargs,
        )


__all__ = [
    "CostFunctionCreator",
]
