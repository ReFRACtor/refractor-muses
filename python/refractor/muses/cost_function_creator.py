from __future__ import annotations
from .cost_function import CostFunction
from .uip_updater import MaxAPosterioriSqrtConstraintUpdateUip
from .current_state import CurrentState, CurrentStateUip
from .forward_model_handle import ForwardModelHandleSet
from .observation_handle import ObservationHandleSet
import refractor.framework as rf  # type: ignore
import copy
from loguru import logger
from collections.abc import Callable
import typing
import weakref
from typing import Any
import numpy as np

if typing.TYPE_CHECKING:
    from .muses_spectral_window import MusesSpectralWindow
    from .muses_observation import MusesObservation, MeasurementId
    from .refractor_uip import RefractorUip
    from .retrieval_strategy import RetrievalStrategy
    from .identifier import InstrumentIdentifier
    from .state_info import StateElement


class CostFunctionStateElementNotify(rf.ObserverMaxAPosterioriSqrtConstraint):
    """Small adapter class to map framework observer to notify StateElement
    when parameters change"""

    def __init__(self, selem: StateElement, slc: slice):
        super().__init__()
        self.slc = slc
        # We want this object to be kept alive, but not force selem to
        # be kept alive. This set of references does that
        self.selem = weakref.ref(selem)
        self._cost_function_notify_helper = self

    def notify_update(self, mstand: rf.MaxAPosterioriSqrtConstraint) -> None:
        s = self.selem()
        if s is not None:
            s.notify_parameter_update(mstand.parameters[self.slc])


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

    def _rf_uip_func_wrap(
        self, instrument: InstrumentIdentifier | None
    ) -> RefractorUip:
        """We need to keep a copy of the UIP if it gets creates by a lower
        level routine, so that we can attach this to the state vector."""
        if self._rf_uip_func is None:
            raise RuntimeError("Need _rf_uip_func to not be None")
        if None in self._rf_uip:
            return self._rf_uip[None]
        if instrument not in self._rf_uip:
            # If we have multiple instruments that need the UIP (e..g,
            # a py-retrieve like retrieval with a joint AIRS OMI step), then
            # recreate the UIP for all the instruments. This is needed to
            # have the update happen correctly - we can't update just half
            # of the UIP.
            if len(self._rf_uip) > 0:
                new_uip = self._rf_uip_func(None)
                k = list(self._rf_uip.keys())[0]
                v = self._rf_uip[k]
                v.uip = new_uip.uip
                self._rf_uip: dict[None | InstrumentIdentifier, RefractorUip] = {}
                self._rf_uip[None] = v
            else:
                self._rf_uip[instrument] = self._rf_uip_func(instrument)
        if None in self._rf_uip:
            return self._rf_uip[None]
        else:
            return self._rf_uip[instrument]

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
        useful if you are just running to forward model. The muses-py
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
        # Keep track of this, in case we create one so we know to attach this to
        # the state vector
        self._rf_uip = {}
        self._rf_uip_func = rf_uip_func
        args = self._forward_model(
            instrument_name_list,
            current_state,
            spec_win_dict,
            self._rf_uip_func_wrap,
            include_bad_sample=include_bad_sample,
            obs_list=obs_list,
            use_empty_apriori=use_empty_apriori,
            **kwargs,
        )
        cfunc = CostFunction(*args)
        # If we have an UIP, then update this when the parameters get
        # updated.  Note the rf_uip.basis_matrix is None handles the
        # degenerate case of when we have no parameters, for example
        # for RetrievalStrategyStepBT. Any time we have parameters,
        # the basis_matrix shouldn't be None.
        for uip in self._rf_uip.values():
            if uip.basis_matrix is not None:
                cfunc.max_a_posteriori.add_observer_and_keep_reference(
                    MaxAPosterioriSqrtConstraintUpdateUip(uip)
                )
        # Attach StateElement to get notified when current state changes.
        #
        # A note on the different lifetimes. For the CostFunction, if we
        # have a UIP than the UIP state observer is *required*. If we pickle
        # and reload the cost function, it should have the UIP and observer.
        # So we use "add_observer_and_keep_reference". However, for the
        # StateElement these are outside of the CostFunction. You can
        # have a CostFunction without any StateElement, if we pickle and reload
        # we don't want to pull all the StateElement along. So for this
        # we use "add_observer" which uses weak pointers - we notify if the
        # object is still there but don't carry around it lifetime and if the
        # object is deleted then we just don't notify it.
        # skip for CurrentStateUip, we already handle the UIP separately.
        if not isinstance(current_state, CurrentStateUip):
            for sid in current_state.retrieval_state_element_id:
                pstart, plen = current_state.retrieval_sv_loc[sid]
                cfunc.max_a_posteriori.add_observer(
                    CostFunctionStateElementNotify(
                        current_state.full_state_element(sid),
                        slice(pstart, pstart + plen),
                    )
                )

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
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
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
        bmatrix = current_state.basis_matrix
        # Leave off the apriori part if requested
        if use_empty_apriori:
            # TODO
            # Note by convention muses-py using a length 1 array here. I think
            # we can eventually relax that, but right now that is assumed in a few
            # places. This was probably to avoid zero size arrays, but there isn't
            # any actually problem in python with those. But for now, fit muses-py
            # convention
            retrieval_sv_apriori = np.zeros((current_state.fm_state_vector_size,))
            retrieval_sv_sqrt_constraint = np.zeros(
                (current_state.fm_state_vector_size, current_state.fm_state_vector_size)
            )
        else:
            retrieval_sv_apriori = current_state.apriori
            retrieval_sv_sqrt_constraint = current_state.sqrt_constraint.transpose()
        return (
            instrument_name_list,
            self.fm_list,
            self.obs_list,
            fm_sv,
            retrieval_sv_apriori,
            retrieval_sv_sqrt_constraint,
            bmatrix,
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
        cstate = CurrentStateUip(rf_uip, ret_info)
        return self.cost_function(
            rf_uip.instrument,
            cstate,
            None,
            rf_uip_func=lambda instrument_name: rf_uip,
            obs_list=obs_list,
            **kwargs,
        )


__all__ = ["CostFunctionCreator"]
