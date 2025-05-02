# Might end up breaking this file up, for now have all the stuff here
from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .creator_handle import CreatorHandle, CreatorHandleSet
from .osp_reader import OspCovarianceMatrixReader, OspSpeciesReader
from .identifier import StateElementIdentifier, RetrievalType
import abc
from loguru import logger
from pathlib import Path
import numpy as np
import numpy.testing as npt
from typing import Any, cast, Self
import typing

if typing.TYPE_CHECKING:
    from .state_element_old_wrapper import (
        StateElementOldWrapper,
        StateElementOldWrapperHandle,
    )
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .error_analysis import ErrorAnalysis
    from .cost_function_creator import CostFunctionStateElementNotify
    from .current_state import SoundingMetadata


# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray
RetrievalGrid2dArray = np.ndarray
ForwardModelGrid2dArray = np.ndarray


class StateElement(object, metaclass=abc.ABCMeta):
    """This handles a single StateElement, identified by
    a StateElementIdentifier (e.g., the name "TATM" or
    "pressure".

    A StateElement has a current value, a apriori value, a
    retrieval_initial_value (at the start of a full multi-step retrieval)
    and step_initial_value (at the start the current retrieval step).
    In addition, we possibly have "true" value, e.g, we know the answer
    because we are simulating data or something like that. Not
    every state element has a "true" value, is None in those cases.

    See the documentation at the start of CurrentState for a discussion
    of the various state vectors.

    As a convention, we always return the values as a np.ndarray, even
    for single scalar values. This just saves on needing to have lots
    of "if ndarray else if scalar" code. A scalar is returned as a
    np.ndarray of size 1.

    Note that we have a separate apriori_cov_fm and constraint_matrix. Most of
    the time these aren't actually independent, for a MaxAPosteriori type cost
    function the constraint matrix is just apriori_cov. However, StateElement
    maintains these as two separate things for two reasons:

    1. One minor, in the existing muses-py code these can come from
       different sources.  It is often the case that while
       constraint_matrix is close to apriori_cov, there may be
       small roundoff differences so if we replace constraint_matrix
       with apriori_cov we get slightly different output. This is a minor
       problem, we currently use in our tests the requirement that we generate
       the same output as muses-py runs. But it is actually ok if these aren't
       the same, we just need to update our expected results. We tend to not
       do this, just because it is easier to find "real problem differences" if
       we keep all differences from occuring.

    2. A bigger difference is actually a real change. muses-py has the
       convention that the constraint_matrix might be different from
       one retrieval step to the next. For example, the ig_refine
       steps used in the brightness temperature steps to determine
       cloud fraction has a constraint matrix inflated by a factor of
       100. This has the effect of reducing the cost of varying a
       parameter like cloud fraction, which is a reasonable thing to
       do when we are really doing something close to calculating
       this. So for an particular step, constraint_matrix might not be
       even close to apriori_cov_fm. There is probably some name for
       a scaled cost function - we might check with Edwin about
       this. But the idea makes sense.  Note that in practice
       apriori_cov_fm gets used just by the ErrorAnalysis, and
       constraint_matrix get used just by CostFunction, so I don't
       think the inconsistency here is an actually problem.

    We have two state mappings, one that goes between the retrieval
    state vector and the forward model state vector, and a second that
    is used in the forward model. These are separated because
    MaxAPosterioriSqrtConstraint needs to be able to go to the forward model
    state vector separately (used by the error analysis). In py-retrieve,
    state_mapping_retrieval_to_fm was the basis matrix, and state_mapping
    what the mapTypeList. But we use the more generate rf.StateMapping here so
    we aren't restricted to just these two.

    The StateElement gets notified when various things happen in a retrieval. These
    are:

    1. notify_start_retrieval called at the very start of a retrieval. So this can
       do things like update the value and step_initial_value to the retrieval_initial_value.
    2. notify_start_step called at the start of a retrieval step. So this can do
       things like update the value and step_initial_value to the next_step_initial_value
       determined in the previous step
    3. notify_step_solution called when a retrieval step reaches a solution. This can
       determine the step_initial_value for the next step.
    4. notify_parameter_update is called when the rf.StateVector in a rf.ForwardModel is updated
       (e.g., the CostFunction gets updated parameters). This is somewhat redundant with
       notify_step_solution, but it allows the StateElement to update its value for each
       change to the rf.StateVector occurs. We don't have any code right now that actually
       makes use of this, but it just seems more consistent with the normal "updating
       a state vector updates objects that use that state" semantics in ReFRACtor.
    """

    def __init__(self, state_element_id: StateElementIdentifier):
        self._state_element_id = state_element_id
        # Used to get notify_parameter_update called with CostFunction parameter
        # gets changed
        self._cost_function_notify_helper: CostFunctionStateElementNotify | None = None

    def __getstate__(self) -> dict[str, Any]:
        # Don't include CostFunctionStateElementNotify when pickling
        state = self.__dict__.copy()
        state["_cost_function_notify_helper"] = None
        return state

    @property
    def metadata(self) -> dict[str, Any]:
        """Some StateElement have extra metadata. There is really only one example
        now, emissivity has camel_distance and prior_source. It isn't clear the best
        way to handle this, but the current design just returns a dictionary with
        any extra metadata values. We can perhaps rework this if needed in the future.
        For most StateElement this will just be a empty dict."""
        res: dict[str, Any] = {}
        return res

    @property
    def state_element_id(self) -> StateElementIdentifier:
        return self._state_element_id

    @property
    def spectral_domain(self) -> rf.SpectralDomain | None:
        """For StateElementWithFrequency, this returns the frequency associated
        with it. For all other StateElement, just return None."""
        return None

    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to forward model
        vector. Would be nice to replace this with a general
        rf.StateMapping, but for now this is assumed in a lot of
        muses-py code."""
        return None

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from the basis matrix, going from
        the forward model vector the retrieval vector."""
        return None

    @abc.abstractproperty
    def retrieval_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def sys_sv_length(self) -> int:
        raise NotImplementedError()

    @abc.abstractproperty
    def forward_model_sv_length(self) -> int:
        raise NotImplementedError()

    @property
    def altitude_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def altitude_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the altitude levels (None otherwise)"""
        return None

    @property
    def pressure_list(self) -> RetrievalGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        return None

    @property
    def pressure_list_fm(self) -> ForwardModelGridArray | None:
        """For state elements that are on pressure level, this returns
        the pressure levels (None otherwise)"""
        return None

    @abc.abstractproperty
    def value(self) -> RetrievalGridArray:
        """Current value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def value_fm(self) -> ForwardModelGridArray:
        """Current value of StateElement"""
        raise NotImplementedError()

    @property
    def value_str(self) -> str | None:
        """A small number of values in the full state are actually str (e.g.,
        StateElementIdentifier("nh3type"). This is like value, but we
        return a str instead. For most StateElement, this returns "None"
        instead which indicates we don't have a str value.

        It isn't clear that this is the best interface, on the other hand these
        str values don't have an obvious other place to go. And these value are
        at least in the spirit of other StateElement. So at least for now we
        will support this, possibly reworking this in the future.
        """
        return None

    @abc.abstractproperty
    def apriori_value(self) -> RetrievalGridArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def apriori_value_fm(self) -> ForwardModelGridArray:
        """Apriori value of StateElement"""
        raise NotImplementedError()

    @abc.abstractproperty
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Constraint matrix, generally the inverse of apriori_cov, although see the
        discussion in StateElement class about how this might be different."""
        raise NotImplementedError()

    def constraint_cross_covariance(
        self, selem2: StateElement
    ) -> RetrievalGrid2dArray | None:
        """Return the constraint cross matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    @abc.abstractproperty
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        """Apriori Covariance"""
        raise NotImplementedError()

    def apriori_cross_covariance_fm(
        self, selem2: StateElement
    ) -> ForwardModelGrid2dArray | None:
        """Return the cross covariance matrix with selem 2. This returns None
        if there is no cross covariance."""
        return None

    @abc.abstractproperty
    def retrieval_initial_value(self) -> RetrievalGridArray:
        """Value StateElement had at the start of the retrieval."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_initial_value(self) -> RetrievalGridArray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        """Value StateElement had at the start of the retrieval step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def state_mapping(self) -> rf.StateMapping:
        """StateMapping used by the forward model (so taking the ForwardModelGridArray
        and mapping to the internal object state)"""
        raise NotImplementedError()

    @abc.abstractproperty
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        """StateMapping used to go between the RetrievalGridArray and
        ForwardModelGridArray (e.g., the basis matrix in muses-py)"""
        raise NotImplementedError()

    @property
    def true_value(self) -> RetrievalGridArray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return None

    @property
    def true_value_fm(self) -> ForwardModelGridArray | None:
        """The "true" value if known (e.g., we are running a simulation).
        "None" if we don't have a value."""
        return None

    @abc.abstractproperty
    def updated_fm_flag(self) -> ForwardModelGridArray:
        """This is array of boolean flag indicating which parts of the forward
        model state vector got updated when we called notify_solution. A 1 means
        it was updated, a 0 means it wasn't. This is used in the ErrorAnalysis."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update_state_element(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        """Update the value of the StateElement. This function updates
        each of the various values passed in.  A value of 'None' (the
        default) means skip updating that part of the StateElement.
        """
        raise NotImplementedError()

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        """Called with the subset of parameters for this StateElement
        when the cost function changes, or a solution has been found."""
        pass

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        pass

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        pass

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        """Called when a retrieval step has a solution.

        A note, the initial_value shouldn't be updated yet. classes that use the results want
        to have the initial guess for *this* step, not what will be used in the next. Instead
        the initial guess should be updated in notify_start_step. We do save what the next
        initial guess should be, assuming it was updated.

        We pass in the slice needed to get this StateElement values, or None if we aren't
        actually retrieving this particular StateElement. It just is more natural to have
        something outside this class maintain this information (e.g the CurrentState), since
        this depends on access to all the StateElement in the StateInfo.
        """
        pass


class StateElementHandle(CreatorHandle):
    """Return StateElement objects, for a given StateElementIdentifier

    Note StateElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        raise NotImplementedError()


class StateElementHandleSet(CreatorHandleSet):
    """This maps a StateElementIdentifier to a StateElement object that handles it.

    Note StatElementHandle can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next."""

    def __init__(self) -> None:
        super().__init__("state_element")

    def state_element(self, state_element_id: StateElementIdentifier) -> StateElement:
        return self.handle(state_element_id)

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(
                    measurement_id,
                    retrieval_config,
                    strategy,
                    observation_handle_set,
                    sounding_metadata,
                )


class StateElementImplementation(StateElement):
    """A very common implementation of a StateElement just populates member variables for
    the value, apriori, etc. This class handles this common case, derived classes should
    fill in handling the various notify functions."""

    # TODO See if we want the RetrievalGrid or ForwardModelGrid here. We'll need to wait
    # until we get to some of the state elements on levels before figuring this out
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        value: RetrievalGridArray,
        apriori_value: RetrievalGridArray,
        apriori_cov_fm: ForwardModelGrid2dArray,
        constraint_matrix: RetrievalGrid2dArray,
        state_mapping_retrieval_to_fm: rf.StateMapping = rf.StateMappingLinear(),
        state_mapping: rf.StateMapping = rf.StateMappingLinear(),
        initial_value: RetrievalGridArray | None = None,
        true_value: RetrievalGridArray | None = None,
        selem_wrapper: StateElementOldWrapper | None = None,
    ) -> None:
        super().__init__(state_element_id)
        self._value = value
        self._apriori_value = apriori_value
        self._constraint_matrix = constraint_matrix
        self._apriori_cov_fm = apriori_cov_fm
        self._state_mapping = state_mapping
        self._state_mapping_retrieval_to_fm = state_mapping_retrieval_to_fm
        self._step_initial_value = (
            initial_value if initial_value is not None else apriori_value
        )
        self._retrieval_initial_value = self._step_initial_value.copy()
        self._true_value = true_value
        self._updated_fm_flag = np.zeros((apriori_cov_fm.shape[0],)).astype(bool)
        self._retrieved_this_step = False
        self._initial_guess_not_updated = False
        self._next_step_initial_value: np.ndarray | None = None
        # Temp, until we have tested everything out
        self._sold = selem_wrapper
        if self._sold is not None and hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        if self._value.shape[0] != param_subset.shape[0]:
            raise RuntimeError(
                f"param_subset doesn't match value size {param_subset.shape[0]} vs {self._value.shape[0]}"
            )
        self._value = param_subset

    def _update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        if self._sold is None:
            raise RuntimeError("This shouldn't happen")
        self._sold.update_initial_guess(current_strategy_step)

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def basis_matrix(self) -> np.ndarray | None:
        res = np.eye(self._value.shape[0])
        if self._sold is not None:
            res2 = self._sold.basis_matrix
            if res2 is None:
                raise RuntimeError("res2 should not be None")
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        res = np.eye(self._value.shape[0])
        if self._sold is not None:
            res2 = self._sold.map_to_parameter_matrix
            if res2 is None:
                raise RuntimeError("res2 should not be None")
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def retrieval_sv_length(self) -> int:
        res = self._value.shape[0]
        if self._sold is not None:
            res2 = self._sold.retrieval_sv_length
            assert res == res2
        return res

    @property
    def sys_sv_length(self) -> int:
        res = self._value.shape[0]
        if self._sold is not None:
            res2 = self._sold.sys_sv_length
            assert res == res2
        return res

    @property
    def forward_model_sv_length(self) -> int:
        res = self._apriori_cov_fm.shape[0]
        if self._sold is not None:
            res2 = self._sold.forward_model_sv_length
            assert res == res2
        return res

    @property
    def value(self) -> RetrievalGridArray:
        res = self._value
        if self._sold is not None:
            res2 = self._sold.value
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def value_fm(self) -> ForwardModelGridArray:
        res = self._state_mapping.mapped_state(rf.ArrayAd_double_1(self.value)).value
        if self._sold is not None:
            res2 = self._sold.value_fm
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def apriori_value(self) -> RetrievalGridArray:
        res = self._apriori_value
        if self._sold is not None:
            try:
                res2 = self._sold.apriori_value
            except (AssertionError, RuntimeError):
                res2 = None
            if res2 is not None:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res

    @property
    def apriori_value_fm(self) -> ForwardModelGridArray:
        res = self._state_mapping.mapped_state(
            rf.ArrayAd_double_1(self.apriori_value)
        ).value
        if self._sold is not None:
            try:
                res2 = self._sold.apriori_value_fm
            except (AssertionError, RuntimeError):
                res2 = None
            if res2 is not None:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res

    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        res = self._constraint_matrix
        if self._sold is not None:
            res2 = self._sold.constraint_matrix
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        res = self._apriori_cov_fm
        if self._sold is not None:
            try:
                res2 = self._sold.apriori_cov_fm
            except AssertionError:
                res2 = None
            if res2 is not None:
                npt.assert_allclose(res, res2)
                assert res.dtype == res2.dtype
        return res

    @property
    def retrieval_initial_value(self) -> RetrievalGridArray:
        res = self._retrieval_initial_value
        if self._sold is not None:
            res2 = self._sold.retrieval_initial_value
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def step_initial_value(self) -> RetrievalGridArray:
        res = self._step_initial_value
        if self._sold is not None:
            res2 = self._sold.step_initial_value
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        res = self._state_mapping.mapped_state(
            rf.ArrayAd_double_1(self.step_initial_value)
        ).value
        if self._sold is not None:
            res2 = self._sold.step_initial_value_fm
            npt.assert_allclose(res, res2)
            assert res.dtype == res2.dtype
        return res

    @property
    def true_value(self) -> RetrievalGridArray | None:
        return self._true_value

    @property
    def true_value_fm(self) -> ForwardModelGridArray | None:
        tv = self.true_value
        if tv is not None:
            return self._state_mapping.mapped_state(rf.ArrayAd_double_1(tv)).value
        return None

    @property
    def state_mapping(self) -> rf.StateMapping:
        """StateMapping used by the forward model (so taking the ForwardModelGridArray
        and mapping to the internal object state)"""
        return self._state_mapping

    @property
    def state_mapping_retrieval_to_fm(self) -> rf.StateMapping:
        """StateMapping used to go between the RetrievalGridArray and
        ForwardModelGridArray (e.g., the basis matrix in muses-py)"""
        return self._state_mapping_retrieval_to_fm

    def update_state_element(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        if current is not None:
            self._value = current
        if apriori is not None:
            self._apriori_value = apriori
        if step_initial is not None:
            self._step_initial_value = step_initial
        if retrieval_initial is not None:
            self._retrieval_initial_value = retrieval_initial
        if true_value is not None:
            self._true_value = true_value
        if self._sold is not None:
            self._sold.update_state_element(
                current, apriori, step_initial, retrieval_initial, true_value
            )

    @property
    def updated_fm_flag(self) -> ForwardModelGridArray:
        res = np.zeros((self.apriori_cov_fm.shape[0],), dtype=bool)
        if self._retrieved_this_step:
            res[:] = True
        if self._sold is not None:
            res2 = self._sold.updated_fm_flag
            npt.assert_allclose(res, res2)
        return res

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if self._sold is not None:
            self._sold.notify_start_retrieval(current_strategy_step, retrieval_config)
        # The value and step initial guess should be set to the retrieval initial value
        self._value = self._retrieval_initial_value.copy()
        self._step_initial_value = self._retrieval_initial_value.copy()
        # This is to support testing. We currently have a way of populate StateInfoOld when
        # we restart a step, but not StateInfo. Longer term we will fix this, but short term
        # just propagate any values in selem_wrapper to this class
        if self._sold:
            self._value = self._sold.value
            self._step_initial_value = self._sold.value
        self._next_step_initial_value = None

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        if self._sold is not None:
            self._sold.notify_start_step(
                current_strategy_step,
                error_analysis,
                retrieval_config,
                skip_initial_guess_update,
            )
        # Update the initial value if we have a setting from the previous step
        if self._next_step_initial_value is not None:
            self._step_initial_value = self._next_step_initial_value
            self._next_step_initial_value = None
        # Set value to initial value
        self._value = self._step_initial_value.copy()
        self._retrieved_this_step = (
            self.state_element_id in current_strategy_step.retrieval_elements
        )
        self._initial_guess_not_updated = (
            self.state_element_id
            in current_strategy_step.retrieval_elements_not_updated
        )

    def notify_step_solution(
        self, xsol: RetrievalGridArray, retrieval_slice: slice | None
    ) -> None:
        # We've already called notify_parameter_update, so no need to update
        # self._value here
        if self._sold is not None:
            self._sold.notify_step_solution(xsol, retrieval_slice)
        # Default is that the next initial value is whatever the solution was from
        # this step. But skip if we are on the not updated list
        self._next_step_initial_value = None
        if retrieval_slice is not None:
            if not self._initial_guess_not_updated:
                self._value = xsol[retrieval_slice]
                self._next_step_initial_value = self._value.copy()
            else:
                # Reset value for any changes in solver run if we aren't allowing this to
                # update
                self._value = self._step_initial_value.copy()


class StateElementOspFile(StateElementImplementation):
    """This implementation of StateElement gets the apriori/initial guess as a hard coded
    value, and the constraint_matrix and apriori_cov_fm from OSP files. This seems a
    bit convoluted to me - why not just have all the values given in the python configuration
    file? But this is the way muses-py works, and at the very least we need to implementation
    for backwards testing.  We may replace this StateElement, there doesn't seem to be any
    good reason to spread everything across multiple files.

    In some cases, we have the species in the covariance species_directory but not the
    covariance_directory. You can optionally request that we just use the constraint
    matrix as the apriori_cov_fm
    """

    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        apriori_value: np.ndarray,
        latitude: float,
        species_directory: Path,
        covariance_directory: Path,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
    ):
        # For OMI and TROPOMI parameters the initial value, and apriori are hard coded.
        # These get set up in script_retrieval_setup_ms.py, so we can look there for the
        # values. The apriori (called stateConstraint) and first guess (called stateInitial)
        # get identically set to this value. The covariance is separately read from a file.
        # Fill these in
        value = apriori_value.copy()
        apriori = apriori_value.copy()
        self.osp_species_reader = OspSpeciesReader.read_dir(species_directory)
        t = self.osp_species_reader.read_file(
            state_element_id, RetrievalType("default")
        )
        map_type = t["mapType"].lower()
        if map_type == "linear":
            smap = rf.StateMappingLinear()
        elif map_type == "log":
            smap = rf.StateMappingLog()
        else:
            raise RuntimeError(f"Don't recognize map_type {map_type}")
        constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            state_element_id, RetrievalType("default")
        )
        if cov_is_constraint:
            apriori_cov_fm = constraint_matrix
        else:
            r = OspCovarianceMatrixReader.read_dir(covariance_directory)
            apriori_cov_fm = r.read_cov(state_element_id, map_type, latitude)
        # This is to support testing. We currently have a way of populate StateInfoOld when
        # we restart a step, but not StateInfo. Longer term we will fix this, but short term
        # just propagate any values in selem_wrapper to this class
        if selem_wrapper is not None:
            value = selem_wrapper.value
            # breakpoint()
        super().__init__(
            state_element_id,
            value,
            apriori,
            apriori_cov_fm,
            constraint_matrix,
            selem_wrapper=selem_wrapper,
            state_mapping=smap,
        )
        # Also update initial value, but not apriori
        if selem_wrapper is not None:
            self._step_initial_value = selem_wrapper.value

    @classmethod
    def create_from_handle(
        cls,
        state_element_id: StateElementIdentifier,
        apriori_value: np.ndarray,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
        selem_wrapper: StateElementOldWrapper | None = None,
        cov_is_constraint: bool = False,
    ) -> Self | None:
        """Create object from the set of parameter the StateElementOspFileHandle supplies.

        We don't actually use all the arguments, but they are there for other classes
        """
        res = cls(
            state_element_id,
            apriori_value,
            sounding_metadata.latitude.value,
            Path(retrieval_config["speciesDirectory"]),
            Path(retrieval_config["covarianceDirectory"]),
            selem_wrapper=selem_wrapper,
            cov_is_constraint=cov_is_constraint,
        )
        return res

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        super().notify_start_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )
        # Most of the time this will just return the same value, but there might be
        # certain steps with a different constraint matrix.
        self._constraint_matrix = self.osp_species_reader.read_constraint_matrix(
            self.state_element_id, current_strategy_step.retrieval_type
        )


class StateElementOspFileHandle(StateElementHandle):
    def __init__(
        self,
        sid: StateElementIdentifier,
        apriori_value: np.ndarray,
        hold: StateElementOldWrapperHandle | None = None,
        cls: type[StateElementOspFile] = StateElementOspFile,
        cov_is_constraint: bool = False,
    ) -> None:
        self.obs_cls = cls
        self.sid = sid
        self.hold = hold
        self.apriori_value = apriori_value
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None
        self.cov_is_constraint = cov_is_constraint

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config
        self.strategy = strategy
        self.observation_handle_set = observation_handle_set
        self.sounding_metadata = sounding_metadata

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        from .state_element_old_wrapper import StateElementOldWrapper

        if state_element_id != self.sid:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target first")
        if self.hold is not None:
            sold = cast(
                StateElementOldWrapper, self.hold.state_element(state_element_id)
            )
        else:
            sold = None
        res = self.obs_cls.create_from_handle(
            state_element_id,
            self.apriori_value,
            self.measurement_id,
            self.retrieval_config,
            self.strategy,
            self.observation_handle_set,
            self.sounding_metadata,
            sold,
            self.cov_is_constraint,
        )
        if res is not None:
            logger.debug(f"Creating {self.obs_cls.__name__} for {state_element_id}")
        return res


class StateElementFillValueHandle(StateElementHandle):
    """There are a few state element (like OMICLOUDFRACTION) that get created even
    when we don't have the instrument data. These should just return a StateElement
    with fill values. This handle is for these."""

    def __init__(
        self,
        sid: StateElementIdentifier,
    ) -> None:
        self.sid = sid

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        logger.debug(f"Creating StateElementFillValue for {state_element_id}")
        fill = np.array(
            [
                -999.0,
            ]
        )
        fill_2d = np.array(
            [
                [
                    -999.0,
                ]
            ]
        )
        return StateElementImplementation(self.sid, fill, fill, fill_2d, fill_2d)


class StateElementFixedValueHandle(StateElementHandle):
    """Create state element from static values, rather than getting this from somewhere else."""

    def __init__(
        self,
        sid: StateElementIdentifier,
        apriori: np.ndarray,
        apriori_cov_fm: np.ndarray,
    ) -> None:
        self.sid = sid
        self.apriori = apriori
        self.apriori_cov_fm = apriori_cov_fm

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        logger.debug(f"Creating StateElementFixedValue for {state_element_id}")
        return StateElementImplementation(
            self.sid,
            self.apriori,
            self.apriori,
            self.apriori_cov_fm,
            np.linalg.inv(self.apriori_cov_fm),
        )


__all__ = [
    "StateElement",
    "StateElementImplementation",
    "StateElementHandle",
    "StateElementHandleSet",
    "StateElementOspFileHandle",
    "StateElementOspFile",
    "StateElementFillValueHandle",
    "StateElementFixedValueHandle",
]
