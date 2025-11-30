from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .mpy import mpy_radiance_data
import numpy as np
from loguru import logger
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .identifier import InstrumentIdentifier
    from .retrieval_array import RetrievalGridArray


# This implements mpy.ReplaceFunctionObject, but we don't actually derive from
# that so we don't depend on mpy being available.
# class CostFunction(rf.NLLSMaxAPosteriori, mpy.ReplaceFunctionObject):
class CostFunction(rf.NLLSMaxAPosteriori):
    """This is the cost function we use to interface between ReFRACtor
    and muses-py. This is just a standard rf.NLLSMaxAPosteriori with
    some extra convenience functions.

    Note although this is labeled MaxAPosteriori, this is actually a
    more general least squares with a quadratic regularization. This
    takes a general constraint matrix and constraint vector. These are
    often but not always the apriori covariance and apriori vector. When
    they are we truly have a MaxAPosteriori, but when not this is a more
    general least squares with a quadratic penalty similar to a MaxAPosteriori.

    We call this a MaxAPosteriori because we already have the machinery in place
    in framework for MaxAPosteriori that can be used directly for the more general
    problem.

    See section III.B of "Tropospheric Emission
    Spectrometer: Retrieval Method and Error Analysis" (IEEE
    TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING, VOL. 44, NO. 5, MAY
    2006) (https://ieeexplore.ieee.org/document/1624609).

    We allow this to replace the functions fm_wrapper and/or
    refractor_residual_fm_jac, as well as just having the functions to
    directly call."""

    def __init__(
        self,
        instrument_name_list: list[InstrumentIdentifier],
        fm_list: list[rf.ForwardModel],
        obs_list: list[rf.Observation],
        fm_sv: rf.StateVector,
        retrieval_sv_constraint_vector: RetrievalGridArray,
        retrieval_sv_sqrt_constraint: RetrievalGridArray,
        mapping: rf.StateMapping,
    ) -> None:
        self.instrument_name_list = instrument_name_list
        self.fm_sv = fm_sv
        self.retrieval_sv_constraint_vector = retrieval_sv_constraint_vector
        self.retrieval_sv_sqrt_constraint = retrieval_sv_sqrt_constraint
        if hasattr(mapping, "basis_matrix"):
            basis_matrix = mapping.basis_matrix
        else:
            basis_matrix = None

        # ------------------------------------------------------
        # Do some sanity checks. This all gets caught at the C++
        # level, but the error messages tend to be more cryptic
        # ------------------------------------------------------

        # Expect sqrt_constraint to be square, and to have the same
        # number of rows as constraint
        if (
            self.retrieval_sv_sqrt_constraint.shape[0]
            != self.retrieval_sv_sqrt_constraint.shape[1]
        ):
            raise RuntimeError(
                f"retrieval_sv_sqrt_constraint should be a square matrix, was {retrieval_sv_sqrt_constraint.shape[0]} x {retrieval_sv_sqrt_constraint.shape[1]}"
            )
        if (
            self.retrieval_sv_sqrt_constraint.shape[0]
            != self.retrieval_sv_constraint_vector.shape[0]
        ):
            raise RuntimeError(
                f"retrieval_sv_constraint_vector size {self.retrieval_sv_constraint_vector.shape[0]} and retrieval_sv_sqrt_constraint size {self.retrieval_sv_sqrt_constraint.shape[0]} should have the same numbers of rows"
            )

        # If we don't have a basis matrix, the forward model state
        # vector size needs to match the retrieval state vector size
        if basis_matrix is None:
            if (
                self.fm_sv.observer_claimed_size
                != self.retrieval_sv_constraint_vector.shape[0]
            ):
                raise RuntimeError(
                    f"Without a basis matrix, fm_sv size of {self.fm_sv.observer_claimed_size} should be same as retrieval_sv_constraint_vector size of {self.retrieval_sv_constraint_vector.shape[0]}"
                )
        else:
            # If we do have a basis matrix, the forward model state
            # vector size needs to match the column size of the basis
            # matrix, and the retrieval state vector size needs to
            # match the rows of the basis matrix
            if self.fm_sv.observer_claimed_size != basis_matrix.shape[0]:
                raise RuntimeError(
                    f"fm_sv size of {self.fm_sv.observer_claimed_size} should match row size of basis matrix of {basis_matrix.shape[0]} x {basis_matrix.shape[1]}"
                )
            if self.retrieval_sv_constraint_vector.shape[0] != basis_matrix.shape[1]:
                raise RuntimeError(
                    f"retrieval_sv_constraint_vector size of {self.retrieval_sv_constraint_vector.shape[0]} should match column size of basis matrix of {basis_matrix.shape[0]} x {basis_matrix.shape[1]}"
                )

        mstand = rf.MaxAPosterioriSqrtConstraint(
            fm_list,
            obs_list,
            self.fm_sv,
            self.retrieval_sv_constraint_vector,
            self.retrieval_sv_sqrt_constraint,
            mapping,
        )
        super().__init__(mstand)
        # Some of the forward models need to know when we have a cost function. In
        # particular, the MusesForwardModels need to know this to attach the UIP
        # correctly. However, our general rf.ForwardModel doesn't have this, since
        # it is completely uncoupled from the CostFunction. Notify the model that
        # have a notify_cost_function function, but skip the rest.
        for fm in fm_list:
            if hasattr(fm, "notify_cost_function"):
                fm.notify_cost_function(self)

    @property
    def obs_list(self) -> list[rf.Observation]:
        return self.max_a_posteriori.observation

    @property
    def fm_list(self) -> list[rf.ForwardModel]:
        return self.max_a_posteriori.forward_model

    def get_state(self) -> dict:
        """Return a dictionary of values that can be used by set_state.
        This allows us to skip running the forward model in unit tests. This
        is similar to a pickle serialization (which we also support), but
        only saves the things that change when we update the parameters.

        Useful for testing when we want to actually test creating this
        CostFunction, but want to skip the solver/forward model step."""
        (msrmnt_is_const, m, k, msrmnt, msrmnt_jacobian, k_x, msrmnt_jacobian_x) = (
            self.max_a_posteriori.get_state()
        )
        # Note we use the "tolist" to translate numpy to a python list. This is
        # so we can dump this to json - json doesn't support np.ndarray types.
        return {
            "parameters": self.parameters.tolist(),
            "msrmnt_is_const": msrmnt_is_const,
            "m": m.tolist(),
            "k": k.tolist(),
            "msrmnt": msrmnt.tolist(),
            "msrmnt_jacobian": msrmnt_jacobian.tolist(),
            "k_x": k_x.tolist(),
            "msrmnt_jacobian_x": msrmnt_jacobian_x.tolist(),
        }

    def set_state(self, d: dict) -> None:
        """Set the state previously saved by get_state"""
        self.parameters = np.array(d["parameters"])
        # Translate the lists back to np.ndarray, with
        # special handling for empty jacobians
        k = np.array(d["k"])
        msrmnt_jacobian = np.array(d["msrmnt_jacobian"])
        k_x = np.array(d["k_x"])
        msrmnt_jacobian_x = np.array(d["msrmnt_jacobian_x"])
        if k.size == 0:
            k = np.zeros((0, 0))
        if msrmnt_jacobian.size == 0:
            msrmnt_jacobian = np.zeros((0, 0))
        if k_x.size == 0:
            k_x = np.zeros((0, 0))
        if msrmnt_jacobian_x.size == 0:
            msrmnt_jacobian_x = np.zeros((0, 0))
        self.max_a_posteriori.set_state(
            d["msrmnt_is_const"],
            np.array(d["m"]),
            k,
            np.array(d["msrmnt"]),
            msrmnt_jacobian,
            k_x,
            msrmnt_jacobian_x,
        )

    # -----------------------------------------------------------------
    # All the functions past this point are just for testing with old
    # py-retrieve code, ReFRACtor retrieval doesn't need any of this.
    # -----------------------------------------------------------------

    def parameters_fm(self) -> np.ndarray:
        """Parameters on the full forward model grid."""
        return self.max_a_posteriori.mapping.mapped_state(
            rf.ArrayAd_double_1(self.parameters)
        ).value

    def should_replace_function(self, func_name: str, parms: list[Any]) -> bool:
        return True

    def replace_function(self, func_name: str, parms: dict) -> Any:
        if func_name == "fm_wrapper":
            return self.fm_wrapper(**parms)
        elif func_name == "residual_fm_jacobian":
            return self.residual_fm_jacobian(**parms)
        elif func_name == "update_uip":
            return self.update_uip(**parms)

    def update_uip(
        self, i_uip: dict, i_ret_info: dict, i_retrieval_vec: dict
    ) -> tuple[None, dict]:
        """This is an adapter that stubs out the updating of the
        UIP. We don't actually need to do this, but
        levmar_nllsq_elanor expects a function here.
        """
        return (None, i_retrieval_vec)

    def fm_wrapper(self, i_uip, i_windows, oco_info):  # type: ignore
        """This uses the CostFunction to calculate the same things that
        muses-py does with it fm_wrapper function. We provide the same
        interface here to 1) provide something that can be used as a
        replacement for mpy.fm_wrapper but possibly using ReFRACtor objects
        and 2) make a more direct comparison between ReFRACtor and muses-py
        (e.g., for testing)"""
        if hasattr(i_uip, "currentGuessList"):
            p = i_uip.currentGuessListFM
        else:
            p = i_uip["currentGuessList"]
        if self.expected_parameter_size != len(p):
            raise RuntimeError(
                "We aren't expecting parameters the size of currentGuessList."
            )
        self.parameters = p
        radiance_fm = self.max_a_posteriori.model
        jac_fm = self.max_a_posteriori.jacobian_fm.transpose()
        bad_flag = 0
        freq_fm = np.concatenate(
            [
                fm.spectral_domain_all().data
                for fm in self.max_a_posteriori.forward_model
            ]
        )
        # This duplicates what mpy.fm_wrapper does. It looks like
        # a number of these are placeholders, but the struct returned
        # by mpy.radiance_data looks like something that is just dumped
        # to a file, so I guess the placeholders make sense in an output
        # file where we don't have these values.

        # Seems to by indexed by detector, of which we only have one
        # dummy one
        detectors = [-1]
        radiance_fm = np.array([radiance_fm])
        nesr_fm = np.zeros(radiance_fm.shape)
        # Oddly frequency isn't indexed by detectors
        # freq_fm = np.array([freq_fm])
        # Not sure what filters is, but fm_wrapper just supplies this
        # as a empty array
        filters = []
        instrument = ""
        o_radiance = mpy_radiance_data(
            radiance_fm, nesr_fm, detectors, freq_fm, filters, instrument
        )
        # We can fill these in if needed, but run_forward_model doesn't
        # actually use these values so we don't bother.
        o_measured_radiance_omi = None
        o_measured_radiance_tropomi = None
        return (
            o_radiance,
            jac_fm,
            bad_flag,
            o_measured_radiance_omi,
            o_measured_radiance_tropomi,
        )

    def good_point(self) -> np.ndarray:
        """Return a boolean array for the full observation size, all
        forward models, including bad samples. True means a good point,
        False means bad."""
        gpt = []
        for obs in self.obs_list:
            s = obs.radiance_all_extended(include_bad_sample=True)
            gpt.append(s.spectral_range.uncertainty >= 0)
        return np.concatenate(gpt)

    def residual_fm_jacobian(self, uip, ret_info, retrieval_vec, iterNum, oco_info={}):  # type: ignore
        """This uses the CostFunction to calculate the same things that
        muses-py does with it residual_fm_jacobian function. We provide the
        same interface here to 1) provide something that can be used as a
        replacement for mpy.fm_wrapper but possibly using ReFRACtor objects
        and 2) make a more direct comparison between ReFRACtor and muses-py
        (e.g., for testing)"""
        # In addition, ret_info has obs_rad and meas_err
        # updated for OMI and TROPOMI. This seems kind of bad to me,
        # but the values get used in run_retrieval of muses-py, so
        # we need to update this.
        # Stub out the UIP, it isn't actually needed for anything. We can have this as None,
        # just because levmar_nllsq_elanor expects to pass in this argument
        if self.expected_parameter_size != len(retrieval_vec):
            raise RuntimeError(
                "We aren't expecting parameters the size of retrieval_vec."
            )
        if False:
            logger.info("Setting parameters in cost function")
            logger.info(f"{retrieval_vec}")
        self.parameters = retrieval_vec
        # obs_rad and meas_err includes bad samples, so we can't use
        # cfunc.max_a_posteriori.measurement here which filters out
        # bad samples. Instead we access the observation list we stashed
        # when we created the cost function directly
        d = []
        u = []
        for obs in self.obs_list:
            s = obs.radiance_all_extended(include_bad_sample=True)
            d.append(s.spectral_range.data)
            u.append(s.spectral_range.uncertainty)
        ret_info["obs_rad"] = np.concatenate(d)
        ret_info["meas_err"] = np.concatenate(u)
        residual = self.residual
        jac_residual = self.jacobian.transpose()

        # TODO Determine what exactly we want to do with bad samples

        # self.max_a_posteriori.model only contains the good samples.
        # The existing muses-py actually just runs the forward model
        # on all points, including bad data. It isn't clear what we want
        # to do here. For now, create the proper size output but use a fill
        # value of -999 for bad data. Do the same for the jacobian, but
        # at least for now with the fill value of 0

        radiance_fm = np.full(ret_info["meas_err"].shape, -999.0)
        gpt = ret_info["meas_err"] >= 0
        radiance_fm[gpt] = self.max_a_posteriori.model
        jac_fm_gpt = self.max_a_posteriori.model_measure_diff_jacobian_fm.transpose()
        jac_fm = np.full((jac_fm_gpt.shape[0], ret_info["meas_err"].shape[0]), -999.0)
        jac_fm[:, gpt] = jac_fm_gpt
        # Need to add handling for bad samples
        stop_flag = 0
        # Muses-py prefers that we just fail if we get nans here. Otherwise we
        # get a pretty obscure error buried in levmar_nllsq_elanor (basically we
        # end up with a rank zero matrix at some point. results in a TypeError).
        # Just cleaner to say we fail because our radiance or jacobian has nans
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Radiance is not finite")
        if not np.all(np.isfinite(jac_fm)):
            raise RuntimeError("Jacobian is not finite")
        if False:
            for fm in self.fm_list:
                logger.debug(f"Forward model: {fm}")
        return (uip, residual, jac_residual, radiance_fm, jac_fm, stop_flag)
