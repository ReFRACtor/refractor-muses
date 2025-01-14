from __future__ import annotations
from .cost_function import CostFunction
import numpy as np
from .replace_function_helper import register_replacement_function_in_block
import refractor.muses.muses_py as mpy  # type: ignore
import os
from pathlib import Path


class MusesLevmarSolver:
    """This is a wrapper around levmar_nllsq_elanor that makes it look like
    a NLLSSolver. Right now we don't actually derive from that, we can perhaps
    put that in place if useful. But for now, just provide a "solve" function.
    """

    def __init__(
        self,
        cfunc: CostFunction,
        max_iter: int,
        delta_value: float,
        conv_tolerance: list[float],
        chi2_tolerance: float,
        log_file: str | os.PathLike[str] | None = None,
        verbose=False,
    ):
        self.cfunc = cfunc
        self.max_iter = max_iter
        self.delta_value = delta_value
        self.conv_tolerance = conv_tolerance
        self.chi2_tolerance = chi2_tolerance
        self.log_file = Path(log_file) if log_file is not None else None
        # Defaults, so if we skip solve we have what is needed for output
        self.success_flag = 1
        self.best_iter = 0
        self.residual_rms = np.asarray([0])
        self.x_iter = np.asarray([0])
        self.diag_lambda_rho_delta = np.zeros((1, 3))
        self.stopcrit = np.zeros(shape=(1, 3), dtype=int)
        self.resdiag = np.zeros(shape=(1, 5), dtype=int)
        self.radiance_iter = np.zeros((1, 1))
        self.iter_num = 0
        self.stop_code = -1
        self.verbose = verbose

    def get_state(self) -> dict:
        """Return a dictionary of values that can be used by set_state.
        This allows us to skip running the solver in unit tests. This
        is similar to a pickle serialization (which we also support), but
        only saves the things that change when we update the parameters.

        Useful for testing when we want to actually test creating this
        CostFunction, but want to skip the solver/forward model step."""
        # Note we use the "tolist" to translate numpy to a python list. This is
        # so we can dump this to json - json doesn't support np.ndarray types.
        return {
            "cfunc": self.cfunc.get_state(),
            "diag_lambda_rho_delta": self.diag_lambda_rho_delta.tolist(),
            "stopcrit": self.stopcrit.tolist(),
            "resdiag": self.resdiag.tolist(),
            "x_iter": self.x_iter.tolist(),
            "radiance_iter": self.radiance_iter.tolist(),
            "iter_num": self.iter_num,
            "stop_code": self.stop_code,
            "success_flag": self.success_flag,
            "best_iter": self.best_iter,
            "residual_rms": self.residual_rms.tolist(),
        }

    def set_state(self, d: dict):
        """Set the state previously saved by get_state"""
        # Translate the lists back to np.ndarray
        self.cfunc.set_state(d["cfunc"])
        self.diag_lambda_rho_delta = np.array(d["diag_lambda_rho_delta"])
        self.stopcrit = np.array(d["stopcrit"])
        self.resdiag = np.array(d["resdiag"])
        self.x_iter = np.array(d["x_iter"])
        self.radiance_iter = np.array(d["radiance_iter"])
        self.iter_num = d["iter_num"]
        self.stop_code = d["stop_code"]
        self.success_flag = d["success_flag"]
        self.best_iter = d["best_iter"]
        self.residual_rms = np.array(d["residual_rms"])

    def retrieval_results(self) -> dict:
        """Return the retrieval results dict. Hopefully this can go away, this
        is just used in mpy.set_retrieval_results (another function we would like
        to remove). It would probably be better for things
        to get this directly from this solver and the cost function. But for
        now we have this.

        Note that this works even in solve() hasn't been called - this returns
        what is expected if max_iter is 0."""
        gpt = self.cfunc.good_point()

        radiance_fm = np.full(gpt.shape, -999.0)
        radiance_fm[gpt] = self.cfunc.max_a_posteriori.model
        jac_fm_gpt = (
            self.cfunc.max_a_posteriori.model_measure_diff_jacobian_fm.transpose()
        )
        jacobian_fm = np.full((jac_fm_gpt.shape[0], gpt.shape[0]), -999.0)
        jacobian_fm[:, gpt] = jac_fm_gpt

        radianceOut2 = {"radiance": radiance_fm}
        jacobianOut2 = {"jacobian_data": jacobian_fm}
        # Oddly, set_retrieval_results expects a different shape for num_iterations
        # = 0. Probably should change the code there, but for now just work around
        # this
        if self.iter_num == 0:
            radianceOut2["radiance"] = radiance_fm[np.newaxis, :]

        return {
            "bestIteration": int(self.best_iter),
            "num_iterations": self.iter_num,
            "stopCode": self.stop_code,
            "xret": self.cfunc.parameters,
            "xretFM": self.cfunc.parameters_fm(),
            "radiance": radianceOut2,
            "jacobian": jacobianOut2,
            "radianceIterations": self.radiance_iter[:, np.newaxis, :],
            "xretIterations": self.cfunc.parameters
            if self.iter_num == 0
            else self.x_iter,
            "stopCriteria": np.copy(self.stopcrit),
            "resdiag": np.copy(self.resdiag),
            "residualRMS": self.residual_rms,
            "delta": self.diag_lambda_rho_delta[:, 2],
            "rho": self.diag_lambda_rho_delta[:, 1],
            "lambda": self.diag_lambda_rho_delta[:, 0],
        }

    def solve(self):
        # py-retrieve expects the directory to already be there, so create if
        # needed.
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with register_replacement_function_in_block("update_uip", self.cfunc):
            with register_replacement_function_in_block(
                "residual_fm_jacobian", self.cfunc
            ):
                # We want some of these to go away
                (
                    xret,
                    self.diag_lambda_rho_delta,
                    self.stopcrit,
                    self.resdiag,
                    self.x_iter,
                    res_iter,
                    radiance_fm,
                    self.radiance_iter,
                    jacobian_fm,
                    self.iter_num,
                    self.stop_code,
                    self.success_flag,
                ) = mpy.levmar_nllsq_elanor(
                    self.cfunc.parameters,
                    None,
                    None,
                    {},
                    self.max_iter,
                    verbose=self.verbose,
                    delta_value=self.delta_value,
                    ConvTolerance=self.conv_tolerance,
                    Chi2Tolerance=self.chi2_tolerance,
                    logWrite=self.log_file is not None,
                    logFile=str(self.log_file),
                )
            # Since xret is the best iteration, which might not be the last,
            # set the cost function to this. Note the cost function does internal
            # caching, so if this is the last one then we don't recalculate
            # residual and jacobian.
            self.cfunc.parameters = xret
            # Find iteration used, only keep the best iteration
            rms = np.array(
                [
                    np.sqrt(np.sum(res_iter[i, :] * res_iter[i, :]) / res_iter.shape[1])
                    for i in range(self.iter_num + 1)
                ]
            )
            self.best_iter = int(np.argmin(rms))
            self.residual_rms = rms
