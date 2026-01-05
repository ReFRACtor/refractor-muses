from __future__ import annotations
from .cost_function import CostFunction
import numpy as np
from .replace_function_helper import register_replacement_function_in_block
import os
from pathlib import Path
from typing import Any
from attrs import frozen
from loguru import logger


@frozen
class SolverResult:
    bestIteration: int
    num_iterations: int
    stopCode: int
    xret: np.ndarray
    xretFM: np.ndarray
    radiance: dict[str, Any]
    jacobian: dict[str, Any]
    radianceIterations: np.ndarray
    xretIterations: np.ndarray
    stopCriteria: np.ndarray
    resdiag: np.ndarray
    residualRMS: np.ndarray
    delta: np.ndarray
    rho: np.ndarray
    lambdav: np.ndarray


class MusesLevmarSolver:
    """This is a wrapper around levmar_nllsq_elanor that makes it look like
    a NLLSSolver. Right now we don't actually derive from that, we can perhaps
    put that in place if useful. But for now, just provide a "solve" function.

    We set up self.cfunc.parameters to whatever the final best iteration solution is.
    """

    def __init__(
        self,
        cfunc: CostFunction,
        max_iter: int,
        delta_value: float,
        conv_tolerance: list[float],
        chi2_tolerance: float,
        log_file: str | os.PathLike[str] | None = None,
        verbose: bool = False,
    ) -> None:
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

    def get_state(self) -> dict[str, Any]:
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

    def set_state(self, d: dict[str, Any]) -> None:
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

    def retrieval_results(self) -> SolverResult:
        """Return the retrieval results.

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

        return SolverResult(
            bestIteration=int(self.best_iter),
            num_iterations=self.iter_num,
            stopCode=self.stop_code,
            xret=self.cfunc.parameters,
            xretFM=self.cfunc.parameters_fm(),
            radiance=radianceOut2,
            jacobian=jacobianOut2,
            radianceIterations=self.radiance_iter[:, np.newaxis, :],
            xretIterations=self.cfunc.parameters if self.iter_num == 0 else self.x_iter,
            stopCriteria=np.copy(self.stopcrit),
            resdiag=np.copy(self.resdiag),
            residualRMS=self.residual_rms,
            delta=self.diag_lambda_rho_delta[:, 2],
            rho=self.diag_lambda_rho_delta[:, 1],
            lambdav=self.diag_lambda_rho_delta[:, 0],
        )

    def solve(self) -> None:
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
                ) = levmar_nllsq_elanor(
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


# See py-retrieve for a lengthy description of this function.
def levmar_nllsq_elanor(
    xInit: np.ndarray,
    i_stepNumber: int | None,
    uip: dict | None,
    ret_info: dict,
    maxIter : int,
    verbose : bool,
    newtonFlag : bool=False,
    sigma : float | None=None,
    delta_value : float | None=None,
    tolerance : float | None=None,
    singTolerance : float | None=None,
    ConvTolerance : list[float] | None=None,
    Chi2Tolerance : float | None=None,
    logWrite : bool=False,
    logFile : str | None=None,
    oco_info : dict={},
):
    # Temp
    import refractor.muses_py as muses_py  # type: ignore

    function_name = "levmar_nllsq_elanor: "

    # Output variables.
    o_x_vector = None
    o_Diag_lambda_rho_delta = None
    o_stopcrit = None
    o_resdiag = None
    o_x_iter = None
    o_res_iter = None
    o_radiance_iter = None
    o_jacobian_fm = None

    o_success_flag = 1

    iterNum = 0  # PYTHON_NOTE: iter is a keyword in Python.
    stopCode = 0

    # AT_LINE 452 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    if sigma is None:
        sigma = 0.1

    if delta_value is None:
        delta_value = 1.0

    if tolerance is None:
        tolerance = 1.0e-7

    if singTolerance is None:
        singTolerance = 1.0e-7

    if maxIter is None:
        maxIter = 100

    # AT_LINE 467 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    if ConvTolerance is None:
        ConvTolerance = [tolerance, tolerance, tolerance]

    if Chi2Tolerance is None:
        Chi2Tolerance = tolerance

    if logWrite:
        assert logFile, "logWrite specified without logFile"
        f = open(logFile, "w")
        f.close()

    # AT_LINE 482 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    # Give this variable a more descriptive name to 'o_x_vector' so we can find it instead of just 'x'.
    o_x_vector = (np.asarray(xInit)).astype(np.float64)

    #  Initialize Stop criteria diagnostics.

    # AT_LINE 486 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    o_stopcrit = np.zeros(shape=(maxIter + 1, 3), dtype=np.float64)
    o_resdiag = np.zeros(shape=(maxIter + 1, 5), dtype=np.float64)

    # Initialize linearity metrics

    # AT_LINE 491 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    o_Diag_lambda_rho_delta = np.zeros(
        shape=(maxIter + 1, 3), dtype=np.float64
    )  # dblarr(maxIter + 1, 3)

    # epsD is numerical precision tolerance

    # AT_LINE 495 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    # epsD = np.MachAr(float_conv=np.float64).eps
    epsD = 2.2204460e-16  # Use the value from IDL running on ponte.

    #  Compute jacobian (nTerms columns by nPoints rows) and residual
    #  (row vector with nPoints elements) based on the initial guess
    #  xInit.
    #

    # UPDATE UIP BASED ON NEW RETRIEVAL VECTOR
    # AT_LINE 504 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

    (uip, o_x_vector) = muses_py.update_uip(uip, ret_info, o_x_vector)

    (uip, residualNext, jacobian_ret, radiance_fm, o_jacobian_fm, stop_flag) = (
        muses_py.residual_fm_jacobian(uip, ret_info, o_x_vector, iterNum, oco_info)
    )

    if stop_flag:
        o_radiance_fm_next = None
        o_success_flag = 0
        return (
            o_x_vector,
            o_Diag_lambda_rho_delta,
            o_stopcrit,
            o_resdiag,
            o_x_iter,
            o_res_iter,
            o_radiance_fm_next,
            o_radiance_iter,
            o_jacobian_fm,
            iterNum,
            stopCode,
            o_success_flag,
        )

    #  A flag to signal the termination of the iterative process.
    stopCode = 0

    resNextNorm2 = np.sum(residualNext**2)

    o_x_iter = np.zeros(shape=(maxIter + 1, len(xInit)), dtype=np.float64)
    o_x_iter[0, :] = o_x_vector[:]

    o_res_iter = np.zeros(shape=(maxIter + 1, len(residualNext)), dtype=np.float64)
    o_res_iter[0, :] = residualNext[:]  # Set the residual for the pre-iteration.

    # AT_LINE 521 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    o_radiance_iter = np.zeros(shape=(maxIter + 1, len(radiance_fm)), dtype=np.float64)
    o_radiance_iter[0, :] = radiance_fm[:]  # Set the radiance for the pre-iteration.

    # AT_LINE 527 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

    #  nPoints must be greater than or equal to nTerms

    nTerms = len(xInit)
    nPoints = len(residualNext)

    #  Compute the L2 norm squared of each column of the jacobian,
    #  and store the results (nTerms of them) in the row array
    #  jacobColumnL2Norm2.

    # AT_LINE 540 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    jacobColumnL2Norm2 = np.zeros(shape=(nTerms), dtype=np.float64)
    for ii in range(nTerms):
        jacobColumnL2Norm2[ii] = np.sum(jacobian_ret[ii] ** 2)

    # AT_LINE 545 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

    # <<<<<<<<<<<<<<< INITIALIZATION OF D (scaleDiag) >>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<< SECTION 6, 6.3 >>>>>>>>>>>>>>>>>>>>>>>>
    #
    #  Initialize scaleDiag, the diagonal of scale diagonal matrix
    #  (a row array with nTerms elements).  If any element of
    #  scaleDiag is equal to zero, then it is set equal to epsD, a
    #  positive small number; therefore, inverting scaleDiag will not
    #  cause division by zero.
    #
    #  If newton keyword is set, then just Newton's method is used,
    #  and there is no need to compute scaleDiag.
    #
    if not newtonFlag:
        scaleDiag = np.sqrt(jacobColumnL2Norm2)
        temp = np.where(scaleDiag == 0)[0]
        if len(temp) > 0:
            scaleDiag[temp] = epsD

    #  <<<<<<<<<<<<<<<<<< END: INITIALIZATION OF D >>>>>>>>>>>>>>>>>>>

    #  Loop counter
    #
    # AT_LINE 569 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
    iterNum = 0  # PYTHON_NOTE: iter is a keyword in Python.

    #  The only reason that rho is set equal to 100 here is to make
    #  sure that the first IF statement in the WHILE loop will be
    #  executed for the first iteration (any value greater than
    #  0.0001 will do.)
    #
    rho = 100.0

    #  A counter for computationally intensive iteration, when a
    #  new jacobian and its QR decomposition are computed.
    #
    nJacobQR = 0

    #  <<<<<<<<<<<< IMPLEMENTATION OF ALGORITHM 7.1, MORE >>>>>>>>>>>>
    #
    # AT_LINE 589 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

    # OPTIMIZATION_LOOP_BEGIN: Inside this loop (iteration), these variables are updated: p_vector,v_vector,x_vector,o_x_vector
    while stopCode == 0:
        #  Update the loop counter
        #
        iterNum = iterNum + 1

        # logger.debug('Iteration: %d, Max: %d' % (iterNum, maxIter))

        #  If rho is greater than 0.0001, then we know that the step p
        #  computed in the previous iteration was accepted, and the
        #  new x is equal to the previous x + p.
        #
        #  The variables that are updated in this IF statement are
        #  those that
        #
        #     - directly or indirectly depend on x (are functions of x)
        #
        #     - and are not used to decide whether to repeat the loop
        #       or not.
        #
        # AT_LINE 609 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        if rho > 0.0001:
            nJacobQR = nJacobQR + 1

            #  Anything "next" in the previous iteration becomes the
            #  current one in this iteration.
            #
            residual = np.copy(residualNext)
            resNorm2 = np.copy(resNextNorm2)

            #  <<<<<<<<<<<<<<<<<<<< QR DECOMPOSITION >>>>>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<<<<<< AND RELATED COMPUTAIONS >>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<<<<<<<<<<< SECTION 3, 3.3 >>>>>>>>>>>>>>>>>>>>
            #
            #  Perform a rank revealing QR decomposition of the
            #  jacobian.  Jacob was updated at the end of the
            #  previous iteration (if rho is greater than 0.0001).
            #
            # AT_LINE 629 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

            # Because the following function RankRevealingQR will mess with jacobColumnL2Norm2, we send in a copy only.
            (R, pivots, rank) = muses_py.rank_revealing_qr(
                jacobian_ret,
                columnL2Norm2=np.copy(jacobColumnL2Norm2),
                epsilon=singTolerance,
            )

            # AT_LINE 636 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            #  Multiply Transpose(residual), a column vector, by Q from
            #  the left.  QResidual is Qf in More's notation, and it is
            #  a column vector with nPoints rows.

            QResidual = muses_py.rrqr_q_mult_a(residual.T, R, rank)

            RN = np.zeros(shape=(nTerms, nTerms), dtype=np.float64)
            RN[0:nTerms, 0:rank] = (muses_py.rrqr_get_rn(R, rank))[:, :]
            T = RN[0:rank, 0:rank]

            #
            #  <<<<<<<<<<<<<<<<< END: QR DECOMPOSITION >>>>>>>>>>>>>>>>>

            #  <<<<<<<<<<<<<<<<<< UPDATE D (scaleDiag) >>>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<<<<<<<< STEP e, ALGORITH 7.1 >>>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<< BASED ON SECTION 6, 6.1 AND 6.3 >>>>>>>>>>>>
            #
            #  Update the elements of scaleDiag if needed.  More has
            #  suggested three ways for updating scaleDiag (D): initial,
            #  adaptive, and continuous.  The adaptive method is used
            #  here, and based on his experience this method is better
            #  than the other two.
            #
            #  If newton keyword is set, then just Newton's method
            #  is used, and there is no need to compute or update
            #  scaleDiag.
            #
            # AT_LINE 658 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            if not newtonFlag:
                temp = np.sqrt(jacobColumnL2Norm2)
                result = np.where(scaleDiag < temp)[0]
                if len(result) > 0:
                    scaleDiag[result] = temp[result]

            #  <<<<<<<<<<<<<<<<<<<<<< END: UPDATE D >>>>>>>>>>>>>>>>>>>>

            #  <<<<<<<<<<<<<<<<<<<<<< NEWTON STEP>>>>>>>>>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<<<< PART OF STEP a, ALGORITH 7.1 >>>>>>>>>>>>>
            #  <<<<<< SOLVE FOR p WHEN LEV. MAR. PARAMETER IS ZERO >>>>>
            #  <<<<<<<<<<<<<<<<<<<<<<< SECTION 3 >>>>>>>>>>>>>>>>>>>>>>>
            #
            #  Compute
            #     - pNewton: Newton step, a row vector with nTerms
            #       elements.
            #     - qNewton: Newton scaled step, a row vector with
            #       nTerms elements.
            #     - qNormNewton: the L2 norm of the Newton scaled step
            #       (step size, a scalar value).
            #
            # AT_LINE 681 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

            pNewton = np.zeros(shape=(nTerms), dtype=np.float64)
            pNewton[0:rank] = -(QResidual[0, 0:rank] @ np.linalg.inv(T))

            # AT_LINE 683 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            (nPermutations, pNewton) = muses_py.column_permute_undo(
                pNewton, pivots[0:rank]
            )

            if not newtonFlag:
                qNewton = scaleDiag * pNewton
                qNormNewton = np.linalg.norm(qNewton)
            else:
                qNormNewton = 0.0

            #
            #  <<<<<<<<<<<<<<<<<<<<<< END: NEWTON >>>>>>>>>>>>>>>>>>>>>>
        # end of if (rho > 0.0001):

        # AT_LINE 697 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

        p_vector = np.copy(pNewton)

        if not newtonFlag:
            q_vector = np.copy(qNewton)
        qNorm = qNormNewton

        if verbose:
            logger.info("***************************************************")
            logger.info(f"Start iteration# = {iterNum}")
            if not newtonFlag:
                logger.info(f"Adaptive D       = {scaleDiag}")
            logger.info(f"At x             = {o_x_vector}")
            logger.info(f"rank of jacobian = {rank}")

            nTemp = min(nTerms, nPoints)
            temp = np.zeros(shape=(nTemp), dtype=np.float64)
            for i in range(nTemp):
                temp[i] = R[i, i]
            logger.info(f"eigenvalues of R = {temp}")
            logger.info(f"Newton step p    = {p_vector}")
            logger.info("")
        # end if verbose:

        if logWrite:
            with open(logFile, "a") as f:
                print("***************************************************", file=f)
                print(f"Start iteration# = {iterNum}", file=f)
                if uip is not None:
                    print(
                        f"Species list = {np.asarray(uip.speciesList)}", file=f
                    )  # use asarray to wrap this list more nicely
                if not newtonFlag:
                    print(f"Adaptive D       = {scaleDiag}", file=f)
                print(f"At x             = {o_x_vector}", file=f)
                print(f"rank of jacobian = {rank}", file=f)

                nTemp = min(nTerms, nPoints)
                temp = np.zeros(shape=(nTemp), dtype=np.float64)
                for i in range(nTemp):
                    temp[i] = R[i, i]
                print(f"eigenvalues of R = {temp}", file=f)
                print(f"Newton step p    = {p_vector}", file=f)
                print("", file=f)
        # end if logWrite:

        #  <<<<<<<<<<<<< ACCEPT NEWTON STEP OR FIND A NEW p >>>>>>>>>>>
        #  <<<<<<<<<<<< WHEN LEV. MAR. PARAMETER IS NOT ZERO >>>>>>>>>>
        #  <<<<<<<<<<<<<<<<<< STEP a, ALGORITH 7.1 >>>>>>>>>>>>>>>>>>
        #
        #  If the size of the Newton step is not too large, then accept
        #  that step and set LevMar parameter (lambda) equal to zero,
        #  else compute another p using LevMar method.
        #
        # AT_LINE 742 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        if qNormNewton <= (1 + sigma) * delta_value:
            lambda_value = 0.0  # The word 'lambda' is a reserved keyword in Python.
        else:  # end part of if (qNormNewton <= (1+sigma)*delta_value):
            #  Compute jacobTilde (More, 5.3)
            #
            # AT_LINE 751 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            jacobTilde = np.copy(jacobian_ret)
            for ii in range(nTerms):
                jacobTilde[ii, :] = jacobTilde[ii, :] / scaleDiag[ii]

            #  <<<< FIND INITIAL UPPER AND LOWER BOUNDS FOR LAMBDA >>>>>
            #  <<<<<<<<<<< INITIALIZATION FOR ALGORITHM 5.5 >>>>>>>>>>>>
            #
            #  In order to find a "correct" value for lambda, LevMar
            #  parameter, we need to apply the algorithm 5.5 in More's
            #  paper; however, first we need to find initial values for
            #  the upper bound and the lower bound for the possible
            #  values of lambda.

            #  Find the initial upper bound, u0, (More, section 5)
            #
            # AT_LINE 767 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            upper_bound = np.linalg.norm(jacobTilde @ residual) / delta_value

            #  Find the initial lower bound, l0, (More, section 5).
            #  There are two cases for the initial lower bound value,
            #  depending on whether the current jacob is rank deficient
            #  or not.
            #
            if rank < nTerms:
                lower_bound = 0.0
            else:
                #  This qNorm is qNormNewton, which is used here.  This
                #  is one part that is different from More's algorithm.
                #  There is no need to use SVD in order to compute some
                #  value that already exists.
                #

                # AT_LINE 785 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                phi0 = qNorm - delta_value

                #  This code segment is an implementation of the formula
                #  at the end of the section 5 of More's paper when
                #  alpha is zero.  Here we need q and qNorm computed when
                #  alpha (lambda) is equal to zero, and they already
                #  exist.  These values are qNewton and qNormNewton, and
                #  the current values of q and qNorm, at this point, are
                #  equal to the values of qNewton and qNormNewton
                #  respectively.
                #
                # AT_LINE 796 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                temp = q_vector * scaleDiag
                (nPermutations, temp) = muses_py.column_permute(temp, pivots)

                phi0Prime = -qNorm * np.sum((np.linalg.inv(RN) @ (temp / qNorm)) ** 2)

                lower_bound = -phi0 / phi0Prime
            # end part of if (rank < nTerms):
            #
            #  <<<<<<<< END: INITIALIZE UPPER AND LOWER BOUNDS >>>>>>>>>

            #  These two statements compute scaleDiagPi which in More's
            #  notation (section 3 of his paper) is equal to
            #
            #     (Greek letter Pi)^T##D##(Greek letter Pi)
            #
            #  The final outcome is D with its diagonal entries, not
            #  its columns or rows, permuted according to pivots.
            #
            #
            # AT_LINE 816 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            scaleDiagPi = np.copy(scaleDiag)
            (nPermutations, scaleDiagPi) = muses_py.column_permute(
                scaleDiagPi, pivots[0:rank]
            )

            #  Set lambda equal to any out of range value so that the IF
            #  statement in the REPEAT/UNTIL loop will get executed when
            #  the loop is entered for the first time.  A negative value
            #  is the best choice.
            #
            # AT_LINE 825 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            lambda_value = -100.0  # Note: the keyword 'lambda' belongs to Python.

            #  A counter for the number of the iterations of the inner
            #  loop (ALGORITHM 5.5, More).
            #
            # AT_LINE 831 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            innerIter = 0

            # AT_LINE 839 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            stopThisLoop = False
            while not stopThisLoop:
                innerIter = innerIter + 1

                #  <<<<<<<<<<<<<<< STEP a, ALGORITHM 5.5 >>>>>>>>>>>>>>>>
                #
                #  Make sure that lambda is within the range.  If not,
                #  then update it.
                #
                # AT_LINE 849 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                if (lambda_value < lower_bound) or (lambda_value > upper_bound):
                    if 0.001 * upper_bound > np.sqrt(lower_bound * upper_bound):
                        lambda_value = 0.001 * upper_bound
                    else:
                        lambda_value = np.sqrt(lower_bound * upper_bound)

                #
                # <<<<<<<<<<<<< END: STEP a, ALGORITHM 5.5 >>>>>>>>>>>>>>

                # If jacobian is rank deficient and lambda is "close" to
                # zero, i.e. lambda is theoretically equal to zero, then
                # 2.4 in More's paper is defined by the limiting process
                # that follows 2.4.
                #
                # Practically this case happens not just when lambda
                # is equal to zero, but also when Abs(lambda) is
                # "very small".
                #
                # We cannot just assume that lambda is "close" to zero
                # when Abs(lambda) is less than some tolerance.  Consider
                # the matrix on the right side of equation 3.4 and the
                # one on the right side of 3.5, after Givens rotations.
                # In More's paper, R becomes R-sub-lambda.  If
                # Abs(lambda) is less than some tolerance, the nonzero
                # entries of the upper triangular R may also be small
                # and close to the tolerance (we are working with values
                # that are small in magnitude,) then relatively speeking
                # Abs(lambda) is not small.
                #
                # To get around this problem, perform the Givens
                # rotations and try to claculate R-sub-lambda (3.5) even
                # if lambda seems to be "close" to zero.  If R-sub-lambda
                # is computed successfully, then it is an upper
                # triangular matrix; therefore, the diagonal entries of
                # R-sub-lambda are the eigenvalues of R-sub-lambda.
                #
                # HOWEVER, while performing Givens rotations and just
                # before starting to perform rotations on a given column,
                # it is possible to compute the Abs(would-be-eigenvalue)
                # in that column.  If the absolute value of this
                # eigenvalue is very small relative to the max of others
                # computed so far, then stop because would-be
                # R-sub-lambda is singular (or nearly singular).
                #
                # Assume e1, e2, ... en are the eigenvalues of
                # R-sub-lambda.  The ratio
                #
                #    max{|ei|}/min{|ei|}
                #
                # is very large (say greater than or equal to 1E7) when
                # R-sub-lambda is singular (or almost singular), which
                # happens when R is rank deficient and Abs(lambda) is
                # "really small".
                #
                # NOTE: this ratio is not the condition number of
                # R-sub-lambda.  This ratio for a square matrix A with
                # eigenvalues e1, e2, ... is the condition number if A
                # is symmetric.

                #  <<<<<<<<<<<< COMPUTE R-sub-lambda AND u >>>>>>>>>>>>>>
                #  <<<<<<<<<<<<< SECOND HALF OF SECTION 3 >>>>>>>>>>>>>>>
                #
                #  In this code segment Givens rotations are used to
                #  compute the QR decomposition of the matrix on the
                #  right side of 3.4 (More).  There is no need to create
                #  that matrix because all enteries below the first
                #  nTerms rows and above the last nTerms rows are zero
                #  therefore, the algorithm used here is not a general QR
                #  decomposition for any matrix.
                #
                #  tempRjY is a matrix that is equal to RN augmented by
                #  the first nTerms rows of the column vector QResidual
                #  (Qf in More's notation).  In this way after Givens
                #  rotations, QResidual is automatically (and implicitly)
                #  multiplied by the matrix w in 3.5.  This
                #  multiplication is in the last equation of section 3
                #  (More).  Because of the shape of the matrices, the
                #  multiplication only changes the first nTerms rows of
                #  QResidual.
                #
                #  After the Givens rotations the last column of tempRjY,
                #  which used to be the first nTerms rows of QResidual,
                #  is converted to the column vector that More calls it
                #  u at the end of section 3.  The first nTerms columns
                #  of tempRjY become the matrix that in More's
                #  notation is refered to as R with the Greek letter
                #  lambda as its subscript (3.5).
                #
                # At this point, we have already allocated the two matricies: tempRjY, tempDiag.  We just need to fill them with zeros.
                # AT_LINE 935 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                tempRjY = np.zeros(shape=(nTerms + 1, nTerms), dtype=np.float64)
                tempRjY[0:nTerms, 0:nTerms] = RN[:, :]

                if nTerms < nPoints:
                    tempRjY[nTerms, 0:nTerms] = QResidual[0, 0:nTerms]
                else:
                    tempRjY[nTerms, 0:nPoints] = QResidual[0, 0:nPoints]

                # AT_LINE 942 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                tempDiag = np.zeros(shape=(nTerms + 1, nTerms), dtype=np.float64)
                lambda_sqrt = np.sqrt(lambda_value)
                for ii in range(nTerms):
                    tempDiag[ii, ii] = scaleDiagPi[ii] * lambda_sqrt

                maxAbsEigVal = np.sqrt(tempRjY[0, 0] ** 2 + tempDiag[0, 0] ** 2)

                # maxAbsEigVal Matches with IDL on 10/08/2018: IDL_OUTPUT: 1654.1574 PYTHON_OUTPUT: 1654.159821244652

                singular_value = False
                ii = 0

                #  Use Givens rotations to compute R-sub-lambda.  Loop
                #  ends either when all rotations are performed or when
                #  it is realized that R-sub-lambda will be singular.
                #
                # AT_LINE 954 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                while (ii < nTerms) and (not singular_value):
                    # Note that we use "ii+1" for the slice of tempDiag in the next line.  Remember that IDL does include
                    # the end of the slice, whereas Python does not.
                    curAbsEigVal = np.sqrt(
                        tempRjY[ii, ii] ** 2 + np.sum(tempDiag[ii, 0 : ii + 1] ** 2)
                    )

                    # AT_LINE 959 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                    if curAbsEigVal / maxAbsEigVal >= singTolerance:
                        rev_loop_count = 0

                        # Note that for Python, we go from ii to 0 with negative step.  IDL goes from ii to 1 with negative step.
                        for jj in range(ii, 0, -1):
                            # AT_LINE 962 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                            (cs, sn) = muses_py.givens(
                                tempDiag[ii, jj - 1], tempDiag[ii, jj]
                            )
                            temp = np.copy(
                                tempDiag[ii:, jj - 1]
                            )  # We have to copy so the memory won't be overwritten.
                            tempDiag[ii:, jj - 1] = (cs * temp) - (
                                sn * tempDiag[ii:, jj]
                            )
                            tempDiag[ii:, jj] = (sn * temp) + (cs * tempDiag[ii:, jj])
                            rev_loop_count += 1
                        # end for jj in reversed(range(1,ii)):

                        # AT_LINE 968 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                        (cs, sn) = muses_py.givens(tempRjY[ii, ii], tempDiag[ii, 0])

                        temp = np.copy(
                            tempRjY[ii:, ii]
                        )  # We have to copy so the memory won't be overwritten.
                        tempRjY[ii:, ii] = (cs * temp) - (sn * tempDiag[ii:, 0])
                        tempDiag[ii:, 0] = (sn * temp) + (cs * tempDiag[ii:, 0])

                        #  Because a column pivoting QR decomposition is
                        #  performed on the jacobian and because of the
                        #  way scaleDiag is updated, I believe this IF
                        #  condition will always be false; however, this
                        #  IF statement will do no harm.
                        #
                        # AT_LINE 979 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                        if curAbsEigVal > maxAbsEigVal:
                            maxAbsEigVal = curAbsEigVal

                        # AT_LINE 982 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                        ii = ii + 1
                    else:
                        # AT_LINE 986 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                        singular_value = True

                # AT_LINE 988 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                # end while (ii < nTerms) and (not singular_value):

                # 10/09/2018: Matches with IDL

                # AT_LINE 990 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

                tempDiag.fill(0.0)

                #
                #  <<<<<<<<<< END: COMPUTE R-sub-lambda AND u >>>>>>>>>>>

                #  <<<<<<<<<<<<<<<< COMPUTE p, q, qNorm >>>>>>>>>>>>>>>>>
                #  <<<<<<<<<<<<<<<<<<< VERY IMPORTANT >>>>>>>>>>>>>>>>>>>
                #
                #  This is one part that is different from More's paper.
                #  Instead of computing qNorm using the singular value
                #  decomposition of jacobTilde, qNorm is computed using
                #  p.  This requires the computation of p for every
                #  lambda (alpha).
                #
                # AT_LINE 1004 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                if not singular_value:
                    #  Compute step p, scaled step q, and the L2 norm of
                    #  the scaled step qNorm.
                    #
                    # pass
                    RJInverse = np.linalg.inv(tempRjY[0:nTerms, 0:nTerms])

                    p_vector = -(tempRjY[nTerms, 0:nTerms] @ RJInverse).T

                    (nPermutations, p_vector) = muses_py.column_permute_undo(
                        p_vector, pivots[0:rank]
                    )

                    q_vector = scaleDiag * p_vector

                    qNorm = np.linalg.norm(q_vector)

                    stopThisLoop = (qNorm >= (1.0 - sigma) * delta_value) and (
                        qNorm <= (1.0 + sigma) * delta_value
                    )

                # AT_LINE 1017 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                else:
                    #  Compute step p, scaled step q, and the L2 norm of
                    #  the scaled step qNorm.
                    #
                    #  Even in this case it may be possible to avoid the
                    #  singular value decomposition of jacobTilde.
                    #

                    # see: https://github.jpl.nasa.gov/MUSES-Processing/idl-retrieve/blob/v1.3.0/Optimization/OPTIMIZATION/Levmar_NLLSQ.pro#L1017

                    tempMatrix = np.zeros(shape=(nTerms, 2 * nTerms), dtype=np.float64)
                    tempMatrix[0:nTerms, 0:nTerms] = RN[:, :]
                    for ii in range(nTerms):
                        tempMatrix[ii, ii + nTerms] = scaleDiagPi[ii] * lambda_sqrt

                    (Rj, pivotsj, rankj) = muses_py.rank_revealing_qr(
                        tempMatrix, None, epsilon=singTolerance
                    )

                    tempMatrix.fill(0.0)

                    Tj = (muses_py.rrqr_get_rn(Rj, rankj))[0:rankj, 0:rankj]

                    temp = np.zeros(shape=(1, 2 * nTerms), dtype=np.float64)
                    if nTerms < nPoints:
                        temp[0, 0:nTerms] = QResidual[0, 0:nTerms]
                    else:
                        temp[0, 0:nTerms] = QResidual[0, 0:nPoints]

                    # TODO: Revisit. This is likely a bug in the IDL code
                    # uj = (rrqr_q_mult_a(temp, Rj, rankj))[0, 0:nTerms]
                    uj = muses_py.rrqr_q_mult_a(temp, Rj, rankj)

                    p_vector = np.zeros(shape=(nTerms), dtype=np.float64)

                    TjInverse = np.linalg.inv(Tj)
                    p_vector[0:rankj] = -(uj[0, 0:rankj] @ TjInverse)

                    (nPermutations, p_vector) = muses_py.column_permute_undo(
                        p_vector, pivots[0:rankj]
                    )
                    (nPermutations, p_vector) = muses_py.column_permute_undo(
                        p_vector, pivots[0:rank]
                    )

                    q_vector = scaleDiag * p_vector

                    qNorm = np.linalg.norm(q_vector)

                    stopThisLoop = True

                # AT_LINE 1056 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

                tempRjY.fill(0.0)

                #
                #  <<<<<<<<<<<<<<<< COMPUTE p, q, qNorm >>>>>>>>>>>>>>>>>

                #  If needed, update lower_bound, lambda_value, and upper_bound for the
                #  next iteration of this loop.
                #
                # AT_LINE 1064 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                if not stopThisLoop:
                    #  <<<<<<<<<<<<<< STEP b, ALGORITHM 5.5 >>>>>>>>>>>>>>
                    #
                    # AT_LINE 1068 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                    phi = qNorm - delta_value
                    temp = q_vector * scaleDiag

                    (nPermutations, temp) = muses_py.column_permute(temp, pivots)

                    phiPrime = -qNorm * np.sum((RJInverse @ (temp / qNorm)) ** 2)

                    # AT_LINE 1073 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                    if phi < 0:
                        upper_bound = lambda_value

                    # AT_LINE 1076 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                    if lower_bound > (lambda_value - phi / phiPrime):
                        # Do nothing since lower_bound is already greater than
                        pass
                    else:
                        lower_bound = lambda_value - phi / phiPrime
                    #
                    #  <<<<<<<<<<< END: STEP b, ALGORITHM 5.5 >>>>>>>>>>>>

                    #  <<<<<<<<<<<<<< STEP c, ALGORITHM 5.5 >>>>>>>>>>>>>>
                    #  <<<<<<<<<< 5.4 IS USED TO COMPUTE lambda >>>>>>>>>>
                    #
                    # AT_LINE 1084 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                    lambda_value = lambda_value - (
                        (phi + delta_value) / delta_value
                    ) * (phi / phiPrime)
                    #
                    #  <<<<<<<<<<< END: STEP c, ALGORITHM 5.5 >>>>>>>>>>>>
                # end if (not stopThisLoop):

                # for some reason it gets hung up in this loop for OMI
                # cloud fraction
                # AT_LINE 1092 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
                if innerIter > 1000:
                    stopThisLoop = True

            # AT_LINE 1094 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            # end while (not stopThisLoop)

            # AT_LINE 1096 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            scaleDiagPi.fill(0.0)
            #
            #  <<<<<<< END: IMPLEMENTATION OF ALGORITHM 5.5, MORE >>>>>>

        # AT_LINE 1101 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        # end else part of if (qNormNewton <= (1 + sigma) * delta_value):

        # UPDATE UIP BASED ON NEW RETRIEVAL VECTOR
        # make sure that updated vector, v, is not negative, if it is
        # linear.  If it is negative, modify p.

        # AT_LINE 1111 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        v_vector = o_x_vector + p_vector

        (uip, v_vector) = muses_py.update_uip(uip, ret_info, v_vector)

        p_vector = v_vector - o_x_vector

        # AT_LINE 1118 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        (
            uip,
            residualNext,
            jacobNext,
            o_radiance_fm_next,
            jacobian_fm_next,
            stop_flag,
        ) = muses_py.residual_fm_jacobian(uip, ret_info, v_vector, iterNum)

        if stop_flag:
            logger.info(function_name, "residual_fm_jacobian failed!")
            o_success_flag = 0
            break

        # AT_LINE 1121 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        resNextNorm2 = np.sum(residualNext**2)
        jacobPNorm2 = np.sum(
            (p_vector.T @ jacobian_ret) ** 2
        )  # jacobian_ret is IDL jacob
        jacResNextNorm2 = np.sum((jacobNext @ residualNext) ** 2)
        jacResNorm2 = np.sum(
            (jacobian_ret @ residual) ** 2
        )  # jacobian_ret is IDL jacob
        xNorm = np.sum(o_x_vector**2)
        pNorm = np.sum(p_vector**2)

        # AT_LINE 1128 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        o_x_iter[iterNum, :] = o_x_vector + p_vector

        o_res_iter[iterNum, :] = residualNext[:]
        o_radiance_iter[iterNum, :] = o_radiance_fm_next[:]

        #  <<<<<<<<<<<<<<<<<<<<<<< COMPUTE rho >>>>>>>>>>>>>>>>>>>>>>>>
        #  <<<<<<<<<<<<<<<<<< STEP b, ALGORITH 7.1 >>>>>>>>>>>>>>>>>>>>
        #
        #  Compute the ratio between the actual reduction and the
        #  predicted reduction (here a linear prediction is used.
        #  (4.1 More)
        #
        #  Before computing rho: if newton keyword is set, then only
        #  Newton's method is used for optimization.  This means that
        #  the step must be accepted anyway; therefore, set rho equal
        #  to some value greater than 0.0001D.
        #

        if newtonFlag:
            rho = 100.0
        elif resNextNorm2 > resNorm2:
            rho = 0.0
        else:
            rho = (1.0 - resNextNorm2 / resNorm2) / (
                jacobPNorm2 / resNorm2 + 2.0 * (lambda_value * (qNorm**2)) / resNorm2
            )

        #
        #  <<<<<<<<<<<<<<<< END: STEP b, ALGORITHM 7.1 >>>>>>>>>>>>>>>>

        # AT_LINE 1156 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        #  <<<<<<<<< UPDATE x AND jacobian IF p IS ACCEPTED >>>>>>>>>>>
        #  <<<<<<<<<<<<<<<<<< STEP C, ALGORITH 7.1 >>>>>>>>>>>>>>>>>>>>
        #
        #  If rho is greater than 0.0001D, then the step p is accepted.
        #
        #  Calculate the thresholds for stopping criteria
        pThresh = pNorm / (1.0 + xNorm)
        costThresh = np.abs(resNorm2 - resNextNorm2) / (1.0 + resNextNorm2)
        JacThresh = np.linalg.norm(jacobian_ret @ residualNext) / (1.0 + resNextNorm2)

        # see: https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf, 4.1.3 Convergence criteria
        # Large values, χ2ν ≫ 1, indicate a poor fit, χ2ν ≈ 1 indicates that the fit error is of the same order as the measurement error (as desired), and
        # χ2ν < 1 indicates that the model is over-fitting the data; that is, the model is fitting the measurement noise.
        # χ2ν = χ2 / (m - n + 1), where m  = # of observations, n = # of parameters
        chi2 = resNextNorm2
        dof = o_radiance_fm_next.shape[0] - p_vector.shape[0] + 1
        chi2_reduced = (chi2 / dof) if dof > 0 else chi2

        if rho > 0.0001:
            o_x_vector = o_x_vector + p_vector
            o_jacobian_fm = np.copy(jacobian_fm_next)
            jacobian_ret = np.copy(jacobNext)

            if (stopCode == 0) and (resNextNorm2 < 1.0 - Chi2Tolerance):
                # model is over-fitting the data; that is, the model is fitting the measurement noise.
                stopCode = 2

            if (stopCode == 0) and (
                (pThresh < ConvTolerance[1])
                and (JacThresh < ConvTolerance[2])
                and (costThresh < ConvTolerance[0])
            ):
                stopCode = 3

            # VK: Add another convergence criteria for χ2 usingthe the reduced χ2, χ2ν = χ2 / (m − n + 1) < ϵ3 (quality of the fit).
            # see: https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf, 4.1.3 Convergence criteria

            # TODO: Make epsilon3 configurable per step in the Strategy table.
            # 1.1 is too low. With the current FM / RTs (OSS or VLIDORT) we rarely get χ2ν (chi2_reduced) lower than 2.0.
            epsilon3 = 1.1
            if (stopCode == 0) and (chi2_reduced >= 1) and (chi2_reduced < epsilon3):
                stopCode = 4
        # end if (rho > 0.0001):

        #
        #  <<<<<<<<<<<<<<<< END: STEP c, ALGORITHM 7.1 >>>>>>>>>>>>>>>>
        # AT_LINE 1180 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        #  If the loop has repeated maxIter times then give up.
        #
        if (stopCode == 0) and (iterNum >= maxIter):
            stopCode = 1

        # Save off stopping criterion
        # AT_LINE 1187 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

        o_stopcrit[iterNum, 0] = costThresh
        o_stopcrit[iterNum, 1] = pThresh
        o_stopcrit[iterNum, 2] = JacThresh

        # Save residual diagnostics
        # AT_LINE 1193 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

        o_resdiag[iterNum, 0] = np.sqrt(resNorm2) / np.sqrt(nPoints)
        o_resdiag[iterNum, 1] = np.sqrt(resNextNorm2) / np.sqrt(nPoints)
        o_resdiag[iterNum, 2] = np.sqrt(jacResNorm2) / np.sqrt(nPoints)
        o_resdiag[iterNum, 3] = np.sqrt(jacResNextNorm2) / np.sqrt(nPoints)
        o_resdiag[iterNum, 4] = pNorm / np.sqrt(nPoints)

        # Save linearity measures
        # AT_LINE 1200 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

        o_Diag_lambda_rho_delta[iterNum, 0] = lambda_value
        o_Diag_lambda_rho_delta[iterNum, 1] = rho
        o_Diag_lambda_rho_delta[iterNum, 2] = delta_value

        if logWrite:
            with open(logFile, "a") as f:
                print("jacobian_ret size", file=f)
                print(
                    f"{len(jacobian_ret.shape)}   {jacobian_ret.shape}   {jacobian_ret.dtype}   {jacobian_ret.size}",
                    file=f,
                )

                # print('jacobian_ret', file=f)
                # print(f'{jacobian_ret}', file=f)

                # print('residualNext', file=f)
                # print(f'{residualNext}', file=f)

                # print('resNextNorm2', file=f)
                # print(f'{resNextNorm2}', file=f)

                print("", file=f)

                print("Convergence Criteria:", file=f)

                print(
                    "costThresh = np.abs(resNorm2 - resNextNorm2) / (1 + resNextNorm2)",
                    file=f,
                )
                print(f"{costThresh}", file=f)

                print("tolerance = ConvTolerance[0]", file=f)
                print(f"{ConvTolerance[0]}", file=f)

                print("pThresh = pNorm / (1 + xNorm)", file=f)
                print(f"{pThresh}", file=f)

                print("tolerance = ConvTolerance[1]", file=f)
                print(f"{ConvTolerance[1]}", file=f)

                print(
                    "JacThresh = np.linalg.norm(jacobian_ret @ residualNext) / (1 + resNextNorm2)",
                    file=f,
                )
                print(f"{JacThresh}", file=f)

                print("tolerance = ConvTolerance[2]", file=f)
                print(f"{ConvTolerance[2]}", file=f)

                print("", file=f)

                if not newtonFlag:
                    if qNormNewton <= (1 + sigma) * delta_value:
                        print("Newton step is chosen.", file=f)
                        print("", file=f)
                        print("qNorm    (1 + sigma) * delta", file=f)
                        print(f"{qNorm}, {(1 + sigma) * delta_value}", file=f)
                    else:
                        print("Newton step is not good; LevMar step is chosen", file=f)
                        print(
                            f"after the inner loop repeated {innerIter} times.", file=f
                        )

                        if singular_value:
                            print("Rj became singular.", file=f)

                        print(f"LevMar step p = {p_vector}", file=f)

                        print("", file=f)

                        print(
                            "(1 - sigma) * delta   qNorm   (1 + sigma) * delta", file=f
                        )
                        print(
                            f"{(1.0 - sigma) * delta_value} {qNorm} {(1.0 + sigma) * delta_value}",
                            file=f,
                        )
                    # end if not newtonFlag:

                    print("", file=f)

                    print(
                        "||F(x)|| / sqrt(m)  ||F(x+p)|| / sqrt(m)   ||J F(x)|| / sqrt(m)   ||J F(x + p)|| / sqrt(m)   ||p|| / sqrt(m)",
                        file=f,
                    )
                    print(
                        "o_resdiag[iterNum, 0]  o_resdiag[iterNum, 1]  o_resdiag[iterNum, 2] o_resdiag[iterNum, 3] o_resdiag[iterNum, 4]",
                        file=f,
                    )
                    print(
                        f"{o_resdiag[iterNum, 0]} {o_resdiag[iterNum, 1]} {o_resdiag[iterNum, 2]} {o_resdiag[iterNum, 3]} {o_resdiag[iterNum, 4]}",
                        file=f,
                    )

                    print("", file=f)

                    print("lambda   rho   delta", file=f)
                    print(f"{lambda_value}   {rho}   {delta_value}", file=f)

                    print("", file=f)

                    if rho > 0.0001:
                        print("The step was accepted, and", file=f)
                        print(f"x + p = {o_x_vector}", file=f)
                    else:
                        print("The step was rejected.", file=f)

                    print(
                        f"{resNextNorm2 = }, {resNorm2 = }, {jacobPNorm2 = }, {lambda_value = }, {qNorm = }",
                        file=f,
                    )
                    print("", file=f)
                else:
                    print("", file=f)

                    print(
                        "||F(x)|| / sqrt(m)    ||F(x+p)|| / sqrt(m)     ||J F(x)|| / sqrt(m)     ||J F(x + p)|| / sqrt(m)",
                        file=f,
                    )
                    print(
                        f"{o_resdiag[iterNum, 0]}   {o_resdiag[iterNum, 1]}   {o_resdiag[iterNum, 2]}   {o_resdiag[iterNum, 3]}",
                        file=f,
                    )

                    print("", file=f)
                    print("After step p", file=f)
                    print(f"x + p = {o_x_vector}", file=f)
                # end if not newtonFlag:
        # end if logWrite:

        # AT_LINE 1297 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
        #  Anything that must be updated at the end of the current
        #  iteration for the next one (if any), and the decision
        #  whether to go through another iteration or not does not
        #  depend on it, is updated in this if statement.
        #
        if stopCode == 0:
            #  Reminder: rho greater than 0.0001 means that the last
            #  step was accepted and we have a new jacob(ian).
            #
            # AT_LINE 1307 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor
            if rho > 0.0001:
                for ii in range(nTerms):
                    jacobColumnL2Norm2[ii] = np.sum(jacobian_ret[ii, :] ** 2)

            #  <<<<<<<<<<<<<<<<<<<<< UPDATE delta >>>>>>>>>>>>>>>>>>>>>>
            #  <<<<<<<<<<<<<<<<<< STEP d, ALGORITH 7.1 >>>>>>>>>>>>>>>>>
            #
            #  Before computing delta: if newton keyword is set, then
            #  only Newton's method is used for optimization, and there
            #  is no need to compute delta.
            #
            if newtonFlag:
                delta_value = 0.0
            elif rho <= 0.25:
                if resNextNorm2 <= resNorm2:
                    tempConst = 0.5
                elif resNextNorm2 > 100.0 * resNorm2:
                    tempConst = 0.1
                else:
                    tempConst = -(
                        jacobPNorm2 / resNorm2 + lambda_value * (qNorm**2) / resNorm2
                    )
                    tempConst = (tempConst / 2.0) / (
                        tempConst + (1.0 - resNextNorm2 / resNorm2) / 2.0
                    )

                    #  If tempConst is not in the closed interval [0.1, 0.5],
                    #  then set it equal to the closest end point, 0.1 or 0.5
                    #
                    if tempConst < 0.1:
                        tempConst = 0.1
                    elif tempConst > 0.5:
                        tempConst = 0.5
                # end else

                delta_value = tempConst * delta_value
                if (rho <= 0.0001) and (qNorm <= (1.0 + sigma) * delta_value):
                    delta_value = 0.75 * (qNorm / (1 + sigma))

            elif (rho >= 0.75) or (
                (rho > 0.25) and (rho < 0.75) and (lambda_value == 0.0)
            ):
                delta_value = 2.0 * qNorm
            # end elif
            #
            #  <<<<<<<<<<<<<<< END: STEP d, ALGORITHM 7.1 >>>>>>>>>>>>>>
        # end if (stopCode == 0):

        # Useful diagnostic while looking at muses-py/ReFRACtor. We
        # can take these out by changing True to False, but since
        # these were useful initially probably worth leaving in.
        if verbose:
            logger.info(f"rho: {rho}")
            logger.info(f"stopCode: {stopCode}")
            logger.info(f"pThresh: {pThresh}")
            logger.info(f"costThresh: {costThresh}")
            logger.info(f"JacThresh: {JacThresh}")
            logger.info(f"Chi2: {resNextNorm2}")
            logger.info(f"resNextNorm2: {resNextNorm2}")
            logger.info(f"resNorm2: {resNorm2}")

        # TODO: print only in verbose mode
        chi2 = resNextNorm2
        dof = o_radiance_fm_next.shape[0] - p_vector.shape[0] + 1
        chi2_reduced = (chi2 / dof) if dof > 0 else chi2

        logger.info(
            f"iter: {iterNum:02}, max: {maxIter:02}, rho = {rho:10.5f}, chi2_reduced = {chi2_reduced:10.5f}, chi2 = {chi2:10.5f}"
        )
    # end while (stopCode == 0):

    # OPTIMIZATION_LOOP_END

    # AT_LINE 1356 Optimization/OPTIMIZATION/Levmar_NLLSQ.pro LevMar_NLLSq_Elanor

    #  Reminder: rho greater than 0.0001 means that the last
    #  step was accepted.
    #
    #  After exiting the loop, if the last step p was accepted,
    #  then residual and resNorm2 must be updated.
    #
    if rho > 0.0001:
        residual = np.copy(residualNext)
        resNorm2 = resNextNorm2

    # This is odd: Not sure why residual and resNorm2 were calculated above.  Nobody uses them after the assignment.

    # Updates the uip so that the atmospheric parameters are consistent with the last accepted state vector.
    (uip, o_x_vector) = muses_py.update_uip(uip, ret_info, o_x_vector)

    # PYTHON_NOTE: In IDL, there is the radiance_fm = radiance_fm_next in the function signature.

    return (
        o_x_vector,
        o_Diag_lambda_rho_delta,
        o_stopcrit,
        o_resdiag,
        o_x_iter,
        o_res_iter,
        o_radiance_fm_next,
        o_radiance_iter,
        o_jacobian_fm,
        iterNum,
        stopCode,
        o_success_flag,
    )


__all__ = ["SolverResult", "MusesLevmarSolver"]
