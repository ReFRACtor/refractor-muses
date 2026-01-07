from __future__ import annotations
from .cost_function import CostFunction
from .identifier import ProcessLocation
import numpy as np
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

class VerboseLogging:
    '''Observer of MusesLevmarSolver that adds some more verbose logging.'''
    def notify_update(self, slv: MusesLevmarSolver, location: ProcessLocation, **kwargs : Any) -> None:
        if location == ProcessLocation("start iteration"):
            self.log_start_iteration(slv)
        elif location == ProcessLocation("end iteration"):
            self.log_end_iteration(slv)
            
    def log_start_iteration(self, slv: MusesLevmarSolver) -> None:
        pass

    def log_end_iteration(self, slv: MusesLevmarSolver) -> None:
        pass

class SolverLogFile:
    '''Observer of MusesLevmarSolver that write information to a separate log file.'''
    def __init__(self, log_file: str | os.PathLike[str]) -> None:
        self.fname = Path(log_file)
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.fname, "w")
        
    def notify_update(self, slv: MusesLevmarSolver, location: ProcessLocation, **kwargs : Any) -> None:
        if location == ProcessLocation("start iteration"):
            self.log_start_iteration(slv)
        elif location == ProcessLocation("end step"):
            self.log_end_step(slv)
            
    def log_start_iteration(self, slv: MusesLevmarSolver) -> None:
        pass

    def log_end_step(self, slv: MusesLevmarSolver) -> None:
        pass
    

class MusesLevmarSolver:
    """This is a wrapper around levmar_nllsq_elanor that makes it look like
    a NLLSSolver. Right now we don't actually derive from that, we can perhaps
    put that in place if useful. But for now, just provide a "solve" function.

    We set up self.cfunc.parameters to whatever the final best
    iteration solution is.

    The solver was written by Edwin, and has some extensive
    documentation. This is for the old IDL code, so stuff needs to be
    mapped a bit to the python class. But this is still pretty
    relevant so we have copied this over.
    ------------------------------------------------------------------------------
    ROUTINE NAME: LevMar_NLLSq

    FILE NAME:
       levmar_nllsq.pro

    ABSTRACT:
       If F(x) is continiously differentiable and F:R^n->R^m, then
       this function could be used to minimize (1/2)*||F(x)||^2.

    AUTHOR:
       Edwin Sarkissian, JPL, Nov. 1998

    SOURCE:
       More, Jorge J. (Proc. Biennial Conference, Dundee 1977), "The
       Levenberg-Marquardt Algorithm: Implementation and Theory"
       see: Page 3,  https://www.osti.gov/servlets/purl/7256021

    DESCRIPTION:
       If F(x) is continiously differentiable and F:R^n->R^m, then
       this function could be used to minimize (1/2)*||F(x)||^2,
       where ||.||^2 is L2 norm raised to power 2,

       This technique is a nonlinear least-squares optimization
       applied to a function that maps from R^n to R^m where n is
       the number of the parameters (the size of the vector x,
       nTerms in this IDL code), and m is the dimension of the vector
       returned by F (the size of the vector F(x), nPoints in this
       IDL code).

       The code in this function is optimized based on the assumption
       that m is much larger than n, and this method works well if
       m >= n.  If m < n, then the performance of this function gets
       worse as the ratio m/n -> 0.

       The search for a minimizer is terminated if one of the
       following five conditions becomes true


          - The first condition (condition # 1) is

            The maximum number of iterations has been reached

          - The second conditon (condition # 2) is

               resNextNorm2 LT 1-Chi2Tolerance

            If the Norm of the residual drops significantly below 1,
            then the minimizer is "fitting noise"

          - The third conditon (condition # 3) is

               (pThresh LT ConvTolerance[1])) AND
                 (JacThresh LT ConvTolerance[2]) AND
                 (costThresh LT ConvTolerance[0]))

            These convergence criteria check whether the change is
            step, p, the change in the cost function, and the
            derivative of the cost function have dropped below
            specified thresholds

            Ideally the search should end when these condition becomes
            true.


       The algorithm used here is based on Jorge J. More's paper
       (check SOURCE) with some differences.  His algorithm requires
       one singular value decomposition of the jacobian (if changed)
       for every iteration of the main loop.  This implementation
       does not use singular value decomposition even when the
       jacobian is rank deficient and lambda (the Lev-Mar parameter is
       very small.

       The second difference is in the implementation of the algorithm
       5.5.  This algorithm only finds the lambda, then p is computed
       based on this lambda; however, in this implementation, because
       there is no singular value decomposition, it is necessary to
       compute p for every value of lambda (alpha).  On average
       algorithm 5.5 (if needed) iterates 2 times; therefore,
       computing p for every iteration of 5.5 is far less expensive
       than computing the singular value decomposition of jacobian any
       time it changes.

       NOTATION: This section will help to relate this IDL code and
       its identifiers to More's paper.

          More                           IDL
          -------------------------      ----------------------------

          lower case Greek sigma         sigma

          lower case Greek lambda        lambda
          or alpha (LevMar param)

          p (step)                       p

          D (scale, a diag. matrix)      scaleDiag (just the diagonal)

          q (=Dp, the scaled step)       q

          ||q|| (=||Dp||)                qNorm

          upper case Greek delta         delta
          upper bound for ||q||


          F(x) or f                      residual

          F(x+p) or f+                   residualNext

          F'(x) or J                     jacob

          ||f||^2                        resNorm2

          ||f+||^2                       resNextNorm2

          J with tilde (=JD^-1)          jacobTilde

          J with bar (section 3          jacobBar (computed explicitly
          of More's paper)               only for testing, if needed)

          lower case Greek rho           rho
          (section 4 of More's
          paper)

          T (More, 3.3)                  T

          R                              RN (not R)

    REVISION HISTORY:
       11/23/98         Initial RCS version
       1/8/99           First version that I was happy with it
       7/26/05          KB-I've made numerous changes, the most recent
                        being the inclusion of new convergence criteria
      10/14/05          Changed bug so that tolerances written to log
                        file correspond to the correct labels

    RETURN:
       A double array of parameters (parameter row vector) where

          resNorm2 = ||F(x)||^2

       is minimized.  The size of the returned vector is the same as
       the size of the input vector x (initial guess).

    INPUT:
       func    (par/req) : The name (uppercase string) of a function
                           that takes a row vector (double 1-Dim
                           array) in R^n space and returns a row
                           vector (double 1-Dim array) in R^m space.

                           This function must be implemented in such a
                           way that its single argument and what it
                           returns are always one dimensional arrays
                           even if n or m are equal to 1, for example,
                           [2.55] or [0.0] but not 2.55 or 0.0.

       xInit   (par/req) : The initial guess of the state vector
                           (Double array).  The size of xInit must be
                           less than or equal to the size of the
                           one-sided input spec.

       jFunc   (key/opt) : The name (uppercase string) of a function
                           that takes a row vector (double 1-Dim
                           array) in R^n space and returns the
                           jacobian (a double array with m rows and n
                           columns) of the function that its name is
                           entered through the first parameter func.

                           This function must be implemented in such a
                           way that its single argument is always a
                           one dimensional array even if n is equal to
                           1, for example, [2.55] but not 2.55.

                           Also, the jacobian that it returns must be
                           a two dimensional array (in IDL sense) even
                           if the jacobian has only one element.

                           If this keyword is not defined, then a
                           finite difference approximation to the
                           jacobian of F(x) is used.

       sigma   (key/opt) : The desired relative error in the norm of
                           the scaled step.  The input value through
                           this keyword is a double greater than zero
                           and less than one.  It has a default value
                           of 0.1D and should not be changed (5.1,
                           5.2, and 5.5 of More's paper).

       delta   (key/opt) : A scalar double value which is the upper
                           limit for the norm of the scaled step.  Its
                           value must be greater than zero, and it has
                           a default value of one (2.1 of More's
                           paper).

       maxIter (key/opt) : A long int.  Maximum iterations performed
                           in optimizaton.  The default value is 100.

       tolerance
               (key/opt) : A scalar double value.  Its value must be
                           greater than zero and small.  The default
                           value is 1D-7.

                           Tolerance is used in the conditional
                           expressions used to decide whether the
                           current point is a minimizer or not.  One
                           of the expressions is

                              (Norm(residual##jacob) LT tolerance)
                                 AND (Norm(p) LT tolerance)

                           the other one is

                              (Norm(p) LT tolerance) AND
                                 (Abs(relErr) LT tolerance)

                           where relErr is

                              (resNorm2-resNextNorm2)/
                                                  resNextNorm2

                           the relative resNorm2 error and

                              p = the current step size that is not
                                  rejected.

       ConvTolerance
               (key/opt) : A vector (n_elements = 3) of tolerance values for the
                           convergence criteria.  The elements of this
                           vector correspond to:
                             ConvTolerance[2] = Convergence threshold for
                             cost function derivative, JacThresh
                             ConvTolerance[0] = Convergence threshold
                             for change in cost function, CostThresh
                             ConvTolerance[1] = Convergence threshold
                             for change in state, pThresh
       Chi2Tolerance
               (key/opt) : Scalar value that determines if the
                           minimizer stops below 1-Chi2Tolerance.  Specifically set
                           up for residuals normalized by a noise
                           value.

       singTolerance
               (key/opt) : A scalar double value.  Its value must be
                           greater than zero and small.  The default
                           value is 1D-7.

                           Any time an upper triangular matrix is
                           formed (R after the QR decomposition of
                           the jacobian, R-sub-lambda after Givens
                           rotations) the ratio

                              Abs(eigenvalue)/max(Abs(eigenvalue))

                           is computed for all eigenvalues.  If any
                           of these ratios is less than singTolerance,
                           then the matrix is considered to be
                           singular.

       newton  (key/opt) : Set this keyword to perform optimization
                           based on Gauss-Newton's method only.

       verbose (key/opt) : Set this keyword to display some search
                           related info on the screen.

       Diag_lambda_rho_delta
                (key/opt): matrix whose columns are dimensioned on the
                           maximum number of iterations and rows are
                           are the lambda, row, and delta (in that
                           order)  for each iteration.

    OUTPUT:
       resNorm2
               (key/opt) : A double scalar value equal to

                              ||F(x)||^2

                           corresponding to the last x, which is
                           returned.

       iter    (key/opt) : The number of the iterations performed,
                           long int.

       jacob   (key/opt) : A double array with n columns and m rows
                           where n is the size x and m is the size of
                           F(x).  The output array is the jacobian
                           corresponding to the returned x.

       residual
               (key/opt) : The residual

                              F(x)

                           corresponding to the returned x.

       stopCode
               (key/opt) : An integer value that indicates how the
                           search for a minimizer ended.  Possible
                           values are

                              - 1: Terminated because maximum number of
                                iterations has been met


                              - 2: cost function dropped below 1-Chi2Tolerance


                              - 3: If change in state, change in cost
                                function, AND cost function derivative
                                have dropped below their tolerances

       nJacobQR
               (key/opt) : The number of times a new jacobian and its
                           QR decomposition are computed.  In other
                           words, the number of computationally
                           intensive iterations.  Type is long int.

       x_iter (key/opt): an array that contains the state vector at
       each iteration.  The size of the array is the maximum number of
       iterations, which is set either by a keyword or by the default
       value (currently 100), by the size of the state vector.  In
       order get this output, an initialized variable must be passed
       in, e.g. x_it = 1, x_iter=x_it.  The returned x_it will be an
       array.

       res_iter (key/opt):  an array that contains the residual vector
       for each iteration.  The size and intiialization is the same for
       x_iter.
    INPUT/OUTPUT:
       oco_info:  has albedo mapping and ray info for oco-2
    SIDE EFFECTS:

    CONSTANTS:

    PRECONDITIONS:
       - func is the name (uppercase string) of a function that is
         implemented in such a way that its single argument and what
         it returns are always one dimensional arrays even if n or m
         are equal to 1, for example, [2.55] or [0.0] but not 2.55 or
         0.0.

       - sigma is a real number greater than 0 and less than 1
         (default value is 0.1 and should not be changed).

       - delta is a real number greater than 0 (default value is 1.0)

    POSTCONDITIONS:
       - returned x minimizes ||F(x)||^2 locally.  It is not
         guaranteed that the returned x is a global minimizer.

    ATBD REFERENCE:

    SRD REFERENCE:

    OTHER TRACEABILITY REFERENCE:

    EXTERNAL ROUTINES:
       Finite_Jacobian    : Finite difference approximation to the
                            jacobian of F(x)
       Rank_Revealing_QR  : To perform a QR decompositon on the
                            jacobian.
       RRQR_Q_Mult_A      : After the QR decomposition of a matrix B,
                            this function is used to multiply another
                            matrix A from left by Q, which is not
                            computed explicitly.
       RRQR_Get_Rn        : To get the top part of the R matrix, after
                            QR decomposition, that has nonzero entries
                            in its rows on or above the diagonal.
       Column_Permute     : To permute the columns of a matrix.
       Column_Permute_Undo: To undo permutation done by Column_Permute.
       Givens             : To get sin and cos for Givens rotation.

    INTERNAL ROUTINES:

    IDL ROUTINES:
       Call_Function: To evaluate call func and jfunc functions
       Keyword_Set : To check the truth value of a keyword
       N_Elements  : To the size of an array
       Double      : Type conversion
       Total       : To get the sum of the elements of an array
       DblArr      : To create a double array (0, 0, 0, ...)
       Size        : To get info. about the structure of a variable
       SqRt        : Square root
       Temporary   : To reuse already allocated memory space
       Transpose   : To get the transpose of a matrix
       Where       : To locate nonzero entries of an array
       Invert      : To invert a square matrix
       Norm        : To get the L2 norm of a vector (IDL Norm function
                     computes L2 norm if and only if the argument is a
                     row vector.)
       Imaginary   : To get the imaginary part of a complex number
       Return      : To return a value

    """

    def __init__(
        self,
        cfunc: CostFunction,
        max_iter: int,
        delta_value: float,
        conv_tolerance: list[float],
        chi2_tolerance: float,
        log_file: str | os.PathLike[str] | None = None,
        newton_flag: bool = False,
        sigma: float = 0.1,
        sing_tolerance: float = 1.0e-7,
        verbose: bool = False,
    ) -> None:
        self.cfunc = cfunc
        self.max_iter = max_iter
        self.delta_value = delta_value
        self.conv_tolerance = conv_tolerance
        self.chi2_tolerance = chi2_tolerance
        self.log_file = Path(log_file) if log_file is not None else None
        self.newton_flag = newton_flag
        self.sigma = sigma
        self.sing_tolerance = sing_tolerance
        # Defaults, so if we skip solve we have what is needed for output
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
        self._observers: set[Any] = set()

    def add_observer(self, obs: Any) -> None:
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._observers.add(obs)
        if hasattr(obs, "notify_add"):
            obs.notify_add(self)

    def remove_observer(self, obs: Any) -> None:
        self._observers.discard(obs)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self) -> None:
        # We change self._observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)

    def notify_update(self, location: ProcessLocation | str, **kwargs: Any) -> None:
        loc = location
        if not isinstance(loc, ProcessLocation):
            loc = ProcessLocation(loc)
        for obs in self._observers:
            obs.notify_update(self, loc, **kwargs)

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
        self.levmar_nllsq_elanor()

    def levmar_nllsq_elanor(
        self,
    ) -> None:
        # Temp
        from refractor.muses_py import (
            rank_revealing_qr,
            rrqr_q_mult_a,
            rrqr_get_rn,
            column_permute_undo,
            column_permute,
            givens,
        )  # type: ignore

        self.stop_code = 0

        if self.log_file is not None:
            f = open(self.log_file, "w")
            f.close()

        x_vector = self.cfunc.parameters.copy()

        #  Initialize Stop criteria diagnostics.

        self.stopcrit = np.zeros(shape=(self.max_iter + 1, 3), dtype=np.float64)
        self.resdiag = np.zeros(shape=(self.max_iter + 1, 5), dtype=np.float64)

        # Initialize linearity metrics

        self.diag_lambda_rho_delta = np.zeros(
            shape=(self.max_iter + 1, 3), dtype=np.float64
        )

        # epsD is numerical precision tolerance
        # epsD = np.MachAr(float_conv=np.float64).eps
        epsD = 2.2204460e-16  # Use the value from IDL running on ponte.

        (residual_next, jacobian_ret, radiance_fm) = (
            self.cfunc.new_residual_fm_jacobian(x_vector)
        )

        #  A flag to signal the termination of the iterative process.
        self.stop_code = 0

        resNextNorm2 = np.sum(residual_next**2)

        self.x_iter = np.zeros(
            shape=(self.max_iter + 1, len(self.cfunc.parameters)), dtype=np.float64
        )
        self.x_iter[0, :] = x_vector[:]

        res_iter = np.zeros(
            shape=(self.max_iter + 1, len(residual_next)), dtype=np.float64
        )
        res_iter[0, :] = residual_next[:]  # Set the residual for the pre-iteration.

        self.radiance_iter = np.zeros(
            shape=(self.max_iter + 1, len(radiance_fm)), dtype=np.float64
        )
        self.radiance_iter[0, :] = radiance_fm[
            :
        ]  # Set the radiance for the pre-iteration.

        #  nPoints must be greater than or equal to nTerms

        nTerms = len(self.cfunc.parameters)
        nPoints = len(residual_next)

        #  Compute the L2 norm squared of each column of the jacobian,
        #  and store the results (nTerms of them) in the row array
        #  jacobColumnL2Norm2.

        jacobColumnL2Norm2 = np.array(
            [np.sum(jacobian_ret[ii, :] ** 2) for ii in range(nTerms)]
        )

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
        if not self.newton_flag:
            scaleDiag = np.sqrt(jacobColumnL2Norm2)
            scaleDiag[scaleDiag == 0] = epsD

        #  <<<<<<<<<<<<<<<<<< END: INITIALIZATION OF D >>>>>>>>>>>>>>>>>>>

        #  Loop counter
        #
        self.iter_num = 0

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
        # OPTIMIZATION_LOOP_BEGIN: Inside this loop (iteration), these
        # variables are updated: p_vector,v_vector,x_vector
        while self.stop_code == 0:
            self.iter_num = self.iter_num + 1

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
            if rho > 0.0001:
                nJacobQR = nJacobQR + 1

                #  Anything "next" in the previous iteration becomes the
                #  current one in this iteration.
                #
                residual = np.copy(residual_next)
                resNorm2 = np.copy(resNextNorm2)

                #  <<<<<<<<<<<<<<<<<<<< QR DECOMPOSITION >>>>>>>>>>>>>>>>>>>
                #  <<<<<<<<<<<<<<<< AND RELATED COMPUTAIONS >>>>>>>>>>>>>>>>
                #  <<<<<<<<<<<<<<<<<<<<< SECTION 3, 3.3 >>>>>>>>>>>>>>>>>>>>
                #
                #  Perform a rank revealing QR decomposition of the
                #  jacobian.  Jacob was updated at the end of the
                #  previous iteration (if rho is greater than 0.0001).
                #

                # Because the following function RankRevealingQR will
                # mess with jacobColumnL2Norm2, we send in a copy
                # only.
                (R, pivots, rank) = rank_revealing_qr(
                    jacobian_ret,
                    columnL2Norm2=np.copy(jacobColumnL2Norm2),
                    epsilon=self.sing_tolerance,
                )

                #  Multiply Transpose(residual), a column vector, by Q
                #  from the left.  QResidual is Qf in More's notation,
                #  and it is a column vector with nPoints rows.

                QResidual = rrqr_q_mult_a(residual.T, R, rank)

                RN = np.zeros(shape=(nTerms, nTerms), dtype=np.float64)
                RN[0:nTerms, 0:rank] = (rrqr_get_rn(R, rank))[:, :]
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
                if not self.newton_flag:
                    temp = np.sqrt(jacobColumnL2Norm2)
                    scaleDiag[scaleDiag < temp] = temp[scaleDiag < temp]

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

                pNewton = np.zeros(shape=(nTerms), dtype=np.float64)
                pNewton[0:rank] = -(QResidual[0, 0:rank] @ np.linalg.inv(T))

                (nPermutations, pNewton) = column_permute_undo(pNewton, pivots[0:rank])

                if not self.newton_flag:
                    qNewton = scaleDiag * pNewton
                    qNormNewton = float(np.linalg.norm(qNewton))
                else:
                    qNormNewton = 0.0

                #
                #  <<<<<<<<<<<<<<<<<<<<<< END: NEWTON >>>>>>>>>>>>>>>>>>>>>>
            # end of if (rho > 0.0001):

            p_vector = np.copy(pNewton)

            if not self.newton_flag:
                q_vector = np.copy(qNewton)
            qNorm = qNormNewton

            self.notify_update("start iteration")
            if self.verbose:
                logger.info("***************************************************")
                logger.info(f"Start iteration# = {self.iter_num}")
                if not self.newton_flag:
                    logger.info(f"Adaptive D       = {scaleDiag}")
                logger.info(f"At x             = {x_vector}")
                logger.info(f"rank of jacobian = {rank}")

                nTemp = min(nTerms, nPoints)
                temp = np.zeros(shape=(nTemp), dtype=np.float64)
                for i in range(nTemp):
                    temp[i] = R[i, i]
                logger.info(f"eigenvalues of R = {temp}")
                logger.info(f"Newton step p    = {p_vector}")
                logger.info("")

            if self.log_file:
                with open(self.log_file, "a") as f:
                    print("***************************************************", file=f)
                    print(f"Start iteration# = {self.iter_num}", file=f)
                    if not self.newton_flag:
                        print(f"Adaptive D       = {scaleDiag}", file=f)
                    print(f"At x             = {x_vector}", file=f)
                    print(f"rank of jacobian = {rank}", file=f)

                    nTemp = min(nTerms, nPoints)
                    temp = np.zeros(shape=(nTemp), dtype=np.float64)
                    for i in range(nTemp):
                        temp[i] = R[i, i]
                    print(f"eigenvalues of R = {temp}", file=f)
                    print(f"Newton step p    = {p_vector}", file=f)
                    print("", file=f)

            #  <<<<<<<<<<<<< ACCEPT NEWTON STEP OR FIND A NEW p >>>>>>>>>>>
            #  <<<<<<<<<<<< WHEN LEV. MAR. PARAMETER IS NOT ZERO >>>>>>>>>>
            #  <<<<<<<<<<<<<<<<<< STEP a, ALGORITH 7.1 >>>>>>>>>>>>>>>>>>
            #
            #  If the size of the Newton step is not too large, then accept
            #  that step and set LevMar parameter (lambda) equal to zero,
            #  else compute another p using LevMar method.
            #
            if qNormNewton <= (1 + self.sigma) * self.delta_value:
                lambda_value = 0.0  # The word 'lambda' is a reserved keyword in Python.
            else:
                #  Compute jacobTilde (More, 5.3)
                #
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
                upper_bound = float(
                    np.linalg.norm(jacobTilde @ residual) / self.delta_value
                )

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
                    phi0 = qNorm - self.delta_value

                    #  This code segment is an implementation of the formula
                    #  at the end of the section 5 of More's paper when
                    #  alpha is zero.  Here we need q and qNorm computed when
                    #  alpha (lambda) is equal to zero, and they already
                    #  exist.  These values are qNewton and qNormNewton, and
                    #  the current values of q and qNorm, at this point, are
                    #  equal to the values of qNewton and qNormNewton
                    #  respectively.
                    #
                    temp = q_vector * scaleDiag
                    (nPermutations, temp) = column_permute(temp, pivots)

                    phi0Prime = -qNorm * np.sum(
                        (np.linalg.inv(RN) @ (temp / qNorm)) ** 2
                    )

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
                scaleDiagPi = np.copy(scaleDiag)
                (nPermutations, scaleDiagPi) = column_permute(
                    scaleDiagPi, pivots[0:rank]
                )

                #  Set lambda equal to any out of range value so that the IF
                #  statement in the REPEAT/UNTIL loop will get executed when
                #  the loop is entered for the first time.  A negative value
                #  is the best choice.
                lambda_value = -100.0  # Note: the keyword 'lambda' belongs to Python.

                #  A counter for the number of the iterations of the inner
                #  loop (ALGORITHM 5.5, More).
                innerIter = 0

                stopThisLoop = False
                while not stopThisLoop:
                    innerIter = innerIter + 1

                    #  <<<<<<<<<<<<<<< STEP a, ALGORITHM 5.5 >>>>>>>>>>>>>>>>
                    #
                    #  Make sure that lambda is within the range.  If not,
                    #  then update it.
                    #
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
                    # At this point, we have already allocated the two
                    # matricies: tempRjY, tempDiag.  We just need to
                    # fill them with zeros.
                    tempRjY = np.zeros(shape=(nTerms + 1, nTerms), dtype=np.float64)
                    tempRjY[0:nTerms, 0:nTerms] = RN[:, :]

                    if nTerms < nPoints:
                        tempRjY[nTerms, 0:nTerms] = QResidual[0, 0:nTerms]
                    else:
                        tempRjY[nTerms, 0:nPoints] = QResidual[0, 0:nPoints]

                    tempDiag = np.zeros(shape=(nTerms + 1, nTerms), dtype=np.float64)
                    lambda_sqrt = np.sqrt(lambda_value)
                    for ii in range(nTerms):
                        tempDiag[ii, ii] = scaleDiagPi[ii] * lambda_sqrt

                    maxAbsEigVal = np.sqrt(tempRjY[0, 0] ** 2 + tempDiag[0, 0] ** 2)

                    singular_value = False
                    ii = 0

                    #  Use Givens rotations to compute R-sub-lambda.  Loop
                    #  ends either when all rotations are performed or when
                    #  it is realized that R-sub-lambda will be singular.
                    while (ii < nTerms) and (not singular_value):
                        # Note that we use "ii+1" for the slice of
                        # tempDiag in the next line.  Remember that
                        # IDL does include the end of the slice,
                        # whereas Python does not.
                        curAbsEigVal = np.sqrt(
                            tempRjY[ii, ii] ** 2 + np.sum(tempDiag[ii, 0 : ii + 1] ** 2)
                        )

                        if curAbsEigVal / maxAbsEigVal >= self.sing_tolerance:
                            rev_loop_count = 0

                            # Note that for Python, we go from ii to 0
                            # with negative step.  IDL goes from ii to
                            # 1 with negative step.
                            for jj in range(ii, 0, -1):
                                (cs, sn) = givens(
                                    tempDiag[ii, jj - 1], tempDiag[ii, jj]
                                )
                                temp = np.copy(
                                    tempDiag[ii:, jj - 1]
                                )  # We have to copy so the memory won't be overwritten.
                                tempDiag[ii:, jj - 1] = (cs * temp) - (
                                    sn * tempDiag[ii:, jj]
                                )
                                tempDiag[ii:, jj] = (sn * temp) + (
                                    cs * tempDiag[ii:, jj]
                                )
                                rev_loop_count += 1

                            (cs, sn) = givens(tempRjY[ii, ii], tempDiag[ii, 0])

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
                            if curAbsEigVal > maxAbsEigVal:
                                maxAbsEigVal = curAbsEigVal
                            ii = ii + 1
                        else:
                            singular_value = True

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
                    if not singular_value:
                        #  Compute step p, scaled step q, and the L2 norm of
                        #  the scaled step qNorm.
                        #
                        # pass
                        RJInverse = np.linalg.inv(tempRjY[0:nTerms, 0:nTerms])

                        p_vector = -(tempRjY[nTerms, 0:nTerms] @ RJInverse).T

                        (nPermutations, p_vector) = column_permute_undo(
                            p_vector, pivots[0:rank]
                        )

                        q_vector = scaleDiag * p_vector

                        qNorm = float(np.linalg.norm(q_vector))

                        stopThisLoop = (
                            qNorm >= (1.0 - self.sigma) * self.delta_value
                        ) and (qNorm <= (1.0 + self.sigma) * self.delta_value)
                    else:
                        #  Compute step p, scaled step q, and the L2 norm of
                        #  the scaled step qNorm.
                        #
                        #  Even in this case it may be possible to avoid the
                        #  singular value decomposition of jacobTilde.
                        #

                        # see: https://github.jpl.nasa.gov/MUSES-Processing/idl-retrieve/blob/v1.3.0/Optimization/OPTIMIZATION/Levmar_NLLSQ.pro#L1017

                        tempMatrix = np.zeros(
                            shape=(nTerms, 2 * nTerms), dtype=np.float64
                        )
                        tempMatrix[0:nTerms, 0:nTerms] = RN[:, :]
                        for ii in range(nTerms):
                            tempMatrix[ii, ii + nTerms] = scaleDiagPi[ii] * lambda_sqrt

                        (Rj, pivotsj, rankj) = rank_revealing_qr(
                            tempMatrix, None, epsilon=self.sing_tolerance
                        )

                        tempMatrix.fill(0.0)

                        Tj = (rrqr_get_rn(Rj, rankj))[0:rankj, 0:rankj]

                        temp = np.zeros(shape=(1, 2 * nTerms), dtype=np.float64)
                        if nTerms < nPoints:
                            temp[0, 0:nTerms] = QResidual[0, 0:nTerms]
                        else:
                            temp[0, 0:nTerms] = QResidual[0, 0:nPoints]

                        # TODO: Revisit. This is likely a bug in the IDL code
                        # uj = (rrqr_q_mult_a(temp, Rj, rankj))[0, 0:nTerms]
                        uj = rrqr_q_mult_a(temp, Rj, rankj)

                        p_vector = np.zeros(shape=(nTerms), dtype=np.float64)

                        TjInverse = np.linalg.inv(Tj)
                        p_vector[0:rankj] = -(uj[0, 0:rankj] @ TjInverse)

                        (nPermutations, p_vector) = column_permute_undo(
                            p_vector, pivots[0:rankj]
                        )
                        (nPermutations, p_vector) = column_permute_undo(
                            p_vector, pivots[0:rank]
                        )

                        q_vector = scaleDiag * p_vector

                        qNorm = float(np.linalg.norm(q_vector))

                        stopThisLoop = True

                    tempRjY.fill(0.0)

                    #
                    #  <<<<<<<<<<<<<<<< COMPUTE p, q, qNorm >>>>>>>>>>>>>>>>>

                    #  If needed, update lower_bound, lambda_value, and upper_bound for the
                    #  next iteration of this loop.
                    #
                    if not stopThisLoop:
                        #  <<<<<<<<<<<<<< STEP b, ALGORITHM 5.5 >>>>>>>>>>>>>>
                        #
                        phi = qNorm - self.delta_value
                        temp = q_vector * scaleDiag

                        (nPermutations, temp) = column_permute(temp, pivots)

                        phiPrime = -qNorm * np.sum((RJInverse @ (temp / qNorm)) ** 2)

                        if phi < 0:
                            upper_bound = lambda_value

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
                        lambda_value = lambda_value - (
                            (phi + self.delta_value) / self.delta_value
                        ) * (phi / phiPrime)
                        #
                        #  <<<<<<<<<<< END: STEP c, ALGORITHM 5.5 >>>>>>>>>>>>
                    # end if (not stopThisLoop):

                    # for some reason it gets hung up in this loop for OMI
                    # cloud fraction
                    if innerIter > 1000:
                        stopThisLoop = True

                # end while (not stopThisLoop)

                scaleDiagPi.fill(0.0)
                #
                #  <<<<<<< END: IMPLEMENTATION OF ALGORITHM 5.5, MORE >>>>>>

            v_vector = x_vector + p_vector

            p_vector = v_vector - x_vector

            (
                residual_next,
                jacobNext,
                radiance_fm_next,
            ) = self.cfunc.new_residual_fm_jacobian(v_vector)

            resNextNorm2 = np.sum(residual_next**2)
            jacobPNorm2 = np.sum(
                (p_vector.T @ jacobian_ret) ** 2
            )  # jacobian_ret is IDL jacob
            jacResNextNorm2 = np.sum((jacobNext @ residual_next) ** 2)
            jacResNorm2 = np.sum(
                (jacobian_ret @ residual) ** 2
            )  # jacobian_ret is IDL jacob
            xNorm = np.sum(x_vector**2)
            pNorm = np.sum(p_vector**2)

            self.x_iter[self.iter_num, :] = x_vector + p_vector

            res_iter[self.iter_num, :] = residual_next[:]
            self.radiance_iter[self.iter_num, :] = radiance_fm_next[:]

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

            if self.newton_flag:
                rho = 100.0
            elif resNextNorm2 > resNorm2:
                rho = 0.0
            else:
                rho = (1.0 - resNextNorm2 / resNorm2) / (
                    jacobPNorm2 / resNorm2
                    + 2.0 * (lambda_value * (qNorm**2)) / resNorm2
                )

            #
            #  <<<<<<<<<<<<<<<< END: STEP b, ALGORITHM 7.1 >>>>>>>>>>>>>>>>

            #  <<<<<<<<< UPDATE x AND jacobian IF p IS ACCEPTED >>>>>>>>>>>
            #  <<<<<<<<<<<<<<<<<< STEP C, ALGORITH 7.1 >>>>>>>>>>>>>>>>>>>>
            #
            #  If rho is greater than 0.0001D, then the step p is accepted.
            #
            #  Calculate the thresholds for stopping criteria
            pThresh = pNorm / (1.0 + xNorm)
            costThresh = np.abs(resNorm2 - resNextNorm2) / (1.0 + resNextNorm2)
            JacThresh = np.linalg.norm(jacobian_ret @ residual_next) / (
                1.0 + resNextNorm2
            )

            # see: https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf, 4.1.3 Convergence criteria
            # Large values, χ2ν ≫ 1, indicate a poor fit, χ2ν ≈ 1 indicates that the fit error is of the same order as the measurement error (as desired), and
            # χ2ν < 1 indicates that the model is over-fitting the data; that is, the model is fitting the measurement noise.
            # χ2ν = χ2 / (m - n + 1), where m  = # of observations, n = # of parameters
            chi2 = resNextNorm2
            dof = radiance_fm_next.shape[0] - p_vector.shape[0] + 1
            chi2_reduced = (chi2 / dof) if dof > 0 else chi2

            if rho > 0.0001:
                x_vector = x_vector + p_vector
                jacobian_ret = np.copy(jacobNext)

                if (self.stop_code == 0) and (resNextNorm2 < 1.0 - self.chi2_tolerance):
                    # model is over-fitting the data; that is, the
                    # model is fitting the measurement noise.
                    self.stop_code = 2

                if (self.stop_code == 0) and (
                    (pThresh < self.conv_tolerance[1])
                    and (JacThresh < self.conv_tolerance[2])
                    and (costThresh < self.conv_tolerance[0])
                ):
                    self.stop_code = 3

                # VK: Add another convergence criteria for χ2 usingthe
                # the reduced χ2, χ2ν = χ2 / (m − n + 1) < ϵ3 (quality
                # of the fit).  see:
                # https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf,
                # 4.1.3 Convergence criteria

                # TODO: Make epsilon3 configurable per step in the
                # Strategy table.  1.1 is too low. With the current FM
                # / RTs (OSS or VLIDORT) we rarely get χ2ν
                # (chi2_reduced) lower than 2.0.
                epsilon3 = 1.1
                if (
                    (self.stop_code == 0)
                    and (chi2_reduced >= 1)
                    and (chi2_reduced < epsilon3)
                ):
                    self.stop_code = 4
            # end if (rho > 0.0001):

            #
            #  <<<<<<<<<<<<<<<< END: STEP c, ALGORITHM 7.1 >>>>>>>>>>>>>>>>
            #  If the loop has repeated self.max_iter times then give up.
            #
            if (self.stop_code == 0) and (self.iter_num >= self.max_iter):
                self.stop_code = 1

            # Save off stopping criterion

            self.stopcrit[self.iter_num, 0] = costThresh
            self.stopcrit[self.iter_num, 1] = pThresh
            self.stopcrit[self.iter_num, 2] = JacThresh

            # Save residual diagnostics

            self.resdiag[self.iter_num, 0] = np.sqrt(resNorm2) / np.sqrt(nPoints)
            self.resdiag[self.iter_num, 1] = np.sqrt(resNextNorm2) / np.sqrt(nPoints)
            self.resdiag[self.iter_num, 2] = np.sqrt(jacResNorm2) / np.sqrt(nPoints)
            self.resdiag[self.iter_num, 3] = np.sqrt(jacResNextNorm2) / np.sqrt(nPoints)
            self.resdiag[self.iter_num, 4] = pNorm / np.sqrt(nPoints)

            # Save linearity measures

            self.diag_lambda_rho_delta[self.iter_num, 0] = lambda_value
            self.diag_lambda_rho_delta[self.iter_num, 1] = rho
            self.diag_lambda_rho_delta[self.iter_num, 2] = self.delta_value

            self.notify_update("end step")
            if self.log_file:
                with open(self.log_file, "a") as f:
                    print("jacobian_ret size", file=f)
                    print(
                        f"{len(jacobian_ret.shape)}   {jacobian_ret.shape}   {jacobian_ret.dtype}   {jacobian_ret.size}",
                        file=f,
                    )
                    print("", file=f)

                    print("Convergence Criteria:", file=f)

                    print(
                        "costThresh = np.abs(resNorm2 - resNextNorm2) / (1 + resNextNorm2)",
                        file=f,
                    )
                    print(f"{costThresh}", file=f)

                    print("tolerance = self.conv_tolerance[0]", file=f)
                    print(f"{self.conv_tolerance[0]}", file=f)

                    print("pThresh = pNorm / (1 + xNorm)", file=f)
                    print(f"{pThresh}", file=f)

                    print("tolerance = self.conv_tolerance[1]", file=f)
                    print(f"{self.conv_tolerance[1]}", file=f)

                    print(
                        "JacThresh = np.linalg.norm(jacobian_ret @ residual_next) / (1 + resNextNorm2)",
                        file=f,
                    )
                    print(f"{JacThresh}", file=f)

                    print("tolerance = self.conv_tolerance[2]", file=f)
                    print(f"{self.conv_tolerance[2]}", file=f)

                    print("", file=f)

                    if not self.newton_flag:
                        if qNormNewton <= (1 + self.sigma) * self.delta_value:
                            print("Newton step is chosen.", file=f)
                            print("", file=f)
                            print("qNorm    (1 + self.sigma) * delta", file=f)
                            print(
                                f"{qNorm}, {(1 + self.sigma) * self.delta_value}",
                                file=f,
                            )
                        else:
                            print(
                                "Newton step is not good; LevMar step is chosen", file=f
                            )
                            print(
                                f"after the inner loop repeated {innerIter} times.",
                                file=f,
                            )

                            if singular_value:
                                print("Rj became singular.", file=f)

                            print(f"LevMar step p = {p_vector}", file=f)

                            print("", file=f)

                            print(
                                "(1 - self.sigma) * delta   qNorm   (1 + self.sigma) * delta",
                                file=f,
                            )
                            print(
                                f"{(1.0 - self.sigma) * self.delta_value} {qNorm} {(1.0 + self.sigma) * self.delta_value}",
                                file=f,
                            )
                        # end if not self.newton_flag:

                        print("", file=f)

                        print(
                            "||F(x)|| / sqrt(m)  ||F(x+p)|| / sqrt(m)   ||J F(x)|| / sqrt(m)   ||J F(x + p)|| / sqrt(m)   ||p|| / sqrt(m)",
                            file=f,
                        )
                        print(
                            "self.resdiag[self.iter_num, 0]  self.resdiag[self.iter_num, 1]  self.resdiag[self.iter_num, 2] self.resdiag[self.iter_num, 3] self.resdiag[self.iter_num, 4]",
                            file=f,
                        )
                        print(
                            f"{self.resdiag[self.iter_num, 0]} {self.resdiag[self.iter_num, 1]} {self.resdiag[self.iter_num, 2]} {self.resdiag[self.iter_num, 3]} {self.resdiag[self.iter_num, 4]}",
                            file=f,
                        )

                        print("", file=f)

                        print("lambda   rho   delta", file=f)
                        print(f"{lambda_value}   {rho}   {self.delta_value}", file=f)

                        print("", file=f)

                        if rho > 0.0001:
                            print("The step was accepted, and", file=f)
                            print(f"x + p = {x_vector}", file=f)
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
                            f"{self.resdiag[self.iter_num, 0]}   {self.resdiag[self.iter_num, 1]}   {self.resdiag[self.iter_num, 2]}   {self.resdiag[self.iter_num, 3]}",
                            file=f,
                        )

                        print("", file=f)
                        print("After step p", file=f)
                        print(f"x + p = {x_vector}", file=f)
                    # end if not self.newton_flag:

            #  Anything that must be updated at the end of the current
            #  iteration for the next one (if any), and the decision
            #  whether to go through another iteration or not does not
            #  depend on it, is updated in this if statement.
            #
            if self.stop_code == 0:
                #  Reminder: rho greater than 0.0001 means that the last
                #  step was accepted and we have a new jacob(ian).
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
                if self.newton_flag:
                    self.delta_value = 0.0
                elif rho <= 0.25:
                    if resNextNorm2 <= resNorm2:
                        tempConst = 0.5
                    elif resNextNorm2 > 100.0 * resNorm2:
                        tempConst = 0.1
                    else:
                        tempConst = -(
                            jacobPNorm2 / resNorm2
                            + lambda_value * (qNorm**2) / resNorm2
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

                    self.delta_value = tempConst * self.delta_value
                    if (rho <= 0.0001) and (
                        qNorm <= (1.0 + self.sigma) * self.delta_value
                    ):
                        self.delta_value = 0.75 * (qNorm / (1 + self.sigma))

                elif (rho >= 0.75) or (
                    (rho > 0.25) and (rho < 0.75) and (lambda_value == 0.0)
                ):
                    self.delta_value = 2.0 * qNorm
                # end elif
                #
                #  <<<<<<<<<<<<<<< END: STEP d, ALGORITHM 7.1 >>>>>>>>>>>>>>
            # end if (self.stop_code == 0):

            # Useful diagnostic while looking at muses-py/ReFRACtor.
            self.notify_update("end iteration")
            if self.verbose:
                logger.info(f"rho: {rho}")
                logger.info(f"stopCode: {self.stop_code}")
                logger.info(f"pThresh: {pThresh}")
                logger.info(f"costThresh: {costThresh}")
                logger.info(f"JacThresh: {JacThresh}")
                logger.info(f"Chi2: {resNextNorm2}")
                logger.info(f"resNextNorm2: {resNextNorm2}")
                logger.info(f"resNorm2: {resNorm2}")

            # TODO: print only in self.verbose mode
            chi2 = resNextNorm2
            dof = radiance_fm_next.shape[0] - p_vector.shape[0] + 1
            chi2_reduced = (chi2 / dof) if dof > 0 else chi2

            logger.info(
                f"iter: {self.iter_num:02}, max: {self.max_iter:02}, rho = {rho:10.5f}, chi2_reduced = {chi2_reduced:10.5f}, chi2 = {chi2:10.5f}"
            )
        # end while (self.stop_code == 0):

        # OPTIMIZATION_LOOP_END

        #  Reminder: rho greater than 0.0001 means that the last
        #  step was accepted.
        #
        #  After exiting the loop, if the last step p was accepted,
        #  then residual and resNorm2 must be updated.
        #
        if rho > 0.0001:
            residual = np.copy(residual_next)
            resNorm2 = resNextNorm2

        # Since x_vector is the best iteration, which might not be the last,
        # set the cost function to this. Note the cost function does internal
        # caching, so if this is the last one then we don't recalculate
        # residual and jacobian.
        self.cfunc.parameters = x_vector

        # Find iteration used, only keep the best iteration
        rms = np.array(
            [
                np.sqrt(np.sum(res_iter[i, :] * res_iter[i, :]) / res_iter.shape[1])
                for i in range(self.iter_num + 1)
            ]
        )
        self.best_iter = int(np.argmin(rms))
        self.residual_rms = rms


__all__ = ["SolverResult", "MusesLevmarSolver"]
