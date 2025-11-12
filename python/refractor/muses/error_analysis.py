from __future__ import annotations
from .mpy import (
    mpy_get_vector,
    mpy_my_total,
)
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
from .refractor_uip import AttrDictAdapter
import numpy as np
import math
from loguru import logger
import typing

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_strategy import CurrentStrategyStep
    from .retrieval_result import RetrievalResult


class ErrorAnalysis:
    """This performs the error analysis.

    This class calls CurrentState.update_previous_aposteriori_cov_fm
    to update the aposteriori covariance for the current
    step. Normally we try to avoid side effects, but in this
    particular case it seems reasonable that calculation the
    aposteriori_cov_fm in ErrorAnalysis updates the current_state. We
    can pull this update out if needed - if we want the update to be
    more explicit. But for now, this is what we do.
    """

    def __init__(
        self,
        current_state: CurrentState,
        current_strategy_step: CurrentStrategyStep,
        retrieval_result: RetrievalResult,
    ) -> None:
        # TODO Clean up passing in RetrievalResult, instead we should just pass in the
        # pieces we need.
        fstate_info = FakeStateInfo(current_state)
        fretrieval_info = FakeRetrievalInfo(current_state)
        offDiagonalSys = self.error_analysis_wrapper(
            retrieval_result.rstep,
            fretrieval_info,
            fstate_info,
            retrieval_result,
        )
        # Update current state with new aposteriori_cov_fm.
        current_state.update_previous_aposteriori_cov_fm(self.Sx, offDiagonalSys)

    def error_analysis_wrapper(
        self,
        radiance: AttrDictAdapter,
        retrieval: FakeRetrievalInfo,
        stateInfo: FakeStateInfo,
        retrieval_result: RetrievalResult,
    ) -> np.ndarray | None:
        # expected noise
        data_error = radiance.NESR
        bad_pixel = data_error < 0

        cstate = retrieval_result.current_state
        if cstate.map_to_parameter_matrix is None or cstate.basis_matrix is None:
            raise RuntimeError("Missing basis matrix")
        my_map = AttrDictAdapter(
            {
                "toPars": np.copy(cstate.map_to_parameter_matrix),
                "toState": np.copy(cstate.basis_matrix),
            }
        )

        # result is jacobian[pars, frequency]
        jacobian = retrieval_result.jacobian[0, :, :].copy()
        jacobian[:, bad_pixel] = 0

        actual_data_residual = retrieval_result.radiance[0, :] - radiance.radiance
        actual_data_residual[bad_pixel] = 0

        if not np.all(np.isfinite(actual_data_residual)):
            raise RuntimeError("actual_data_residual is not finite")
        if not np.all(np.isfinite(data_error)):
            raise RuntimeError("data_error is not finite")
        if not np.all(np.isfinite(jacobian)):
            raise RuntimeError("jacobian is not finite")

        # Get normalized (by nesr) systematic jacobian
        jacobian_sys = retrieval_result.jacobian_sys

        # For now, we need to assign these variables below so they have some values because
        # for AIRS run, the value of nSys is 0.
        Sb = None
        self._Sb = None
        if jacobian_sys is not None:
            jacobian_sys = jacobian_sys[0, :, :].copy()
            jacobian_sys[:, bad_pixel] = 0

            if not np.all(np.isfinite(jacobian_sys)):
                raise RuntimeError("jacobian_sys is not finite")

            for i in range(jacobian_sys.shape[0]):
                jacobian_sys[i, :] /= data_error

            Sb = cstate.Sb
            self._Sb = Sb

        for ii in range(jacobian.shape[0]):
            jacobian[ii, :] /= data_error

        jacobian_fm = np.copy(jacobian)
        jacobian = my_map.toState @ jacobian_fm

        # AT_LINE 177 Error_Analysis_Wrapper.pro
        Sa = cstate.Sa
        self._Sa = Sa

        ret_vector = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        con_vector = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        FM_Flag = True
        INITIAL_Flag = True
        TRUE_Flag = False
        CONSTRAINT_Flag = False

        for ispecie in range(retrieval.n_species):
            species_name = retrieval.species[ispecie]
            ind1FM = retrieval.parameterStartFM[ispecie]
            ind2FM = retrieval.parameterEndFM[ispecie]

            ret_vector[ind1FM : ind2FM + 1] = mpy_get_vector(
                retrieval_result.resultsList,
                retrieval,
                species_name,
                FM_Flag,
                INITIAL_Flag,
                TRUE_Flag,
                CONSTRAINT_Flag,
            )
            con_vector[ind1FM : ind2FM + 1] = mpy_get_vector(
                retrieval.constraintVector,
                retrieval,
                species_name,
                FM_Flag,
                INITIAL_Flag,
                TRUE_Flag,
                CONSTRAINT_Flag,
            )

            if (
                retrieval.mapType[ispecie].lower() == "log"
            ):  # Note the spelling of 'mapType' in retrieval object.
                ret_vector[ind1FM : ind2FM + 1] = np.log(
                    ret_vector[ind1FM : ind2FM + 1]
                )
                con_vector[ind1FM : ind2FM + 1] = np.log(
                    con_vector[ind1FM : ind2FM + 1]
                )

        constraintVector = con_vector
        resultVector = ret_vector

        offDiagonalSys = self.error_analysis(
            my_map,
            jacobian,
            jacobian_fm,
            Sa,
            jacobian_sys,
            Sb,
            cstate.constraint_matrix,
            constraintVector,
            resultVector,
            data_error,
            actual_data_residual,
            retrieval_result,
            retrieval,
            cstate.error_current_values,
        )
        return offDiagonalSys

    # see: https://ieeexplore.ieee.org/document/1624609
    # Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
    # V. ERROR CHARACTERIZATION

    def error_analysis(
        self,
        my_map: AttrDictAdapter,
        jacobian: np.ndarray,
        jacobianFM: np.ndarray,
        Sa: np.ndarray,
        jacobianSys: np.ndarray | None,
        Sb: np.ndarray | None,
        constraintMatrix: np.ndarray,
        constraintVector: np.ndarray,
        resultVector: np.ndarray,
        dataError: np.ndarray,
        actualDataResidual: np.ndarray,
        retrieval_result: RetrievalResult,
        retrieval: FakeRetrievalInfo,
        errorCurrentValues: np.ndarray,
    ) -> np.ndarray | None:
        o_offDiagonalSys = None

        self._ch4_evs = np.zeros(shape=(0,))
        kappa = jacobian @ jacobian.T
        kappaFM = jacobianFM @ jacobian.T  # [parameter,frequency]
        S_inv = np.linalg.inv(kappa + constraintMatrix) @ my_map.toState
        self._KtSyK = kappa
        self._KtSyKFM = kappaFM

        doUpdateFM = retrieval.doUpdateFM[0 : retrieval.n_totalParametersFM]
        dontUpdateFM = 1 - doUpdateFM

        my_id = np.asarray(np.identity(S_inv.shape[1]), dtype=np.float64)
        self._Sx_smooth = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)
        self._Sx_rand = S_inv.T @ kappa @ S_inv

        if jacobianSys is not None and Sb is not None:
            kappaInt = np.matmul(jacobianSys, np.transpose(jacobian))
            self._Sx_sys = np.matmul(
                np.matmul(
                    np.matmul(np.transpose(np.matmul(kappaInt, S_inv)), Sb), kappaInt
                ),
                S_inv,
            )
            sys = np.zeros(shape=(len(jacobianSys[0, :])), dtype=np.float64)
            for ii in range(len(jacobianSys[0, :])):
                sys[ii] = np.matmul(
                    np.matmul(np.transpose(jacobianSys[:, ii]), Sb), jacobianSys[:, ii]
                )
            self._radianceResidualRMSSys = math.sqrt(np.sum(sys) / sys.size)
            o_offDiagonalSys = np.matmul(np.matmul(Sb, kappaInt), -S_inv)
        else:
            self._Sx_sys = np.zeros(self._Sx_rand.shape)
            self._radianceResidualRMSSys = 0.0

        self._Sx = self._Sx_smooth + self._Sx_rand + self._Sx_sys
        self._A = np.matmul(kappaFM, S_inv)

        G_matrix = np.matmul(np.transpose(jacobian), S_inv)
        for jj in range(G_matrix.shape[1]):
            G_matrix[:, jj] /= dataError
        self._GMatrix = np.matmul(G_matrix, my_map.toPars)
        self._GMatrixFM = G_matrix

        # some species, like emis, are retrieved in step 1 for a
        # particular spectral region and not updated following this.
        # In that case, keep errors when they are not moved.  If this
        # is the case, set the error to the previous error, and all
        # error components to zero.
        indw = np.where(dontUpdateFM == 1)[0]

        if len(indw) > 0:
            self._Sx[indw, indw] = errorCurrentValues[indw, indw]
            self._Sx_rand[indw, indw] = 0
            self._Sx_sys[indw, indw] = 0

        my_id = np.identity(S_inv.shape[1], dtype=np.float64)
        self._Sx_smooth = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)

        self._Sx_smooth_self = np.zeros(self._Sx.shape)
        self._Sx_crossState = self._Sx_smooth.copy()

        # override the block diagonal terms with smooth_self and crossstate
        species_list_fs = np.asarray(retrieval.speciesListFM)
        for ii in range(retrieval.n_species):
            species = retrieval.species[ii]
            indMe0 = np.where(species_list_fs == species)[0]

            if len(indMe0) > 1:
                ss = mpy_my_total(abs(Sa[indMe0, :]), 0)
                indMe = np.where(ss > 0)[0]
                indYou = np.where(ss == 0)[0]
            else:
                ss = Sa[indMe0, :][0]
                indMe = np.where(ss > 0)[0]
                indYou = np.where(ss == 0)[0]
            if len(indMe) == 1:
                self._Sx_smooth_self[indMe, indMe] = (
                    (1 - self._A[indMe, indMe])
                    * Sa[indMe, indMe]
                    * (1 - self._A[indMe, indMe])
                )
            else:
                my_id = np.identity(len(indMe))
                ind_me_2d = np.ix_(indMe, indMe)
                self._Sx_smooth_self[ind_me_2d] = (
                    (my_id - self._A[ind_me_2d]).T
                    @ Sa[ind_me_2d]
                    @ (my_id - self._A[ind_me_2d])
                )

            # interferent on and off diagonal
            if len(indYou) == 0:
                self._Sx_crossState[:, :] = 0
            elif len(indMe) == 1 and len(indYou) == 1:
                self._Sx_crossState[indMe, indMe] = (
                    self._A[indYou, indMe] * Sa[indYou, indYou] * self._A[indYou, indMe]
                )
            elif len(indMe) == 1 and len(indYou) > 1:
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe)
                self._Sx_crossState[np.ix_(indMe, indMe)] = (
                    self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
                )
            elif len(indMe) > 1 and len(indYou) == 1:
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe)
                self._Sx_crossState[np.ix_(indMe, indMe)] = (
                    self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
                )
            else:
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe)
                self._Sx_crossState[np.ix_(indMe, indMe)] = (
                    self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
                )

        self._Sa_ret = my_map.toPars.T @ Sa @ my_map.toPars

        S_inv_ret = np.linalg.inv(kappa + constraintMatrix)
        my_id = np.identity(S_inv_ret.shape[1], dtype=np.float64)
        Sx_smooth_ret = (
            (my_id - kappa @ S_inv_ret).T @ self._Sa_ret @ (my_id - kappa @ S_inv_ret)
        )
        Sx_smooth_ret_fm = my_map.toState.T @ Sx_smooth_ret @ my_map.toState

        self._Sx_mapping = self._Sx_smooth - Sx_smooth_ret_fm

        self._Sx_ret_smooth = my_map.toPars.T @ self._Sx_smooth_self @ my_map.toPars
        self._Sx_ret_crossState = my_map.toPars.T @ self._Sx_crossState @ my_map.toPars
        self._Sx_ret_rand = my_map.toPars.T @ self._Sx_rand @ my_map.toPars
        self._Sx_ret_sys = my_map.toPars.T @ self._Sx_sys @ my_map.toPars

        self._Sx_ret_mapping = (
            self._Sx_ret_smooth + self._Sx_ret_crossState - Sx_smooth_ret
        )
        self._A_ret = kappa @ S_inv @ my_map.toPars
        self._deviationVsErrorSpecies = np.zeros((retrieval.n_species,))
        self._deviationVsRetrievalCovarianceSpecies = np.zeros((retrieval.n_species,))
        self._deviationVsAprioriCovarianceSpecies = np.zeros((retrieval.n_species,))
        self._degreesOfFreedomForSignal = np.zeros((retrieval.n_species,))
        self._degreesOfFreedomNoise = np.zeros((retrieval.n_species,))
        for ii in range(retrieval.n_species):
            m1f = retrieval.parameterStartFM[ii]
            m2f = retrieval.parameterEndFM[ii]
            x = self._A[m1f : m2f + 1, m1f : m2f + 1]

            if (m2f - m1f) > 0:
                self._degreesOfFreedomForSignal[ii] = np.trace(x)
                self._degreesOfFreedomNoise[ii] = (m2f - m1f + 1) - np.trace(x)
            else:
                self._degreesOfFreedomForSignal[ii] = x
                self._degreesOfFreedomNoise[ii] = 1 - x

            m1 = retrieval.parameterStart[ii]
            m2 = retrieval.parameterEnd[ii]

            if not np.all(np.isfinite(kappa)):
                raise RuntimeError("kappa is not finite")

            if not np.all(np.isfinite(kappaFM)):
                raise RuntimeError("kappaFM is not finite")

            if not np.all(np.isfinite(S_inv)):
                raise RuntimeError("S_inv  is not finite")

            if not np.all(np.isfinite(self._Sx_smooth)):
                raise RuntimeError("self._Sx_smooth is not finite")

            if Sb is not None:
                if not np.all(np.isfinite(Sb)):
                    raise RuntimeError("Sb is not finite")

                if not np.all(np.isfinite(kappaInt)):
                    raise RuntimeError("kappaInt is not finite")

                if not np.all(np.isfinite(self._Sx_sys)):
                    raise RuntimeError("self._Sx_sys is not finite")

        # these quantities should be done on the retrieval grid because they
        # are better posed on that grid.
        indf = np.arange(len(retrieval_result.frequency))

        # PYTHON_NOTE: It is possible that the size of self.frequency is greater than size of jacobian.shape[1]
        #
        #     jacobian.shape (x,220)
        #
        # so we cannot not use ind as is an index because that would cause an out of bound:
        # IndexError: index 220 is out of bounds for axis 1 with size 220
        # To fix the issue, we make a reduced_index with indices smaller than the size of jacobian.shape[1]
        if len(retrieval_result.frequency) > jacobian.shape[1]:
            reduced_index = np.where(indf < jacobian.shape[1])[0]
            indf = indf[reduced_index]

        value = np.zeros(shape=(retrieval.n_totalParameters), dtype=np.float64)
        for ii in range(retrieval.n_totalParameters):
            K = jacobian[ii, indf]
            resid_vector = actualDataResidual[indf]
            if np.sum(K * K) > 0:  # when K is very small, k*k becomes zero.
                value[ii] = (
                    np.sum(K * resid_vector)
                    / math.sqrt(np.sum(K * K))
                    / np.sqrt(np.sum(resid_vector * resid_vector))
                )
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL of my_map.toPars NOT FINITE!")

        K = np.copy(jacobian)
        dL = actualDataResidual / dataError
        for jj in range(retrieval.n_totalParameters):
            K[jj, :] = K[jj, :] * dataError

        valueRet = np.zeros(shape=(retrieval.n_totalParameters), dtype=np.float64)
        for ii in range(retrieval.n_totalParameters):
            myK = K[ii, :]
            resid_vector = dL
            # K.dL / |K| / |NESR|
            if np.sum(myK * myK) > 0:  # when K is very small, k*k becomes zero.
                valueRet[ii] = (
                    np.sum(myK * resid_vector)
                    / math.sqrt(np.sum(myK * myK))
                    / math.sqrt(np.sum(resid_vector * resid_vector))
                )
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL is not finite")

        dL = actualDataResidual / dataError
        K = np.copy(jacobianFM)
        for jj in range(retrieval.n_totalParametersFM):
            K[jj, :] = K[jj, :] * dataError

        value = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        for ii in range(retrieval.n_totalParametersFM):
            myK = K[ii, :]
            resid_vector = dL
            # K.dL / |K| / |NESR|
            if np.sum(myK * myK) > 0:  # when K is very small, k*k becomes zero.
                value[ii] = (
                    np.sum(myK * resid_vector)
                    / math.sqrt(np.sum(myK * myK))
                    / math.sqrt(np.sum(resid_vector * resid_vector))
                )
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL is not not finite")

        self._KDotDL_list = valueRet
        self._KDotDL = np.amax(np.abs(value))
        self._KDotDL_species = []
        self._KDotDL_byspecies = np.zeros((retrieval.n_species,))
        for ii in range(retrieval.n_species):
            m1 = retrieval.parameterStart[ii]
            m2 = retrieval.parameterEnd[ii]
            self._KDotDL_species.append(retrieval.speciesList[m1])
            self._KDotDL_byspecies[ii] = np.amax(np.abs(self._KDotDL_list[m1 : m2 + 1]))

        # dict preserves order
        unique_filters = list(dict.fromkeys(retrieval_result.filter_list))
        self._LDotDL_byfilter = np.zeros((len(unique_filters),))
        for ii in range(len(unique_filters)):
            start = retrieval_result.filterStart[ii]
            endd = retrieval_result.filterEnd[ii]
            if start >= 0:
                x1 = actualDataResidual[start : endd + 1]
                x2 = retrieval_result.radiance[0, start : endd + 1]
                self._LDotDL_byfilter[ii] = (
                    np.sum(x1 * x2)
                    / math.sqrt(np.sum(x1 * x1))
                    / math.sqrt(np.sum(x2 * x2))
                )

        self._LDotDL = self._LDotDL_byfilter[0]
        self._KDotDL_byfilter = np.zeros((len(unique_filters),))
        for ii in range(len(unique_filters)):
            v1 = retrieval_result.filterStart[ii]
            v2 = retrieval_result.filterEnd[ii]
            if v1 > 0:
                dL = actualDataResidual / dataError
                K = np.copy(jacobianFM)
                for jj in range(retrieval.n_totalParametersFM):
                    K[jj, :] = K[jj, :] * dataError
                value = np.zeros(
                    shape=(retrieval.n_totalParametersFM), dtype=np.float64
                )
                for kk in range(retrieval.n_totalParametersFM):
                    myK = K[kk, v1 : v2 + 1]
                    resid_vector = dL[v1 : v2 + 1]
                    # K.dL / |K| / |NESR|
                    if np.sum(myK * myK) > 0:  # when K is very small, k*k becomes zero.
                        value[kk] = (
                            np.sum(myK * resid_vector)
                            / math.sqrt(np.sum(myK * myK))
                            / math.sqrt(np.sum(resid_vector * resid_vector))
                        )
                        if not np.all(np.isfinite(value[kk])):
                            raise RuntimeError("KDotDL of jacobianFM is not finite")
                self._KDotDL_byfilter[ii] = np.amax(np.abs(value))

        if retrieval.n_totalParametersSys > 0 and jacobianSys is not None:
            logger.warning(
                "This section of the code for retrieval.n_totalParametersSys has not been tested."
            )
            value = np.zeros(shape=(retrieval.n_totalParametersSys), dtype=np.float64)
            for ii in range(retrieval.n_totalParametersSys):
                K = jacobianSys[ii, :]
                # K.dL / |K| / |NESR|
                if np.sum(K * K) > 0:  # when K is very small, K*K becomes zero.
                    value[ii] = (
                        np.sum(K * actualDataResidual[:])
                        / math.sqrt(np.sum(K * K))
                        / math.sqrt(np.sum(actualDataResidual * actualDataResidual))
                    )
                    if not np.all(np.isfinite(value[ii])):
                        raise RuntimeError("KDotDL of jacobianSys is not finite")
            self._maxKDotDLSys = np.amax(np.abs(value))
        else:
            self._maxKDotDLSys = 0.0

        Sx = self._Sx
        Sx_rand = self._Sx_rand

        self._errorFM = np.zeros((retrieval.n_totalParametersFM,))
        self._precision = np.zeros((retrieval.n_totalParametersFM,))
        for is_index in range(retrieval.n_totalParametersFM):
            self._errorFM[is_index] = np.sqrt(Sx[is_index, is_index])
            self._precision[is_index] = np.sqrt(Sx_rand[is_index, is_index])

        # this is a new diagnostic 3/13/2015
        # it is the change in all parameters from each spectral point
        # the result is something the size of the FM Jacobian
        # see if (1) spikey things causing issues
        # (2) one band forcing one way vs. another forcing another
        # (3) see which residuals are trying to pull where

        # G is gain matrix
        # actualDataResidual is fit - observed; also includes measurement error
        # G ## actualDataResidual should be 0

        if retrieval.n_totalParametersFM == 1:
            GdL = G_matrix * actualDataResidual
        else:
            GdL = G_matrix * 0
            for jj in range(len(actualDataResidual)):
                GdL[jj, :] = G_matrix[jj, :] * actualDataResidual[jj]
        self._GdL = GdL
        ind = [
            idx for idx, value in enumerate(retrieval.speciesListFM) if value == "CH4"
        ]

        # Eventhough we are not doing special processing, we will still need to calculate the self._ch4_evs field.
        # Also, only calculate the self._ch4_evs field. if 'CH4' is in retrieval.speciesListFM.
        calculate_evs_field_flag = True
        if len(ind) > 10 and calculate_evs_field_flag:
            pp = retrieval.pressureListFM[ind]
            (wmatrix, svmatrix, vmatrix) = np.linalg.svd(
                np.transpose(jacobianFM[ind, :]), full_matrices=True
            )
            num_pp_elements = len(pp)
            mydiff = resultVector[ind] - constraintVector[ind]
            dots = np.zeros(shape=(num_pp_elements), dtype=np.float64)
            for jj in range(num_pp_elements):
                dots[jj] = np.sum(vmatrix[jj, :] * mydiff)
            self._ch4_evs = np.abs(
                dots[0 : 9 + 1]
            )  # ratio of use of 1st two vs. next 8 eVs

        return o_offDiagonalSys

    @property
    def Sb(self) -> np.ndarray | None:
        return self._Sb

    @property
    def Sa(self) -> np.ndarray:
        return self._Sa

    @property
    def Sa_ret(self) -> np.ndarray:
        return self._Sa_ret

    @property
    def KtSyK(self) -> np.ndarray:
        return self._KtSyK

    @property
    def KtSyKFM(self) -> np.ndarray:
        return self._KtSyKFM

    @property
    def Sx(self) -> np.ndarray:
        return self._Sx

    @property
    def Sx_smooth(self) -> np.ndarray:
        return self._Sx_smooth

    @property
    def Sx_ret_smooth(self) -> np.ndarray:
        return self._Sx_ret_smooth

    @property
    def Sx_smooth_self(self) -> np.ndarray:
        return self._Sx_smooth_self

    @property
    def Sx_crossState(self) -> np.ndarray:
        return self._Sx_crossState

    @property
    def Sx_ret_crossState(self) -> np.ndarray:
        return self._Sx_ret_crossState

    @property
    def Sx_rand(self) -> np.ndarray:
        return self._Sx_rand

    @property
    def Sx_ret_rand(self) -> np.ndarray:
        return self._Sx_ret_rand

    @property
    def Sx_ret_mapping(self) -> np.ndarray:
        return self._Sx_ret_mapping

    @property
    def Sx_sys(self) -> np.ndarray:
        return self._Sx_sys

    @property
    def Sx_ret_sys(self) -> np.ndarray:
        return self._Sx_ret_sys

    @property
    def radianceResidualRMSSys(self) -> float:
        return self._radianceResidualRMSSys

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def A_ret(self) -> np.ndarray:
        return self._A_ret

    @property
    def GMatrix(self) -> np.ndarray:
        return self._GMatrix

    @property
    def GMatrixFM(self) -> np.ndarray:
        return self._GMatrixFM

    @property
    def deviationVsErrorSpecies(self) -> np.ndarray:
        return self._deviationVsErrorSpecies

    @property
    def deviationVsRetrievalCovarianceSpecies(self) -> np.ndarray:
        return self._deviationVsRetrievalCovarianceSpecies

    @property
    def deviationVsAprioriCovarianceSpecies(self) -> np.ndarray:
        return self._deviationVsAprioriCovarianceSpecies

    @property
    def degreesOfFreedomForSignal(self) -> np.ndarray:
        return self._degreesOfFreedomForSignal

    @property
    def degreesOfFreedomNoise(self) -> np.ndarray:
        return self._degreesOfFreedomNoise

    @property
    def KDotDL_list(self) -> np.ndarray:
        return self._KDotDL_list

    @property
    def KDotDL(self) -> float:
        return self._KDotDL

    @property
    def KDotDL_species(self) -> list[str]:
        return self._KDotDL_species

    @property
    def KDotDL_byspecies(self) -> np.ndarray:
        return self._KDotDL_byspecies

    @property
    def LDotDL(self) -> float:
        return self._LDotDL

    @property
    def KDotDL_byfilter(self) -> np.ndarray:
        return self._KDotDL_byfilter

    @property
    def maxKDotDLSys(self) -> float:
        return self._maxKDotDLSys

    @property
    def errorFM(self) -> np.ndarray:
        return self._errorFM

    @property
    def precision(self) -> np.ndarray:
        return self._precision

    @property
    def GdL(self) -> np.ndarray:
        return self._GdL

    @property
    def ch4_evs(self) -> np.ndarray:
        return self._ch4_evs


__all__ = [
    "ErrorAnalysis",
]
