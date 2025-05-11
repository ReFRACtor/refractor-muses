from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
import copy
import numpy as np
import numpy.testing as npt
import math
from loguru import logger
from scipy.linalg import block_diag  # type: ignore
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .current_state import CurrentState
    from .muses_strategy import CurrentStrategyStep
    from .retrieval_result import RetrievalResult
    from .identifier import StateElementIdentifier


class ErrorAnalysis:
    """This just groups together some of the error analysis stuff
    together, to put this together. Nothing more than a shuffling
    around of stuff already in muses-py

    """

    def __init__(
        self,
        current_state: CurrentState,
        current_strategy_step: CurrentStrategyStep,
        covariance_state_element_name: list[StateElementIdentifier],
    ) -> None:
        # Need to remove usage in retrieval_info.py and cloud_result_summary before
        # removing here. However we don't use this in ErrorAnalysis any longer
        self.error_current = mpy.ObjectView(self.initialize_error_initial(
            current_state, current_strategy_step, covariance_state_element_name
        ))
        current_state.setup_previous_aposteriori_cov_fm(covariance_state_element_name,
                                                        current_strategy_step)

    def initialize_error_initial(
        self,
        current_state: CurrentState,
        current_strategy_step: CurrentStrategyStep,
        covariance_state_element_name: list[StateElementIdentifier],
    ) -> dict | mpy.ObjectView:
        """covariance_state_element_name should be the list of state
        elements we need covariance from. This is all the elements we
        will retrieve, plus any interferents that get added in. This
        list is unique elements, sorted by the order_species sorting

        """
        selem_list = []
        for sname in covariance_state_element_name:
            selem = current_state.full_state_element(sname)
            # Note clear why, but we get slightly different results if we
            # update the original state_info. May want to track this down,
            # but as a work around we just copy this. This is just needed
            # to get the mapping type, I don't think anything else is
            # needed. We should be able to pull that out from the full
            # initial guess update at some point, so we don't need to do
            # the full initial guess
            selem = copy.deepcopy(selem)
            if hasattr(selem, "update_initial_guess"):
                selem.update_initial_guess(current_strategy_step)
            selem_list.append(selem)

        # Note the odd seeming "capitalize" here. This is because
        # get_prior_error uses the map type to look up files, and
        # rather than "linear" or "log" it needs "Linear" or "Log"
        pressure_list: list[float] = []
        species_list = []
        map_list = []

        # Make block diagonal covariance.
        matrix_list = []
        # AT_LINE 25 get_prior_error.pro
        for selem in selem_list:
            matrix = selem.apriori_cov_fm
            plist = selem.pressure_list_fm
            if plist is not None:
                pressure_list.extend(plist)
            else:
                # Convention for elements not on a pressure grid is to use a single
                # -2
                pressure_list.extend(np.array([-2.0]))
            species_list.extend([str(selem.state_element_id)] * matrix.shape[0])
            matrix_list.append(matrix)
            smap = selem.state_mapping
            if isinstance(smap, rf.StateMappingLinear):
                mtype = "linear"
            elif isinstance(smap, rf.StateMappingLog):
                mtype = "log"
            else:
                raise RuntimeError(f"Don't recognize state mapping {smap}")
            map_list.extend([mtype] * matrix.shape[0])

        initial = block_diag(*matrix_list)
        # Off diagonal blocks for covariance.
        for i, selem1 in enumerate(selem_list):
            for selem2 in selem_list[i + 1 :]:
                matrix2 = selem1.apriori_cross_covariance_fm(selem2)
                if matrix2 is not None:
                    initial[np.array(species_list) == str(selem1.state_element_id), :][
                        :, np.array(species_list) == str(selem2.state_element_id)
                    ] = matrix2
                    initial[np.array(species_list) == str(selem2.state_element_id), :][
                        :, np.array(species_list) == str(selem1.state_element_id)
                    ] = np.transpose(matrix2)
        return mpy.constraint_data(
            initial, pressure_list, [str(i) for i in species_list], map_list
        )

    def update_retrieval_result(self, retrieval_result: RetrievalResult) -> None:
        """Update the retrieval_result and ErrorAnalysis. The retrieval_result
        are updated in place."""
        fstate_info = FakeStateInfo(retrieval_result.current_state)
        fretrieval_info = FakeRetrievalInfo(retrieval_result.current_state)
        # Updates self.error_current and retrieval_result in place
        self.error_analysis_wrapper(
            retrieval_result.rstep,
            fretrieval_info,
            fstate_info,
            retrieval_result,
        )
        retrieval_result.update_error_analysis(self)

    def error_analysis_wrapper(
            self,
            radiance : mpy.ObjectView,
            retrieval : FakeRetrievalInfo,
            stateInfo : FakeStateInfo,
            retrieval_result : RetrievalResult) -> None:
        self.set_retrieval_results(retrieval_result)
        # expected noise
        data_error = radiance.NESR
        bad_pixel = data_error < 0

        cstate = retrieval_result.current_state
        if(cstate.map_to_parameter_matrix is None or cstate.basis_matrix is None):
            raise RuntimeError("Missing basis matrix")
        my_map = mpy.ObjectView({
            'toPars': np.copy(cstate.map_to_parameter_matrix), 
            'toState': np.copy(cstate.basis_matrix)
        })
    
        # result is jacobian[pars, frequency]
        jacobian = retrieval_result.jacobian[0,:,:].copy() # type: ignore[attr-defined]
        jacobian[:, bad_pixel] = 0
    
        actual_data_residual = retrieval_result.radiance[0,:] - radiance.radiance  # type: ignore[attr-defined]
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
    
        if jacobian_sys is not None:
            jacobian_sys = jacobian_sys[0,:,:].copy()
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
            
            # PYTHON_NOTE: Because the slices does not include the end, we have to add 1 to the end of the slice.
            ret_vector[ind1FM:ind2FM+1] = mpy.get_vector(retrieval_result.resultsList, retrieval, species_name, FM_Flag, INITIAL_Flag, TRUE_Flag, CONSTRAINT_Flag)  # type: ignore[attr-defined]
            con_vector[ind1FM:ind2FM+1] = mpy.get_vector(retrieval.constraintVector, retrieval, species_name, FM_Flag, INITIAL_Flag, TRUE_Flag, CONSTRAINT_Flag)
    
            if retrieval.mapType[ispecie].lower() == 'log':  # Note the spelling of 'mapType' in retrieval object.
                # PYTHON_NOTE: Because the slices does not include the end, we have to add 1 to the end of the slice.
                ret_vector[ind1FM:ind2FM+1] = np.log(ret_vector[ind1FM:ind2FM+1])
                con_vector[ind1FM:ind2FM+1] = np.log(con_vector[ind1FM:ind2FM+1])
        # end for ispecie in range(retrieval.n_species):
    
        # AT_LINE 204 Error_Analysis_Wrapper.pro
        constraintVector = con_vector
        resultVector = ret_vector
    
    
        # if not updating, keep current error analysis
        currentSpecies = retrieval.species[0:retrieval.n_species]
    
        # AT_LINE 256 Error_Analysis_Wrapper.pro
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
            cstate.error_current_values)

        cstate.update_previous_aposteriori_cov_fm(self.Sx, offDiagonalSys)

        # Only needed by CloudResultSummary
        currentSpecies = retrieval.species[0:retrieval.n_species]
        self.error_current = mpy.constraint_clear(self.error_current.__dict__, currentSpecies)
        self.error_current = mpy.constraint_set(self.error_current.__dict__, self.Sx, currentSpecies)  # type: ignore[attr-defined]
        if jacobian_sys is not None and offDiagonalSys is not None:
            # set block and transpose of block
            my_ind = np.where(np.asarray(retrieval.parameterEndSys) > 0)[0]
    
            # The following function constraint_set_subblock() updates the 'data' portion of errorCurrent structure.
            # Call twice with the last two parameters swapped.
            self.error_current = mpy.constraint_set_subblock(
                self.error_current, 
                np.transpose(offDiagonalSys), 
                np.asarray(currentSpecies), 
                np.asarray(retrieval.speciesSys[0:len(my_ind)])
            )
            
            self.error_current = mpy.constraint_set_subblock(
                self.error_current, 
                offDiagonalSys, 
                np.asarray(retrieval.speciesSys[0:len(my_ind)]), 
                np.asarray(currentSpecies)
            )

    # see: https://ieeexplore.ieee.org/document/1624609
    # Tropospheric Emission Spectrometer: Retrieval Method and Error Analysis
    # V. ERROR CHARACTERIZATION
    
    def error_analysis(
            self,
            my_map : mpy.ObjectView,
            jacobian : np.ndarray,
            jacobianFM : np.ndarray,
            Sa : np.ndarray,
            jacobianSys : np.ndarray | None,
            Sb : np.ndarray | None,
            constraintMatrix : np.ndarray,
            constraintVector : np.ndarray,
            resultVector : np.ndarray,
            dataError : np.ndarray,
            actualDataResidual : np.ndarray,
            retrieval_result : RetrievalResult,
            retrieval : FakeRetrievalInfo,
            errorCurrentValues : mpy.ObjectView) -> np.ndarray | None:
        
        o_offDiagonalSys = None
    
        kappa = jacobian @ jacobian.T
        kappaFM = jacobianFM @ jacobian.T # [parameter,frequency]
        S_inv = np.linalg.inv(kappa + constraintMatrix) @ my_map.toState
        self._KtSyK = kappa
        self._KtSyKFM = kappaFM
    
        doUpdateFM = retrieval.doUpdateFM[0:retrieval.n_totalParametersFM]
        dontUpdateFM = 1 - doUpdateFM
    
        my_id = np.asarray(np.identity(S_inv.shape[1]), dtype=np.float64)
        self._Sx_smooth = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)
        self._Sx_rand = S_inv.T @ kappa @ S_inv
    
        if jacobianSys is not None and Sb is not None:
            kappaInt = np.matmul(jacobianSys, np.transpose(jacobian))
            self._Sx_sys = np.matmul(np.matmul(np.matmul(np.transpose(np.matmul(kappaInt, S_inv)), Sb), kappaInt), S_inv)
            sys = np.zeros(shape=(len(jacobianSys[0, :])), dtype=np.float64)
            for ii in range(len(jacobianSys[0, :])):
                sys[ii] = np.matmul(np.matmul(np.transpose(jacobianSys[:, ii]), Sb), jacobianSys[:, ii])
            self._radianceResidualRMSSys = math.sqrt(np.sum(sys) / sys.size)
            o_offDiagonalSys = np.matmul(np.matmul(Sb, kappaInt), -S_inv)
        else:
            self._Sx_sys = np.zeros(self._Sx_rand.shape)
            self._radianceResidualRMSSys = 0.0
    
        self._Sx = (self._Sx_smooth + self._Sx_rand + self._Sx_sys)
        self._A = np.matmul(kappaFM, S_inv)
    
        G_matrix = np.matmul(np.transpose(jacobian), S_inv) 
        for jj in range(G_matrix.shape[1]):
            G_matrix[:, jj] /=  dataError
        self._GMatrix = np.matmul(G_matrix, my_map.toPars)
        self._GMatrixFM = G_matrix
    
        # AT_LINE 211 Error_Analysis.pro
        # some species, like emis, are retrieved in step 1 for a particular spectral region and not updated following this.  In that case, keep errors when they are not moved.  If this is the case, set the error to the previous error, and all error components to zero.
        ind = np.where(dontUpdateFM == 1)[0]
    
        if len(ind) > 0:
            self._Sx[ind, ind] = errorCurrentValues[ind, ind]
            self._Sx_rand[ind, ind] = 0
            self._Sx_smooth[ind, ind] = 0
            self._Sx_sys[ind, ind] = 0
            
        my_id = np.identity(S_inv.shape[1], dtype=np.float64)
        self._Sx_smooth[:, :] = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)
        
        self._Sx_smooth_self[:, :] = 0
        self._Sx_crossState[:, :] = self._Sx_smooth[:, :]
    
        # override the block diagonal terms with smooth_self and crossstate
        species_list_fs = np.asarray(retrieval.speciesListFM)
        for ii in range(retrieval.n_species):
    
            # previous (wrong) code
            # species = retrieval.species[ii]
            # indMe = np.where(species_list_fs == species)[0]
            # indYou = np.where(species_list_fs != species)[0]
    
            # update 4/13/2021 to match ssund version
            species = retrieval.species[ii]
            indMe0 = np.where(species_list_fs == species)[0]
    
            #  also include in indMe anything that has off-diagonal Sa elements
            if len(indMe0) > 1:
                ss = mpy.my_total(abs(Sa[indMe0, :]), 0)
                indMe = np.where(ss > 0)[0]
                indYou = np.where(ss == 0)[0]
            else:
                ss = Sa[indMe0, :][0]
                indMe = np.where(ss > 0)[0]
                indYou = np.where(ss == 0)[0]
    
            # self-smooth
            if len(indMe) == 1:
                self._Sx_smooth_self[indMe, indMe] = (1 - self._A[indMe, indMe]) * Sa[indMe, indMe] * (1 - self._A[indMe, indMe])
            else:
                # IDL:
                # id = IDENTITY(N_ELEMENTS(indMe))
                # res.Sx_Smooth_self[indMe,indMe,*] = $
                # (id - res.A[indMe,indMe,*]) ## Sa[indMe,indMe,*] ## Transpose(id - res.A[indMe, indMe,*])
                
                my_id = np.identity(len(indMe))
                ind_me_2d = np.ix_(indMe, indMe)
                self._Sx_smooth_self[ind_me_2d] = (my_id - self._A[ind_me_2d]).T @ Sa[ind_me_2d] @ (my_id - self._A[ind_me_2d])
            # end if len(indMe) == 1:
    
            # interferent on and off diagonal
            if len(indYou) == 0:
                self._Sx_crossState[:, :] = 0
            elif len(indMe) == 1 and len(indYou) == 1:
                # IDL:
                # temp = res.A[indYou,indMe,*] * Sa[indYou,indYou] * res.A[indYou, indMe]
                # res.Sx_crossState[indMe,indMe] = temp
                
                temp = self._A[indYou, indMe] * Sa[indYou, indYou] * self._A[indYou, indMe]
                self._Sx_crossState[indMe, indMe] = temp
            elif len(indMe) == 1 and len(indYou) > 1:
                # IDL:
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou,*] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe] = temp[*]                  
    
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
                
                temp = self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
    
                ind_me_me = np.ix_(indMe, indMe) 
                self._Sx_crossState[indMe, indMe] = temp 
            elif len(indMe) > 1 and len(indYou) == 1:
                # IDL: 
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe,*] = temp
    
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
    
                temp = self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
                
                ind_me_me = np.ix_(indMe, indMe) 
                self._Sx_crossState[ind_me_me] = temp[:, :]
            else:
                # IDL:
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou,*] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe,*] = temp
                
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
    
                temp = self._A[ind_you_me].T @ Sa[ind_you_you] @ self._A[ind_you_me]
                
                ind_me_me = np.ix_(indMe, indMe) 
                self._Sx_crossState[ind_me_me] = temp[:, :]
        # end for ii in range(retrieval.n_species):
    
        # TODO: implement
        # only sa_ret and sx_smooth_ret is needed from the block below
        #  
        # ; estimate mapping error
        # ; Compare error analysis on FM grid vs error analysis on retrieval
        # ; grid mapped to FM grid
        # ; Calculate total errors, then subtract
    
        # ak_ret = INVERT(kappa + constraint) ## kappa
        
        # IDL:
        # m = map.toPars
        # Sa_ret = m ## Sa ## Transpose(m)
        # res.sa_ret = sa_ret
    
        Sa_ret = my_map.toPars.T @ Sa @ my_map.toPars
        self._Sa_ret[:, :] = Sa_ret
    
        # IDL:
        # S_inv_ret = invert(kappa + constraint)
        # id = IDENTITY(N_ELEMENTS(S_inv_ret[0,*]),/DOUBLE)
        # sx_smooth_ret = (id - S_inv_ret ## kappa) ## Sa_ret ## TRANSPOSE(id - S_inv_ret ## kappa)
        # sx_smooth_ret_fm = map.toState ## sx_smooth_ret ## transpose(map.toState)
        
        S_inv_ret = np.linalg.inv(kappa + constraintMatrix)
        my_id = np.identity(S_inv_ret.shape[1], dtype=np.float64)
        Sx_smooth_ret = (my_id - kappa @ S_inv_ret).T @ Sa_ret @ (my_id - kappa @ S_inv_ret)
        Sx_smooth_ret_fm = my_map.toState.T @ Sx_smooth_ret @ my_map.toState
    
        # mapping error:  compare sx_smooth_ret_fm and res.Sx_smooth
        # mapping error from choice of retrieval levels
        self._Sx_mapping[:, :] = self._Sx_smooth - Sx_smooth_ret_fm
    
        # map error covariances to retrieval grid
    
        # IDL:
        # smooth = map.toPars ## res.Sx_Smooth_self ## transpose(map.toPars)
        # cross = map.toPars ## res.Sx_crossState ## transpose(map.toPars)
        # rand = map.toPars ## res.Sx_rand ## transpose(map.toPars)
        # sys = map.toPars ## res.Sx_sys ## transpose(map.toPars)
    
        smooth = my_map.toPars.T @ self._Sx_smooth_self @ my_map.toPars
        cross = my_map.toPars.T @ self._Sx_crossState @ my_map.toPars
        rand = my_map.toPars.T @ self._Sx_rand @ my_map.toPars
        sys = my_map.toPars.T @ self._Sx_sys @ my_map.toPars
    
        self._Sx_ret_smooth[:, :] = smooth
        self._Sx_ret_crossState[:, :] = cross
        self._Sx_ret_rand[:, :] = rand
        self._Sx_ret_sys[:, :] = sys
    
        #  mapping error from choice of retrieval levels
        self._Sx_ret_mapping[:, :] = smooth + cross - Sx_smooth_ret    
        
        # IDL:
        # res.A_ret = map.toPars ## S_Inv ## kappa
        self._A_ret = kappa @ S_inv @ my_map.toPars
    
        # AT_LINE 524 Error_Analysis.pro
        self._deviationVsErrorSpecies[:] = 0
        self._deviationVsRetrievalCovarianceSpecies[:] = 0
        self._deviationVsAprioriCovarianceSpecies[:] = 0
        
        # AT_LINE 531 Error_Analysis.pro
        for ii in range(retrieval.n_species):
            m1f = retrieval.parameterStartFM[ii]
            m2f = retrieval.parameterEndFM[ii]
            x = self._A[m1f:m2f+1, m1f:m2f+1]
    
            # AT_LINE 535 Error_Analysis.pro
            if (m2f-m1f) > 0:
                self._degreesOfFreedomForSignal[ii] = np.trace(x)
                self._degreesOfFreedomNoise[ii] = (m2f-m1f+1) - np.trace(x)
            else:
                self._degreesOfFreedomForSignal[ii] = x
                self._degreesOfFreedomNoise[ii] = 1 - x
            # end else part of if (m2f-m1f) > 0:
    
            # AT_LINE 545 Error_Analysis.pro
            m1 = retrieval.parameterStart[ii]
            m2 = retrieval.parameterEnd[ii]
    
            # if retrieval.species[ii] != 'EMIS' and retrieval.species[ii] != 'CLOUDEXT':
            #     resolution = calculate_resolution(result.A[m1f:m2f+1, m1f:m2f+1], heightKm)
            #     result.resolution[m1f:m2f+1] = resolution[:]
    
            # AT_LINE 554 Error_Analysis.pro
            # check finite for all results
            if not np.all(np.isfinite(kappa)):
                raise RuntimeError("kappa is not finite")
            
            if not np.all(np.isfinite(kappaFM)):
                raise RuntimeError("kappaFM is not finite")
    
            if not np.all(np.isfinite(S_inv)):
                raise RuntimeError("S_inv  is not finite")
            
            if not np.all(np.isfinite(self._Sx_smooth)):
                raise RuntimeError("self._Sx_smooth is not finite")
    
            # AT_LINE 579 Error_Analysis.pro
            if Sb is not None:
                if not np.all(np.isfinite(Sb)):
                    raise RuntimeError("Sb is not finite")
                
                if not np.all(np.isfinite(kappaInt)):
                    raise RuntimeError("kappaInt is not finite")
    
                if not np.all(np.isfinite(self._Sx_sys)):
                    raise RuntimeError("self._Sx_sys is not finite")
                
        # end for ii in range(retrieval.n_species):
        # AT_LINE 607 Error_Analysis.pro
    
        # AT_LINE 621 Error_Analysis.pro
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
            if np.sum(K*K) > 0: # when K is very small, k*k becomes zero.
                value[ii] = np.sum(K *resid_vector) / math.sqrt(np.sum(K * K)) / np.sqrt(np.sum(resid_vector * resid_vector))
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL of my_map.toPars NOT FINITE!")
            # end part of if np.sum(K*K) >  0: # when K is very small, k*k becomes zero.
        # end for ii in range(retrieval.n_totalParameters):
    
        # AT_LINE 643 Error_Analysis.pro
        #ret grid for vector
        K = np.copy(jacobian)  # Make a copy so we can mess with K.
        dL = actualDataResidual / dataError
        for jj in range(retrieval.n_totalParameters):
            K[jj, :] = K[jj, :] * dataError
        
        valueRet = np.zeros(shape=(retrieval.n_totalParameters), dtype=np.float64)
        for ii in range(retrieval.n_totalParameters):
            myK = K[ii, :]
            resid_vector = dL
            # K.dL / |K| / |NESR|
            if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
                valueRet[ii] = np.sum(myK *resid_vector) / math.sqrt(np.sum(myK * myK)) / math.sqrt(np.sum(resid_vector * resid_vector))
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL is not finite")
                # end if (np.all(np.isfinite(value[ii])) == False):
            # end part of  if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
        # end for ii in range(retrieval.n_totalParameters):
    
        # AT_LINE 663 Error_Analysis.pro
        # now duplicate PGE, which is on FM grid and with unnormalized K and
        # normalized L
        dL = actualDataResidual / dataError
        K = np.copy(jacobianFM)
        for jj in range(retrieval.n_totalParametersFM):
            K[jj, :] = K[jj, :] * dataError
        
        value = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        for ii in range(retrieval.n_totalParametersFM):
            myK = K[ii, :]
            resid_vector = dL
            # K.dL / |K| / |NESR|
            if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
                value[ii] = np.sum(myK *resid_vector) / math.sqrt(np.sum(myK * myK)) / math.sqrt(np.sum(resid_vector * resid_vector))
                if not np.all(np.isfinite(value[ii])):
                    raise RuntimeError("KDotDL is not not finite")
            # end if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
        # end part of for ii in range(retrieval.n_totalParametersFM):
    
        # AT_LINE 687 Error_Analysis.pro
        self._KDotDL_list = valueRet
        self._KDotDL = np.amax(np.abs(value))
    
        # get max for each species 
        for ii in range(retrieval.n_species):
            m1 = retrieval.parameterStart[ii]
            m2 = retrieval.parameterEnd[ii]
            self._KDotDL_species[ii] = retrieval.speciesList[m1]
            self._KDotDL_byspecies[ii] = np.amax(np.abs(self._KDotDL_list[m1:m2+1]))
        # end for ii in range(retrieval.n_species):
    
    
        # AT_LINE 699 Error_Analysis.pro
        # get kdotDL and LDotDL by filter
    
        # Just in case the filter_list is not unique, we try to make a unique list with GetUniqueValues() function.
        utilList = mpy.UtilList()
        unique_filters = utilList.GetUniqueValues(retrieval_result.filter_list)
        for ii in range(len(unique_filters)):
            start = retrieval_result.filterStart[ii]
            endd = retrieval_result.filterEnd[ii]
            if start >= 0:
                x1 = actualDataResidual[start:endd+1]
                # For some reason, the radiance dimension may be 2 with the first dimension being 1 [1,442]
                if len(retrieval_result.radiance.shape) == 2:
                    if retrieval_result.radiance.shape[0] == 1:         # This case: [1,442]
                        x2 = retrieval_result.radiance[0, start:endd+1]   # For [1,442] we just get the first row.
                    else:
                        logger.error("This function does not know how to handle retrieval_result.radiance with shape", retrieval_result.radiance.shape)
                        assert False
                else:
                    x2 = retrieval_result.radiance[start:endd+1]
    
                self._LDotDL_byfilter[ii] = np.sum(x1*x2) / math.sqrt(np.sum(x1*x1)) / math.sqrt(np.sum(x2*x2))
            # end if start >= 0:
        # end for ii in range(len(unique_filters)):
    
        self._LDotDL = self._LDotDL_byfilter[0]
    
        # AT_LINE 712 Error_Analysis.pro
        for ii in range(len(unique_filters)):
            v1 = retrieval_result.filterStart[ii]
            v2 = retrieval_result.filterEnd[ii]
            if v1 > 0:
                dL = actualDataResidual / dataError
                K = np.copy(jacobianFM)
                for jj in range(retrieval.n_totalParametersFM):
                    K[jj, :] = K[jj, :] * dataError
                value = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
                for kk in range(retrieval.n_totalParametersFM):
                    myK = K[kk, v1:v2+1]
                    resid_vector = dL[v1:v2+1]
                    # K.dL / |K| / |NESR|
                    if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
                        value[kk] = np.sum(myK *resid_vector) / math.sqrt(np.sum(myK * myK)) / math.sqrt(np.sum(resid_vector * resid_vector))
                        if not np.all(np.isfinite(value[kk])):
                            raise RuntimeError("KDotDL of jacobianFM is not finite")
                    # end if np.sum(myK*myK) > 0: # when K is very small, k*k becomes zero.
                # end for kk in range(retrieval.n_totalParametersFM):
                
                self._KDotDL_byfilter[ii] = np.amax(np.abs(value))
        # end for ii in range(len(unique_filters)):
    
        # AT_LINE 740 Error_Analysis.pro
        if retrieval.n_totalParametersSys > 0 and jacobianSys is not None:
            logger.warning("This section of the code for retrieval.n_totalParametersSys has not been tested.")
            # now look at residual dotted into the systematic species
            value = np.zeros(shape=(retrieval.n_totalParametersSys), dtype=np.float64)
            for ii in range(retrieval.n_totalParametersSys):
                K = jacobianSys[ii, :]
                # K.dL / |K| / |NESR|
                if np.sum(K*K) > 0: # when K is very small, K*K becomes zero.
                    value[ii] = np.sum(K * actualDataResidual[:]) / math.sqrt(np.sum(K*K)) / math.sqrt(np.sum(actualDataResidual * actualDataResidual))
                    if not np.all(np.isfinite(value[ii])):
                        raise RuntimeError("KDotDL of jacobianSys is not finite")
                # end if np.sum(K*K) > 0: # when K is very small, K*K becomes zero.
            # end for ii in range(retrieval.n_totalParametersSys):
    
            self._maxKDotDLSys = np.amax(np.abs(value))
        # end part of if retrieval.n_totalParametersSys > 0:
    
        # AT_LINE 758 Error_Analysis.pro
        Sx = self._Sx
        Sx_rand = self._Sx_rand
    
        # AT_LINE 761 Error_Analysis.pro
        for is_index in range(retrieval.n_totalParametersFM):
            self._errorFM[is_index] = np.sqrt(Sx[is_index, is_index])
            self._precision[is_index] = np.sqrt(Sx_rand[is_index, is_index])
        # end for is_index in range(retrieval.n_totalParametersFM):
    
    
        # AT_LINE 769 Error_Analysis.pro
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
    
        # AT_LINE 790 Error_Analysis.pro
        #self._GdL[:] = GdL[:]
        self._GdL = GdL
    
        # AT_LINE 792 Error_Analysis.pro
        ind = utilList.WhereEqualIndices(retrieval.speciesListFM, 'CH4')
        
        # Eventhough we are not doing special processing, we will still need to calculate the self._ch4_evs field.
        # Also, only calculate the self._ch4_evs field. if 'CH4' is in retrieval.speciesListFM.
        calculate_evs_field_flag = True
        if len(ind) > 10 and calculate_evs_field_flag:
            pp = retrieval.pressureListFM[ind]
    
            # IDL:
            # svdc, jacobianfm[ind,*], w, u, v  
    
            # TODO: IDL code uses SVDC which is diffrent from np.linalg.svd
            (wmatrix, svmatrix, vmatrix) = np.linalg.svd(np.transpose(jacobianFM[ind, :]), full_matrices=True)
            
            num_pp_elements = len(pp) # PYTHON_NOTE: Variable 'np' is a reserved word, changed np to num_pp_elements.
    
            # Error_Analysis:jacobianfm[ind,*]
            # <Expression>    DOUBLE    = Array[64, 442]
            # Error_Analysis:id
            # ID              DOUBLE    = Array[64, 2]
            # Error_Analysis:v
            # V               DOUBLE    = Array[64, 64]
            # Error_Analysis:vt
            # VT              DOUBLE    = Array[64, 2]
            # (442,442) and (64,2) not aligned: 442 (dim 1) != 64 (dim 0)
    
            my_id = np.zeros(shape=(num_pp_elements, 3), dtype=np.float64)
            my_id[0, 0] = 1
            my_id[1, 1] = 1
            my_id[2, 2] = 1
            
            # these on FM grid, and log
            mydiff = resultVector[ind] - constraintVector[ind]
    
            # figure out amount of each vector in the result
            dots = np.zeros(shape=(num_pp_elements), dtype=np.float64)
            for jj in range(num_pp_elements):
                dots[jj] = np.sum(vmatrix[jj, :] * mydiff)
    
            # look at the first 2 eV's versus eV 3-10
            self._ch4_evs = np.abs(dots[0:9+1]) # ratio of use of 1st two vs. next 8 eVs
    
        return o_offDiagonalSys

    def set_retrieval_results(self, retrieval_result: RetrievalResult) -> None:
        """This is our own copy of mpy.set_retrieval_results, so we
        can start making changes to clean up the coupling of this.

        """
        # Convert any dict to ObjectView so we can have a consistent
        # way of referring to our input.
        num_species = len(retrieval_result.current_state.retrieval_state_element_id)
        nfreqs = len(retrieval_result.rstep.frequency)
        num_filters = len(retrieval_result.filter_index)
        
        # get the total number of frequency points in all microwindows for the
        # gain matrix
        rows = len(retrieval_result.current_state.retrieval_state_vector_element_list)
        rowsSys = len(retrieval_result.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(retrieval_result.current_state.forward_model_state_vector_element_list)
        if rowsSys == 0:
            rowsSys = 1

        o_results: dict[str, Any] = {
            "_error": np.zeros(shape=(rows), dtype=np.float64),
            "_precision": np.zeros(shape=(rowsFM), dtype=np.float64),
            "_resolution": np.zeros(shape=(rowsFM), dtype=np.float64),
            # jacobians - for last outputStep
            "_GdL": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            "_jacobianSys": None,
            # error stuff follows - calc later
            "_A_ret": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sa_ret": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_ret_smooth": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_ret_crossState": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_ret_rand": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_ret_sys": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_ret_mapping": np.zeros(shape=(rows, rows), dtype=np.float64),
            "_Sx_smooth_self": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "_Sx_crossState": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "_Sx_mapping": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            # by species
            "_informationContentSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "_degreesOfFreedomNoise": np.zeros(shape=(num_species), dtype=np.float64),
            "_degreesOfFreedomForSignal": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "_degreesOfFreedomForSignalTrop": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "_bestDegreesOfFreedomList": ["" for x in range(num_species)],
            "_bestDegreesOfFreedomTotal": ["" for x in range(num_species)],
            "_verticalResolution": np.zeros(shape=(num_species), dtype=np.float64),
            "_deviationVsError": 0.0,
            "_deviationVsRetrievalCovariance": 0.0,
            "_deviationVsAprioriCovariance": 0.0,
            "_deviationVsErrorSpecies": np.zeros(shape=(num_species), dtype=np.float64),
            "_deviationVsRetrievalCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "_deviationVsAprioriCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            # quality and general
            "_KDotDL": 0.0,
            "_KDotDL_list": np.zeros(shape=(rows), dtype=np.float64),
            "_KDotDL_byspecies": np.zeros(shape=(num_species), dtype=np.float64),
            "_KDotDL_species": ["" for x in range(num_species)],
            "_KDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float64),
            "_maxKDotDLSys": 0.0,
        }

        struct2 = {
            "_LDotDL": 0.0,
            "_LDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float64),
            "_calscaleMean": 0.0,
            "_masterQuality": -999,
            # EM NOTE - Modified to increase vector size to allow for stratosphere capture
            "_tsur_minus_tatm0": -999.0,
            "_tsur_minus_prior": -999.0,
            "_ch4_evs": np.zeros(shape=(10), dtype=np.float64),  # FLTARR(10)
        }
        o_results.update(struct2)
        self.__dict__.update(o_results)
        self._errorFM = np.zeros(shape=(rowsFM), dtype=np.float64)
        self._A =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self._Sx =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)

    @property
    def Sb(self) -> np.ndarray:
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
    def LDotDL(self) -> np.ndarray:
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
