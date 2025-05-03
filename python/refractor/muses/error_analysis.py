from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
import copy
import numpy as np
import sys
import os
import math
from loguru import logger
from pprint import pprint
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
        self.error_initial = self.initialize_error_initial(
            current_state, current_strategy_step, covariance_state_element_name
        )
        self.error_current: dict | mpy.ObjectView = copy.deepcopy(self.error_initial)
        # Code seems to assume these are object view.
        self.error_initial = mpy.ObjectView(self.error_initial)
        self.error_current = mpy.ObjectView(self.error_current)

    def snapshot_to_file(self, fname: str | os.PathLike[str]) -> None:
        """py-retrieve is big on having functions with unintended side effects. It can
        be hard to determine what changes when. This writes a complete text dump of
        this object, which we can then diff against other snapshots to see what has
        changed."""
        with np.printoptions(precision=None, threshold=sys.maxsize):
            with open(fname, "w") as fh:
                pprint(
                    {
                        "error_initial": self.error_initial.__dict__,
                        "error_current": self.error_current.__dict__,
                    },
                    stream=fh,
                )

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
        # Updates retrieval_result in place
        self.write_retrieval_summary(
            fretrieval_info,
            fstate_info,
            retrieval_result,
            self.error_current,
        )

    def error_analysis_wrapper(
            self,
            radiance : mpy.ObjectView,
            retrieval : FakeRetrievalInfo,
            stateInfo : FakeStateInfo,
            retrieval_result : RetrievalResult) -> None:
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
    
            # AT_LINE 107 Error_Analysis_Wrapper.pro
            speciesList = retrieval.speciesSys[0:retrieval.n_speciesSys]
            Sb = mpy.constraint_get(self.error_current.__dict__, speciesList)
            retrieval_result.Sb = Sb  # type: ignore[attr-defined]
    
        for ii in range(jacobian.shape[0]):
            jacobian[ii, :] /= data_error
    
        jacobian_fm = np.copy(jacobian)
        jacobian = my_map.toState @ jacobian_fm
    
        # AT_LINE 177 Error_Analysis_Wrapper.pro
        Sa = mpy.constraint_get(self.error_initial.__dict__, retrieval.species[0:retrieval.n_species])
    
        retrieval_result.Sa[:, :] = Sa[:, :]  # type: ignore[attr-defined]
        #breakpoint()
    
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
        errorCurrentValues = mpy.constraint_get(self.error_current.__dict__, currentSpecies)
    
        # AT_LINE 251 Error_Analysis_Wrapper.pro
    
        speciesList = retrieval.speciesList
    
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
            errorCurrentValues)
    
        # AT_LINE 280 Error_Analysis_Wrapper.pro
        speciesList = retrieval.speciesList
    
        # AT_LINE 336 Error_Analysis_Wrapper.pro
        # update errorCurrent
        # first clear correlations between current retrieved species and all
        # others then set error based on current error analysis
        self.error_current = mpy.constraint_clear(self.error_current.__dict__, currentSpecies)
        self.error_current = mpy.constraint_set(self.error_current.__dict__, retrieval_result.Sx, currentSpecies)  # type: ignore[attr-defined]
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
        # end if jacobian_sys is not None:


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
            retrieval_result : Any, # Lots of attributes mypy can't see, so just have the
                                    # type here Any
            retrieval : FakeRetrievalInfo,
            errorCurrentValues : mpy.ObjectView) -> np.ndarray | None:
        
        # Output variables
        o_offDiagonalSys = None
        retrieval_result_dict = retrieval_result.__dict__
    
        # It is possible that these variables can be None: {Sb, Db, constraintMatrix}
    
        #;;;;;;;;;;; intermediate equations (Section 4.2.2) ;;;;;;;;;;
        # AT_LINE 47 Error_Analysis.pro
    
        # IDL:
        # kappa = TRANSPOSE(jacobian) ## jacobian
        # kappaFM = TRANSPOSE(jacobian) ## jacobianFM
    
        kappa = jacobian @ jacobian.T
        kappaFM = jacobianFM @ jacobian.T # [parameter,frequency]
    
        # IDL:
        # S_inv = map.toState ## INVERT(kappa + constraint)
    
        S_inv = np.linalg.inv(kappa + constraintMatrix) @ my_map.toState
        
        # AT_LINE 100 Error_Analysis.pro
        retrieval_result.KtSyK = kappa
    
        if 'KtSyKFM' in retrieval_result_dict:
            retrieval_result.KtSyKFM = kappaFM
        retrieval_result.KtSyKFM = kappaFM
    
        doUpdateFM = retrieval.doUpdateFM[0:retrieval.n_totalParametersFM]
        dontUpdateFM = 1 - doUpdateFM
    
        # equations (Section 4.2.3)
        # AT_LINE 109 Error_Analysis.pro
        if constraintMatrix is None:
            logger.warning("constraintMatrix is None. Cannot calculate my_id, Sx_smooth and Sx_rand")
        else:
            my_id = np.asarray(np.identity(S_inv.shape[1]), dtype=np.float64)
            retrieval_result.Sx_smooth = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)
            retrieval_result.Sx_rand = S_inv.T @ kappa @ S_inv
    
        # AT_LINE 120 Error_Analysis.pro
        if jacobianSys is not None and Sb is not None:
            kappaInt = np.matmul(jacobianSys, np.transpose(jacobian))
            retrieval_result.Sx_sys = np.matmul(np.matmul(np.matmul(np.transpose(np.matmul(kappaInt, S_inv)), Sb), kappaInt), S_inv)
    
            # get expected error in radianceResidualRMS from sys error
            sys = np.zeros(shape=(len(jacobianSys[0, :])), dtype=np.float64)
            for ii in range(len(jacobianSys[0, :])):
                sys[ii] = np.matmul(np.matmul(np.transpose(jacobianSys[:, ii]), Sb), jacobianSys[:, ii])
            retrieval_result.radianceResidualRMSSys = math.sqrt(np.sum(sys) / sys.size)
    
            # AT_LINE 134 Error_Analysis.pro
            o_offDiagonalSys = np.matmul(np.matmul(Sb, kappaInt), -S_inv)
        else:
            retrieval_result.Sx_sys = copy.deepcopy(retrieval_result.Sx_rand)
            retrieval_result.Sx_sys[:] = 0
        # end if jacobianSys is not None:
    
        # AT_LINE 162 Error_Analysis.pro
        retrieval_result.Sx[:, :] = (retrieval_result.Sx_smooth + retrieval_result.Sx_rand + retrieval_result.Sx_sys)[:, :]
        retrieval_result.A[:, :] = np.matmul(kappaFM, S_inv)[:, :]
    
        # AT_LINE 179 Error_Analysis.pro
        # find actual GN step
        G_matrix = np.matmul(np.transpose(jacobian), S_inv)    # PYTHON_NOTE: Changed G to G_matrix so we can easily find this variable.
        for jj in range(G_matrix.shape[1]):
            G_matrix[:, jj] = G_matrix[:, jj] / dataError
    
        # AT_LINE 183 Error_Analysis.pro
        gainRet = np.matmul(G_matrix, my_map.toPars)
    
        # PYTHON_NOTE: It is possible that the size of gainRet and  result.GMatrix are different.
        # If that is the case, we shrink to the smaller of the two.
        # ValueError: could not broadcast input array from shape (220,62) into shape (370,62)
        if retrieval_result.GMatrix.shape[0] > gainRet.shape[0]:
            retrieval_result.GMatrix = np.resize(retrieval_result.GMatrix, (gainRet.shape[0], retrieval_result.GMatrix.shape[1]))
    
        # PYTHON_NOTE: It is possible that the size of G_matrix and result.GMatrixFM are different.
        if retrieval_result.GMatrixFM.shape[0] > G_matrix.shape[0]:
            retrieval_result.GMatrixFM = np.resize(retrieval_result.GMatrixFM, (G_matrix.shape[0], retrieval_result.GMatrixFM.shape[1]))
    
        retrieval_result.GMatrix[:, :] = gainRet[:, :]
        retrieval_result.GMatrixFM[:, :] = G_matrix[:, :]
    
        # AT_LINE 211 Error_Analysis.pro
        # some species, like emis, are retrieved in step 1 for a particular spectral region and not updated following this.  In that case, keep errors when they are not moved.  If this is the case, set the error to the previous error, and all error components to zero.
        ind = np.where(dontUpdateFM == 1)[0]
    
        if len(ind) > 0:
            retrieval_result.Sx[ind, ind] = errorCurrentValues[ind, ind]
            retrieval_result.Sx_rand[ind, ind] = 0
            retrieval_result.Sx_smooth[ind, ind] = 0
            retrieval_result.Sx_sys[ind, ind] = 0
            
        my_id = np.identity(S_inv.shape[1], dtype=np.float64)
        retrieval_result.Sx_smooth[:, :] = (my_id - kappaFM @ S_inv).T @ Sa @ (my_id - kappaFM @ S_inv)
        
        retrieval_result.Sx_smooth_self[:, :] = 0
        retrieval_result.Sx_crossState[:, :] = retrieval_result.Sx_smooth[:, :]
    
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
                retrieval_result.Sx_smooth_self[indMe, indMe] = (1 - retrieval_result.A[indMe, indMe]) * Sa[indMe, indMe] * (1 - retrieval_result.A[indMe, indMe])
            else:
                # IDL:
                # id = IDENTITY(N_ELEMENTS(indMe))
                # res.Sx_Smooth_self[indMe,indMe,*] = $
                # (id - res.A[indMe,indMe,*]) ## Sa[indMe,indMe,*] ## Transpose(id - res.A[indMe, indMe,*])
                
                my_id = np.identity(len(indMe))
                ind_me_2d = np.ix_(indMe, indMe)
                retrieval_result.Sx_smooth_self[ind_me_2d] = (my_id - retrieval_result.A[ind_me_2d]).T @ Sa[ind_me_2d] @ (my_id - retrieval_result.A[ind_me_2d])
            # end if len(indMe) == 1:
    
            # interferent on and off diagonal
            if len(indYou) == 0:
                retrieval_result.Sx_crossState[:, :] = 0
            elif len(indMe) == 1 and len(indYou) == 1:
                # IDL:
                # temp = res.A[indYou,indMe,*] * Sa[indYou,indYou] * res.A[indYou, indMe]
                # res.Sx_crossState[indMe,indMe] = temp
                
                temp = retrieval_result.A[indYou, indMe] * Sa[indYou, indYou] * retrieval_result.A[indYou, indMe]
                retrieval_result.Sx_crossState[indMe, indMe] = temp
            elif len(indMe) == 1 and len(indYou) > 1:
                # IDL:
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou,*] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe] = temp[*]                  
    
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
                
                temp = retrieval_result.A[ind_you_me].T @ Sa[ind_you_you] @ retrieval_result.A[ind_you_me]
    
                ind_me_me = np.ix_(indMe, indMe) 
                retrieval_result.Sx_crossState[indMe, indMe] = temp 
            elif len(indMe) > 1 and len(indYou) == 1:
                # IDL: 
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe,*] = temp
    
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
    
                temp = retrieval_result.A[ind_you_me].T @ Sa[ind_you_you] @ retrieval_result.A[ind_you_me]
                
                ind_me_me = np.ix_(indMe, indMe) 
                retrieval_result.Sx_crossState[ind_me_me] = temp[:, :]
            else:
                # IDL:
                # temp = res.A[indYou,indMe,*] ## Sa[indYou,indYou,*] ## Transpose(res.A[indYou, indMe,*])
                # res.Sx_crossState[indMe,indMe,*] = temp
                
                ind_you_you = np.ix_(indYou, indYou)
                ind_you_me = np.ix_(indYou, indMe) 
    
                temp = retrieval_result.A[ind_you_me].T @ Sa[ind_you_you] @ retrieval_result.A[ind_you_me]
                
                ind_me_me = np.ix_(indMe, indMe) 
                retrieval_result.Sx_crossState[ind_me_me] = temp[:, :]
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
        retrieval_result.Sa_ret[:, :] = Sa_ret
    
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
        retrieval_result.Sx_mapping[:, :] = retrieval_result.Sx_smooth - Sx_smooth_ret_fm
    
        # map error covariances to retrieval grid
    
        # IDL:
        # smooth = map.toPars ## res.Sx_Smooth_self ## transpose(map.toPars)
        # cross = map.toPars ## res.Sx_crossState ## transpose(map.toPars)
        # rand = map.toPars ## res.Sx_rand ## transpose(map.toPars)
        # sys = map.toPars ## res.Sx_sys ## transpose(map.toPars)
    
        smooth = my_map.toPars.T @ retrieval_result.Sx_smooth_self @ my_map.toPars
        cross = my_map.toPars.T @ retrieval_result.Sx_crossState @ my_map.toPars
        rand = my_map.toPars.T @ retrieval_result.Sx_rand @ my_map.toPars
        sys = my_map.toPars.T @ retrieval_result.Sx_sys @ my_map.toPars
    
        retrieval_result.Sx_ret_smooth[:, :] = smooth
        retrieval_result.Sx_ret_crossState[:, :] = cross
        retrieval_result.Sx_ret_rand[:, :] = rand
        retrieval_result.Sx_ret_sys[:, :] = sys
    
        #  mapping error from choice of retrieval levels
        retrieval_result.Sx_ret_mapping[:, :] = smooth + cross - Sx_smooth_ret    
        
        # IDL:
        # res.A_ret = map.toPars ## S_Inv ## kappa
        retrieval_result.A_ret = kappa @ S_inv @ my_map.toPars
    
        # AT_LINE 524 Error_Analysis.pro
        retrieval_result.deviationVsErrorSpecies[:] = 0
        retrieval_result.deviationVsRetrievalCovarianceSpecies[:] = 0
        retrieval_result.deviationVsAprioriCovarianceSpecies[:] = 0
        
        # AT_LINE 531 Error_Analysis.pro
        for ii in range(retrieval.n_species):
            m1f = retrieval.parameterStartFM[ii]
            m2f = retrieval.parameterEndFM[ii]
            x = retrieval_result.A[m1f:m2f+1, m1f:m2f+1]
    
            # AT_LINE 535 Error_Analysis.pro
            if (m2f-m1f) > 0:
                retrieval_result.degreesOfFreedomForSignal[ii] = np.trace(x)
                retrieval_result.degreesOfFreedomNoise[ii] = (m2f-m1f+1) - np.trace(x)
            else:
                retrieval_result.degreesOfFreedomForSignal[ii] = x
                retrieval_result.degreesOfFreedomNoise[ii] = 1 - x
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
            
            if not np.all(np.isfinite(retrieval_result.Sx_smooth)):
                raise RuntimeError("retrieval_result.Sx_smooth is not finite")
    
            # AT_LINE 579 Error_Analysis.pro
            if Sb is not None:
                if not np.all(np.isfinite(Sb)):
                    raise RuntimeError("Sb is not finite")
                
                if not np.all(np.isfinite(kappaInt)):
                    raise RuntimeError("kappaInt is not finite")
    
                if not np.all(np.isfinite(retrieval_result.Sx_sys)):
                    raise RuntimeError("retrieval_result.Sx_sys is not finite")
                
        # end for ii in range(retrieval.n_species):
        # AT_LINE 607 Error_Analysis.pro
    
        # AT_LINE 621 Error_Analysis.pro
        # these quantities should be done on the retrieval grid because they
        # are better posed on that grid.
        indf = np.arange(len(retrieval_result.frequency))
    
        # PYTHON_NOTE: It is possible that the size of retrieval_result.frequency is greater than size of jacobian.shape[1]
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
        retrieval_result.KDotDL_list = valueRet
        retrieval_result.KDotDL = np.amax(np.abs(value))
    
        # get max for each species 
        for ii in range(retrieval.n_species):
            m1 = retrieval.parameterStart[ii]
            m2 = retrieval.parameterEnd[ii]
            retrieval_result.KDotDL_species[ii] = retrieval.speciesList[m1]
            retrieval_result.KDotDL_byspecies[ii] = np.amax(np.abs(retrieval_result.KDotDL_list[m1:m2+1]))
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
    
                retrieval_result.LDotDL_byfilter[ii] = np.sum(x1*x2) / math.sqrt(np.sum(x1*x1)) / math.sqrt(np.sum(x2*x2))
            # end if start >= 0:
        # end for ii in range(len(unique_filters)):
    
        retrieval_result.LDotDL = retrieval_result.LDotDL_byfilter[0]
    
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
                
                retrieval_result.KDotDL_byfilter[ii] = np.amax(np.abs(value))
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
    
            retrieval_result.maxKDotDLSys = np.amax(np.abs(value))
        # end part of if retrieval.n_totalParametersSys > 0:
    
        # AT_LINE 758 Error_Analysis.pro
        Sx = retrieval_result.Sx
        Sx_rand = retrieval_result.Sx_rand
    
        # AT_LINE 761 Error_Analysis.pro
        for is_index in range(retrieval.n_totalParametersFM):
            retrieval_result.errorFM[is_index] = np.sqrt(Sx[is_index, is_index])
            retrieval_result.precision[is_index] = np.sqrt(Sx_rand[is_index, is_index])
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
        #retrieval_result.GdL[:] = GdL[:]
        retrieval_result.GdL = GdL
    
        # AT_LINE 792 Error_Analysis.pro
        ind = utilList.WhereEqualIndices(retrieval.speciesListFM, 'CH4')
        
        # Eventhough we are not doing special processing, we will still need to calculate the retrieval_result.ch4_evs field.
        # Also, only calculate the retrieval_result.ch4_evs field. if 'CH4' is in retrieval.speciesListFM.
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
            dots = np.zeros(shape=(num_pp_elements), dtype=np.float32)
            for jj in range(num_pp_elements):
                dots[jj] = np.sum(vmatrix[jj, :] * mydiff)
    
            # look at the first 2 eV's versus eV 3-10
            retrieval_result.ch4_evs = np.abs(dots[0:9+1]) # ratio of use of 1st two vs. next 8 eVs
    
        return o_offDiagonalSys

    def write_retrieval_summary(
            self,
            retrievalInfo : FakeRetrievalInfo,
            stateInfo : FakeStateInfo,
            retrieval_result : Any, # Lots of attributes mypy can't see, so just have the
                                    # type here Any
            errorCurrent : mpy.ObjectView) -> None:
        utilGeneral = mpy.UtilGeneral()
        utilList = mpy.UtilList()
    
    
        if np.max(stateInfo.true['values']) > 0:
            have_true = True
        else:
            have_true = False
    
        # calculate many parameters in results.
        # including c-curve flag
        # column quantities
        # column errors
        # column DOFs
        # emis mean
        # cloud variability
        # QF file
    
        # AT_LINE 35 Write_Retrieval_Summary.pro
        num_species = retrievalInfo.n_species
        
        # get tropopause presure
        indTATM = utilList.WhereEqualIndices(stateInfo.species, 'TATM')
    
        # PYTHON_NOTE: The shape of stateInfo.current['values'][indTATM,:] is (1,64), we need to reshape it back to (64,) to make life easier for TropopauseTES() function.
        tropopause_level = mpy.tropopause_tes(
            stateInfo.current['pressure'], 
            np.reshape(stateInfo.current['values'][indTATM, :], (stateInfo.current['values'].shape[1])),
            stateInfo.current['latitude'],
            np.amax(stateInfo.current['pressure'])
        )
        
        # AT_LINE 42 Write_Retrieval_Summary.pro
        tropopausePressure = stateInfo.current['pressure'][tropopause_level]
        retrieval_result.tropopausePressure = tropopausePressure
    
        if stateInfo.gmaoTropopausePressure > -990:
            # AT_LINE 46 Write_Retrieval_Summary.pro
            tropopausePressure = stateInfo.gmaoTropopausePressure
            retrieval_result.tropopausePressure = stateInfo.gmaoTropopausePressure
        else:
            raise RuntimeError("GMAO tropopause pressure is not defined")
        # AT_LINE 54 Write_Retrieval_Summary.pro
    
        retrieval_result.cloudODAve = 0
        retrieval_result.cloudODVar = 0
        retrieval_result.cloudODAveError = 0
    
        # AT_LINE 57 Write_Retrieval_Summary.pro
        indt = np.where(np.array(stateInfo.species) == 'TATM')[0][0]
        indh = np.where(np.array(stateInfo.species) == 'H2O')[0][0]
        if stateInfo.cloudPars['use'] == 'yes':
            factor = mpy.compute_cloud_factor(
                stateInfo.current['pressure'],
                stateInfo.current['values'][indt, :],
                stateInfo.current['values'][indh, :],
                stateInfo.current['PCLOUD'][0],
                stateInfo.current['scalePressure'],
                stateInfo.current['tsa']['surfaceAltitudeKm']*1000,
                stateInfo.current['latitude']
            )
            # Do some rounding to match IDL: 
            #     IDL output for factor: 1.2542310
            #     Python output  factor: 1.2542897673405733 
            factor = round(factor, 7)
    
            # AT_LINE 67 Write_Retrieval_Summary.pro
            ind = np.where(
                (stateInfo.cloudPars['frequency'] >= 974) & 
                (stateInfo.cloudPars['frequency'] <= 1201)
            )[0]
    
            if len(ind) > 0:
                retrieval_result.cloudODAve = np.sum(stateInfo.current['cloudEffExt'][0, ind]) / len(stateInfo.current['cloudEffExt'][0, ind]) * factor
    
            # step = 1
            # stepName = TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT,EMIS
            # product name Products_Jacobian-TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT-bar_land.nc
            
            ind = np.where(np.asarray(retrievalInfo.speciesList) == 'CLOUDEXT')[0]
            indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == 'CLOUDEXT')[0]
    
            # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
            my_map = mpy.get_one_map(retrievalInfo, 'CLOUDEXT')
    
            # AT_LINE 77 Write_Retrieval_Summary.pro
            if len(ind) > 0:
                # map error to ret
                errlog = retrieval_result.errorFM[indFM] @ my_map['toPars']
    
                cloudod = np.exp(retrieval_result.resultsList[ind]) * factor
                err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
                myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
    
                if myMean == np.nan:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                if np.nan in err:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                if np.inf in err:
                    myMean = np.mean(cloudod)
                    err[:] = myMean
    
                x = np.var((cloudod - myMean) / err, ddof=1)
                retrieval_result.cloudODVar = math.sqrt(x)
            else:
                # cloud not retrieved... use 975-1200
                # NOTE: code has not been tested.
    
                # Look through all indices that meet the pressure criteria, and look to see if same index in errorCurrent.species matches with 'CLOUDEXT'.
                ind4 : list[int] = []
                for ii in range(0, len(errorCurrent.pressure)):
                    if (errorCurrent.pressure[ii] >= 975 and errorCurrent.pressure[ii] <= 1200) and (errorCurrent.species[ii] == 'CLOUDEXT'):
                        if ii not in ind4:
                            ind4.append(ii)
    
                if len(ind4) > 0:
                    # found 975-1200
                    # error_current = utilGeneral.ManualArrayGetWithRHSIndices(errorCurrent.data, ind, ind)
                    error_current = errorCurrent.data[ind4, ind4]
    
                    retrieval_result.cloudODAveError = math.sqrt(np.sum(error_current)) / len(ind4) * retrieval_result.cloudODAve
    
                    ind3 = np.where(
                        (stateInfo.cloudPars['frequency'] >= 974) & 
                        (stateInfo.cloudPars['frequency'] <= 1201)
                    )[0]
    
                    cloudod = stateInfo.current['cloudEffExt'][0, ind3] * factor
    
                    # error_current = utilGeneral.ManualArrayGetWithRHSIndices(errorCurrent.data, ind, ind)
                    error_current = errorCurrent.data[ind3, ind3]
    
                    err = stateInfo.current['cloudEffExt'][0, ind3] * np.sqrt(error_current)
                    myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
    
                    if myMean == np.nan:
                        myMean = np.mean(cloudod)
                        err[:] = myMean
    
                    if np.nan in err:
                        myMean = np.mean(cloudod)
                        err[:] = myMean
    
                    if np.inf in err:
                        myMean = np.mean(cloudod)
                        err[:] = myMean
    
                    x = np.var((cloudod - myMean) / err, ddof=1)
                # end if len(ind) > 0:
            # end else part of if (len(ind) > 0):
        # end if stateInfo.cloudPars['use'] == 'yes':
    
        # AT_LINE 107 Write_Retrieval_Summary.pro
        if stateInfo.cloudPars['use'] == 'yes':
            if stateInfo.current['scalePressure'] == 0:
                stateInfo.current['scalePressure'] = 0.1
    
            factor = mpy.compute_cloud_factor(
                stateInfo.current['pressure'],
                stateInfo.current['values'][stateInfo.species.index('TATM'), :],
                stateInfo.current['values'][stateInfo.species.index('H2O'), :],
                stateInfo.current['PCLOUD'][0],
                stateInfo.current['scalePressure'],
                stateInfo.current['tsa']['surfaceAltitudeKm']*1000,
                stateInfo.current['latitude']
            )
    
            # Do some rounding to match IDL: 
            #     IDL output for factor: 1.2542310
            #     Python output  factor: 1.2542897673405733 
            factor = round(factor, 7)
            
            ind = np.where(np.asarray(retrievalInfo.speciesList) == 'CLOUDEXT')[0]
            indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == 'CLOUDEXT')[0]
    
            # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
            my_map = mpy.get_one_map(retrievalInfo, 'CLOUDEXT')
            
            if len(ind) > 0:
                errlog = retrieval_result.errorFM[indFM] @ my_map['toPars']
    
                cloudod = np.exp(retrieval_result.resultsList[ind]) * factor
                err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
                myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)
                
                x = np.var((cloudod - myMean) / err, ddof=1)
                retrieval_result.cloudODVar = math.sqrt(x)
            # end if (len(ind) > 0):
        # end if stateInfo.cloudPars['use'] == 'yes':
    
        # AT_LINE 144 Write_Retrieval_Summary.pro
        # calculate emisDev, the emissivity difference from the true state
        # between 975 and 1200 cm-1
        retrieval_result.emisDev = 0
        
        ind = np.where(
            (stateInfo.emisPars['frequency'] >= 975) & 
            (stateInfo.emisPars['frequency'] <= 1200)
        )[0]
    
        if len(ind) > 0:
            retrieval_result.emisDev = np.mean(stateInfo.current['emissivity'][ind]) - np.mean(stateInfo.constraint['emissivity'][ind])
    
        # AT_LINE 159 Write_Retrieval_Summary.pro
        # emission layer flag
        retrieval_result.emissionLayer = 0
    
        ind10 = np.asarray([])  # Start with an empty list so we have the variable set.
        if 'O3' in retrievalInfo.speciesListFM:
            ind10 = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'O3')
    
        # AT_LINE 162 Write_Retrieval_Summary.pro
        if len(ind10) > 0:
            indt = np.where(np.array(stateInfo.species) == 'TATM')[0][0]
            TATM = stateInfo.current['values'][indt, :]
            TSUR = stateInfo.current['TSUR']
            
            indo3 = np.where(np.array(stateInfo.species) == 'O3')[0][0]
            o3 = stateInfo.current['values'][indo3, :]
            o3ig = stateInfo.constraint['values'][indo3, :]
            
            aveTATM = 0.0
            my_sum = 0.0
            for ii in range(0, 3):
                my_sum = my_sum + o3[ii] - o3ig[ii]
                aveTATM = aveTATM + TATM[ii]
            
            aveTATM = aveTATM / 3
            if my_sum/3 >= 1.5e-9:
                retrieval_result.emissionLayer = aveTATM - TSUR
    
        # AT_LINE 181 Write_Retrieval_Summary.pro
        # ozone ccurve flag PRE-R12
        retrieval_result.ozoneCcurve = 1
    
        ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'O3')
        
        if len(ind) > 0:
            pressure = stateInfo.current['pressure']
            
            indo3 = np.where(np.array(stateInfo.species) == 'O3')[0][0]
            o3 = stateInfo.current['values'][indo3, :]
            o3ig = stateInfo.initial['values'][indo3, :]
            
            indLow = np.where(pressure >= 700)[0]
            indHigh = np.where((pressure >= 200) & (pressure <= 350))[0]
    
            if len(indLow) > 0 and len(indHigh) > 0:
                ratio1 = np.mean(o3[indLow]) / np.mean(o3ig[indLow])
                ratio2 = np.mean(o3[indLow]) / np.mean(o3[indHigh])
                if ratio1 >= 1.6 and ratio2 >= 1.4:
                    retrieval_result.ozoneCcurve = 0
            # end if len(indLow) > 0 and len(indHigh) > 0
        # if len(ind) > 0:
    
        # AT_LINE 200 Write_Retrieval_Summary.pro
        # ozone ccurve flag R12 and beyond (from c++ code 9/2012)
        # test 1 max ozone below 700 mb > 150 ppb
        # test 2 below 700 mb:  ozone > 100 ppb or retrieval / prior > 1.8 and average
        # diagonal AK for these levels < 0.1
        # test 3 c-shaped 
        #                //let maxlo be the largest value for levels greater than 700 hPa
        #                // and minhi be the smallest value for levels from 700 hPa to 200 hPa
        #                // and surf be the surface concentration
        #                // the C-curve is considered if :
        #                // 1) maxlo/minhi > 2.5
        #                // or 2) maxlo/minhi > 2 and maxlo/surf > 1.1 (use of 1.05 - 1.1
        # if any condition is true then ccurve = 1
    
        retrieval_result.ozoneCcurve = 1
        retrieval_result.ozone_slope_QA = 1
    
        ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'O3')
        
        if len(ind) > 0:
            pressure = stateInfo.current['pressure']
            o3 = stateInfo.current['values'][stateInfo.species.index('O3'),:]
            o3ig = stateInfo.initial['values'][stateInfo.species.index('O3'),:]
            indLow = np.where(pressure >= 700)[0]
            indHigh = np.where((pressure >= 200) & (pressure <= 700))[0]
            if len(indLow) > 0 and len(indHigh) > 0:
                maxlo = np.amax(o3[indLow])
                minhi = np.amin(o3[indHigh])
                meanlo = np.mean(o3[indLow])
                meanloig = np.mean(o3ig[indLow])
                surf = o3[np.amin(indLow)]
    
                # pull out mapToState and mapToPars
                # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
                my_map = mpy.get_one_map(retrievalInfo, 'O3')
    
                # This may not be correct.
                # AK = results.A[ind, ind, :]
                # Let's use the manual way.
                AK = utilGeneral.ManualArrayGetWithRHSIndices(retrieval_result.A, ind, ind)
    
                AKzz = np.matmul(np.matmul(my_map['toState'], AK), my_map['toPars'])
                meanAKlo = np.var(AKzz[indLow, indLow])
    
                if maxlo * 1e9 > 150 or \
                    ((maxlo*1e9 > 100 or meanlo / meanloig > 1.8) and (meanAKlo < 0.1)) or \
                    ((maxlo / minhi > 2.5 or maxlo / minhi >= 2) and (maxlo / surf >= 1.1)):
                    retrieval_result.ozoneCcurve = 0
            # end if len(indLow) > 0 and len(indHigh) > 0:
    
            # slope c-curve flag
            o3 = stateInfo.current['values'][stateInfo.species.index('O3'), :]
            indp = np.where(o3 > 0)[0]
            altitude = stateInfo.current['heightKm'][indp]
            o3 = o3[indp]
    
            slope = mpy.ccurve_jessica(altitude, o3)
            retrieval_result.ozone_slope_QA = slope
        # end if len(ind) > 0:
    
        # AT_LINE 255 Write_Retrieval_Summary.pro
        # Now get species dependent preferences
        for ispecie in range(0, num_species):
            my_sum = 0.0
    
            species_name = retrievalInfo.species[ispecie]
    
            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]
    
            # AT_LINE 269 Write_Retrieval_Summary.pro
            # deviation quality flag - refined for O3 and HCN
            retrieval_result.deviation_QA[ispecie] = 1
            retrieval_result.num_deviations_QA[ispecie] = 1
            retrieval_result.DeviationBad_QA[ispecie] = 1
            pressure = stateInfo.current['pressure']
            
            if loc != -1:
                # Only use the index() function if the species_name is in the stateInfo.species array.
                profile = stateInfo.current['values'][loc, :]
                constraint = stateInfo.constraint['values'][loc, :]
            
            ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, species_name)
    
            ak_diag = retrieval_result.A[ind, ind]
    
            # Also note that the value of profile is only set above if loc is not -1 above.
            if loc != -1:
                # AT_LINE 279 Write_Retrieval_Summary.pro
                result_quality = mpy.quality_deviation(pressure, profile, constraint, ak_diag, species_name)
                retrieval_result.deviation_QA[ispecie] = result_quality.deviation_QA
                retrieval_result.num_deviations_QA[ispecie] = result_quality.num_deviations
                retrieval_result.DeviationBad_QA[ispecie] = result_quality.deviationBad
    
            # AT_LINE 292 Write_Retrieval_Summary.pro
            species_name = retrievalInfo.species[ispecie]
            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]
    
            # AT_LINE 294 Write_Retrieval_Summary.pro
            if (loc >= 0) and (species_name != 'TATM'):
                # get index of column species
                # Note: When results.columnSpecies was first allocated, all the elements are empty strings.  As we are processing
                #       each species, the specie name will be added to the array.
                indcol = utilList.WhereEqualIndices(retrieval_result.columnSpecies, '')
    
                indcol = indcol[0]
    
                # Add the species_name to the current index.
                retrieval_result.columnSpecies[indcol] = species_name
    
                # AT_LINE 301 Write_Retrieval_Summary.pro
    
                # EM NOTE - Adding stratosphere to the column for analysis
                for ij in range(0, 5):
                    # AT_LINE 303 Write_Retrieval_Summary.pro
                    if ij == 0:
                        my_type = 'Column'
    
                        minPressure = 0
                        minIndex = np.int64(len(stateInfo.current['pressure']) - 1)
    
                        maxPressure = np.amax(stateInfo.current['pressure'])
                    elif ij == 1:
                        my_type = 'Trop' 
    
                        minPressure = retrieval_result.tropopausePressure
                        minIndex = np.argmin(np.abs(stateInfo.current['pressure'] - minPressure))
    
                        maxPressure = np.amax(stateInfo.current['pressure'])
                    elif ij == 2:
                        # upper tropopause
                        my_type = 'UpperTrop'
    
                        maxPressure = 500
    
                        minPressure = retrieval_result.tropopausePressure
                        minIndex = np.argmin(np.abs(stateInfo.current['pressure'] - minPressure))
                    elif ij == 3:
                        # lower troposphere
                        my_type = 'LowerTrop'
                        
                        minPressure = 500
                        minIndex = np.argmin(np.abs(stateInfo.current['pressure'] - minPressure))
                        
                        maxPressure = np.amax(stateInfo.current['pressure'])
                    elif ij == 4:
                        # Stratosphere
                        my_type = 'Strato'
                        minPressure = 0
                        minIndex = np.int64(len(stateInfo.current['pressure']) - 1)
    
                        maxPressure = retrieval_result.tropopausePressure
                        if maxPressure == 0:
                            maxPressure = 200.
                    else:
                        raise RuntimeError("Type not found")
                    # end if (ij == 0):
    
                    # AT_LINE 336 Write_Retrieval_Summary.pro
                    retrieval_result.columnPressureMin[ij] = minPressure
                    retrieval_result.columnPressureMax[ij] = maxPressure
    
                    
                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    my_map = retrievalInfo.mapToState
                    if(my_map is None):
                        raise RuntimeError("basis matrix missing")
                    
                    mapType = retrievalInfo.mapType[ispecie]
                    
                    linear = 0
                    if mapType.lower() == 'linear':
                        linear = 1
                    if mapType.lower() == 'linearpca':
                        linear = 1
    
                    indSpecie = loc
                    indH2O = utilList.WhereEqualIndices(stateInfo.species, 'H2O')[0]
                    indTATM = utilList.WhereEqualIndices(stateInfo.species, 'TATM')[0]
    
                    # AT_LINE 357 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.constraint['values'][indSpecie, :],
                        stateInfo.constraint['pressure'],
                        stateInfo.constraint['values'][indTATM, :],
                        stateInfo.constraint['values'][indH2O, :],
                        stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                        stateInfo.current['latitude'],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None
                    )
    
                    retrieval_result.columnPrior[ij, indcol] = x['column']
    
                    # AT_LINE 368 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.initial['values'][indSpecie, :],
                        stateInfo.initial['pressure'],
                        stateInfo.initial['values'][indTATM, :],
                        stateInfo.initial['values'][indH2O, :],
                        stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                        stateInfo.current['latitude'],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None
                    )
                    
                    retrieval_result.columnInitial[ij, indcol] = x['column']
    
                    # AT_LINE 379 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.initialInitial['values'][indSpecie, :],
                        stateInfo.initialInitial['pressure'],
                        stateInfo.initialInitial['values'][indTATM, :],
                        stateInfo.initialInitial['values'][indH2O, :],
                        stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                        stateInfo.current['latitude'],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None
                    )
    
                    retrieval_result.columnInitialInitial[ij, indcol] = x['column']
    
                    # AT_LINE 390 Write_Retrieval_Summary.pro
                    x = mpy.column(
                        stateInfo.current['values'][indSpecie, :],
                        stateInfo.current['pressure'],
                        stateInfo.current['values'][indTATM, :],
                        stateInfo.current['values'][indH2O, :],
                        stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                        stateInfo.current['latitude'],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None)
                    
                    retrieval_result.column[ij, indcol] = x['column']
    
                    # AT_LINE 400 Write_Retrieval_Summary.pro
                    # air column
                    x = mpy.column(
                        stateInfo.current['values'][indSpecie, :] * 0 + 1,
                        stateInfo.current['pressure'],
                        stateInfo.current['values'][indTATM, :],
                        stateInfo.current['values'][indH2O, :],
                        stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                        stateInfo.current['latitude'],
                        minPressure,
                        maxPressure,
                        linear,
                        pge=None
                    )
    
                    retrieval_result.columnAir[ij] = x['columnAir']
    
                    # AT_LINE 411 Write_Retrieval_Summary.pro
                    if species_name == 'O3' and my_type == 'Trop':
                        # compare initial gues for this step to retrieved.
                        ret = retrieval_result.column[ij, indcol]
                        ig = retrieval_result.columnInitial[ij, indcol]
                        ratio = (ret / ig) - 1.0
                        retrieval_result.O3_tropo_consistency = ratio
                    # end if species_name == 'O3' and my_type == 'Trop':
    
                    # AT_LINE 420 Write_Retrieval_Summary.pro
                    # true values
                    # only for synthetic data
                    if have_true:
                        x = mpy.column(
                            stateInfo.true['values'][indSpecie, :],
                            stateInfo.true['pressure'],
                            stateInfo.true['values'][indTATM, :],
                            stateInfo.true['values'][indH2O, :],
                            stateInfo.current['tsa']['surfaceAltitudeKm'] * 1000,
                            stateInfo.current['latitude'],
                            minPressure,
                            maxPressure,
                            linear,
                            pge=None
                        )
    
                        retrieval_result.columnTrue[ij, indcol] = x['column']
    
                    # AT_LINE 435 Write_Retrieval_Summary.pro
                    Sx = retrieval_result.Sx[ind1FM:ind2FM+1, ind1FM:ind2FM+1]
    
                    derivativeFinal = np.copy(x['derivative'])
    
                    mapType = mapType.lower()
                    if mapType == 'log':
                        # PYTHON_NOTE: It is possible that the length of x['derivative'] is greater than stateInfo.current['values'][indSpecie,:]
                        #              In that case, we make sure both terms below on the right hand side are the same sizes.
                        rhs_term_sizes = len(stateInfo.current['values'][indSpecie, :]) 
                        derivativeFinal = x['derivative'][0:rhs_term_sizes] * stateInfo.current['values'][indSpecie, 0:rhs_term_sizes]
    
                    # IDL: 
                    # error = SQRT(derivativeFinal[0:minIndex] ## Sx[0:minIndex,0:minIndex] ## TRANSPOSE(derivativeFinal[0:minIndex])
    
                    error = np.sqrt(derivativeFinal[0:minIndex+1].T @ Sx[0:minIndex+1, 0:minIndex+1] @ derivativeFinal[0:minIndex+1])
                    retrieval_result.columnError[ij, indcol] = error
    
                    # AT_LINE 446 Write_Retrieval_Summary.pro
                    # multipy prior covariance to calc predicted prior error
                    if mapType == 'log':
                        # PYTHON_NOTE: It is possible that the length of x['derivative'] is greater than stateInfo.current['values'][indSpecie,:]
                        #              In that case, we make sure both terms below on the right hand side are the same sizes.
                        rhs_term_sizes = len(stateInfo.initial['values'][indSpecie, :]) 
                        derivativeFinal = x['derivative'][0:rhs_term_sizes] * stateInfo.initial['values'][indSpecie, 0:rhs_term_sizes]
                    
                    # IDL: 
                    # error = SQRT(derivativeFinal[0:minIndex] ## results.Sa[0:minIndex,0:minIndex] ## TRANSPOSE(derivativeFinal[0:minIndex])
                    
                    error = np.sqrt(derivativeFinal[0:minIndex+1].T @ retrieval_result.Sa[0:minIndex+1, 0:minIndex+1] @ derivativeFinal[0:minIndex+1])
    
                    retrieval_result.columnPriorError[ij, indcol] = error
    
                    # AT_LINE 455 Write_Retrieval_Summary.pro
                    if species_name == 'O3' and my_type == 'Column':
                        retrieval_result.O3_columnErrorDU = retrieval_result.columnError[ij, indcol] / 2.69e+16
                        # temporary fix
                        retrieval_result.omi_cloudfraction = stateInfo.current['omi']['cloud_fraction']
                        retrieval_result.tropomi_cloudfraction = stateInfo.current['tropomi']['cloud_fraction']
                    # end if species_name == 'O3' and my_type == 'Column':
    
                    if my_type == 'Column' and species_name == 'H2O':
                        ind4 = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'H2O')
                        ind5 = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, 'HDO')
                        
                        if len(ind4) > 0 and len(ind5) > 0:
                            # in H2O/HDO step, check H2O column - H2O column from O3 step / error
                            retrieval_result.H2O_H2OQuality = \
                                (retrieval_result.column[ij, indcol] - retrieval_result.columnInitial[ij, indcol]) / retrieval_result.columnPriorError[ij, indcol]
                        # end if len(ind4) > 0 and len(ind5) > 0
                    # end if my_type == 'Column' and species_name == 'H2O':
    
                    # AT_LINE 474 Write_Retrieval_Summary.pro
                    # calculate DOFs for different ranges
                    # based on layers
                    # each level corresponds to a layer which ranges to 1/2
                    # between it and level below to halfway between it and
                    # level above.  So if range was 1000-100 hPa, and there
                    # was a level at 100 hPa, only half of the AK at 100 hPa would be
                    # included because only half of the above described layer is inclu
    
                    ispecie = utilList.WhereEqualIndices(retrievalInfo.species, species_name)[0]
                    ind1FM = retrievalInfo.parameterStartFM[ispecie]
                    ind2FM = retrievalInfo.parameterEndFM[ispecie]
                    ak = mpy.get_diagonal(retrieval_result.A)
                    ak = ak[ind1FM:ind2FM+1]
                    na = len(ak)
    
                    pressureLayers = np.asarray(stateInfo.current['pressure'][0])
                    pressureLayers = np.append(pressureLayers, (stateInfo.current['pressure'][1:] + stateInfo.current['pressure'][0:na-1]) / 2)
    
                    indp = np.where(
                        (pressureLayers >= (minPressure - 0.0001)) & 
                        (pressureLayers < (maxPressure + 0.0001))
                    )[0]
                    
                    dof = np.sum(ak[indp[0:len(indp)-1]])
    
                    # PYTHON_NOTE: It is possible with the where() function below, the array returned is empty.
                    #              If it is empty, we cannot use np.amax() so we have to do two separated steps.
                    #              as opposed to IDL which does it it one step: indp1 = max(where(pressureLayers GT maxPressure))
                    max_indices = np.where(pressureLayers > maxPressure)[0]
    
                    indp1 = np.int64(-1)
                    if len(max_indices) > 0:
                        indp1 = np.amax(max_indices)
    
                    if indp1 != -1:
                        fraction1 = (maxPressure - pressureLayers[indp1+1]) / (pressureLayers[indp1] - pressureLayers[indp1+1])
                        dof = dof + fraction1 * ak[indp1]
    
                    indp2 = np.int64(-1)
                    min_indices = np.where(pressureLayers < minPressure)[0]
                    if len(min_indices) > 0:
                        indp2 = np.amin(min_indices)
    
                    if indp2 != -1:
                        fraction2 = (minPressure - pressureLayers[indp2-1]) / (pressureLayers[indp2] - pressureLayers[indp2-1])
                        dof = dof + fraction2 * ak[indp2]
    
                    retrieval_result.columnDOFS[ij, indcol] = dof
                # end for ij in range(0, 5):
    
                # AT_LINE 505 Write_Retrieval_Summary.pro
            # end if (loc >= 0) and (species_name != 'TATM'):
            continue
        # end for ispecie in range(0,num_species):

        # This gets filled in later
        retrieval_result.masterQuality = 0
        
__all__ = [
    "ErrorAnalysis",
]
