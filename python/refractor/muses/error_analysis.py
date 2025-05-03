from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
import refractor.framework as rf  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
import copy
import numpy as np
import sys
import os
from pprint import pprint
from scipy.linalg import block_diag  # type: ignore
import typing

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
        # Updates error_current and retrieval_result in place
        self.error_analysis_wrapper(
            retrieval_result.rstep,
            fretrieval_info,
            fstate_info,
            self.error_initial,
            self.error_current,
            retrieval_result,
        )
        # Updates retrieval_result in place
        _ = mpy.write_retrieval_summary(
            None,
            fretrieval_info,
            fstate_info,
            None,
            retrieval_result,
            {},
            None,
            None,
            None,
            self.error_current,
            writeOutputFlag=False,
            errorInitial=self.error_initial,
        )

    def error_analysis_wrapper(
            self,
            radiance : mpy.ObjectView,
            retrieval : FakeRetrievalInfo,
            stateInfo : FakeStateInfo,
            errorInitial : mpy.ObjectView,
            errorCurrent : mpy.ObjectView,
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
        constraintMatrix = None
    
        if jacobian_sys is not None:
            jacobian_sys = jacobian_sys[0,:,:].copy()
            jacobian_sys[:, bad_pixel] = 0
            
            if not np.all(np.isfinite(jacobian_sys)):
                raise RuntimeError("jacobian_sys is not finite")
            
            for i in range(jacobian_sys.shape[0]):
                jacobian_sys[i, :] /= data_error
    
            # AT_LINE 107 Error_Analysis_Wrapper.pro
            speciesList = retrieval.speciesSys[0:retrieval.n_speciesSys]
            Sb = mpy.constraint_get(errorCurrent.__dict__, speciesList)
            retrieval_result.Sb = Sb  # type: ignore[attr-defined]
    
        for ii in range(jacobian.shape[0]):
            jacobian[ii, :] /= data_error
    
        jacobian_fm = np.copy(jacobian)
        jacobian = my_map.toState @ jacobian_fm
    
        # AT_LINE 177 Error_Analysis_Wrapper.pro
        n = retrieval.n_totalParameters
        Sa = mpy.constraint_get(errorInitial.__dict__, retrieval.species[0:retrieval.n_species])
    
        constraintMatrix = np.copy(retrieval.Constraint[0:n, 0:n])
        retrieval_result.Sa[:, :] = Sa[:, :]  # type: ignore[attr-defined]
    
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
        errorCurrentValues = mpy.constraint_get(errorCurrent.__dict__, currentSpecies)
    
        # AT_LINE 251 Error_Analysis_Wrapper.pro
    
        speciesList = retrieval.speciesList
    
        # AT_LINE 256 Error_Analysis_Wrapper.pro
        (result, offDiagonalSys) = mpy.error_analysis(
            None,
            my_map,
            jacobian,
            jacobian_fm,
            Sa,
            jacobian_sys,
            Sb,
            None,
            constraintMatrix,
            constraintVector,
            None,
            None,
            resultVector,
            data_error,
            None,
            actual_data_residual,
            retrieval_result,
            retrieval,
            stateInfo.current['heightKm'],
            errorCurrentValues)
    
        # AT_LINE 280 Error_Analysis_Wrapper.pro
        speciesList = retrieval.speciesList
    
        # AT_LINE 336 Error_Analysis_Wrapper.pro
        # update errorCurrent
        # first clear correlations between current retrieved species and all
        # others then set error based on current error analysis
        errorCurrent = mpy.constraint_clear(errorCurrent.__dict__, currentSpecies)
        errorCurrent = mpy.constraint_set(errorCurrent.__dict__, retrieval_result.Sx, currentSpecies)  # type: ignore[attr-defined]
        if jacobian_sys is not None:
            # set block and transpose of block
            my_ind = np.where(np.asarray(retrieval.parameterEndSys) > 0)[0]
    
            # The following function constraint_set_subblock() updates the 'data' portion of errorCurrent structure.
            # Call twice with the last two parameters swapped.
            errorCurrent = mpy.constraint_set_subblock(
                errorCurrent, 
                np.transpose(offDiagonalSys), 
                np.asarray(currentSpecies), 
                np.asarray(retrieval.speciesSys[0:len(my_ind)])
            )
            
            errorCurrent = mpy.constraint_set_subblock(
                errorCurrent, 
                offDiagonalSys, 
                np.asarray(retrieval.speciesSys[0:len(my_ind)]), 
                np.asarray(currentSpecies)
            )
        # end if jacobian_sys is not None:

__all__ = [
    "ErrorAnalysis",
]
