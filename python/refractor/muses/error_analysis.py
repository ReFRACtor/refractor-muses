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
        # Both these functions update retrieval_result in place, and
        # also returned. We don't need the return value, it is just the
        # same as retrieval_result
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
            results : RetrievalResult) -> None:
        # expected noise
        dataError = radiance.NESR
        if len(dataError.shape) == 1:
            my_shape = (1, len(dataError)) # Convert (442,) to (1,442)
        else:
            my_shape = dataError.shape
    
        dataError = mpy.glom(np.reshape(dataError, my_shape), 0, 1)  # Function glom() likes 2-D array or more
    
        # AT_LINE 25 Error_Analysis_Wrapper.pro
        # Because the code was IDL specific of restoring variable, Python did not implement lines 25 through 35.
    
        # get good frequencies
        # AT_LINE 38 Error_Analysis_Wrapper.pro
        indPosFreq = np.where(dataError > 0)[0]
        indNegFreq = np.where(dataError < 0)[0]
    
        # make maps:  FM -> ret & vice versa
        # AT_LINE 42 Error_Analysis_Wrapper.pro
        mmm = retrieval.n_totalParameters
        nnn = retrieval.n_totalParametersFM
        my_map = {
            'toPars': np.copy(retrieval.mapToParameters[0:nnn, 0:mmm]), 
            'toState': np.copy(retrieval.mapToState[0:mmm, 0:nnn])
        }      
    
        my_map = mpy.ObjectView(my_map)
    
        # result is jacobian[pars, frequency]
        jacobian = mpy.glom(results.jacobian, 0, 2)  # Function glom() likes 2-D array or more
        jacobian[:, indNegFreq] = 0
    
        # actual residual, fit - observed
        # AT_LINE 62 Error_Analysis_Wrapper.pro
    
        # PYTHON_NOTE: It is possible that the size of results.radiance and radiance.radiance are different.
        # If that is the case, we shrink to the smaller of the two.
        # ValueError: operands could not be broadcast together with shapes (1,370) (220,) 
        if len(results.radiance.shape) == 2:
            if results.radiance.shape[1] > len(radiance.radiance):
                results.radiance = np.resize(results.radiance, (results.radiance.shape[0], len(radiance.radiance)))  # Shrink from (1,370) to (1,220) 
    
        actualDataResidual = results.radiance - radiance.radiance
        if len(actualDataResidual.shape) == 1:
            my_shape = (1, len(actualDataResidual)) # Convert (442,) to (1,442)
        else:
            my_shape = actualDataResidual.shape
    
        actualDataResidual = mpy.glom(np.reshape(actualDataResidual, my_shape), 0, 1)
        actualDataResidual[indNegFreq] = 0
    
        # AT_LINE 67 Error_Analysis_Wrapper.pro
        if np.all(np.isfinite(actualDataResidual)) == False:
            logger.error("actualDataResidual NOT FINITE!")
            assert False
    
        if np.all(np.isfinite(dataError)) == False:
            logger.error("dataError NOT FINITE!")
            assert False
    
        if np.all(np.isfinite(jacobian)) == False:
            logger.error("jacobian NOT FINITE!")
            assert False
    
        # AT_LINE 90 Error_Analysis_Wrapper.pro
    
        # Get normalized (by nesr) systematic jacobian
        nSys = retrieval.n_totalParametersSys
        jacobianSys = None
    
        # For now, we need to assign these variables below so they have some values because
        # for AIRS run, the value of nSys is 0.
        Sb = None
        Db = None
        constraintMatrix = None
    
        if nSys > 0:
            jacobianSys = results.jacobianSys
            if len(jacobianSys.shape) == 1:
                my_shape = (1, len(jacobianSys)) # Convert (442,) to (1,442)
            else:
                my_shape = jacobianSys.shape
            jacobianSys = mpy.glom(jacobianSys, 0, 2)
            jacobianSys[:, indNegFreq] = 0
    
            # AT_LINE 97 Error_Analysis_Wrapper.pro
            if np.all(np.isfinite(jacobianSys)) == False:
                logger.error("jacobianSys NOT FINITE!")
                assert False
    
            # AT_LINE 104 Error_Analysis_Wrapper.pro
            for ii in range(retrieval.n_totalParametersSys):
                jacobianSys[ii, :] = jacobianSys[ii, :] / dataError
    
            # AT_LINE 107 Error_Analysis_Wrapper.pro
            speciesList = retrieval.speciesSys[0:retrieval.n_speciesSys]
            Sb = mpy.constraint_get(errorCurrent.__dict__, speciesList)
            results.Sb = Sb
            SbSpecies = mpy.constraint_get_species(errorCurrent, speciesList)
            SbPressures = mpy.constraint_get_pressures(errorCurrent.__dict__, speciesList)
    
            # AT_LINE 107 Error_Analysis_Wrapper.pro
            # get actual error in systematic species
            specSys = []
            for one_species in SbSpecies:
                if one_species not in specSys:
                    specSys.append(one_species)
    
            # loop through each systematic species
            # get the true for it 
            # get the estimate for it
    
            for ii in range(0, len(specSys)):
                if specSys[ii] == 'EMIS':
                    n_count = stateInfo.emisPars['num_frequencies']
                    myState = stateInfo.current['emissivity'][0:n_count]
                    myTrue = stateInfo.true['emissivity'][0:n_count]
                elif specSys[ii] == 'CLOUDEXT':
                    n_count = stateInfo.cloudPars['num_frequencies']
                    myState = stateInfo.current['cloudEffExt'][0, 0:n_count]
                    myTrue = stateInfo.true['cloudEffExt'][0, 0:n_count] + stateInfo.true['cloudEffExt'][1, 0:n_count]
                elif specSys[ii] == 'CALSCALE':
                    n_count = stateInfo.calibrationPars['num_frequencies']
                    myState = stateInfo.current['calibrationScale'][0:n_count]
                    myTrue = stateInfo.true['calibrationScale'][0:n_count]
                elif specSys[ii] == 'PCLOUD':
                    myState = stateInfo.current['PCLOUD'][0]
                    myTrue = stateInfo.true['PCLOUD'][0]
                elif specSys[ii] == 'TSUR':
                    myState = stateInfo.current['TSUR']
                    myTrue = stateInfo.current['TSUR']
                elif specSys[ii] == 'PTGANG':
                    myState = stateInfo.current['tes']['boresightNadirRadians']
                    myTrue = stateInfo.current['tes']['boresightNadirRadians']
                elif specSys[ii] == 'PSUR':
                    myState = stateInfo.current['PSUR']
                    myTrue = stateInfo.current['PSUR']
                else:
                    n_count = stateInfo.num_pressures
                    loc = np.where(np.asarray(stateInfo.species) == specSys[ii])[0]
                    myState = copy.deepcopy(stateInfo.current['values'][loc, 0:n_count])
                    myTrue = copy.deepcopy(stateInfo.true['values'][loc, 0:n_count])
    
                db0 = myTrue - myState  # Calculate the delta.
    
                # Because the shape of db0 may be weird: (1, 64) (1, 64), we fix it to just a vector.
                if isinstance(db0, np.ndarray):
                    if len(db0.shape) == 2 and db0.shape[0] == 1:
                        db0 = np.reshape(db0, (db0.shape[1]))    # Change shape from (1,64) to (64)
                elif np.isscalar(db0):
                    db0 = np.asarray([db0])  # Make a list of one element.
                else:
                    logger.error("This function does not handle this type of db0 yet", type(db0))
                    assert False
    
                linear = (specSys[ii] == 'TATM' or specSys[ii] == 'TSUR' or specSys[ii] == 'EMIS')
                if not linear:
                    db0 = np.log(myTrue)-np.log(myState)
    
                # Because the shape of db0 may be weird: (1, 64) (1, 64), we fix it to just a vector.
                if isinstance(db0, np.ndarray):
                    if len(db0.shape) == 2 and db0.shape[0] == 1:
                        db0 = np.reshape(db0, (db0.shape[1]))    # Change shape from (1,64) to (64)
                elif np.isscalar(db0):
                    db0 = np.asarray([db0])  # Make a list of one element.
                else:
                    logger.error("This function does not handle this type of db0 yet", type(db0))
                    assert False
    
                if ii == 0:
                    Db = db0   # Use uppercase 'D' for Db because variables below uses 'Db'
                else:
                    Db = np.append(Db, db0, axis=0)
                # end else portion of if (ii == 0):
            # end for ii in range(0,len(specSys)):
        # end part of if (nSys > 0):
    
        # AT_LINE 167 Error_Analysis_Wrapper.pro
        # Get normalized (by nesr) jacobian
        nf = retrieval.n_totalParametersFM
    
        # PYTHON_NOTE: It is possible that the size of jacobian.shape[1] is greater than dataError.
        # If that is the case, we shrink to the smaller of the two.
        # ValueError: operands could not be broadcast together with shapes (370,) (220,) 
        if jacobian.shape[1] > len(dataError):
            jacobian = np.resize(jacobian, (jacobian.shape[0], len(dataError)))
    
        for ii in range(nf):
            jacobian[ii, :] = np.divide(jacobian[ii, :], dataError)
    
        jacobianFM = np.copy(jacobian)
        jacobian = my_map.toState @ jacobianFM
    
        # AT_LINE 177 Error_Analysis_Wrapper.pro
        n = retrieval.n_totalParameters
        Sa = mpy.constraint_get(errorInitial.__dict__, retrieval.species[0:retrieval.n_species])
    
        constraintMatrix = np.copy(retrieval.Constraint[0:n, 0:n])
        results.Sa[:, :] = Sa[:, :]
    
        # Nobody uses these 2 variables yet.
        trueVector = retrieval.trueParameterListFM[0:nf]
        initialVector = retrieval.initialGuessListFM[0:nf]
    
        ret_vector = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        con_vector = np.zeros(shape=(retrieval.n_totalParametersFM), dtype=np.float64)
        FM_Flag = True
        INITIAL_Flag = True
        TRUE_Flag = False
        CONSTRAINT_Flag = False
    
        for ispecie in range(retrieval.n_species):
            species_name = retrieval.species[ispecie]
            ind1 = retrieval.parameterStart[ispecie]
            ind2 = retrieval.parameterEnd[ispecie]
            ind1FM = retrieval.parameterStartFM[ispecie]
            ind2FM = retrieval.parameterEndFM[ispecie]
            
            # PYTHON_NOTE: Because the slices does not include the end, we have to add 1 to the end of the slice.
            ret_vector[ind1FM:ind2FM+1] = mpy.get_vector(results.resultsList, retrieval, species_name, FM_Flag, INITIAL_Flag, TRUE_Flag, CONSTRAINT_Flag)
            con_vector[ind1FM:ind2FM+1] = mpy.get_vector(retrieval.constraintVector, retrieval, species_name, FM_Flag, INITIAL_Flag, TRUE_Flag, CONSTRAINT_Flag)
    
            if retrieval.mapType[ispecie].lower() == 'log':  # Note the spelling of 'mapType' in retrieval object.
                # PYTHON_NOTE: Because the slices does not include the end, we have to add 1 to the end of the slice.
                ret_vector[ind1FM:ind2FM+1] = np.log(ret_vector[ind1FM:ind2FM+1])
                con_vector[ind1FM:ind2FM+1] = np.log(con_vector[ind1FM:ind2FM+1])
        # end for ispecie in range(retrieval.n_species):
    
        # AT_LINE 204 Error_Analysis_Wrapper.pro
        constraintVector = con_vector
        resultVector = ret_vector
    
        # AT_LINE 208 Error_Analysis_Wrapper.pro
    
        # select out non-spike indices
        ind = np.where(dataError < 0)[0]
        if len(ind) > 0:
            jacobian[:, ind] = 0
            jacobianFM[:, ind] = 0
            if jacobianSys is not None:
                jacobianSys[:, ind] = 0
            actualDataResidual[ind] = 0
        # end part of if len(ind) > 0:
    
        # if not updating, keep current error analysis
        currentSpecies = retrieval.species[0:retrieval.n_species]
        errorCurrentValues = mpy.constraint_get(errorCurrent.__dict__, currentSpecies)
    
        # AT_LINE 251 Error_Analysis_Wrapper.pro
    
        speciesList = retrieval.speciesList
        speciesListFM = retrieval.speciesListFM
    
        # AT_LINE 256 Error_Analysis_Wrapper.pro
        (result, offDiagonalSys) = mpy.error_analysis(
            None,
            my_map,
            jacobian,
            jacobianFM,
            Sa,
            jacobianSys,
            Sb,
            Db,
            constraintMatrix,
            constraintVector,
            trueVector,
            initialVector,
            resultVector,
            dataError,
            None,
            actualDataResidual,
            results,
            retrieval,
            stateInfo.current['heightKm'],
            errorCurrentValues)
    
        # AT_LINE 280 Error_Analysis_Wrapper.pro
        speciesList = retrieval.speciesList
        speciesListFM = retrieval.speciesListFM
    
        # AT_LINE 336 Error_Analysis_Wrapper.pro
        # update errorCurrent
        # first clear correlations between current retrieved species and all
        # others then set error based on current error analysis
        errorCurrent = mpy.constraint_clear(errorCurrent.__dict__, currentSpecies)
        errorCurrent = mpy.constraint_set(errorCurrent.__dict__, results.Sx, currentSpecies)
        if jacobianSys is not None:
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
        # end if jacobianSys is not None:
    
        return (result, errorCurrent)

__all__ = [
    "ErrorAnalysis",
]
