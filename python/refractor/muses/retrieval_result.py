from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .observation_handle import mpy_radiance_from_observation_list
from .identifier import StateElementIdentifier, FilterIdentifier
import math
import numpy as np
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .cost_function import CostFunction
    from .muses_observation import MusesObservation
    from .current_state import CurrentState, PropagatedQA


class RetrievalResult:
    """There are a few top level functions that work with a structure called
    retrieval_results. Pull all this together into an object so we can clearly
    see the interface and possibly change things.

    The coupling of this isn't great, we may want to break this up. The current
    set of steps needed to fully generate this can be found in
    RetrievalStrategyStepRetrieve

    1. Use results of the run_retrieval to create RetrievalResult using the
       constructor (__init__).
    2. Generate the systematic Jacobian using this updated StateInfo, and
       store results in RetrievalResult
    3. Run the error analysis (ErrorAnalysis.update_retrieval_result) and
       store the results in RetrievalResult
    4. Run the QA analysis (QaDataHandleSet.qa_update_retrieval_result) and
       store the results in RetrievalResult

    We may also want to break this up into smaller pieces (e.g., put all the column stuff
    together). But this really is just "a bunch of stuff we generate from a retrieval step
    solution", so the design may be fine.
    """

    def __init__(
        self,
        ret_res: dict,
        current_state: CurrentState,
        obs_list: list[MusesObservation],
        radiance_full: dict,
        propagated_qa: PropagatedQA,
        jacobian_sys: np.ndarray | None = None,
    ):
        """ret_res is what we get returned from MusesLevmarSolver"""
        self.rstep = mpy.ObjectView(
            mpy_radiance_from_observation_list(obs_list, include_bad_sample=True)
        )
        self.radiance_full = radiance_full
        self.obs_list = obs_list
        self.instruments = [obs.instrument_name for obs in self.obs_list]
        self.current_state = current_state
        self.sounding_metadata = current_state.sounding_metadata
        self.ret_res = mpy.ObjectView(ret_res)
        self.jacobianSys = jacobian_sys
        self.propagated_qa = propagated_qa
        # Get old retrieval results structure, and merge in with this object
        d = self.set_retrieval_results()
        self.__dict__.update(d)
        self.set_retrieval_results_derived()

    def update_jacobian_sys(self, cfunc_sys: CostFunction) -> None:
        """Run the forward model in cfunc to get the jacobian_sys set."""
        self.jacobianSys = (
            cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[
                np.newaxis, :, :
            ]
        )

    def state_value(self, state_name : str) -> float:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name : str) -> np.ndarray:
        return self.current_state.full_state_value(StateElementIdentifier(state_name))
    
    @property
    def tropopause_pressure(self) -> float:
        res = self.state_value("gmaoTropopausePressure")
        if(res <= -990):
            raise RuntimeError("GMA tropopause pressure is not defined")
        return res

    @property
    def tropopausePressure(self) -> float:
        return self.tropopause_pressure
    
    @property
    def omi_cloudfraction(self) -> float:
        return self.state_value("OMICLOUDFRACTION")

    @property
    def tropomi_cloudfraction(self) -> float:
        return self.state_value("TROPOMICLOUDFRACTION")

    @property
    def radiance_initial(self) -> np.ndarray:
        # Not sure how important this is, but old muses-py code had this
        # as float32. Match this for now, we might remove this at some point,
        # but this does has small changes to the output
        if(self.num_iterations == 0):
            return self.ret_res.radiance["radiance"][:, :].astype(np.float32)
        return self.ret_res.radianceIterations[0, :, :].astype(np.float32)

    @property
    def radianceInitial(self) -> np.ndarray:
        return self.radiance_initial
    
    @property
    def LMResults_costThresh(self) -> np.ndarray:
        return self.ret_res.stopCriteria[:, 0]

    @property
    def LMResults_resNorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 0]

    @property
    def LMResults_resNormNext(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 1]

    @property
    def LMResults_jacresNorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 2]

    @property
    def LMResults_jacResNormNext(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 3]

    @property
    def LMResults_pnorm(self) -> np.ndarray:
        return self.ret_res.resdiag[:, 4]

    @property
    def LMResults_delta(self) -> np.ndarray:
        return self.ret_res.delta

    @property
    def LMResults_iterList(self) -> np.ndarray:
        res = [self.current_state.initial_guess]
        for i in range(1, self.ret_res.num_iterations + 1):
            res.append(self.ret_res.xretIterations[i, :])
        return np.vstack(res)

    @property
    def frequency(self) -> np.ndarray:
        return self.rstep.frequency

    @property
    def radiance(self) -> np.ndarray:
        # Not sure how important this is, but old muses-py code had this
        # as float32. Match this for now, we might remove this at some point,
        # but this does has small changes to the output
        if(len(self.ret_res.radiance["radiance"].shape) == 1):
            return self.ret_res.radiance["radiance"][np.newaxis,:].astype(np.float32)
        return self.ret_res.radiance["radiance"].astype(np.float32)

    @property
    def radianceObserved(self) -> np.ndarray:
        # Not sure how important this is, but old muses-py code had this
        # as float32. Match this for now, we might remove this at some point,
        # but this does has small changes to the output
        return self.rstep.radiance.astype(np.float32)

    @property
    def NESR(self) -> np.ndarray:
        # Not sure how important this is, but old muses-py code had this
        # as float32. Match this for now, we might remove this at some point,
        # but this does has small changes to the output
        return self.rstep.NESR.astype(np.float32)

    @property
    def jacobian(self) -> np.ndarray:
        return self.ret_res.jacobian["jacobian_data"][np.newaxis,:,:]
    
    @property
    def resultsList(self) -> np.ndarray:
        return self.current_state.initial_guess if self.best_iteration == 0 else self.ret_res.xretIterations[self.best_iteration, :]

    @property
    def resultsListFM(self) -> np.ndarray:
        return self.ret_res.xretFM

    @property
    def is_ocean(self) -> bool:
        return self.current_state.sounding_metadata.is_ocean

    @property
    def retIteration(self) -> int:
        return self.ret_res.xretIterations
    
    @property
    def bestIteration(self) -> int:
        return self.ret_res.bestIteration
    
    @property
    def num_iterations(self) -> int:
        return self.ret_res.num_iterations

    @property
    def stopCode(self) -> int:
        return self.ret_res.stopCode

    @property
    def Desert_Emiss_QA(self) -> float:
        wlen = self.current_state.full_state_spectral_domain_wavelength(
            StateElementIdentifier("emissivity")
        )
        if wlen is None:
            raise RuntimeError("Expected to find emissivity frequencies")
        ind = np.argmin(np.abs(wlen - 1025))
        return self.state_value_vec("emissivity")[ind]
        
    @property
    def species_list_fm(self) -> list[str]:
        """This is the length of the forward model state vector, with a
        retrieval_element name for each location."""
        return [
            str(i) for i in self.current_state.forward_model_state_vector_element_list
        ]

    @property
    def species_list_retrieval(self) -> list[str]:
        """This is the length of the retrieval state vector, with a
        retrieval_element name for each location."""
        return [str(i) for i in self.current_state.retrieval_state_element_id]

    @property
    def pressure_list_fm(self) -> np.ndarray:
        pdata = []
        # Convention of muses-py is to use [-2] for items that aren't on
        # pressure levels
        for sid in self.current_state.retrieval_state_element_id:
            d = self.current_state.pressure_list_fm(sid)
            if d is not None:
                pdata.append(d)
            else:
                pdata.append(np.array([-2.0]))
        return np.concatenate(pdata)

    @property
    def best_iteration(self) -> int:
        return self.bestIteration  # type: ignore[attr-defined]

    @property
    def results_list(self) -> np.ndarray:
        return self.resultsList  # type: ignore[attr-defined]

    @property
    def master_quality(self) -> int:
        return self.masterQuality

    @master_quality.setter
    def master_quality(self, val: int) -> None:
        self.masterQuality = val

    @property
    def jacobian_sys(self) -> np.ndarray | None:
        return self.jacobianSys

    @property
    def filter_index(self) -> list[int]:
        res = [0]
        res.extend(i.filter_index for i in self.rstep.filterNames)
        # Reorder, if needed
        sorder = FilterIdentifier.spectral_order([FilterIdentifier("ALL"), *self.rstep.filterNames])
        return [res[i] for i in sorder]
    
    @property
    def filter_list(self) -> list[str]:
        res = ['ALL']
        res.extend(i.spectral_name for i in self.rstep.filterNames)
        # Reorder, if needed
        sorder = FilterIdentifier.spectral_order([FilterIdentifier("ALL"), *self.rstep.filterNames])
        return [res[i] for i in sorder]
    
    @property
    def filterStart(self) -> list[int]:
        res = [0]
        istart = 0
        for fs in self.rstep.filterSizes:
            res.append(istart)
            istart += fs
        # Reorder, if needed
        sorder = FilterIdentifier.spectral_order([FilterIdentifier("ALL"), *self.rstep.filterNames])
        return [res[i] for i in sorder]

    @property
    def filterEnd(self) -> list[int]:
        res = [0]
        iend = -1
        for fs in self.rstep.filterSizes:
            iend += fs
            res.append(iend)
        res[0] = iend
        # Reorder, if needed
        sorder = FilterIdentifier.spectral_order([FilterIdentifier("ALL"), *self.rstep.filterNames])
        return [res[i] for i in sorder]

    @property
    def propagatedTATMQA(self) -> int:
        return self.propagated_qa.tatm_qa
    
    @property
    def propagatedO3QA(self) -> int:
        return self.propagated_qa.o3_qa
    
    @property
    def propagatedH2OQA(self) -> int:
        return self.propagated_qa.h2o_qa
        
    def set_retrieval_results(self) -> dict:
        """This is our own copy of mpy.set_retrieval_results, so we
        can start making changes to clean up the coupling of this.

        """
        # Convert any dict to ObjectView so we can have a consistent
        # way of referring to our input.
        num_species = len(self.current_state.retrieval_state_element_id)
        nfreqs = len(self.rstep.frequency)

        detectorsUsed = 0
        num_detectors = 1

        num_filters = len(self.filter_index)
        
        # get the total number of frequency points in all microwindows for the
        # gain matrix
        rows = len(self.current_state.retrieval_state_vector_element_list)
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        if rowsSys == 0:
            rowsSys = 1

        o_results: dict[str, Any] = {
            "badRetrieval": -999,
            "radianceResidualMean": np.zeros(shape=(num_filters), dtype=np.float32),
            "radianceResidualMeanInitial": np.zeros(
                shape=(num_filters), dtype=np.float32
            ),
            "radianceResidualRMS": np.zeros(shape=(num_filters), dtype=np.float32),
            "radianceResidualRMSInitial": np.zeros(
                shape=(num_filters), dtype=np.float32
            ),
            "radianceResidualRMSMean": np.zeros(shape=(num_filters), dtype=np.float32),
            "radianceResidualRMSRelativeContinuum": np.zeros(
                shape=(num_filters), dtype=np.float32
            ),
            "residualSlope": np.zeros(shape=(num_filters), dtype=np.float32),
            "residualQuadratic": np.zeros(shape=(num_filters), dtype=np.float32),
            "radianceContinuum": np.zeros(shape=(num_filters), dtype=np.float32),
            "radianceSNR": np.zeros(shape=(num_filters), dtype=np.float32),
            "residualNormInitial": 0.0,
            "residualNormFinal": 0.0,
            "detectorsUsed": detectorsUsed,
            "radianceResidualMeanDet": np.zeros(
                shape=(num_detectors), dtype=np.float64
            ),
            "radianceResidualRMSDet": np.zeros(shape=(num_detectors), dtype=np.float64),
            "radianceResidualRMSSys": 0.0,
            "freqShift": 0.0,
            "chiApriori": 0.0,
            "radianceResidualMax": 0.0,
            "chiLM": np.copy(self.ret_res.residualRMS[self.ret_res.bestIteration]),
            "num_radiance ": 0,
            "error": np.zeros(shape=(rows), dtype=np.float64),
            "errorFM": np.zeros(shape=(rowsFM), dtype=np.float64),
            "precision": np.zeros(shape=(rowsFM), dtype=np.float64),
            "resolution": np.zeros(shape=(rowsFM), dtype=np.float64),
            # jacobians - for last outputStep
            "GdL": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            "jacobianSys": None,
            # error stuff follows - calc later
            "A": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float32),
            "A_ret": np.zeros(shape=(rows, rows), dtype=np.float32),
            "KtSyK": np.zeros(shape=(rows, rows), dtype=np.float32),
            "Sx": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sa_ret": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_smooth": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_crossState": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_rand": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_sys": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_mapping": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sa": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sb": np.zeros(shape=(rowsSys, rowsSys), dtype=np.float64),
            "Sx_smooth_self": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_smooth": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_crossState": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_sys": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_rand": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "Sx_mapping": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "SxActual": np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64),
            "GMatrix": np.zeros(shape=(nfreqs, rows), dtype=np.float64),
            "GMatrixFM": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            # by species
            "informationContentSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "degreesOfFreedomNoise": np.zeros(shape=(num_species), dtype=np.float64),
            "degreesOfFreedomForSignal": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "degreesOfFreedomForSignalTrop": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "bestDegreesOfFreedomList": ["" for x in range(num_species)],
            "bestDegreesOfFreedomTotal": ["" for x in range(num_species)],
            "verticalResolution": np.zeros(shape=(num_species), dtype=np.float64),
            "deviationVsError": 0.0,
            "deviationVsRetrievalCovariance": 0.0,
            "deviationVsAprioriCovariance": 0.0,
            "deviationVsErrorSpecies": np.zeros(shape=(num_species), dtype=np.float64),
            "deviationVsRetrievalCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            "deviationVsAprioriCovarianceSpecies": np.zeros(
                shape=(num_species), dtype=np.float64
            ),
            # quality and general
            "KDotDL": 0.0,
            "KDotDL_list": np.zeros(shape=(rows), dtype=np.float32),
            "KDotDL_byspecies": np.zeros(shape=(num_species), dtype=np.float32),
            "KDotDL_species": ["" for x in range(num_species)],
            "KDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float32),
            "maxKDotDLSys": 0.0,
        }

        struct2 = {
            "LDotDL": 0.0,
            "LDotDL_byfilter": np.zeros(shape=(num_filters), dtype=np.float32),
            "cloudODAve": 0.0,
            "cloudODAveError": 0.0,
            "emisDev": 0.0,
            "cloudODVar": 0.0,
            "calscaleMean": 0.0,
            "H2O_H2OQuality": 0.0,
            "emissionLayer": 0.0,
            "ozoneCcurve": 0.0,
            "ozone_slope_QA": -999.0,
            "propagatedTATMQA": 0.0,
            "propagatedO3QA": 0.0,
            "propagatedH2OQA": 0.0,
            "masterQuality": -999,
            "columnAir": np.full((5), -999, dtype=np.float64),
            "column": np.full((5, 20), -999, dtype=np.float64),  # DBLARR(4, 20)-999.0
            "columnError": np.full(
                (5, 20), -999, dtype=np.float64
            ),  # DBLARR(4, 20)-999.0
            "columnPriorError": np.full(
                (5, 20), -999, dtype=np.float64
            ),  #  DBLARR(4, 20)-999.0
            "columnInitialInitial": np.full(
                (5, 20), -999, dtype=np.float64
            ),  # DBLARR(4, 20)-999.0
            "columnInitial": np.full(
                (5, 20), -999, dtype=np.float64
            ),  # DBLARR(4, 20)-999.0
            "columnPrior": np.full(
                (5, 20), -999, dtype=np.float64
            ),  # DBLARR(4, 20)-999.0
            "columnTrue": np.full(
                (5, 20), -999, dtype=np.float64
            ),  # DBLARR(4, 20)-999.0
            "columnSpecies": ["" for x in range(20)],  # STRARR(20)
            # EM NOTE - Modified to increase vector size to allow for stratosphere capture
            "columnPressureMax": np.zeros(shape=(5), dtype=np.float32),  # FLTARR(4)
            "columnPressureMin": np.zeros(shape=(5), dtype=np.float32),  # FLTARR(4)
            "columnDOFS": np.zeros(shape=(5, 20), dtype=np.float32),  # FLTARR(4, 20)
            "tsur_minus_tatm0": -999.0,
            "tsur_minus_prior": -999.0,
            "deviation_QA": np.full(
                (num_species), -999, dtype=np.float32
            ),  # FLTARR(num_species)-999
            "num_deviations_QA": np.full(
                (num_species), -999, dtype=np.int32
            ),  # INTARR(num_species)-999
            "DeviationBad_QA": np.full(
                (num_species), -999, dtype=np.int32
            ),  # INTARR(num_species)-999
            "O3_columnErrorDU": 0.0,  # total colummn error
            "O3_tropo_consistency": 0.0,  # tropospheric column change from initial
            "ch4_evs": np.zeros(shape=(10), dtype=np.float32),  # FLTARR(10)
        }
        o_results.update(struct2)
        return o_results

    def set_retrieval_results_derived(self) -> None:
        # calc radianceResidualMean and radianceResidualRMS
        rObs = self.rstep.radiance
        rIter = self.radiance
        rInitial = self.radianceInitial
    
        NESR = self.rstep.NESR
        freq = self.rstep.frequency
    
        # AT_LINE 26 Set_Retrieval_Results_Derived.pro Set_Retrieval_Results_Derived
        ind = np.where(self.rstep.frequency > 0.0)[0]
    
        # PYTHON_NOTE: It is possible that the size of freq is greater than NESR and rObs
        #
        #     NESR.shape (220, )
        #     freq.shape (370, )
        #     ind.shape 370
        #     rObs.shape (220, )
        #
        # so we cannot not use ind as is an index because that would cause an out of bound:
        # IndexError: index 220 is out of bounds for axis 1 with size 220
        # To fix the issue, we make a reduced_index with indices smaller than the size of NESR.
    
        if len(freq) > len(NESR):
            # Get the indices of ind where the values are smaller than the size of NESR.
            reduced_index = np.where(ind < NESR.shape[0])[0]
            # Use the reduced_index to index ind so we don't index pass the size of rObs and NESR 
            self.radianceMaximumSNR = np.amax(rObs[ind[reduced_index]] / NESR[ind[reduced_index]])
        else:
            self.radianceMaximumSNR = np.amax(rObs[ind] / NESR[ind])
    
        # take out spikes
        # AT_LINE 30 Set_Retrieval_Results_Derived.pro Set_Retrieval_Results
        # We have to use slow method.
        indv = []
        for ii in range(len(NESR)):
            if NESR[ii] >= 0.0 and NESR[ii] <= abs(np.mean(NESR)*100.0):
                indv.append(ii)
    
        ind = np.asarray(indv)
    
        # AT_LINE 32 Set_Retrieval_Results_Derived.pro Set_Retrieval_Results_Derived
        if len(ind) > 0:
            rObs = rObs[ind]
            rIter = rIter[0, ind]
            rInitial = rInitial[0, ind]
    
            NESR = NESR[ind]
    
            freq = freq[ind]
        else:
            raise RuntimeError("No good radiances, all NESRs < 0")
    
        self.radianceResidualRMS[1:] = -999.0
        self.radianceResidualMean[1:] = -999.0
        self.residualSlope[:] = -999 
        self.residualQuadratic[:] = -999 
    
    
        if len(self.filter_list) > 1:
            # Start at 1 for the loop.
            for jj in range(0, len(self.filter_list)):
                if jj == 0:
                    start = 0
                    endd = len(self.rstep.NESR)
                else:
                    start = self.filterStart[jj]
                    endd = self.filterEnd[jj]
    
                if start >= 0 and endd >= 0:
                    ind = np.where(self.rstep.NESR[start:endd+1] > 0)[0]  # Note that we use 'endd+1' because in Python, the slice does not include the end point.
                    
                    if len(ind) > 5:
                        scaledDifference = (self.rstep.radiance[start+ind]- self.radiance[0, start+ind]) / self.rstep.NESR[start+ind]
                        uu_mean = np.mean(scaledDifference)
                        uu_var = np.var(scaledDifference)
                        self.radianceResidualRMS[jj] = math.sqrt(uu_var) # actual stdev NOT RMS
                        self.radianceResidualMean[jj] = uu_mean
    
                        scaledDifferenceInitial = (self.rstep.radiance[start+ind]- self.radianceInitial[0, start+ind]) / self.rstep.NESR[start+ind]
                        uu_mean = np.mean(scaledDifferenceInitial)
                        uu_var = np.var(scaledDifferenceInitial)
                        self.radianceResidualRMSInitial[jj] = math.sqrt(uu_var) # actual stdev NOT RMS
                        self.radianceResidualMeanInitial[jj] = uu_mean
    
                        # get stdev relative to maximum (top 2% of radiances in fit)
                        vals = np.sort(self.radiance[0, start+ind])
                        nx = len(vals)
                        if nx > 50:
                            vals = np.mean(vals[int(len(vals)*49/50):len(vals)])
                        else:
                            vals = np.max(vals)
                        difference = (self.rstep.radiance[start+ind]- self.radiance[0, start+ind])
                        uu_var = np.var(difference)
                        uu_mean = np.mean(difference)
                        self.radianceResidualRMSRelativeContinuum[jj] = math.sqrt(uu_var + uu_mean * uu_mean) / vals
    
                        self.radianceContinuum[jj] = vals
                        self.radianceSNR[jj] = np.mean(self.radiance[0, start+ind] / self.rstep.NESR[start+ind])
    
                        # get first and second derivative of normalized residual versus radiance / continuum
                        # this looks for patterns such as issues at the line core
                        myx = self.radiance[0, start+ind].copy() / vals
                        myy = scaledDifference.copy()
                        indx = np.argsort(myx)
                        myx = myx[indx]
                        myy = myy[indx]
    
                        # cut off the very few points "above" the continuum
                        indx = (np.where((myx > 0.0)*(myx < 1.00)))[0]
                        myx = myx[indx]
                        myy = myy[indx]
    
    
                        # linear fit and quadratic fit.  Save linear value from linear fit, and quadratic value from quadratic fit.
                        linear_fit = np.polyfit(myx, myy, 1) # return has highest order fit first
                        quadratic_fit = np.polyfit(myx, myy, 2) # return has highest order fit first
                        self.residualSlope[jj] = linear_fit[0]
                        self.residualQuadratic[jj] = quadratic_fit[0]
    
                        if jj < -999:
                            # alternative method
                            indstart, indend = mpy.frequency_get_bands(self.frequency)
                            radiance = self.radiance[0, :]
                            radiance_obs = self.rstep.radiance
                            nesr = self.rstep.NESR
    
                            inds = np.argsort(radiance[indstart[jj-1]:indend[jj-1]+1])
                            # get top 5% of values in each band
                            # sort by radiance size
                            if len(inds) > 20:
                                indxv = int(len(inds)*0.98)
                                continuum = np.mean(radiance[indstart[jj-1]+inds[indxv]])
                            else:
                                continuum = radiance[indstart[jj-1]+inds[len(inds)-1]]
    
                            # band values
                            # divide radiance by continuum to get a relative radiance (0-1.0+)
                            myrad = radiance[indstart[jj-1]:indend[jj-1]+1].copy() / continuum
                            myerror = (radiance_obs[indstart[jj-1]:indend[jj-1]+1]-radiance[indstart[jj-1]:indend[jj-1]+1])/nesr[indstart[jj-1]:indend[jj-1]+1]
    
                            # order by radiance size
                            inds = np.argsort(myrad)
                            myrad = myrad[inds]
                            myerror = myerror[inds]
    
                            inds = (np.where((myrad > 0.0)*(myrad < 1.00)))[0]
                            myrad = myrad[inds]
                            myerror = myerror[inds]
    
                # end if start >= 0 and endd >= 0:
            # end for jj in range(1, len(self.filter)):
        # end if len(self.filter) > 1:
    
        # AT_LINE 124 Set_Retrieval_Results_Derived.pro Set_Retrieval_Results_Derived
        # calc residualNormInitial and residualNormFinal
        # NOT chi2, but chi
        self.residualNormInitial = math.sqrt(
            self.radianceResidualMeanInitial[0] * self.radianceResidualMeanInitial[0] + \
            self.radianceResidualRMSInitial[0] * self.radianceResidualRMSInitial[0]
        )
        
        self.residualNormFinal = math.sqrt(
            self.radianceResidualMean[0] * self.radianceResidualMean[0] + \
            self.radianceResidualRMS[0] * self.radianceResidualRMS[0]
        )

__all__ = [
    "RetrievalResult",
]
