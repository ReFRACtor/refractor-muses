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
        # Filled in with ErrorAnalysis.write_retrieval_summary. We may move some of
        # this calculation to this class, but for now it gets done there.
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        num_species = len(self.current_state.retrieval_state_element_id)
        # This really is exactly 5. See the column calculation. This is
        # ["Column", "Trop", "UpperTrop", "LowerTrop", ""Strato
        num_col = 5 
        # TODO Would be good to calculate this somewhat
        max_num_species = 20
        if rowsSys == 0:
            rowsSys = 1
        self.cloudODAve = 0.0
        self.cloudODVar = 0.0
        self.cloudODAveError = 0.0
        self.emisDev = 0.0
        self.emissionLayer = 0.0
        self.H2O_H2OQuality = 0.0
        self.O3_columnErrorDU = 0.0
        self.O3_tropo_consistency = 0.0
        self.ozoneCcurve = 0.0
        self.ozone_slope_QA = -999.0
        self.errorFM = np.zeros(shape=(rowsFM), dtype=np.float64)
        self.columnDOFS = np.zeros(shape=(num_col, max_num_species), dtype=np.float32)
        self.A =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float32)
        self.Sa =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.Sb =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.Sx =  np.zeros(shape=(rowsFM, rowsFM), dtype=np.float64)
        self.columnPriorError = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnInitial = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnInitialInitial = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnError= np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnPrior= np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.column = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnAir = np.full((num_col), -999, dtype=np.float64)
        self.columnTrue = np.full((num_col, max_num_species), -999, dtype=np.float64)
        self.columnPressureMax = np.zeros(shape=(num_col), dtype=np.float32)
        self.columnPressureMin = np.zeros(shape=(num_col), dtype=np.float32)
        self.columnSpecies = ["",] * max_num_species
        self.DeviationBad_QA = np.full((num_species), -999, dtype=np.int32)
        self.num_deviations_QA = np.full((num_species), -999, dtype=np.int32)
        self.deviation_QA = np.full((num_species), -999, dtype=np.float32)

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

    @property
    def radianceMaximumSNR(self) -> float:
        return np.amax(self.rstep.radiance[self.rstep.frequency > 0.0] / self.rstep.NESR[self.rstep.frequency > 0.0])
    
    @property
    def radianceResidualMean(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                scaled_diff = (self.rstep.radiance[slc][gpt] - self.radiance[0,slc][gpt])/ self.rstep.NESR[slc][gpt]
                res.append(np.mean(scaled_diff))
            else:
                res.append(0 if len(res) == 0 else -999.0)
        return np.array(res).astype(np.float32)

    @property
    def radianceResidualRMS(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                scaled_diff = (self.rstep.radiance[slc][gpt] - self.radiance[0,slc][gpt])/ self.rstep.NESR[slc][gpt]
                res.append(math.sqrt(np.var(scaled_diff)))
            else:
                res.append(0 if len(res) == 0 else -999.0)
        return np.array(res).astype(np.float32)

    @property
    def radianceResidualMeanInitial(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                scaled_diff = (self.rstep.radiance[slc][gpt] - self.radianceInitial[0,slc][gpt])/ self.rstep.NESR[slc][gpt]
                res.append(np.mean(scaled_diff))
            else:
                res.append(0 if len(res) == 0 else -999.0)
        return np.array(res).astype(np.float32)

    @property
    def radianceResidualRMSInitial(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                scaled_diff = (self.rstep.radiance[slc][gpt] - self.radianceInitial[0,slc][gpt])/ self.rstep.NESR[slc][gpt]
                res.append(math.sqrt(np.var(scaled_diff)))
            else:
                res.append(0 if len(res) == 0 else -999.0)
        return np.array(res).astype(np.float32)

    @property
    def radianceResidualRMSRelativeContinuum(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                valsv = np.sort(self.radiance[0,slc][gpt])
                if(len(valsv) > 50):
                    # TODO I don't think this actually does what is intended.
                    # Should this be np.mean(vals[-50:]?
                    vals = np.mean(valsv[int(len(valsv)*49/50):len(valsv)])
                else:
                    vals = np.max(valsv)
                diff = (self.rstep.radiance[slc][gpt] - self.radiance[0,slc][gpt])
                uu_var = np.var(diff)
                uu_mean = np.mean(diff)
                res.append(math.sqrt(uu_var + uu_mean * uu_mean) / vals)
            else:
                res.append(0)
        return np.array(res).astype(np.float32)

    @property
    def radianceContinuum(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                valsv = np.sort(self.radiance[0,slc][gpt])
                if(len(valsv) > 50):
                    # TODO I don't think this actually does what is intended.
                    # Should this be np.mean(vals[-50:]?
                    vals = np.mean(valsv[int(len(valsv)*49/50):len(valsv)])
                else:
                    vals = np.max(valsv)
                res.append(vals)
            else:
                res.append(0)
        return np.array(res).astype(np.float32)

    @property
    def radianceSNR(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                res.append(np.mean(self.radiance[0,slc][gpt] / self.rstep.NESR[slc][gpt]))
            else:
                res.append(0)
        return np.array(res).astype(np.float32)
    

    @property
    def residualNormInitial(self) -> float:
        t1 = self.radianceResidualMeanInitial[0]
        t2 = self.radianceResidualRMSInitial[0]
        return math.sqrt(t1*t1+t2*t2)

    @property
    def residualNormFinal(self) -> float:
        t1 = self.radianceResidualMean[0]
        t2 = self.radianceResidualRMS[0]
        return math.sqrt(t1*t1+t2*t2)

    @property
    def residualSlope(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                valsv = np.sort(self.radiance[0,slc][gpt])
                if(len(valsv) > 50):
                    # TODO I don't think this actually does what is intended.
                    # Should this be np.mean(vals[-50:]?
                    vals = np.mean(valsv[int(len(valsv)*49/50):len(valsv)])
                else:
                    vals = np.max(valsv)
                myx = self.radiance[0, slc][gpt] / vals
                myy = (self.rstep.radiance[slc][gpt] - self.radiance[0, slc][gpt]) / self.rstep.NESR[slc][gpt]
                # cut off the very few points "above" the continuum
                indx = (np.where((myx > 0.0)*(myx < 1.00)))[0]
                myx = myx[indx]
                myy = myy[indx]
                linear_fit = np.polyfit(myx, myy, 1)                
                res.append(linear_fit[0])
            else:
                res.append(-999.0)
        return np.array(res).astype(np.float32)

    @property
    def residualQuadratic(self) -> np.ndarray:
        res = []
        for s,e in zip(self.filterStart, self.filterEnd):
            slc = slice(s,e+1)
            gpt = self.rstep.NESR[slc] > 0
            if(np.count_nonzero(gpt) > 5):
                valsv = np.sort(self.radiance[0,slc][gpt])
                if(len(valsv) > 50):
                    # TODO I don't think this actually does what is intended.
                    # Should this be np.mean(vals[-50:]?
                    vals = np.mean(valsv[int(len(valsv)*49/50):len(valsv)])
                else:
                    vals = np.max(valsv)
                myx = self.radiance[0, slc][gpt] / vals
                myy = (self.rstep.radiance[slc][gpt] - self.radiance[0, slc][gpt]) / self.rstep.NESR[slc][gpt]
                # cut off the very few points "above" the continuum
                indx = (np.where((myx > 0.0)*(myx < 1.00)))[0]
                myx = myx[indx]
                myy = myy[indx]
                quadratic_fit = np.polyfit(myx, myy, 2) 
                res.append(quadratic_fit[0])
            else:
                res.append(-999.0)
        return np.array(res).astype(np.float32)
    
    def set_retrieval_results(self) -> dict:
        """This is our own copy of mpy.set_retrieval_results, so we
        can start making changes to clean up the coupling of this.

        """
        # Convert any dict to ObjectView so we can have a consistent
        # way of referring to our input.
        num_species = len(self.current_state.retrieval_state_element_id)
        nfreqs = len(self.rstep.frequency)
        num_filters = len(self.filter_index)
        
        # get the total number of frequency points in all microwindows for the
        # gain matrix
        rows = len(self.current_state.retrieval_state_vector_element_list)
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        if rowsSys == 0:
            rowsSys = 1

        o_results: dict[str, Any] = {
            "radianceResidualRMSSys": 0.0,
            "error": np.zeros(shape=(rows), dtype=np.float64),
            "precision": np.zeros(shape=(rowsFM), dtype=np.float64),
            "resolution": np.zeros(shape=(rowsFM), dtype=np.float64),
            # jacobians - for last outputStep
            "GdL": np.zeros(shape=(nfreqs, rowsFM), dtype=np.float64),
            "jacobianSys": None,
            # error stuff follows - calc later
            "A_ret": np.zeros(shape=(rows, rows), dtype=np.float32),
            "KtSyK": np.zeros(shape=(rows, rows), dtype=np.float32),
            "Sa_ret": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_smooth": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_crossState": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_rand": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_sys": np.zeros(shape=(rows, rows), dtype=np.float64),
            "Sx_ret_mapping": np.zeros(shape=(rows, rows), dtype=np.float64),
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
            "calscaleMean": 0.0,
            "masterQuality": -999,
            # EM NOTE - Modified to increase vector size to allow for stratosphere capture
            "tsur_minus_tatm0": -999.0,
            "tsur_minus_prior": -999.0,
            "ch4_evs": np.zeros(shape=(10), dtype=np.float32),  # FLTARR(10)
        }
        o_results.update(struct2)
        return o_results

__all__ = [
    "RetrievalResult",
]
