from __future__ import annotations
import refractor.muses.muses_py as mpy  # type: ignore
from .observation_handle import mpy_radiance_from_observation_list
from .identifier import StateElementIdentifier
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

    Unlike a number of things that we want to elevate to a class, this
    really does look like just a structure of various calculated
    things that then get reported in the output files - so I think
    this is probably little more than wrapping up stuff in one place.

    The coupling of this isn't great, we may want to break this up. The current
    set of steps needed to fully generate this can be found in
    RetrievalStrategyStepRetrieve

    1. Use results of the run_retrieval to create RetrievalResult using the
       constructor (__init__).
    2. Pass this object along with other object to update the StateInfo
       (StateInfo.update_state)
    3. Generate the systematic Jacobian using this updated StateInfo, and
       store results in RetrievalResult
    4. Run the error analysis (ErrorAnalysis.update_retrieval_result) and
       store the results in RetrievalResult
    5. Run the QA analysis (QaDataHandleSet.qa_update_retrieval_result) and
       store the results in RetrievalResult

    TODO Not sure how to unpack this, but we'll work on it.
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
        # Get old retrieval results structure, and merge in with this object
        d = self.set_retrieval_results()
        self.__dict__.update(d)
        mpy.set_retrieval_results_derived(
            self,
            self.rstep,
            propagated_qa.tatm_qa,
            propagated_qa.o3_qa,
            propagated_qa.h2o_qa,
        )

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

    @frequency.setter
    def frequency(self, v : np.ndarray) -> None:
        # Kind of a kludge, but set_retrieval_results_derived wants to set this
        # value, even though we already have it. Just ignore this, so we
        # don't need to mess with set_retrieval_results_derived. Long term,
        # we should pull out the set_retrieval_results_derived stuff into this class, but
        # for now just punt on that.
        pass

    @property
    def radiance(self) -> np.ndarray:
        # Not sure how important this is, but old muses-py code had this
        # as float32. Match this for now, we might remove this at some point,
        # but this does has small changes to the output
        return self.ret_res.radiance["radiance"][np.newaxis,:].astype(np.float32)

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

        # get filters start and end
        # have standard list of filters
        # We need a better way of instantiating this
        ff0 = []
        for i in range(len(self.rstep.filterNames)):
            ff0.append(
                [
                    str(self.rstep.filterNames[i]),
                ]
                * self.rstep.filterSizes[i]
            )
        ff1 = np.concatenate(ff0)
        # TODO Replace this logic
        filtersName = [
            "ALL",
            "UV1",
            "UV2",
            "VIS",
            "CrIS-fsr-lw",
            "CrIS-fsr-mw",
            "CrIS-fsr-sw",
            "2B1",
            "1B2",
            "2A1",
            "1A1",
            "BAND1",
            "BAND2",
            "BAND3",
            "BAND4",
            "BAND5",
            "BAND6",
            "BAND7",
            "BAND8",
            "O2A",
            "WCO2",
            "SCO2",
            "CH4",
        ]
        filtersMap = [
            "ALL",
            "UV1",
            "UV2",
            "VIS",
            "TIR1",
            "TIR3",
            "TIR4",
            "TIR1",
            "TIR2",
            "TIR3",
            "TIR4",
            "UV1",
            "UV2",
            "UVIS",
            "VIS",
            "NIR1",
            "NIR2",
            "SWIR3",
            "SWIR4",
            "NIR1",
            "SWIR1",
            "SWIR2",
            "SWIR3",
        ]
        filters = [
            "ALL",
            "UV1",
            "UV2",
            "VIS",
            "UVIS",
            "NIR1",
            "NIR2",
            "SWIR1",
            "SWIR2",
            "SWIR3",
            "SWIR4",
            "TIR1",
            "TIR2",
            "TIR3",
            "TIR4",
        ]
        num_filters = len(filters)
        filterStart = [0]
        filterEnd = [len(ff1) - 1]
        filter_index = [0]
        filter_list = ["ALL"]

        for jj in range(
            1, len(filtersName)
        ):  # Start jj with 1 since we are leaving the first value alone.
            ind1 = np.where(ff1 == filtersName[jj])[0]
            ind2 = np.where(np.array(filters) == filtersMap[jj])[0][0]
            if len(ind1) > 0:
                filter_index.append(ind2)
                filter_list.append(filters[ind2])
                filterStart.append(int(np.amin(ind1)))
                filterEnd.append(int(np.amax(ind1)))

        num_filters = len(filter_index)
        if num_filters == 1:
            raise RuntimeError(
                "Update set_retrieval_results.  Filter not found in list: ", ff1
            )

        # get the total number of frequency points in all microwindows for the
        # gain matrix
        rows = len(self.current_state.retrieval_state_vector_element_list)
        rowsSys = len(self.current_state.systematic_model_state_vector_element_list)
        rowsFM = len(self.current_state.forward_model_state_vector_element_list)
        if rowsSys == 0:
            rowsSys = 1

        o_results: dict[str, Any] = {
            "badRetrieval": -999,
            "filter_index": filter_index,
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
            "filter_list": filter_list,  #  Note, we changed 'filter' to 'filter_list' since 'filter' is a Python keyword.
            "filterStart": filterStart,
            "filterEnd": filterEnd,
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


__all__ = [
    "RetrievalResult",
]
