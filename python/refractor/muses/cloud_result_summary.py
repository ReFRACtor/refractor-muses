from __future__ import annotations
from . import fake_muses_py as mpy  # type: ignore
from .fake_state_info import FakeStateInfo
from .fake_retrieval_info import FakeRetrievalInfo
from .identifier import StateElementIdentifier
import numpy as np
import math
import typing

if typing.TYPE_CHECKING:
    from .error_analysis import ErrorAnalysis
    from .current_state import CurrentState


# Needs a lot of cleanup, we are just shoving stuff into place
class CloudResultSummary:
    def __init__(
        self,
        current_state: CurrentState,
        result_list: np.ndarray,
        error_analysis: ErrorAnalysis,
    ) -> None:
        self.current_state = current_state
        utilList = mpy.UtilList()
        stateInfo = FakeStateInfo(self.current_state)
        retrievalInfo = FakeRetrievalInfo(self.current_state)

        num_species = retrievalInfo.n_species

        self._cloudODVar = 0.0
        self._cloudODAveError = 0.0

        # AT_LINE 57 Write_Retrieval_Summary.pro
        factor = self.cloud_factor

        # step = 1
        # stepName = TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT,EMIS
        # product name Products_Jacobian-TATM,H2O,HDO,N2O,CH4,TSUR,CLOUDEXT-bar_land.nc

        ind = np.where(np.asarray(retrievalInfo.speciesList) == "CLOUDEXT")[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == "CLOUDEXT")[0]

        # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
        my_map = mpy.get_one_map(retrievalInfo, "CLOUDEXT")

        # AT_LINE 77 Write_Retrieval_Summary.pro
        if len(ind) > 0:
            # map error to ret
            errlog = error_analysis.errorFM[indFM] @ my_map["toPars"]

            cloudod = np.exp(result_list[ind]) * factor
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
            self._cloudODVar = math.sqrt(x)
        else:
            # cloud not retrieved... use 975-1200
            # NOTE: code has not been tested.
            cov = None
            selem = self.current_state.state_element(StateElementIdentifier("CLOUDEXT"))
            if selem is not None and selem.pressure_list_fm is not None:
                plist = selem.pressure_list_fm
                plist_ind = (plist >= 975) & (plist <= 1200)
                try:
                    cov = self.current_state.previous_aposteriori_cov_fm(
                        [StateElementIdentifier("CLOUDEXT")]
                    )
                except KeyError:
                    # If not in previous_aposteriori_cov_fm, then we just skip the next part
                    pass
            if cov is not None and np.count_nonzero(plist_ind) > 0:
                error_current = np.diag(cov[plist_ind, :][:, plist_ind])
                self._cloudODAveError = (
                    math.sqrt(np.sum(error_current))
                    / error_current.size
                    * self.cloudODAve
                )
        # end else part of if (len(ind) > 0):

        # AT_LINE 107 Write_Retrieval_Summary.pro
        ind = np.where(np.asarray(retrievalInfo.speciesList) == "CLOUDEXT")[0]
        indFM = np.where(np.asarray(retrievalInfo.speciesListFM) == "CLOUDEXT")[0]

        # NOTE: mpy.get_one_map will return maps that have columns and rows switched compared to the IDL implementation
        my_map = mpy.get_one_map(retrievalInfo, "CLOUDEXT")

        if len(ind) > 0:
            errlog = error_analysis.errorFM[indFM] @ my_map["toPars"]

            cloudod = np.exp(result_list[ind]) * factor
            err = (np.exp(np.log(cloudod) + errlog) - cloudod) * factor
            myMean = np.sum(cloudod / err / err) / np.sum(1 / err / err)

            x = np.var((cloudod - myMean) / err, ddof=1)
            self._cloudODVar = math.sqrt(x)

        ind = np.where(
            (stateInfo.emisPars["frequency"] >= 975)
            & (stateInfo.emisPars["frequency"] <= 1200)
        )[0]

        self._emissionLayer = 0
        self._emisDev = -999.0
        if len(ind) > 0:
            self._emisDev = np.mean(stateInfo.current["emissivity"][ind]) - np.mean(
                stateInfo.constraint["emissivity"][ind]
            )

        ind10 = np.asarray([])  # Start with an empty list so we have the variable set.
        if "O3" in retrievalInfo.speciesListFM:
            ind10 = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, "O3")

        # AT_LINE 162 Write_Retrieval_Summary.pro
        if len(ind10) > 0:
            indt = np.where(np.array(stateInfo.species) == "TATM")[0][0]
            TATM = stateInfo.current["values"][indt, :]
            TSUR = stateInfo.current["TSUR"]

            indo3 = np.where(np.array(stateInfo.species) == "O3")[0][0]
            o3 = stateInfo.current["values"][indo3, :]
            o3ig = stateInfo.constraint["values"][indo3, :]

            aveTATM = 0.0
            my_sum = 0.0
            for ii in range(0, 3):
                my_sum = my_sum + o3[ii] - o3ig[ii]
                aveTATM = aveTATM + TATM[ii]

            aveTATM = aveTATM / 3
            if my_sum / 3 >= 1.5e-9:
                self._emissionLayer = aveTATM - TSUR

        self._ozoneCcurve = 1
        self._ozone_slope_QA = 1
        ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, "O3")
        if len(ind) > 0:
            pressure = stateInfo.current["pressure"]
            o3 = stateInfo.current["values"][stateInfo.species.index("O3"), :]
            o3ig = stateInfo.initial["values"][stateInfo.species.index("O3"), :]
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
                my_map = mpy.get_one_map(retrievalInfo, "O3")

                AK = error_analysis.A[ind, :][:, ind]
                AKzz = np.matmul(np.matmul(my_map["toState"], AK), my_map["toPars"])
                meanAKlo = np.var(AKzz[indLow, indLow])

                if (
                    maxlo * 1e9 > 150
                    or (
                        (maxlo * 1e9 > 100 or meanlo / meanloig > 1.8)
                        and (meanAKlo < 0.1)
                    )
                    or (
                        (maxlo / minhi > 2.5 or maxlo / minhi >= 2)
                        and (maxlo / surf >= 1.1)
                    )
                ):
                    self._ozoneCcurve = 0
            # end if len(indLow) > 0 and len(indHigh) > 0:

            # slope c-curve flag
            o3 = stateInfo.current["values"][stateInfo.species.index("O3"), :]
            indp = np.where(o3 > 0)[0]
            altitude = stateInfo.current["heightKm"][indp]
            o3 = o3[indp]

            slope = mpy.ccurve_jessica(altitude, o3)
            self._ozone_slope_QA = slope
        # end if len(ind) > 0:

        # AT_LINE 255 Write_Retrieval_Summary.pro
        # Now get species dependent preferences
        self._deviation_QA = np.zeros((num_species,))
        self._num_deviations_QA = np.zeros((num_species,), dtype=int)
        self._DeviationBad_QA = np.zeros((num_species,), dtype=int)
        for ispecie in range(0, num_species):
            my_sum = 0.0

            species_name = retrievalInfo.species[ispecie]

            loc = -1
            if species_name in stateInfo.species:
                loc = np.where(np.array(stateInfo.species) == species_name)[0][0]

            # AT_LINE 269 Write_Retrieval_Summary.pro
            # deviation quality flag - refined for O3 and HCN
            self._deviation_QA[ispecie] = 1
            self._num_deviations_QA[ispecie] = 1
            self._DeviationBad_QA[ispecie] = 1
            pressure = stateInfo.current["pressure"]

            if loc != -1:
                # Only use the index() function if the species_name is in the stateInfo.species array.
                profile = stateInfo.current["values"][loc, :]
                constraint = stateInfo.constraint["values"][loc, :]

            ind = utilList.WhereEqualIndices(retrievalInfo.speciesListFM, species_name)

            ak_diag = error_analysis.A[ind, ind]

            # Also note that the value of profile is only set above if loc is not -1 above.
            if loc != -1:
                # AT_LINE 279 Write_Retrieval_Summary.pro
                result_quality = mpy.quality_deviation(
                    pressure, profile, constraint, ak_diag, species_name
                )
                self._deviation_QA[ispecie] = result_quality.deviation_QA
                self._num_deviations_QA[ispecie] = result_quality.num_deviations
                self._DeviationBad_QA[ispecie] = result_quality.deviationBad

    @property
    def cloudODAve(self) -> float:
        freq = self.current_state.state_spectral_domain_wavelength(
            StateElementIdentifier("CLOUDEXT")
        )
        if freq is None:
            raise RuntimeError("This shouldn't happen")
        ind = np.where((freq >= 974) & (freq <= 1201))[0]
        ceffect = self.state_value_vec("CLOUDEXT")
        if len(ind) > 0:
            res = np.sum(ceffect[ind]) / len(ceffect[ind]) * self.cloud_factor
        else:
            res = 0
        return res

    @property
    def cloudODVar(self) -> float:
        return self._cloudODVar

    @property
    def cloudODAveError(self) -> float:
        return self._cloudODAveError

    @property
    def emisDev(self) -> float:
        return self._emisDev

    @property
    def emissionLayer(self) -> float:
        return self._emissionLayer

    @property
    def ozoneCcurve(self) -> float:
        return self._ozoneCcurve

    @property
    def ozone_slope_QA(self) -> float:
        return self._ozone_slope_QA

    @property
    def deviation_QA(self) -> np.ndarray:
        return self._deviation_QA

    @property
    def num_deviations_QA(self) -> np.ndarray:
        return self._num_deviations_QA

    @property
    def DeviationBad_QA(self) -> np.ndarray:
        return self._DeviationBad_QA

    def state_value(self, state_name: str) -> float:
        return self.current_state.state_value(StateElementIdentifier(state_name))[0]

    def state_value_vec(self, state_name: str) -> np.ndarray:
        return self.current_state.state_value(StateElementIdentifier(state_name))

    @property
    def cloud_factor(self) -> float:
        scale_pressure = self.state_value("scalePressure")
        if scale_pressure == 0:
            scale_pressure = 0.1
        res = mpy.compute_cloud_factor(
            self.state_value_vec("pressure"),
            self.state_value_vec("TATM"),
            self.state_value_vec("H2O"),
            self.state_value("PCLOUD"),
            scale_pressure,
            self.current_state.sounding_metadata.surface_altitude.convert("m").value,
            self.current_state.sounding_metadata.latitude.value,
        )
        # TODO Rounding currently done. I', not sure this makes a lot of sense,
        # this was to match the old IDL code. I don't know that we actually want
        # to do that, but for now have this in place.
        res = round(res, 7)
        return res


__all__ = ["CloudResultSummary"]
