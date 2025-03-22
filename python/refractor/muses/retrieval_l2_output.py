from __future__ import annotations
from loguru import logger
import refractor.muses.muses_py as mpy  # type: ignore
import os
import copy
from .retrieval_output import RetrievalOutput, CdfWriteTes
from .identifier import InstrumentIdentifier, ProcessLocation, StateElementIdentifier
from pathlib import Path
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .retrieval_strategy import RetrievalStrategy
    from .retrieval_strategy_step import RetrievalStrategyStep


def _new_from_init(cls, *args):
    """For use with pickle, covers common case where we just store the
    arguments needed to create an object."""
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst


class FileNumberHandle:
    """The RetrievalL2Output uses a numbering system to capture species from different
    steps. The number convention is based on the number of files left, so files might
    be names foo-2.nc, foo-1.nc, foo-0.nc (generated in that order). The original
    code would count the number of species left in the strategy table.

    The problem with this is that we don't want to assume that a strategy is fixed,
    we want to allow for changes in the strategy based on results or whatever criteria.
    So instead, we create the files within one name, foo-initial0.nc, foo-initial1.nc,
    foo-initial2.nc etc. Then, at the end of processing a sounding we go back and
    rename them."""

    def __init__(self, basename: Path):
        self.count = 0
        self.basename = basename

    def current_file_name(self) -> Path:
        return Path(str(self.basename) + f"-initial-{self.count}.nc")

    def next(self):
        self.count += 1

    def finalize(self):
        """Go back through and rename file to the final name"""
        for i in range(self.count + 1):
            f1 = Path(str(self.basename) + f"-initial-{i}.nc")
            f2 = Path(str(self.basename) + f"-{self.count - i}.nc")
            os.rename(f1, f2)


class RetrievalL2Output(RetrievalOutput):
    """Observer of RetrievalStrategy, outputs the Products_L2 files."""

    def __init__(self):
        self.dataTATM = None
        self.dataH2O = None
        self.dataN2O = None
        self.file_number_dict = {}

    def __reduce__(self):
        return (_new_from_init, (self.__class__,))

    @property
    def retrieval_info(self):
        return self.retrieval_strategy.retrieval_info

    def file_number_handle(self, basefname: Path) -> FileNumberHandle:
        """Return the FileNumberHandle for working the basefname. This handles numbering
        L2 output files if we have the same species in different strategy steps"""
        if basefname in self.file_number_dict:
            self.file_number_dict[basefname].next()
        else:
            self.file_number_dict[basefname] = FileNumberHandle(basefname)
        return self.file_number_dict[basefname]

    @property
    def species_list(self):
        """List of species, partially ordered so TATM comes before H2O, H2O before HDO,
        and N2O before CH4.

        The ordering is because TATM, H2O and N2O are used in making the lite files
        of CH4, HDO and H2O lite files, so we need to data from these before we get
        to the lite files."""
        if self._species_list is None:
            self._species_list = list(np.unique(self.species_list_fm))
            for spc in ("N2O", "H2O", "TATM"):
                if spc in self._species_list:
                    self._species_list.remove(spc)
                    self._species_list.insert(0, spc)
        return self._species_list

    def finalize_file_number(self):
        """Rename all the files that our FileNumberHandle is handling."""
        logger.debug("Finalizing file number is output file names")
        for fnum in self.file_number_dict.values():
            fnum.finalize()
        self.file_number_dict = {}

    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        retrieval_strategy_step: RetrievalStrategyStep | None = None,
        **kwargs,
    ):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        # Start of a retrieval
        if location == ProcessLocation("update target"):
            # Save these, used in later lite files. Note these actually get
            # saved between steps, so we initialize these for the first step but
            # then leave them alone
            self.dataTATM = None
            self.dataH2O = None
            self.dataN2O = None
            self.file_number_dict = {}
        if location == ProcessLocation("retrieval done"):
            self.finalize_file_number()
        if location != ProcessLocation("retrieval step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        # Regenerate this for the current step
        self._species_list = None
        for self.spcname in self.species_list:
            if (
                self.species_list_fm.count(self.spcname) <= 1
                or self.spcname in ("CLOUDEXT", "EMIS")
                or self.spcname.startswith("OMI")
                or self.spcname.startswith("NIR")
            ):
                continue
            self.out_fname = self.file_number_handle(
                self.output_directory / "Products" / f"Products_L2-{self.spcname}"
            ).current_file_name()
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Not sure about the logic here, but this is what script_retrieval_ms does
            if not os.path.exists(self.out_fname) or self.spcname in (
                "TATM",
                "H2O",
                "N2O",
            ):
                dataInfo = self.write_l2()
                if self.spcname == "TATM":
                    self.dataTATM = dataInfo
                elif self.spcname == "H2O":
                    self.dataH2O = dataInfo
                elif self.spcname == "N2O":
                    self.dataN2O = dataInfo
                self.lite_file(dataInfo)

    def lite_file(self, dataInfo):
        """Create lite file."""
        if self.spcname == "CH4":
            if self.dataN2O is not None:
                data2 = self.dataN2O
            else:
                # Fake the data
                logger.warning(
                    "code has not been tested for species_name CH4 and dataN2O is None"
                )
                data2 = copy.deepcopy(dataInfo)
                value = self.current_state.full_state_initial_value(
                    StateElementIdentifier("N2O")
                )
                data2["SPECIES"][data2["SPECIES"] > 0] = copy.deepcopy(value)
                data2["INITIAL"][data2["SPECIES"] > 0] = copy.deepcopy(value)
                data2["CONSTRAINTVECTOR"][data2["SPECIES"] > 0] = copy.deepcopy(value)
                data2["AVERAGINGKERNEL"].fill(0.0)
                data2["OBSERVATIONERRORCOVARIANCE"].fill(0.0)
        elif self.spcname == "HDO":
            data2 = self.dataH2O
        else:
            data2 = None

        state_element_out = []
        for sid in self.current_state.full_state_element_id:
            t = self.current_state.full_state_element(sid)
            if t.should_write_to_l2_product(self.instruments):
                state_element_out.append(t)

        if self.spcname == "H2O" and self.dataTATM is not None:
            self.out_fname = self.file_number_handle(
                self.output_directory / "Products" / "Lite_Products_L2-RH"
            ).current_file_name()
            if InstrumentIdentifier("OCO2") not in self.instruments:
                t = CdfWriteTes()
                t.write_lite(
                    self.step_number,
                    str(self.out_fname),
                    self.instruments,
                    str(self.lite_directory) + "/",
                    dataInfo,
                    self.dataTATM,
                    "RH",
                    state_element_out=state_element_out,
                )

        self.out_fname = self.file_number_handle(
            self.output_directory / "Products" / f"Lite_Products_L2-{self.spcname}"
        ).current_file_name()
        if InstrumentIdentifier("OCO2") not in self.instruments:
            t = CdfWriteTes()
            data2 = t.write_lite(
                self.step_number,
                str(self.out_fname),
                self.instruments,
                str(self.lite_directory) + "/",
                dataInfo,
                data2,
                self.spcname,
                state_element_out=state_element_out,
            )

    def generate_geo_data(self, species_data):
        """Generate the geo_data, pulled out just to keep write_l2 from getting
        too long."""
        nobs = 1
        geo_data = {
            "DAYNIGHTFLAG": None,
            "landFlag".upper(): np.zeros(shape=(nobs), dtype=np.int32) - 999,
            "LATITUDE": None,
            "LONGITUDE": None,
            "TIME": None,
            "surfaceTypeFootprint".upper(): np.zeros(shape=(nobs), dtype=np.int64)
            - 999,
            "SOUNDINGID": None,
        }

        geo_data = mpy.ObjectView(geo_data)
        smeta = self.sounding_metadata
        geo_data.TIME = np.int32(smeta.wrong_tai_time)
        geo_data.LATITUDE = smeta.latitude.convert("deg").value
        geo_data.LONGITUDE = smeta.longitude.convert("deg").value
        geo_data.SOUNDINGID = smeta.sounding_id
        geo_data.LANDFLAG = np.int32(0 if smeta.is_ocean else 1)
        geo_data.SURFACETYPEFOOTPRINT = np.int32(2 if smeta.is_ocean else 3)

        for i, inst in enumerate(self.instruments):
            geo_data.__dict__.update(self.obs_list[i].sounding_desc)

        # get surface type using hres database
        if self.results.is_ocean:
            geo_data.LANDFLAG = np.int32(0)
            geo_data.SURFACETYPEFOOTPRINT = np.int32(2)

            if np.amin(np.abs(smeta.height.convert("km").value)) > 0.1:
                geo_data.SURFACETYPEFOOTPRINT = 1

        hour = smeta.local_hour
        if hour >= 8 and hour <= 22:
            geo_data.DAYNIGHTFLAG = np.int16(1)
        elif hour <= 5 or hour >= 22:
            geo_data.DAYNIGHTFLAG = np.int16(0)
        else:
            geo_data.DAYNIGHTFLAG = np.int16(-999)

        # if sza defined, then use this for daynight, update 12/2017, DF, SSK
        if "omi_sza_uv2" in species_data:
            geo_data.DAYNIGHTFLAG = 0
            if species_data["OMI_SZA_UV2"] < 85:
                geo_data.DAYNIGHTFLAG = np.int16(1)

        geo_data = geo_data.__dict__

        for k, v in geo_data.items():
            if isinstance(v, list):
                geo_data[k] = np.asarray(v)
        return geo_data

    def write_l2(self):
        """Create L2 product file"""
        runtime_attributes = dict()

        # AT_LINE 7 write_products_one.pro
        # num_pressures varies based on surface pressure.  We set it to max here.
        num_pressures = 67
        nfreqEmis = 121

        if InstrumentIdentifier("OCO-2") in self.instruments:
            # TODO If we get sample data, we can put this back in
            logger.warning(
                "There is a block of code in the muses-py for reporting OCO-2 values. We don't have that code, because we don't have any test data for this. So skipping."
            )

        nfilter = len(self.results.filter_index) - 1

        species_data = {
            "SPECIES".upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            "PRIORCOVARIANCE".upper(): np.zeros(
                shape=(num_pressures, num_pressures), dtype=np.float32
            )
            - 999,
            "cloudTopPressureDOF".upper(): 0.0,
            "PRECISION".upper(): np.zeros(shape=(num_pressures), dtype=np.float32)
            - 999,
            "airDensity".upper(): np.zeros(shape=(num_pressures), dtype=np.float32)
            - 999,
            "altitude".upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            "AVERAGECLOUDEFFOPTICALDEPTH".upper(): 0.0,
            "AVERAGINGKERNEL".upper(): np.zeros(
                shape=(num_pressures, num_pressures), dtype=np.float32
            )
            - 999,
            "AVERAGINGKERNELDIAGONAL".upper(): np.zeros(
                shape=(num_pressures), dtype=np.float32
            )
            - 999,
            "CLOUDEFFECTIVEOPTICALDEPTH".upper(): np.zeros(shape=(28), dtype=np.float32)
            - 999,
            "CLOUDEFFECTIVEOPTICALDEPTHERROR".upper(): np.zeros(
                shape=(28), dtype=np.float32
            )
            - 999,
            "cloudTopPressure".upper(): 0.0,
            "cloudTopPressureError".upper(): 0.0,
            "CLOUDVARIABILITY_QA".upper(): 0.0,
            "CONSTRAINTVECTOR".upper(): np.zeros(
                shape=(num_pressures), dtype=np.float32
            )
            - 999,
            "DOFS".upper(): 0.0,
            "deviation_QA".upper(): -999,
            "num_deviations_QA".upper(): -999,
            "DeviationBad_QA".upper(): -999,
            "H2O_H2O_corr_QA".upper(): 0.0,
            "INITIAL".upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            "KDotDL_QA".upper(): 0.0,
            "KDotDLSys_QA".upper(): 0.0,
            "LDotDL_QA".upper(): 0.0,
            "MEASUREMENTERRORCOVARIANCE".upper(): np.zeros(
                shape=(num_pressures, num_pressures), dtype=np.float32
            )
            - 999,
            "OBSERVATIONERRORCOVARIANCE".upper(): np.zeros(
                shape=(num_pressures, num_pressures), dtype=np.float32
            )
            - 999,
            "PRESSURE".upper(): np.zeros(shape=(num_pressures), dtype=np.float32) - 999,
            "propagated_H2O_QA".upper(): 0,
            "propagated_O3_QA".upper(): 0,
            "propagated_TATM_QA".upper(): 0,
            "radianceResidualRMS".upper(): 0.0,
            "radianceResidualMean".upper(): 0.0,
            "radiance_residual_stdev_change".upper(): 0.0,
            "FILTER_INDEX": np.zeros(shape=(nfilter), dtype=np.int16),
            "radianceResidualRMS_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "radianceResidualMean_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "radianceResidualRMSRelativeContinuum_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "radiance_continuum_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "radianceSNR_filter".upper(): np.zeros(shape=(nfilter), dtype=np.float32),
            "radianceResidualSlope_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "radianceResidualQuadratic_filter".upper(): np.zeros(
                shape=(nfilter), dtype=np.float32
            ),
            "residualNormFinal".upper(): 0.0,
            "residualNormInitial".upper(): 0.0,
            "retrieveInLog".upper(): -999,
            "Quality".upper(): np.int16(-999),
            "Desert_Emiss_QA".upper(): 0.0,
            "EMISSIVITY_CONSTRAINT".upper(): np.zeros(
                shape=(nfreqEmis), dtype=np.float32
            )
            - 999,
            "EMISSIVITY_ERROR".upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32)
            - 999,
            "EMISSIVITY_INITIAL".upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32)
            - 999,
            "EMISSIVITY".upper(): np.zeros(shape=(nfreqEmis), dtype=np.float32) - 999,
            "emissivity_Wavenumber".upper(): np.zeros(
                shape=(nfreqEmis), dtype=np.float32
            )
            - 999,
            "SURFACEEMISSMEAN_QA".upper(): 0.0,
            "SURFACEEMISSIONLAYER_QA".upper(): 0.0,
            "SURFACETEMPCONSTRAINT".upper(): 0.0,
            "SURFACETEMPDEGREESOFFREEDOM".upper(): 0.0,
            "SURFACETEMPERROR".upper(): 0.0,
            "SURFACETEMPINITIAL".upper(): 0.0,
            "SURFACETEMPOBSERVATIONERROR".upper(): 0.0,
            "SURFACETEMPPRECISION".upper(): 0.0,
            "SURFACETEMPVSAPRIORI_QA".upper(): 0.0,
            "SURFACETEMPVSATMTEMP_QA".upper(): 0.0,
            "SURFACETEMPERATURE".upper(): 0.0,
            "column".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_air".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_DOFS".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_error".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_initial".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_PressureMax".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_PressureMin".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "column_Units".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "COLUMN_PRIOR".upper(): np.zeros(shape=(5), dtype=np.float32) - 999,
            "TOTALERROR".upper(): np.zeros(shape=(num_pressures), dtype=np.float32)
            - 999,
            "TOTALERRORCOVARIANCE".upper(): np.zeros(
                shape=(num_pressures, num_pressures), dtype=np.float32
            )
            - 999,
            "CLOUDFREQUENCY".upper(): np.zeros(shape=(28), dtype=np.float32) - 999,
            "tropopausePressure".upper(): 0.0,
            "lmresults_delta".upper(): self.results.LMResults_delta[
                self.results.bestIteration
            ],
        }

        if (
            self.current_state.full_state_spectral_domain_wavelength(
                StateElementIdentifier("emissivity")
            ).shape[0]
            == 0
        ):
            del species_data["EMISSIVITY_CONSTRAINT"]
            del species_data["EMISSIVITY_ERROR"]
            del species_data["EMISSIVITY_INITIAL"]
            del species_data["EMISSIVITY"]
            del species_data["EMISSIVITY_WAVENUMBER"]

        species_data = mpy.ObjectView(species_data)

        # AT_LINE 121 write_products_one.pro
        gpress = self.state_value("gmaoTropopausePressure")
        species_data.TROPOPAUSEPRESSURE = (
            gpress if gpress > 0 else self.results.tropopausePressure
        )

        # AT_LINE 126 write_products_one.pro
        species_data.DESERT_EMISS_QA = self.results.Desert_Emiss_QA
        species_data.PROPAGATED_TATM_QA = np.int32(self.results.propagatedTATMQA)
        species_data.PROPAGATED_O3_QA = np.int32(self.results.propagatedO3QA)
        species_data.PROPAGATED_H2O_QA = np.int32(self.results.propagatedH2OQA)
        species_data.RADIANCEMAXIMUMSNR = self.results.radianceMaximumSNR
        species_data.RESIDUALNORMFINAL = self.results.residualNormFinal
        species_data.RESIDUALNORMINITIAL = self.results.residualNormInitial

        smeta = self.sounding_metadata

        # Determine subset of the max num_pressures that we actually have
        # data for
        num_actual_pressures = self.current_state.full_state_value("TATM").shape[0]
        # And get the range of data we use to fill in our fields
        pslice = slice(num_pressures - num_actual_pressures, num_pressures)
        # get column / altitude / air density / trop column stuff
        altitudeResult, _ = mpy.compute_altitude_pge(
            self.state_value_vec("pressure"),
            self.state_value_vec("TATM"),
            self.state_value_vec("H2O"),
            smeta.surface_altitude.convert("m").value,
            smeta.latitude.convert("deg").value,
            i_waterType=None,
            i_pge=True,
        )

        altitudeResult = mpy.ObjectView(altitudeResult)
        species_data.AIRDENSITY[pslice] = (altitudeResult.airDensity * 1e6)[
            :
        ]  # convert molec/cm3 -> molec/m3

        species_data.ALTITUDE[pslice] = altitudeResult.altitude[:]

        # AT_LINE 169 write_products_one.pro
        if self.spcname == "O3":
            species_data.O3_CCURVE_QA = np.int32(self.results.ozoneCcurve)
            species_data.O3_SLOPE_QA = self.results.ozone_slope_QA

            species_data.O3_COLUMNERRORDU = self.results.O3_columnErrorDU
            species_data.O3_TROPO_CONSISTENCY_QA = self.results.O3_tropo_consistency

        if self.spcname in self.results.columnSpecies:
            indcol = self.results.columnSpecies.index(self.spcname)

            species_data.COLUMN = copy.deepcopy(self.results.column[:, indcol])
            species_data.COLUMN_AIR = copy.deepcopy(self.results.columnAir[:])
            species_data.COLUMN_DOFS = copy.deepcopy(self.results.columnDOFS[:, indcol])
            species_data.COLUMN_ERROR = copy.deepcopy(
                self.results.columnError[:, indcol]
            )
            species_data.COLUMN_INITIAL = copy.deepcopy(
                self.results.columnInitial[:, indcol]
            )
            species_data.COLUMN_PRESSUREMAX = copy.deepcopy(
                self.results.columnPressureMax[:]
            )
            species_data.COLUMN_PRESSUREMIN = copy.deepcopy(
                self.results.columnPressureMin[:]
            )
            species_data.COLUMN_PRIOR = copy.deepcopy(
                self.results.columnPrior[:, indcol]
            )
        species_data.RADIANCERESIDUALRMS_FILTER = self.results.radianceResidualRMS[1:]
        species_data.RADIANCERESIDUALMEAN_FILTER = self.results.radianceResidualMean[1:]
        species_data.radianceResidualRMSRelativeContinuum_FILTER = (
            self.results.radianceResidualRMSRelativeContinuum[1:]
        )
        species_data.RADIANCE_CONTINUUM_FILTER = self.results.radianceContinuum[1:]
        species_data.RADIANCESNR_FILTER = self.results.radianceSNR[1:]
        species_data.FILTER_INDEX = self.results.filter_index[1:]
        species_data.RADIANCERESIDUALSLOPE_FILTER = self.results.residualSlope[1:]
        species_data.RADIANCERESIDUALQUADRATIC_FILTER = self.results.residualQuadratic[
            1:
        ]
        species_data.RADIANCERESIDUALRMS = self.results.radianceResidualRMS[0]
        species_data.RADIANCERESIDUALMEAN = self.results.radianceResidualMean[0]
        species_data.RADIANCE_RESIDUAL_STDEV_CHANGE = (
            self.results.radianceResidualRMSInitial[0]
            - self.results.radianceResidualRMS[0]
        )

        if InstrumentIdentifier("OMI") in self.instruments:
            # Make all names uppercased to make life easier.
            obs = self.observation("OMI")
            # List of bands we have data for
            blist = [str(i[0]) for i in obs.filter_data]
            sza = obs.solar_zenith
            raz = obs.relative_azimuth
            vza = obs.observation_zenith
            sca = obs.scattering_angle
            d = species_data.__dict__
            for bout in ["UV1", "UV2"]:
                if bout in blist:
                    i = blist.index(bout)
                    d[f"OMI_SZA_{bout}"] = sza[i]
                    d[f"OMI_RAZ_{bout}"] = raz[i]
                    d[f"OMI_VZA_{bout}"] = vza[i]
                    d[f"OMI_SCA_{bout}"] = sca[i]
                    d[f"OMI_SURFACEALBEDO{bout}"] = self.state_value(
                        f"OMISURFACEALBEDO{bout}"
                    )
                    d[f"OMI_SURFACEALBEDO{bout}CONSTRAINTVECTOR"] = self.state_apriori(
                        f"OMISURFACEALBEDO{bout}"
                    )
                    if bout != "UV1":
                        # For who knows what reason this isn't present for UV1.
                        d[f"OMI_SURFACEALBEDOSLOPE{bout}"] = self.state_value(
                            f"OMISURFACEALBEDOSLOPE{bout}"
                        )
                        d[f"OMI_SURFACEALBEDOSLOPE{bout}CONSTRAINTVECTOR"] = (
                            self.state_apriori(f"OMISURFACEALBEDOSLOPE{bout}")
                        )
                    d[f"OMI_NRADWAV{bout}"] = self.state_value(f"OMINRADWAV{bout}")
                    d[f"OMI_ODWAV{bout}"] = self.state_value(f"OMIODWAV{bout}")
                    d[f"OMI_RINGSF{bout}"] = self.state_value(f"OMIRINGSF{bout}")
                else:
                    d[f"OMI_SZA_{bout}"] = 0.0
                    d[f"OMI_RAZ_{bout}"] = 0.0
                    d[f"OMI_VZA_{bout}"] = 0.0
                    d[f"OMI_SCA_{bout}"] = 0.0
                    d[f"OMI_SURFACEALBEDO{bout}"] = 0.0
                    d[f"OMI_SURFACEALBEDO{bout}CONSTRAINTVECTOR"] = 0.0
                    if bout != "UV1":
                        # For who knows what reason this isn't present for UV1.
                        d[f"OMI_SURFACEALBEDOSLOPE{bout}"] = 0.0
                        d[f"OMI_SURFACEALBEDOSLOPE{bout}CONSTRAINTVECTOR"] = 0.0
                    d[f"OMI_NRADWAV{bout}"] = 0.0
                    d[f"OMI_ODWAV{bout}"] = 0.0
                    d[f"OMI_RINGSF{bout}"] = 0.0
            species_data.OMI_CLOUDFRACTION = self.state_value("OMICLOUDFRACTION")
            species_data.OMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.state_apriori(
                "OMICLOUDFRACTION"
            )
            species_data.OMI_CLOUDTOPPRESSURE = obs.cloud_pressure.value
        if InstrumentIdentifier("TROPOMI") in self.instruments:
            # As with OMI, make all names uppercased to make life easier.
            # EM NOTE - This will have to be expanded if additional tropomi bands are used
            obs = self.observation("TROPOMI")
            # List of bands we have data for
            blist = [str(i[0]) for i in obs.filter_data]
            sza = obs.solar_zenith
            raz = obs.relative_azimuth
            vza = obs.observation_zenith
            sca = obs.scattering_angle
            d = species_data.__dict__
            for bout in ["BAND1", "BAND2", "BAND3"]:
                if bout in blist:
                    i = blist.index(bout)
                    d[f"TROPOMI_SZA_{bout}"] = sza[i]
                    d[f"TROPOMI_RAZ_{bout}"] = raz[i]
                    d[f"TROPOMI_VZA_{bout}"] = vza[i]
                    d[f"TROPOMI_SCA_{bout}"] = sca[i]
                    d[f"TROPOMI_SURFACEALBEDO{bout}"] = self.state_value(
                        f"TROPOMISURFACEALBEDO{bout}"
                    )
                    d[f"TROPOMI_SURFACEALBEDO{bout}CONSTRAINTVECTOR"] = (
                        self.state_apriori(f"TROPOMISURFACEALBEDO{bout}")
                    )
                    d[f"TROPOMI_SOLARSHIFT{bout}"] = self.state_value(
                        f"TROPOMISOLARSHIFT{bout}"
                    )
                    d[f"TROPOMI_RADIANCESHIFT{bout}"] = self.state_value(
                        f"TROPOMIRADIANCESHIFT{bout}"
                    )
                    d[f"TROPOMI_RADSQUEEZE{bout}"] = self.state_value(
                        f"TROPOMIRADSQUEEZE{bout}"
                    )
                    d[f"TROPOMI_RINGSF{bout}"] = self.state_value(
                        f"TROPOMIRINGSF{bout}"
                    )
                    if bout != "BAND1":
                        # For who knows what reason this isn't present for band 1.
                        d[f"TROPOMI_SURFACEALBEDOSLOPE{bout}"] = self.state_value(
                            f"TROPOMISURFACEALBEDOSLOPE{bout}"
                        )
                        d[f"TROPOMI_SURFACEALBEDOSLOPE{bout}CONSTRAINTVECTOR"] = (
                            self.state_apriori(f"TROPOMISURFACEALBEDOSLOPE{bout}")
                        )
                        d[f"TROPOMI_SURFACEALBEDOSLOPEORDER2{bout}"] = self.state_value(
                            f"TROPOMISURFACEALBEDOSLOPEORDER2{bout}"
                        )
                        d[f"TROPOMI_SURFACEALBEDOSLOPEORDER2{bout}CONSTRAINTVECTOR"] = (
                            self.state_apriori(f"TROPOMISURFACEALBEDOSLOPEORDER2{bout}")
                        )
                    if bout == "BAND3":
                        d[f"TROPOMI_TEMPSHIFT{bout}"] = self.state_value(
                            f"TROPOMITEMPSHIFT{bout}"
                        )

                else:
                    d[f"TROPOMI_SZA_{bout}"] = 0.0
                    d[f"TROPOMI_RAZ_{bout}"] = 0.0
                    d[f"TROPOMI_VZA_{bout}"] = 0.0
                    d[f"TROPOMI_SCA_{bout}"] = 0.0
                    d[f"TROPOMI_SURFACEALBEDO{bout}"] = 0.0
                    d[f"TROPOMI_SURFACEALBEDO{bout}CONSTRAINTVECTOR"] = 0.0
                    d[f"TROPOMI_SOLARSHIFT{bout}"] = 0.0
                    d[f"TROPOMI_RADIANCESHIFT{bout}"] = 0.0
                    d[f"TROPOMI_RADSQUEEZE{bout}"] = 0.0
                    d[f"TROPOMI_RINGSF{bout}"] = 0.0
                    if bout != "BAND1":
                        # For who knows what reason this isn't present for band 1.
                        d[f"TROPOMI_SURFACEALBEDOSLOPE{bout}"] = 0.0
                        d[f"TROPOMI_SURFACEALBEDOSLOPE{bout}CONSTRAINTVECTOR"] = 0.0
                        d[f"TROPOMI_SURFACEALBEDOSLOPEORDER2{bout}"] = 0.0
                        d[f"TROPOMI_SURFACEALBEDOSLOPEORDER2{bout}CONSTRAINTVECTOR"] = (
                            0.0
                        )
                    if bout == "BAND3":
                        d[f"TROPOMI_TEMPSHIFT{bout}"] = 0.0

            species_data.TROPOMI_CLOUDFRACTION = self.state_value(
                "TROPOMICLOUDFRACTION"
            )
            species_data.TROPOMI_CLOUDFRACTIONCONSTRAINTVECTOR = self.state_apriori(
                "TROPOMICLOUDFRACTION"
            )
            species_data.TROPOMI_CLOUDTOPPRESSURE = obs.cloud_pressure.value

        # species_data.TROPOMI_EOF1 = 1.0
        species_data.SPECIES[pslice] = self.retrieval_info.species_results(
            self.results, self.spcname
        )
        species_data.INITIAL[pslice] = self.retrieval_info.species_initial(self.spcname)
        species_data.CONSTRAINTVECTOR[pslice] = self.retrieval_info.species_constraint(
            self.spcname
        )
        species_data.PRESSURE[pslice] = self.state_value_vec("pressure")
        species_data.CLOUDTOPPRESSURE = self.state_value("PCLOUD")

        utilList = mpy.UtilList()
        indx = utilList.WhereEqualIndices(self.species_list_fm, "PCLOUD")
        if len(indx) > 0:
            indx = indx[0]
            species_data.CLOUDTOPPRESSUREDOF = self.results.A[indx, indx]
            species_data.CLOUDTOPPRESSUREERROR = self.results.errorFM[indx]

        # AT_LINE 288 write_products_one.pro
        species_data.AVERAGECLOUDEFFOPTICALDEPTH = np.nan_to_num(
            self.results.cloudODAve
        )
        species_data.CLOUDVARIABILITY_QA = np.int32(
            np.nan_to_num(self.results.cloudODVar)
        )
        species_data.H2O_H2O_CORR_QA = self.results.H2O_H2OQuality
        species_data.KDOTDL_QA = self.results.KDotDL
        species_data.KDOTDLSYS_QA = self.results.maxKDotDLSys
        species_data.LDOTDL_QA = self.results.LDotDL
        species_data.QUALITY = np.int16(self.results.masterQuality)
        species_data.SURFACEEMISSMEAN_QA = self.results.emisDev
        species_data.SURFACEEMISSIONLAYER_QA = self.results.emissionLayer
        species_data.SURFACETEMPVSAPRIORI_QA = self.results.tsur_minus_prior
        species_data.SURFACETEMPVSATMTEMP_QA = self.results.tsur_minus_tatm0

        # AT_LINE 300 write_products_one.pro
        species_data.SURFACETEMPERATURE = self.state_value("TSUR")
        unique_speciesListFM = utilList.GetUniqueValues(self.species_list_fm)

        indx = utilList.WhereEqualIndices(unique_speciesListFM, "TSUR")
        if len(indx) > 0:
            indx = indx[0]

            indxRet = utilList.WhereEqualIndices(
                self.retrieval_info.species_list, "TSUR"
            )[0]
            species_data.SURFACETEMPCONSTRAINT = self.retrieval_info.constraint_vector[
                indxRet
            ]

            # AT_LINE 306 src_ms-2018-12-10/write_products_one.pro
            indy = utilList.WhereEqualIndices(
                self.retrieval_info.species_names, "TSUR"
            )[0]
            species_data.SURFACETEMPDEGREESOFFREEDOM = (
                self.results.degreesOfFreedomForSignal[indy]
            )

            species_data.SURFACETEMPERROR = self.results.errorFM[indx]
            species_data.SURFACETEMPINITIAL = self.retrieval_info.initial_guess_list[
                indxRet
            ]

        # AT_LINE 324 write_products_one.pro
        ispecie = utilList.WhereEqualIndices(
            self.retrieval_info.species_names, self.spcname
        )
        ispecie = ispecie[
            0
        ]  # We just need one from the list so we can index into various variables.

        species_data.DEVIATION_QA = self.results.deviation_QA[ispecie]
        species_data.NUM_DEVIATIONS_QA = self.results.num_deviations_QA[ispecie]
        species_data.DEVIATIONBAD_QA = self.results.DeviationBad_QA[ispecie]

        ind1FM = self.retrieval_info.retrieval_info_obj.parameterStartFM[ispecie]
        ind2FM = self.retrieval_info.retrieval_info_obj.parameterEndFM[ispecie]

        if self.retrieval_info.retrieval_info_obj.mapType[ispecie].lower() == "linear":
            species_data.RETRIEVEINLOG = np.int32(0)
        elif (
            self.retrieval_info.retrieval_info_obj.mapType[ispecie].lower()
            == "linearpca"
        ):
            species_data.RETRIEVEINLOG = np.int32(0)
        else:
            species_data.RETRIEVEINLOG = np.int32(1)

        # AT_LINE 342 write_products_one.pro
        species_data.DOFS = np.sum(
            mpy.get_diagonal(self.results.A[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1])
        )
        species_data.PRECISION[pslice] = np.sqrt(
            mpy.get_diagonal(
                self.results.Sx_rand[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]
            )
        )

        # Build a 3D array so we can use it to access the below assignments.
        # third_index = np.asarray([0 for ii in range(1)])  # Set the 3rd index all to 0.
        # array_3d_indices = np.ix_(indConv, indConv, third_index)  # (64, 64, 1)

        # Generate an array of indices for the right hand side based on the slice information.
        # rhs_range_index = np.asarray([ii for ii in range(ind1FM, ind2FM + 1)])

        # Using slow method because fast method (using Python awesome list of locations as arrays for indices) is not working.
        species_data.AVERAGINGKERNEL[pslice, pslice] = self.results.A[
            ind1FM : ind2FM + 1, ind1FM : ind2FM + 1
        ]
        species_data.MEASUREMENTERRORCOVARIANCE[pslice, pslice] = self.results.Sx_rand[
            ind1FM : ind2FM + 1, ind1FM : ind2FM + 1
        ]
        species_data.TOTALERRORCOVARIANCE[pslice, pslice] = self.results.Sx[
            ind1FM : ind2FM + 1, ind1FM : ind2FM + 1
        ]

        # We pass in for rhs_start_index because sum_Sx_Sx_sys_Sx_crossState already contain the correct shape.
        sum_Sx_Sx_sys_Sx_crossState = (
            self.results.Sx_rand[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]
            + self.results.Sx_sys[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]
            + self.results.Sx_crossState[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]
        )

        species_data.OBSERVATIONERRORCOVARIANCE[pslice, pslice] = (
            sum_Sx_Sx_sys_Sx_crossState
        )

        #
        # Not sure if the right hand side indices are correct for Python.
        #
        #

        species_data.PRIORCOVARIANCE[pslice, pslice] = self.results.Sa[
            ind1FM : ind2FM + 1, ind1FM : ind2FM + 1
        ]
        species_data.AVERAGINGKERNELDIAGONAL[pslice] = mpy.get_diagonal(
            self.results.A[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1]
        )  ## utilGeneral.ManualArraySets(species_data.AVERAGINGKERNELDIAGONAL, get_diagonal(self.results.A[ind1FM:ind2FM+1, ind1FM:ind2FM+1]), indConv, rhs_start_index=0)
        species_data.TOTALERROR[pslice] = np.sqrt(
            mpy.get_diagonal(self.results.Sx[ind1FM : ind2FM + 1, ind1FM : ind2FM + 1])
        )

        # AT_LINE 355 write_products_one.pro
        if self.state_sd_wavelength("cloudEffExt").shape[0] > 0:
            species_data.CLOUDFREQUENCY = [
                600,
                650,
                700,
                750,
                800,
                850,
                900,
                950,
                975,
                100,
                1025,
                1050,
                1075,
                1100,
                1150,
                1200,
                1250,
                1300,
                1350,
                1400,
                1900,
                2000,
                2040,
                2060,
                2080,
                2100,
                2200,
                2250,
            ]

            # AT_LINE 365 src_ms-2018-12-10/write_products_one.pro
            factor = mpy.compute_cloud_factor(
                self.state_value_vec("pressure"),
                self.state_value_vec("TATM"),
                self.state_value_vec("H2O"),
                self.state_value("PCLOUD"),
                self.state_value("scalePressure"),
                self.current_state.sounding_metadata.surface_altitude.value * 1000,
                self.current_state.sounding_metadata.latitude.value,
            )

            convertToOD = factor

            # AT_LINE 363 write_products_one.pro
            # AT_LINE 374 src_ms-2018-12-10/write_products_one.pro
            species_data.CLOUDEFFECTIVEOPTICALDEPTH = (
                self.state_value_vec("cloudEffExt")[0, :] * convertToOD
            )

            indf = utilList.WhereEqualIndices(self.species_list_fm, "CLOUDEXT")
            if len(indf) > 0:
                species_data.CLOUDEFFECTIVEOPTICALDEPTHERROR = (
                    self.results.errorFM[indf] * convertToOD
                )
        # end if self.state_info.state_info_obj.cloudPars['num_frequencies'] > 0:

        # add special fields for HDO
        # AT_LINE 383 src_ms-2018-12-10/write_products_one.pro
        if self.spcname == "HDO":
            indfh = utilList.WhereEqualIndices(self.species_list_fm, "H2O")
            indfd = utilList.WhereEqualIndices(self.species_list_fm, "HDO")

            indp = np.where(species_data.SPECIES > 0)[0]

            matrix = species_data.AVERAGINGKERNEL * 0 - 999
            species_data.HDO_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.HDO_H2OAVERAGINGKERNEL[pslice, pslice] = self.results.A[
                indfh, indfd
            ]

            species_data.H2O_HDOAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_HDOAVERAGINGKERNEL[pslice, pslice] = self.results.A[
                indfd, indfh
            ]

            # AT_LINE 407 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OAVERAGINGKERNEL = copy.deepcopy(matrix)
            species_data.H2O_H2OAVERAGINGKERNEL[pslice, pslice] = self.results.A[
                indfh, indfh
            ]

            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OMEASUREMENTERRORCOVARIANCE[pslice, pslice] = (
                self.results.Sx_rand[indfh, indfd]
            )

            # AT_LINE 396 write_products_one.pro
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOMEASUREMENTERRORCOVARIANCE[pslice, pslice] = (
                self.results.Sx_rand[indfd, indfh]
            )

            # AT_LINE 417 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OMEASUREMENTERRORCOVARIANCE[pslice, pslice] = (
                self.results.Sx_rand[indfh, indfh]
            )

            # AT_LINE 400 write_products_one.pro
            error = (
                self.results.Sx_rand + self.results.Sx_crossState + self.results.Sx_sys
            )

            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[
                indfh, indfd
            ]

            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[
                indfd, indfh
            ]

            # AT_LINE 434 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OOBSERVATIONERRORCOVARIANCE[pslice, pslice] = error[
                indfh, indfh
            ]

            # AT_LINE 408 write_products_one.pro
            error = self.results.Sx

            species_data.HDO_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.HDO_H2OTOTALERRORCOVARIANCE[pslice, pslice] = error[
                indfh, indfd
            ]

            species_data.H2O_HDOTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_HDOTOTALERRORCOVARIANCE[pslice, pslice] = error[
                indfd, indfh
            ]

            # AT_LINE 445 src_ms-2018-12-10/write_products_one.pro
            species_data.H2O_H2OTOTALERRORCOVARIANCE = copy.deepcopy(matrix)
            species_data.H2O_H2OTOTALERRORCOVARIANCE[pslice, pslice] = error[
                indfh, indfh
            ]

            # AT_LINE 448 src_ms-2018-12-10/write_products_one.pro
            vector_of_fills = np.ndarray(shape=(num_pressures), dtype=np.float32)
            vector_of_fills.fill(-999.0)

            # add H2O constraint and result
            species_data.H2O_CONSTRAINTVECTOR = copy.deepcopy(vector_of_fills)
            species_data.H2O_CONSTRAINTVECTOR[indp] = (
                self.retrieval_info.species_constraint("H2O")
            )

            species_data.H2O_SPECIES = copy.deepcopy(vector_of_fills)
            species_data.H2O_SPECIES[indp] = self.retrieval_info.species_results(
                self.results, "H2O"
            )

            species_data.H2O_INITIAL = copy.deepcopy(vector_of_fills)

            # TODO
            # This looks wrong to me. Although this is marked initial, it is getting
            # this from the results. It is possible this is correct, perhaps this
            # the value used for the HDO? We should double check this. But this
            # is what the current muses-py code does
            species_data.H2O_INITIAL[indp] = self.retrieval_info.species_results(
                self.results, "H2O", INITIAL_Flag=True
            )
        # end if self.spcname == 'HDO':

        ind = utilList.WhereEqualIndices(self.species_list_fm, "EMIS")
        if len(ind) > 0:
            # Create an array of indices so we can access i_results.Sx matrix.
            array_2d_indices = np.ix_(ind, ind)  # (64, 64)
            species_data.EMISSIVITY_ERROR = np.sqrt(
                mpy.get_diagonal(self.results.Sx[array_2d_indices])
            )

        if self.state_sd_wavelength("emissivity").shape[0] > 0:
            species_data.EMISSIVITY_CONSTRAINT = self.state_apriori_vec("emissivity")
            species_data.EMISSIVITY_INITIAL = (
                self.current_state.full_state_initial_initial_value(
                    StateElementIdentifier("emissivity")
                )
            )
            species_data.EMISSIVITY = self.state_value_vec("emissivity")
            species_data.EMISSIVITY_WAVENUMBER = self.state_sd_wavelength("emissivity")

            # This test doesn't work, since nothing has accessed this yet. I believe this
            # is always in the state, get_state_initial.py in py-retrieve always fills this
            # in if emissivity is filled in. Comment out this test, we can return to
            # this if needed.
            # if StateElementIdentifier("native_emissivity") in self.current_state.full_state_element_id:
            if True:
                species_data.NATIVE_HSR_EMISSIVITY_INITIAL = (
                    self.current_state.full_state_initial_initial_value(
                        StateElementIdentifier("native_emissivity")
                    )
                )
                species_data.NATIVE_HSR_EMIS_WAVENUMBER = self.state_sd_wavelength(
                    "native_emissivity"
                )

            selem = self.current_state.full_state_element(
                StateElementIdentifier("emissivity")
            )
            species_data.EMISSIVITY_OFFSET_DISTANCE = np.array(
                [
                    selem.camel_distance,
                ]
            )
            runtime_attributes.setdefault("EMISSIVITY_INITIAL", dict())
            runtime_attributes["EMISSIVITY_INITIAL"]["database"] = selem.prior_source

        # AT_LINE 631 write_products_one.pro
        # for CH4 add in N2O results, constraint vector, calculate
        # n2o-corrected, save original_species
        if self.spcname == "CH4" and "CH4" in self.retrieval_info.species_names:
            species_data.N2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.N2O_CONSTRAINTVECTOR = np.zeros(
                shape=(num_pressures), dtype=np.float32
            )

            # AT_LINE 676 src_ms-2018-12-10/write_products_one.pro
            species_data.N2O_DOFS = 0.0

            ispecieN2O = -1
            if "N2O" in self.retrieval_info.species_names:
                ispecieN2O = self.retrieval_info.species_names.index("N2O")
                ind1FMN2O = self.retrieval_info.retrieval_info_obj.parameterStartFM[
                    ispecieN2O
                ]
                ind2FMN2O = self.retrieval_info.retrieval_info_obj.parameterEndFM[
                    ispecieN2O
                ]

            # AT_LINE 683 write_products_one.pro
            if ispecieN2O >= 0:
                # AT_LINE 642 write_products_one.pro
                species_data.N2O_DOFS = 0.0
                species_data.N2O_DOFS = np.sum(
                    mpy.get_diagonal(
                        self.results.A[
                            ind1FMN2O : ind2FMN2O + 1, ind1FMN2O : ind2FMN2O + 1
                        ]
                    )
                )

                species_data.N2O_SPECIES[pslice] = self.retrieval_info.species_results(
                    self.results, "N2O"
                )
                species_data.N2O_CONSTRAINTVECTOR[pslice] = (
                    self.retrieval_info.species_constraint("N2O")
                )
            else:
                # N2O not retrieved... use values from initial guess
                logger.warning("code has not been tested for N2O not retrieved.")
                species_data.N2O_SPECIES[pslice] = (
                    self.current_state.full_state_initial_value("N2O")
                )
                species_data.N2O_CONSTRAINTVECTOR[pslice] = (
                    self.current_state.full_state_initial_value("N2O")
                )

            # correct ch4 from n2o
            species_data.ORIGINAL_SPECIES = copy.deepcopy(species_data.SPECIES)

            # AT_LINE 649 write_products_one.pro
            # AT_LINE 699 src_ms-2018-12-10/write_products_one.pro
            n2o = species_data.N2O_SPECIES[pslice]
            n2o_xa = species_data.N2O_CONSTRAINTVECTOR[pslice]
            ch4 = species_data.SPECIES[pslice]

            species_data.SPECIES[pslice] = np.exp(
                np.log(ch4) + np.log(n2o_xa) - np.log(n2o)
            )

            # track ev's used
            # AT_LINE 657 write_products_one.pro
            species_data.CH4_EVS = np.zeros(shape=(10), dtype=np.float32)
            species_data.CH4_EVS[:] = self.results.ch4_evs[:]
        # end if self.spcname == 'CH4' and 'CH4' in self.retrieval_info.retrieval_info_obj.species:

        # for CH4 if jointly retrieved with TATM add TATM
        if self.spcname == "CH4" and "TATM" in self.retrieval_info.species_names:
            species_data.TATM_SPECIES = np.zeros(
                shape=(num_pressures), dtype=np.float32
            )
            species_data.TATM_CONSTRAINTVECTOR = np.zeros(
                shape=(num_pressures), dtype=np.float32
            )
            species_data.TATM_DEVIATION = 0.0

            species_data.TATM_SPECIES[pslice] = self.retrieval_info.species_results(
                self.results, "TATM"
            )
            species_data.TATM_CONSTRAINTVECTOR[pslice] = (
                self.retrieval_info.species_constraint("TATM")
            )

            # AT_LINE 725 src_ms-2018-12-10/write_products_one.pro

            # add in H2O results
            species_data.H2O_SPECIES = np.zeros(shape=(num_pressures), dtype=np.float32)
            species_data.H2O_CONSTRAINTVECTOR = np.zeros(
                shape=(num_pressures), dtype=np.float32
            )

            species_data.H2O_SPECIES[pslice] = self.retrieval_info.species_results(
                self.results, "H2O"
            )
            species_data.H2O_CONSTRAINTVECTOR[pslice] = (
                self.retrieval_info.species_constraint("H2O")
            )

            indp = np.where(species_data.TATM_SPECIES > 0)[0]
            maxx = np.amax(
                np.abs(
                    species_data.TATM_SPECIES[indp]
                    - species_data.TATM_CONSTRAINTVECTOR[indp]
                )
            )
            species_data.TATM_DEVIATION = maxx  # maximum deviation from prior
        # end if species_name == 'CH4' and 'TATM' in self.retrieval_info.retrieval_info_obj.species:

        species_data = species_data.__dict__
        for k, v in species_data.items():
            if isinstance(v, list):
                species_data[k] = np.asarray(v)

        state_element_out = []
        for sid in self.current_state.full_state_element_id:
            t = self.current_state.full_state_element(sid)
            if t.should_write_to_l2_product(self.instruments):
                state_element_out.append(t)

        #######
        # write with lite format using cdf_write_tes

        o_data = species_data
        o_data.update(self.generate_geo_data(species_data))
        t = CdfWriteTes()
        t.write(
            o_data,
            str(self.out_fname),
            runtimeAttributes=runtime_attributes,
            state_element_out=state_element_out,
        )

        return o_data


__all__ = [
    "RetrievalL2Output",
]
