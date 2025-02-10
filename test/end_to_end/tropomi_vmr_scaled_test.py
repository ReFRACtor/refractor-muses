from __future__ import annotations
from functools import cached_property
import numpy as np
from refractor.tropomi import TropomiFmObjectCreator
import refractor.framework as rf
from refractor.muses import (
    CurrentState,
    CurrentStrategyStep,
    ForwardModelHandle,
    MeasurementId,
    MusesObservation,
    MusesRunDir,
    RefractorUip,
    RetrievableStateElement,
    RetrievalInfo,
    RetrievalConfiguration,
    RetrievalStrategy,
    RetrievalStrategyCaptureObserver,
    SingleSpeciesHandle,
    StateInfo,
    InstrumentIdentifier,
    StateElementIdentifier,
)
from typing import Callable
import subprocess
from loguru import logger
import pytest


class O3ScaledStateElement(RetrievableStateElement):
    """Note that we may rework this. Not sure how much we need specific
    StateElement vs. handling a class of them. But for now, we have
    the O3 scaled as a separate StateElement as we work out what exactly we
    want to do with new ReFRACtor only StateElement.

    We can use the SingleSpeciesHandle to add this in, e.g.,

    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("O3_SCALED", O3SCaledStateElement, pass_state=False))
    """

    def __init__(self, state_info: StateInfo, name=StateElementIdentifier("O3_SCALED")):
        super().__init__(state_info, name)
        self._value = np.array(
            [
                1.0,
            ]
        )
        self._constraint = self._value.copy()

    def sa_covariance(self):
        """Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs."""
        # TODO, Double check this. Not sure of the connection between this
        # and the constraintMatrix. Note the pressure is right, this
        # indicates we aren't on levels so we don't need a pressure
        return np.diag([10 * 10.0] * 1), [-999.0] * 1

    @property
    def value(self):
        return self._value

    def should_write_to_l2_product(self, instruments: list[InstrumentIdentifier]):
        if InstrumentIdentifier("TROPOMI") in instruments:
            return True
        return False

    def net_cdf_variable_name(self):
        # Want names like OMI_EOF_UV1
        return str(self.name)

    def net_cdf_struct_units(self):
        """Returns the attributes attached to a netCDF write out of this
        StateElement."""
        return {
            "Longname": "O3 VMR scale factor",
            "Units": "",
            "FillValue": "",
            "MisingValue": "",
        }

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        update_next: bool,
        retrieval_config: RetrievalConfiguration,
        step: int,
        do_update_fm: np.ndarray,
    ):
        # If we are requested not to update the next step, then save a copy
        # of this to reset the value
        if not update_next:
            self.state_info.next_state[self.name] = self.clone_for_other_state()
        self._value = results_list[retrieval_info.species_list == str(self._name)]

    def update_initial_guess(self, current_strategy_step: CurrentStrategyStep):
        self.mapType = "linear"
        self.pressureList = np.full((1,), -2.0)
        self.altitudeList = np.full((1,), -2.0)
        self.pressureListFM = self.pressureList
        self.altitudeListFM = self.altitudeList
        # Apriori
        self.constraintVector = self._constraint.copy()
        # Normally the same as apriori, but doesn't have to be
        self.initialGuessList = self.value.copy()
        self.trueParameterList = np.zeros((1))
        self.constraintVectorFM = self.constraintVector
        self.initialGuessListFM = self.initialGuessList
        self.trueParameterListFM = self.trueParameterList
        self.minimum = np.full((1), -999.0)
        self.maximum = np.full((1), -999.0)
        self.maximum_change = np.full((1), -999.0)
        self.mapToState = np.eye(1)
        self.mapToParameters = np.eye(1)
        # Not sure if the is covariance, or sqrt covariance. Note this
        # does not seem to the be the same as the Sa used in the error
        # analysis. I think muses-py uses the constraintMatrix sort of
        # like a weighting that is independent of apriori covariance.
        self.constraintMatrix = np.diag(np.full((1,), 10 * 10.0))


class ScaledTropomiFmObjectCreator(TropomiFmObjectCreator):
    @cached_property
    def absorber_vmr(self):
        vmrs = []
        # Get the VMR profile. This will remain at the initial guess
        vmr_profile, _ = self.current_state.object_state(
            [
                StateElementIdentifier("O3"),
            ]
        )
        # And get the scaling
        selem = [
            StateElementIdentifier("O3_SCALED"),
        ]
        coeff, mp = self.current_state.object_state(selem)
        vmr_o3 = rf.AbsorberVmrLevelScaled(
            self.pressure_fm, vmr_profile, coeff[0], "O3"
        )
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv,
            selem,
            [
                vmr_o3,
            ],
        )
        vmrs.append(vmr_o3)
        return vmrs


class ScaledTropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.measurement_id = None

    def notify_update_target(self, measurement_id: MeasurementId):
        """Clear any caching associated with assuming the target being retrieved is fixed"""
        self.measurement_id = measurement_id

    def forward_model(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState,
        obs: MusesObservation,
        fm_sv: rf.StateVector,
        rf_uip_func: Callable[[InstrumentIdentifier | None], RefractorUip] | None,
        **kwargs,
    ):
        if instrument_name != InstrumentIdentifier("TROPOMI"):
            return None
        obj_creator = ScaledTropomiFmObjectCreator(
            current_state,
            self.measurement_id,
            obs,
            rf_uip_func=rf_uip_func,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Scaled Tropomi Forward model\n{fm}")
        return fm


@pytest.mark.long_test
def test_tropomi_vrm_scaled(
    osp_dir, gmao_dir, vlidort_cli, end_to_end_run_dir, tropomi_test_in_dir
):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one."""
    dir = end_to_end_run_dir / "tropomi_vmr_scaled"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(tropomi_test_in_dir, osp_dir, gmao_dir, path_prefix=dir)
    # Modify the Table.asc to add a scaled element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    subprocess.run(
        f'sed -i -e "s/O3,/O3_SCALED,/" {str(r.run_dir / "Table.asc")}', shell=True
    )
    # For faster turn around time, set number of iterations to 1. We can test
    # everything, even though the final residual will be pretty high
    subprocess.run(f'sed -i -e "s/15/1 /" {str(r.run_dir / "Table.asc")}', shell=True)

    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    try:
        lognum = logger.add("tropomi_vmr_scaled/retrieve.log")
        # Save data so we can work on getting output in isolation
        rscap = RetrievalStrategyCaptureObserver(
            "retrieval_strategy_retrieval_step", "starting run_step"
        )
        rs.add_observer(rscap)
        ihandle = ScaledTropomiForwardModelHandle(
            use_pca=False, use_lrad=False, lrad_second_order=False
        )
        rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
        rs.state_element_handle_set.add_handle(
            SingleSpeciesHandle(
                StateElementIdentifier("O3_SCALED"),
                O3ScaledStateElement,
                pass_state=False,
                name=StateElementIdentifier("O3_SCALED"),
            )
        )
        rs.update_target(r.run_dir / "Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    if True:
        # The L2-O3 product doesn't get generated, since "O3-SCALED" isn't the "O3"
        # looked for in the code. Fixing this looks a bit involved, and we really should
        # just rework the output anyways. So for now just skip this.
        pass
        # Print out output of EOF, just so we have something to see
        # subprocess.run("h5dump -d OMI_EOF_UV1 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
        # subprocess.run("h5dump -d OMI_EOF_UV2 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
