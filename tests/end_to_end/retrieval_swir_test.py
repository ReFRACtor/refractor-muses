from __future__ import annotations
from refractor.muses import (
    CurrentState,
    CurrentStrategyStep,
    ForwardModelHandle,
    MeasurementId,
    MusesObservation,
    MusesRunDir,
    RetrievalConfiguration,
    RetrievalStrategy,
    SimulatedObservation,
    SimulatedObservationHandle,
    InstrumentIdentifier,
    StateElementIdentifier,
    ProcessLocation,
    modify_strategy_table,
)
from refractor.old_py_retrieve_wrapper import (
    RetrievableStateElementOld,
    StateInfoOld,
    SingleSpeciesHandleOld,
    RetrievalInfo,
)
from refractor.tropomi import TropomiSwirForwardModelHandle, TropomiSwirFmObjectCreator
import refractor.framework as rf
from functools import cached_property
import subprocess
import copy
import os
import pickle
from loguru import logger
import numpy as np
import pytest


# This actually runs ok, but it fails with a LIDORT error when one of the steps goes out of
# range. Not really an error so much as we just need to work out what the strategy is and
# possibly pick a different sounding. But skip for now so we don't have a failing unit test
@pytest.mark.skip
@pytest.mark.long_test
def test_retrieval(tropomi_swir, ifile_hlp):
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # Grab each step so we can separately test output
    # rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    # rs.add_observer(rscap)
    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_swir.run_dir}/Table.asc")
    rs.retrieval_ms()


class CostFunctionCapture:
    """Grab cost function, and then raise a exception to break out of retrieval."""

    def __init__(self):
        self.location_to_capture = ProcessLocation("create_cost_function")

    def notify_update(
        self, retrieval_strategy, location, retrieval_strategy_step=None, **kwargs
    ):
        if location != self.location_to_capture:
            return
        self.cost_function = retrieval_strategy_step.cfunc
        raise StopIteration()


class PrintSpectrum(rf.ObserverPtrNamedSpectrum):
    def __init__(self):
        super().__init__()
        self.data = {}

    def notify_update(self, o):
        if o.name not in self.data:
            self.data[o.name] = []
        self.data[o.name].append(
            [o.spectral_domain.wavelength("nm"), o.spectral_range.data]
        )
        print("---------")
        print(o.name)
        print(o.spectral_domain.wavelength("nm"))
        print(o.spectral_range.data)
        print("---------")


# Look just at the forward model
@pytest.mark.long_test
def test_co_fm(tropomi_swir, ifile_hlp):
    """Look just at the forward model"""
    # This is slightly convoluted, but we want to make sure we have the cost
    # function/ForwardModel that is being used in the retrieval. So we
    # start running the retrieval, and then stop when have the cost function.
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # Just retrieve CO
    modify_strategy_table(
        rs,
        0,
        [
            StateElementIdentifier("CO"),
        ],
    )
    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
        # absorption_gases=["CO",]
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(tropomi_swir.run_dir / "Table.asc")
    cfcap = CostFunctionCapture()
    rs.add_observer(cfcap)
    try:
        rs.retrieval_ms()
    except StopIteration:
        pass
    cfunc = cfcap.cost_function
    # Save in case we want to access directly
    pickle.dump(cfunc, open("cfunc.pkl", "wb"))
    fm = cfunc.fm_list[0]
    pspec = PrintSpectrum()
    fm.underlying_forward_model.add_observer(pspec)
    # spec = fm.radiance_all()
    p = cfunc.parameters
    print("p: ", p)
    atmosphere = fm.underlying_forward_model.radiative_transfer.lidort.atmosphere
    absorber = atmosphere.absorber
    # VMR values, we'll want to make sure this actually changes
    vmr_val = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    # residual = copy.copy(cfunc.residual)
    print("vmr_val: ", vmr_val)
    coeff = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff: ", coeff)
    hresgrid_wn = fm.underlying_forward_model.spectral_grid.high_resolution_grid(
        0
    ).wavenumber()
    od = np.vstack(
        [
            np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis, :, :]
            for wn in hresgrid_wn
        ]
    )
    tod = np.vstack(
        [
            atmosphere.optical_properties(wn, 0).total_optical_depth().value
            for wn in hresgrid_wn
        ]
    )

    # After the first step in levmar_nllsq, this is the parameter. We just got
    # this by setting a breakpoint in levmar_nllsq and looking. But this is enough
    # for us to try to figure out what is going on
    p2 = [
        -15.74241805,
        -16.20146017,
        -16.16204205,
        -16.18824768,
        -16.24048363,
        -16.3996463,
        -16.59970071,
        -16.79074649,
        -16.9677597,
        -17.49288014,
        -17.79474739,
        -17.46974549,
        -17.36741773,
    ]
    cfunc.parameters = p2
    # residual2 = copy.copy(cfunc.residual)
    print("p-p2: ", p - p2)
    vmr_val2 = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    # Very small change
    print("vmr-vmr_val2: ", vmr_val - vmr_val2)
    coeff2 = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff2-coeff: ", coeff2 - coeff)
    od2 = np.vstack(
        [
            np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis, :, :]
            for wn in hresgrid_wn
        ]
    )
    print("od-od2: ", np.abs(od - od2)[:, :, 1].max())

    # Make a big change
    p3 = np.array(p2) * 0.75
    cfunc.parameters = p3
    coeff3 = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff3 - coeff2: ", coeff3 - coeff2)
    vmr_val3 = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    print("vmr_val3 - vmr_val2: ", vmr_val3 - vmr_val2)
    od3 = np.vstack(
        [
            np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis, :, :]
            for wn in hresgrid_wn
        ]
    )
    tod3 = np.vstack(
        [
            atmosphere.optical_properties(wn, 0).total_optical_depth().value
            for wn in hresgrid_wn
        ]
    )
    print("od3-od2: ", np.abs(od3 - od2)[:, :, 1].max())
    # residual3 = copy.copy(cfunc.residual)
    # print("residual3-residual: ", residual3-residual)
    print("tod3-tod: ", np.abs(tod3 - tod).max())

    # This is the portion that isn't the parameter constraint
    # print((residual2-residual)[:112])


@pytest.mark.long_test
@pytest.mark.parametrize(
    "use_oss,oss_training_data",
    [(True, "../OSS_file_all_1243_0_1737006075.1163344.npz"), (False, None)],
)
def test_simulated_retrieval(
    ifile_hlp,
    end_to_end_run_dir,
    tropomi_band7_test_in_dir2,
    use_oss,
    oss_training_data,
):
    """Do a simulation, and then a retrieval to get this result"""
    test_dir = end_to_end_run_dir / f"swir_simulation{'_oss' if use_oss else ''}"
    subprocess.run(["rm", "-r", str(test_dir)])
    mrdir = MusesRunDir(
        tropomi_band7_test_in_dir2,
        ifile_hlp,
        path_prefix=test_dir,
    )
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # Just retrieve CO
    modify_strategy_table(
        rs,
        0,
        [
            StateElementIdentifier("CO"),
        ],
    )
    if oss_training_data is not None:
        oss_training_data = tropomi_band7_test_in_dir2 / oss_training_data
        # oss_training_data = "/home/mthill/muses/py-retrieve/mthill/refractor_example_notebooks/OSS_file_all_1243_2486_2_1737006088.359634.npz"

    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
        use_oss=use_oss,
        oss_training_data=oss_training_data,
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(mrdir.run_dir / "Table.asc")
    os.chdir(mrdir.run_dir)

    # Do all the setup etc., but stop the retrieval at step 0 (i.e., before we
    # do the first retrieval step). We then grab the CostFunction for that step,
    # which we can use for simulation purposes.
    rs.strategy_executor.execute_retrieval(stop_at_step=0)
    cfunc = rs.strategy_executor.create_cost_function()
    pickle.dump(cfunc, open(test_dir / "cfunc_initial_guess.pkl", "wb"))

    # Get the log vmr values set in the state vector. This is the initial guess.
    # For purposes of a simulation, we will say the "right" answer is to reduce the
    # VMR by 25%. So calculate the "true" log vmr and update the cost function with
    # this set of parameters.
    vmr_log_initial = cfunc.parameters
    vmr_initial = np.exp(vmr_log_initial)
    vmr_true = 0.75 * vmr_initial
    vmr_log_true = np.log(vmr_true)
    cfunc.parameters = vmr_log_true

    # Run forward model and get "true" radiance.
    rad_true = [
        cfunc.fm_list[0].radiance(0, True).spectral_range.data,
    ]
    obs_sim = SimulatedObservation(cfunc.obs_list[0], rad_true)
    pickle.dump(obs_sim, open(test_dir / "obs_sim.pkl", "wb"))

    # Have simulated observation, and do retrieval
    ohandle = SimulatedObservationHandle(
        StateElementIdentifier("TROPOMI"),
        pickle.load(open(test_dir / "obs_sim.pkl", "rb")),
    )
    rs.observation_handle_set.add_handle(ohandle, priority_order=100)
    rs.update_target(mrdir.run_dir / "Table.asc")
    rs.retrieval_ms()


@pytest.mark.parametrize(
    "use_oss,oss_training_data",
    [(True, "../OSS_file_all_1243_0_1737006075.1163344.npz"), (False, None)],
)
def test_radiance(
    ifile_hlp,
    tmpdir,
    tropomi_band7_test_in_dir2,
    use_oss,
    oss_training_data,
):
    """Do a simulation, and then a retrieval to get this result"""
    mrdir = MusesRunDir(
        tropomi_band7_test_in_dir2,
        ifile_hlp,
        path_prefix=tmpdir,
    )
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # Just retrieve CO
    modify_strategy_table(
        rs,
        0,
        [
            StateElementIdentifier("CO"),
        ],
    )
    if oss_training_data is not None:
        oss_training_data = tropomi_band7_test_in_dir2 / oss_training_data

    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
        use_oss=use_oss,
        oss_training_data=oss_training_data,
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(mrdir.run_dir / "Table.asc")
    os.chdir(mrdir.run_dir)

    # Do all the setup etc., but stop the retrieval at step 0 (i.e., before we
    # do the first retrieval step). We then grab the CostFunction for that step,
    # which we can use for simulation purposes.
    rs.strategy_executor.execute_retrieval(stop_at_step=0)
    cfunc = rs.strategy_executor.create_cost_function()
    cstate = rs.current_state
    # Print out a description of the full state, so we can look at the problem
    # with the albedo
    print(cstate.state_desc())
    print([str(sid) for sid in cstate.forward_model_state_vector_element_list])

    # Run forward model
    rad_spectrum = cfunc.fm_list[0].radiance(0, True)
    pickle.dump(rad_spectrum, open(tmpdir / f"radiance_oss_{use_oss}.pkl", "wb"))


@pytest.mark.long_test
def test_sim_albedo_0_9_retrieval(
    gmao_dir,
    ifile_hlp,
    python_fp_logger,
    end_to_end_run_dir,
    tropomi_band7_sim_alb_dir,
):
    """Use simulated data Josh generated"""
    dir = end_to_end_run_dir / "synth_alb_0_9"
    subprocess.run(["rm", "-r", str(dir)])
    mrdir = MusesRunDir(
        tropomi_band7_sim_alb_dir,
        ifile_hlp=ifile_hlp,
        path_prefix=dir,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
        ihandle = TropomiSwirForwardModelHandle(
            use_pca=True, use_lrad=False, lrad_second_order=False
        )
        rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
        rs.update_target(mrdir.run_dir / "Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)


class ScaledStateElement(RetrievableStateElementOld):
    """Note that we may rework this. Not sure how much we need specific
    StateElement vs. handling a class of them.
    We can use the SingleSpeciesHandle to add this in, e.g.,

    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("H2O_SCALED", ScaledStateElement, pass_state=False, name="H2O_SCALED"))
    """

    def __init__(self, state_info: StateInfoOld, name=None):
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

    @property
    def apriori_value(self) -> np.ndarray:
        return np.array(
            [
                1.0,
            ]
        )

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

    def update_state(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        initial_initial: np.ndarray | None = None,
        true: np.ndarray | None = None,
    ) -> None:
        """We have a few places where we want to update a state element other than
        update_initial_guess. This function updates each of the various values passed in.
        A value of 'None' (the default) means skip updating that part of the state."""
        raise NotImplementedError

    def update_state_element(
        self,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
        retrieval_config: RetrievalConfiguration,
        step: int,
        do_update_fm: np.ndarray,
    ):
        self._value = results_list[
            np.array(retrieval_info.species_list) == str(self._name)
        ]

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


class ScaledTropomiFmObjectCreator(TropomiSwirFmObjectCreator):
    @cached_property
    def absorber_vmr(self):
        vmrs = []
        for gas in self.absorption_gases:
            selem = [
                StateElementIdentifier(gas),
            ]
            coeff, mp = self.current_state.object_state(selem)
            # Need to get mp to be the log mapping in current_state, but for
            # now just work around this
            mp = rf.StateMappingLog()
            # Scaled retrievals other than CO
            if gas == "CO":
                vmr = rf.AbsorberVmrLevel(self.pressure_fm, coeff, gas, mp)
            else:
                selem = [
                    StateElementIdentifier(f"{gas}_SCALED"),
                ]
                coeff2, _ = self.current_state.object_state(selem)
                vmr = rf.AbsorberVmrLevelScaled(self.pressure_fm, coeff, coeff2[0], gas)
            self.current_state.add_fm_state_vector_if_needed(
                self.fm_sv,
                selem,
                [
                    vmr,
                ],
            )
            vmrs.append(vmr)
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
        **kwargs,
    ):
        if instrument_name != InstrumentIdentifier("TROPOMI"):
            return None
        obj_creator = ScaledTropomiFmObjectCreator(
            current_state,
            self.measurement_id,
            obs,
            fm_sv=fm_sv,
            **self.creator_kwargs,
        )
        fm = obj_creator.forward_model
        logger.info(f"Scaled Tropomi Forward model\n{fm}")
        return fm


# Currently doesn't work, since we have reworked the StateElement stuff.
# We'll fix this, but wait until we have everything completed in the StateElement to come
# back to this
@pytest.mark.skip
@pytest.mark.long_test
def test_scaled_sim_albedo_0_9_retrieval(
    ifile_hlp,
    python_fp_logger,
    end_to_end_run_dir,
    tropomi_band7_sim_alb_dir,
):
    """Use simulated data Josh generated"""
    dir = end_to_end_run_dir / "synth_alb_0_9_scaled"
    subprocess.run(["rm", "-r", str(dir)])
    mrdir = MusesRunDir(
        tropomi_band7_sim_alb_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # Change table to use scaled versions
    modify_strategy_table(
        rs,
        0,
        [
            StateElementIdentifier("CO"),
            StateElementIdentifier("CH4_SCALED"),
            StateElementIdentifier("H2O_SCALED"),
            StateElementIdentifier("HDO_SCALED"),
            StateElementIdentifier("TROPOMISOLARSHIFTBAND7"),
            StateElementIdentifier("TROPOMIRADIANCESHIFTBAND7"),
            StateElementIdentifier("TROPOMISURFACEALBEDOBAND7"),
            StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEBAND7"),
            StateElementIdentifier("TROPOMISURFACEALBEDOSLOPEORDER2BAND7"),
        ],
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        ihandle = ScaledTropomiForwardModelHandle(
            use_pca=True, use_lrad=False, lrad_second_order=False
        )
        rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
        rs.state_element_handle_set.add_handle(
            SingleSpeciesHandleOld(
                "H2O_SCALED",
                ScaledStateElement,
                pass_state=False,
                name=StateElementIdentifier("H2O_SCALED"),
            )
        )
        rs.state_element_handle_set.add_handle(
            SingleSpeciesHandleOld(
                "CH4_SCALED",
                ScaledStateElement,
                pass_state=False,
                name=StateElementIdentifier("CH4_SCALED"),
            )
        )
        rs.state_element_handle_set.add_handle(
            SingleSpeciesHandleOld(
                "HDO_SCALED",
                ScaledStateElement,
                pass_state=False,
                name=StateElementIdentifier("HDO_SCALED"),
            )
        )
        rs.update_target(mrdir.run_dir / "Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
