import numpy.testing as npt
import refractor.framework as rf
from refractor.omi import OmiFmObjectCreator
from refractor.muses import InstrumentIdentifier


def test_solar_model(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.omi_solar_model[0])


def test_spec_win(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.spec_win)


def test_spectrum_sampling(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.spectrum_sampling)


def test_instrument(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.instrument)


def test_atmosphere(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.atmosphere)


def test_radiative_transfer(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.radiative_transfer)


def test_forward_model(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.forward_model)


class PrintSpectrum(rf.ObserverPtrNamedSpectrum):
    def notify_update(self, o):
        print("---------")
        print(o.name)
        print(o.spectral_domain.wavelength("nm"))
        print(o.spectral_range.data)
        print("---------")


class SaveSpectrum(rf.ObserverPtrNamedSpectrum):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def notify_update(self, o):
        import pickle

        fn = self.filename.format(name=o.name.replace(" ", "_"))
        with open(fn, "wb") as out:
            print(f"Saving {o.name} to {fn}")
            data = {
                "name": o.name,
                "wavelength": o.spectral_domain.wavelength("nm"),
                "radiance": o.spectral_range.data,
            }
            pickle.dump(data, out)


def test_fm_run(omi_fm_object_creator_step_0):
    fm = omi_fm_object_creator_step_0.forward_model
    fm.add_observer_and_keep_reference(PrintSpectrum())
    print(fm.radiance(1, True).value)


def test_state_vector(omi_fm_object_creator_step_0):
    omi_fm_object_creator_step_0.fm_sv.update_state(
        omi_fm_object_creator_step_0.current_state.initial_guess_full
    )
    print(omi_fm_object_creator_step_0.fm_sv)


def test_state_vector_step2(omi_fm_object_creator_step_1):
    omi_fm_object_creator_step_1.fm_sv.update_state(
        omi_fm_object_creator_step_1.current_state.initial_guess_full
    )
    print(omi_fm_object_creator_step_1.fm_sv)


def test_raman_effect(omi_fm_object_creator_step_0):
    print(omi_fm_object_creator_step_0.raman_effect)


def test_forward_model_step2(omi_fm_object_creator_step_1):
    """Step 2, which has two microwindows"""
    print(omi_fm_object_creator_step_1.forward_model)


def test_fm_run_step2(omi_fm_object_creator_step_1):
    omi_fm_object_creator_step_1.use_pca = False
    omi_fm_object_creator_step_1.use_lrad = False
    omi_fm_object_creator_step_1.lrad_second_order = False
    fm = omi_fm_object_creator_step_1.forward_model

    spectrum_lidort = fm.radiance(0, True)

    fm = OmiFmObjectCreator(
        omi_fm_object_creator_step_1.current_state,
        omi_fm_object_creator_step_1.measurement_id,
        omi_fm_object_creator_step_1.rs.observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            omi_fm_object_creator_step_1.current_state,
            omi_fm_object_creator_step_1.rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("OMI")
            ],
            None,
            osp_dir=omi_fm_object_creator_step_1.osp_dir,
        ),
        osp_dir=omi_fm_object_creator_step_1.osp_dir,
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
    ).forward_model

    spectrum_pca = fm.radiance(0, True)

    npt.assert_allclose(
        spectrum_lidort.spectral_domain.data,
        spectrum_pca.spectral_domain.data,
        rtol=1e-10,
    )
    npt.assert_allclose(
        spectrum_lidort.spectral_range.data, spectrum_pca.spectral_range.data, rtol=2e-2
    )
