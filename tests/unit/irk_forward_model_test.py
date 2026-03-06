from refractor.muses import (
    InstrumentIdentifier,
    StateElementIdentifier,
    CrisFmObjectCreator,
    AirsFmObjectCreator,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import pprint
import subprocess


@require_muses_py_fm
def test_muses_cris_forward_model_irk(joint_tropomi_step_12):
    """We don't normally run IRK for CRIS (only for AIRS), but put this test
    in because it doesn't have the added complication that AIRS uses the
    TES frequencies. This tests the IRK code without including that extra
    piece (which we test separately"""
    rs, rstep, _ = joint_tropomi_step_12
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    obs_cris.spectral_window.include_bad_sample = True
    # For the purpose of testing, add some extra jacobians in so we can check
    # that part of the code.
    rs.current_state.testing_add_retrieval_state_element_id(
        StateElementIdentifier("TATM")
    )
    ocreator = CrisFmObjectCreator(rs.current_state, rs.retrieval_config, obs_cris)
    fm = ocreator.forward_model
    # Set up jacobian of state vector
    fm_sv = ocreator.fm_sv
    fm_sv.update_state(fm_sv.state)
    rirk = fm.irk(rs.current_state)

    fmcmp = MusesCrisForwardModel(rs.current_state, obs_cris, rs.retrieval_config)
    rirkcmp = fmcmp.irk(rs.current_state)
    with open("rirk.txt", "w") as fh:
        pprint.pprint(rirk, fh)
    with open("rirkcmp.txt", "w") as fh:
        pprint.pprint(rirkcmp, fh)
    subprocess.run(["diff", "-u", "rirk.txt", "rirkcmp.txt"], check=True)


@require_muses_py_fm
def test_muses_airs_forward_model_irk(airs_irk_step_6):
    rs, rstep, _ = airs_irk_step_6
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    obs_airs.spectral_window.include_bad_sample = True
    ocreator = AirsFmObjectCreator(rs.current_state, rs.retrieval_config, obs_airs)
    fm = ocreator.forward_model
    # Set up jacobian of state vector
    fm_sv = ocreator.fm_sv
    fm_sv.update_state(fm_sv.state)
    rirk = fm.irk(rs.current_state)

    fmcmp = MusesAirsForwardModel(rs.current_state, obs_airs, rs.retrieval_config)
    rirkcmp = fmcmp.irk(rs.current_state)
    with open("rirk.txt", "w") as fh:
        pprint.pprint(rirk, fh)
    with open("rirkcmp.txt", "w") as fh:
        pprint.pprint(rirkcmp, fh)
    import pickle

    rirk_py_retrieve = pickle.load(
        open("/home/smyth/Local/refractor-muses/py-retrieve_result_irk.pkl", "rb")
    )
    with open("rirk_py_retrieve.txt", "w") as fh:
        pprint.pprint(rirk_py_retrieve, fh)
    subprocess.run(["diff", "-u", "rirk.txt", "rirkcmp.txt"], check=True)
    # subprocess.run(["diff", "-u", "rirk.txt", "rirk_py_retrieve.txt"], check=True)
