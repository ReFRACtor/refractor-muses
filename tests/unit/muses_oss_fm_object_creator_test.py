from refractor.muses import (
    InstrumentIdentifier,
    StateElementIdentifier,
    CrisFmObjectCreator,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import pytest

@pytest.mark.skip
def test_airs_oss_init(ifile_hlp):
    # Do multiple times, to make sure we have the handling for this in place
    for i in range(10):
        oss_handle.oss_init(
            ifile_hlp,
            [
                StateElementIdentifier(i)
                for i in ["H2O", "O3", "TSUR", "CLOUDEXT", "PCLOUD"]
            ],
            [
                StateElementIdentifier(i)
                for i in [
                    "PRESSURE",
                    "TATM",
                    "H2O",
                    "CO2",
                    "O3",
                    "N2O",
                    "CO",
                    "CH4",
                    "SO2",
                    "NH3",
                    "HNO3",
                    "OCS",
                    "N2",
                    "HCN",
                    "SF6",
                    "HCOOH",
                    "CCL4",
                    "CFC11",
                    "CFC12",
                    "CFC22",
                    "HDO",
                    "CH3OH",
                    "C2H4",
                    "PAN",
                ]
            ],
            64,
            121,
            InstrumentIdentifier("AIRS"),
        )


@require_muses_py_fm
def test_muses_cris_forward_model_oss(joint_tropomi_step_12_no_run_dir):
    rs, rstep, _ = joint_tropomi_step_12_no_run_dir
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    obs_cris.spectral_window.include_bad_sample = True
    ocreator = CrisFmObjectCreator(rs.current_state, rs.retrieval_config, obs_cris)
    fm = ocreator.forward_model
    s = fm.radiance(0)
    #rad = s.spectral_range.data
    #jac = s.spectral_range.data_ad.jacobian
    fmcmp = MusesCrisForwardModel(rs.current_state, obs_cris, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian


@require_muses_py_fm
def test_muses_airs_forward_model_oss(joint_omi_step_8_no_run_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    obs_airs.spectral_window.include_bad_sample = True
    fmcmp = MusesAirsForwardModel(rs.current_state, obs_airs, rs.retrieval_config)
    scmp = fmcmp.radiance(0)
    radcmp = scmp.spectral_range.data
    jaccmp = scmp.spectral_range.data_ad.jacobian
