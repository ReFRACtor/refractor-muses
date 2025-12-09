from __future__ import annotations
from refractor.muses import (
    StateElement,
    StateElementEmis,
    StateElementNativeEmis,
    StateElementCloudExt,
)
import numpy.testing as npt
import pickle
from pathlib import Path


def check_selem(selem: StateElement, fexpect: Path, save: bool = False) -> None:
    # We validated the results against the old state elements from muses-py.
    # Remove that so we don't depend on having muses-py available, but we want to
    # know if the value has changed indicating a possible problem.
    if save:
        pickle.dump(
            {
                "value_fm": selem.value_fm,
                "constraint_vector_fm": selem.constraint_vector_fm,
                "sd": selem.spectral_domain.data,
            },
            open(fexpect, "wb"),
        )
    expected = pickle.load(open(fexpect, "rb"))
    npt.assert_allclose(selem.constraint_vector_fm, expected["constraint_vector_fm"])
    npt.assert_allclose(selem.value_fm, expected["value_fm"])
    npt.assert_allclose(selem.spectral_domain.data, expected["sd"])


def test_state_element_emis(cris_tropomi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = cris_tropomi_shandle
    s = StateElementEmis.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_freq" / "emis_expect.pkl",
        save=False,
    )


def test_state_element_native_emis(cris_tropomi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = cris_tropomi_shandle
    s = StateElementNativeEmis.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_freq" / "native_emis_expect.pkl",
        save=False,
    )


def test_state_element_cloudext(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    s = StateElementCloudExt.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_freq" / "cloudext_expect.pkl",
        save=False,
    )
