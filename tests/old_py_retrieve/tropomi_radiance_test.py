import numpy as np
import numpy.testing as npt
from refractor.old_py_retrieve_wrapper import (
    TropomiRadiancePyRetrieve,
    TropomiRadianceRefractor,
)
import refractor.framework as rf
import pandas as pd
import pickle
import pytest
from pathlib import Path


@pytest.mark.old_py_retrieve_test
def test_tropomi_radiance(tropomi_uip_step_2):
    # The initial shift for everything is 0. Change to something so we can test that
    # this actually gets used.
    rf_uip = tropomi_uip_step_2
    rf_uip.tropomi_params["solarshift_BAND3"] = 0.01
    rf_uip.tropomi_params["radianceshift_BAND3"] = 0.02
    rf_uip.tropomi_params["radsqueeze_BAND3"] = 0.03
    rf_uip.uip_tropomi["jacobians"] = np.append(
        rf_uip.uip_tropomi["jacobians"], ("TROPOMIRADSQUEEZEBAND3")
    )
    mrad = TropomiRadiancePyRetrieve(rf_uip)
    sv = rf.StateVector()
    sv.add_observer(mrad)
    x = [
        rf_uip.tropomi_params["solarshift_BAND3"],
        rf_uip.tropomi_params["radianceshift_BAND3"],
        rf_uip.tropomi_params["radsqueeze_BAND3"],
    ]
    sv.update_state(x)
    r = mrad.radiance(0)
    fname = str(next((Path(rf_uip.run_dir) / "Input").glob("Radiance_TROPOMI*.pkl")))
    mrad2 = TropomiRadianceRefractor(
        rf_uip,
        [
            "BAND3",
        ],
        fname,
    )
    sv2 = rf.StateVector()
    sv2.add_observer(mrad2)
    x2 = np.array(
        [
            rf_uip.tropomi_params["solarshift_BAND3"],
            rf_uip.tropomi_params["radianceshift_BAND3"],
            rf_uip.tropomi_params["radsqueeze_BAND3"],
        ]
    )
    sv2.update_state(x2)
    r2 = mrad2.radiance(0)
    npt.assert_allclose(r.spectral_range.data, r2.spectral_range.data, atol=3e-4)
    npt.assert_allclose(
        r.spectral_range.uncertainty, r2.spectral_range.uncertainty, rtol=2e-5
    )
    # Expect jacobian to be different, because we are fixing this in
    # the refractor version.
    if False:
        print(r.spectral_range.data_ad.jacobian)
        print(r2.spectral_range.data_ad.jacobian)
    fdlist = [0.001, 0.001, 0.0001]
    y0 = r2.spectral_range.data
    for i, fd in enumerate(fdlist):
        xdelta = x2.copy()
        xdelta[i] += fd
        sv2.update_state(xdelta)
        r2delta = mrad2.radiance(0)
        yd = r2delta.spectral_range.data
        jfd = (yd - y0) / fd
        jcalc = r2.spectral_range.data_ad.jacobian[:, i]
        print(f"Find difference jacobian for index {i}")
        if False:
            print("   Jac finite difference: ", jfd)
            print("   Jac calculate: ", jcalc)
            print("   Jac relative difference: ", np.abs((jfd - jcalc) / jcalc))
        print("   Summary abs difference")
        print(pd.DataFrame(np.abs(jfd - jcalc)).describe())
        print("   Summary relative difference")
        print(pd.DataFrame(np.abs((jfd - jcalc) / jcalc)).describe())


@pytest.mark.old_py_retrieve_test
def test_bad_sample_tropomi_radiance(tropomi_uip_step_2):
    """Test bad sample handling in TropomiRadiance."""
    # Add some fake bad data
    rf_uip = tropomi_uip_step_2
    rdata = pickle.load(
        open(Path(rf_uip.run_dir) / "Input/Radiance_TROPOMI_.pkl", "rb")
    )
    rdata["Earth_Radiance"]["EarthRadianceNESR"][1::15] = -999
    pickle.dump(rdata, open(Path(rf_uip.run_dir) / "Input/Radiance_TROPOMI_.pkl", "wb"))
    mrad = TropomiRadiancePyRetrieve(rf_uip)
    fname = str(next((Path(rf_uip.run_dir) / "Input").glob("Radiance_TROPOMI*.pkl")))
    mrad2 = TropomiRadianceRefractor(
        rf_uip,
        [
            "BAND3",
        ],
        fname,
    )
    npt.assert_allclose(mrad.bad_sample_mask(0), mrad2.bad_sample_mask(0))
    npt.assert_allclose(mrad.spectral_domain(0).data, mrad2.spectral_domain(0).data)
    npt.assert_allclose(
        mrad.spectral_domain(0, inc_bad_sample=True).data,
        mrad2.spectral_domain(0, inc_bad_sample=True).data,
    )
    r1 = mrad.radiance_all()
    r2 = mrad2.radiance_all()
    npt.assert_allclose(r1.spectral_domain.data, r2.spectral_domain.data)
    npt.assert_allclose(r1.spectral_range.data, r2.spectral_range.data)
    npt.assert_allclose(r1.spectral_range.uncertainty, r2.spectral_range.uncertainty)
    r1 = mrad.radiance_all_with_bad_sample()
    r2 = mrad2.radiance_all_with_bad_sample()
    npt.assert_allclose(r1.spectral_domain.data, r2.spectral_domain.data)
    npt.assert_allclose(r1.spectral_range.data, r2.spectral_range.data)
    npt.assert_allclose(r1.spectral_range.uncertainty, r2.spectral_range.uncertainty)
