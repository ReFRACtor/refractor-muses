from refractor.muses import (
    OspCovarianceMatrixReader,
    StateElementIdentifier,
    OspSpeciesReader,
    OspL2SetupControlInitial,
    RetrievalType,
)
import numpy as np
import numpy.testing as npt


def test_covariance(osp_dir):
    latitude = 30.0
    map_type = "linear"
    r = OspCovarianceMatrixReader.read_dir(osp_dir / "Covariance" / "Covariance")
    cov_matrix = r.read_cov(StateElementIdentifier("OMIODWAVUV1"), map_type, latitude)
    npt.assert_allclose(cov_matrix.original_cov, np.array([[0.0004]]))
    pressure = np.array(
        [
            1.0067983e03,
            1.0000000e03,
            9.0851400e02,
            8.2540200e02,
            7.4989300e02,
            6.8129100e02,
            6.1896600e02,
            5.6234200e02,
            5.1089800e02,
            4.6416000e02,
            4.2169800e02,
            3.8311700e02,
            3.4806900e02,
            3.1622700e02,
            2.8729800e02,
            2.6101600e02,
            2.3713700e02,
            2.1544400e02,
            1.9573500e02,
            1.7782900e02,
            1.6156100e02,
            1.4677900e02,
            1.3335200e02,
            1.2115200e02,
            1.1006900e02,
            1.0000000e02,
            9.0851800e01,
            8.2540600e01,
            7.4989600e01,
            6.8129500e01,
            6.1896300e01,
            5.6233900e01,
            5.1089600e01,
            4.6415800e01,
            4.2169600e01,
            3.8311900e01,
            3.4807100e01,
            3.1622900e01,
            2.8729900e01,
            2.6101700e01,
            2.3713600e01,
            2.1544300e01,
            1.9573400e01,
            1.7782800e01,
            1.6156000e01,
            1.4678000e01,
            1.3335200e01,
            1.2115300e01,
            1.1007000e01,
            1.0000000e01,
            9.0851400e00,
            8.2540200e00,
            6.8129100e00,
            5.1089800e00,
            4.6416000e00,
            3.1622700e00,
            2.6101600e00,
            2.1544300e00,
            1.6156000e00,
            1.3335200e00,
            1.0000000e00,
            6.8129200e-01,
            3.8311800e-01,
            2.1544300e-01,
            1.0000000e-01,
        ]
    )
    cov_matrix = r.read_cov(StateElementIdentifier("TATM"), map_type, latitude)
    d = cov_matrix.interpolated_covariance(pressure)
    assert d.shape == (65, 65)
    cov_matrix = r.read_cov(StateElementIdentifier("NH3"), map_type, latitude, "CLN")
    cov_matrix = r.read_cov(StateElementIdentifier("NH3"), map_type, latitude, "ENH")


def test_species(osp_dir):
    r = OspSpeciesReader.read_dir(
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    t = r.read_file(
        StateElementIdentifier("OMIODWAVUV1"), RetrievalType("a_retrieval_type")
    )
    t2 = r.read_file(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("a_retrieval_type")
    )
    t3 = r.read_file(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("omicloud_ig_refine")
    )
    assert t["mapType"].lower() == "linear"
    assert t2["mapType"].lower() == "linear"
    assert t3["mapType"].lower() == "linear"
    cmatrix = r.read_constraint_matrix(
        StateElementIdentifier("OMIODWAVUV1"), RetrievalType("a_retrieval_type"), 1
    )
    npt.assert_allclose(cmatrix, [[2500.0]])
    # Repeat, to make sure caching works
    cmatrix = r.read_constraint_matrix(
        StateElementIdentifier("OMIODWAVUV1"), RetrievalType("a_retrieval_type"), 1
    )
    npt.assert_allclose(cmatrix, [[2500.0]])
    cmatrix2 = r.read_constraint_matrix(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("a_retrieval_type"), 1
    )
    cmatrix3 = r.read_constraint_matrix(
        StateElementIdentifier("OMICLOUDFRACTION"),
        RetrievalType("omicloud_ig_refine"),
        1,
    )
    npt.assert_allclose(cmatrix2, [[400.0]])
    npt.assert_allclose(cmatrix3, [[4.0]])


def test_osp_l2_setup_control_initial(osp_dir):
    f = OspL2SetupControlInitial.read(
        osp_dir / "L2_Setup" / "ops" / "L2_Setup_ms-CrIS-TROPOMI-CAMEL"
    )
    assert f["Single_State_Directory"] == osp_dir / "L2_Setup" / "ops" / "L2_Setup"
    assert f.sid_to_type[StateElementIdentifier("PCLOUD")] == "Single"
    assert f.sid_to_type[StateElementIdentifier("TATM")] == "GMAO"


def test_species_premade(osp_dir):
    r = OspSpeciesReader.read_dir(
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    cov = r.read_constraint_matrix(
        StateElementIdentifier("TATM"), RetrievalType("default"), 30
    )
    assert cov.shape == (30, 30)


def test_species_h2o_hdo(osp_dir):
    r = OspSpeciesReader.read_dir(
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    # Test handling cross terms when we have both H2O and HDO.
    cov = r.read_constraint_matrix(
        StateElementIdentifier("H2O"),
        RetrievalType("default"),
        16,
        sid2=StateElementIdentifier("H2O"),
    )
    cov2 = r.read_constraint_matrix(
        StateElementIdentifier("HDO"),
        RetrievalType("default"),
        16,
        sid2=StateElementIdentifier("HDO"),
    )
    cov3 = r.read_constraint_matrix(
        StateElementIdentifier("H2O"),
        RetrievalType("default"),
        16,
        sid2=StateElementIdentifier("HDO"),
    )
    assert cov.shape == (16, 16)
    assert cov2.shape == (16, 16)
    assert cov3.shape == (16, 16)


def test_species_covariance(osp_dir):
    r = OspSpeciesReader.read_dir(
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    # Note that this returns a dummy identify matrix. See the comment in the code about
    # this
    cov = r.read_constraint_matrix(
        StateElementIdentifier("HDO"), RetrievalType("default"), 30
    )
    assert cov.shape == (30, 30)
