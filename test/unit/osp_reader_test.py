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
    r = OspCovarianceMatrixReader.read_dir(osp_dir / "Covariance" / "Covariance")
    d = r.read_cov(StateElementIdentifier("OMIODWAVUV1"), "linear", 10)
    npt.assert_allclose(d, np.array([[0.0004]]))


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
        StateElementIdentifier("OMIODWAVUV1"), RetrievalType("a_retrieval_type")
    )
    npt.assert_allclose(cmatrix, [[2500.0]])
    # Repeat, to make sure caching works
    cmatrix = r.read_constraint_matrix(
        StateElementIdentifier("OMIODWAVUV1"), RetrievalType("a_retrieval_type")
    )
    npt.assert_allclose(cmatrix, [[2500.0]])
    cmatrix2 = r.read_constraint_matrix(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("a_retrieval_type")
    )
    cmatrix3 = r.read_constraint_matrix(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("omicloud_ig_refine")
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
