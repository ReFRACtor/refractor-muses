from refractor.muses import (
    OspCovarianceMatrixReader,
    StateElementIdentifier,
    OspSpeciesReader,
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
        StateElementIdentifier("OMIODWAVSLOPEUV1"), RetrievalType("a_retrieval_type")
    )
    t2 = r.read_file(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("a_retrieval_type")
    )
    t3 = r.read_file(
        StateElementIdentifier("OMICLOUDFRACTION"), RetrievalType("omicloud_ig_refine")
    )
    #breakpoint()
