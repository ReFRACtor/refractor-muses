from refractor.muses import OspCovarianceMatrixReader, StateElementIdentifier
import numpy as np
import numpy.testing as npt


def test_covariance(osp_dir):
    r = OspCovarianceMatrixReader.read_dir(osp_dir / "Covariance" / "Covariance")
    d = r.read_cov(StateElementIdentifier("OMIODWAVUV1"), "linear", 10)
    npt.assert_allclose(d, np.array([[0.0004]]))
