from refractor.muses import greatcircle
from pytest import approx


def test_greatcircle():
    res = greatcircle(62.86464309692383, 81.03790283203125, 62.875, 81.02499389648438)
    # Got this results from calling muses_py.greatcircle
    assert res == approx(1.3261137928940256 * 1e3)
