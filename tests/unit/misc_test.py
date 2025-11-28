from refractor.muses import greatcircle
from pytest import approx


def test_greatcircle():
    # Temp
    import refractor.muses.muses_py as mpy

    res = greatcircle(62.86464309692383, 81.03790283203125, 62.875, 81.02499389648438)
    res2 = mpy.greatcircle(
        62.86464309692383, 81.03790283203125, 62.875, 81.02499389648438, meters=True
    )
    assert res2 == approx(1.3261137928940256 * 1e3)
    assert res == approx(1.3261137928940256 * 1e3)
