import refractor.muses_py as mpy
import refractor.muses_py_fm as mpy_fm
import pytest

# Marker that skips a test if we don't have muses-py
require_muses_py = pytest.mark.skipif(
    not mpy.have_muses_py,
    reason="test requires that the muses-py library is available",
)

require_muses_py_fm = pytest.mark.skipif(
    not mpy_fm.have_muses_py,
    reason="test requires that the muses-py-fm library is available",
)
