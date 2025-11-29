import refractor.muses_py as mpy
import pytest

# Marker that skips a test if we don't have muses-py
require_muses_py = pytest.mark.skipif(
    not mpy.have_muses_py, reason="test requires that the muses-py library is available"
)
