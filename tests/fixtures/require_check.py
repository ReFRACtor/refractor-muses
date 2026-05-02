import refractor.muses_py as mpy
import refractor.muses_py_fm as mpy_fm
from refractor.muses import muses_oss_handle
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

# muses-oss is closed source. We normally require this, but if we are
# working in open source only then this isn't available.
require_oss = pytest.mark.skipif(
    not muses_oss_handle.have_liboss, reason="test requires that muses-oss be installed"
)
