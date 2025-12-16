# Just import any files we find in this directory, rather than listing
# everything.

import os as _os
import re as _re
import glob as _glob
from .version import __version__
import typing as _typing

for _i in _glob.glob(_os.path.dirname(__file__) + "/*.py"):
    mname = _os.path.basename(_i).split(".")[0]
    # Don't load ipython, which is ipython magic extensions, or unit tests
    if (
        not mname == "ipython"
        and not mname == "version"
        and not mname == "cython_try"
        and not _re.search("_test", mname)
    ):
        exec("from .%s import *" % mname)

if _typing.TYPE_CHECKING:
    # mypy doesn't correctly support import *. Pretty annoying, there are threads going
    # back years about why this doesn't work. We don't want to spend a whole lot of
    # time working around this, the point of mypy is to help us and reduce our work, not
    # to make a bunch of make work. But to the degree useful, we can work around this by
    # having an explicit imports for things needed by mypy. We don't want this in general, it
    # is fragile (did you remember to update __init__ here when you added that new
    # class?). So just as much as it is useful we do a whack a mole here of quieting errors
    # we get in things like refractor.omi.
    #
    # Note we guard this with the standard "if typing.TYPE_CHECKING", so this code doesn't
    # appear in real python usage of this module.
    from .refractor_fm_object_creator import RefractorFmObjectCreator
    from .forward_model_handle import ForwardModelHandle, ForwardModelHandleSet
    from .muses_raman import MusesRaman, SurfaceAlbedo
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import (
        CurrentState,
        PropagatedQA,
    )
    from .state_info import StateInfo
    from .sounding_metadata import SoundingMetadata
    from .retrieval_array import (
        RetrievalGridArray,
        FullGridArray,
        FullGridMappedArray,
        RetrievalGrid2dArray,
        FullGrid2dArray,
    )
    from .misc import osp_setup, AttrDictAdapter, ResultIrk

    from .muses_observation import (
        MeasurementId,
        MusesObservation,
    )
    from .muses_airs_observation import MusesAirsObservation
    from .muses_cris_observation import MusesCrisObservation
    from .muses_reflectance_observation import (
        MusesOmiObservation,
        MusesTropomiObservation,
    )
    from .muses_tes_observation import MusesTesObservation
    from .muses_strategy import (
        CurrentStrategyStep,
        CurrentStrategyStepDict,
        MusesStrategy,
        MusesStrategyHandle,
        MusesStrategyImp,
    )
    from .state_element import (
        StateElement,
        StateElementImplementation,
        StateElementHandle,
        StateElementHandleSet,
        StateElementFillValueHandle,
        StateElementFixedValueHandle,
        StateElementWithCreate,
        StateElementWithCreateHandle,
    )
    from .state_element_osp import (
        StateElementOspFile,
        StateElementOspFileFixedValue,
        OspSetupReturn,
    )
    from .identifier import (
        FilterIdentifier,
        InstrumentIdentifier,
        RetrievalType,
        StateElementIdentifier,
        StrategyStepIdentifier,
    )
    from .fake_state_info import FakeStateInfo
    from .fake_retrieval_info import FakeRetrievalInfo
    from .filter_metadata import FileFilterMetadata, DictFilterMetadata
    from .replace_function_helper import (
        suppress_replacement,
        register_replacement_function_in_block,
    )
    from .input_file_monitor import InputFileMonitor
    from .refractor_capture_directory import RefractorCaptureDirectory
    from .retrieval_result import RetrievalResult
    from .spectral_window_handle import SpectralWindowHandleSet
    from .cost_function import CostFunction
    from .cost_function_creator import CostFunctionCreator
    from .priority_handle_set import PriorityHandleSet
    from .retrieval_configuration import RetrievalConfiguration
    from .observation_handle import (
        ObservationHandleSet,
        mpy_radiance_from_observation_list,
    )
    from .tes_file import TesFile
    from .order_species import order_species

del _i
del _re
del _os
del _glob
del _typing
