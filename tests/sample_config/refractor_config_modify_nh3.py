from __future__ import annotations
from refractor.muses import (
    RetrievalStrategy,
    RetrievalStrategyCaptureObserver,
    StateElementFromClimatologyNh3,
    OspSetupReturn,
    StateElementWithCreateHandle,
    StateElementIdentifier,
    StateInfo,
    ObservationHandleSet,
    MusesStrategy,
    SoundingMetadata,
    RetrievalConfiguration,
    FullGridMappedArray,
    InstrumentIdentifier,
    mpy_radiance_from_observation_list,
)
from refractor.tropomi import TropomiForwardModelHandle
from refractor.omi import OmiForwardModelHandle
import refractor.framework as rf
from typing import Any
from loguru import logger


# Configuration to use our own TROPOMI and OMI forward models.
# Note that if we want to use VLIDORT instead, can just skip the forward model

# Turn on logging for refractor. This is independent of python logger, although we
# could probably integrate this together if it is important
rf.Logger.set_implementation(rf.FpLogger())

rs = RetrievalStrategy(None)
rs.forward_model_handle_set.add_handle(
    TropomiForwardModelHandle(use_pca=True, use_lrad=False, lrad_second_order=False),
    priority_order=100,
)
rs.forward_model_handle_set.add_handle(
    OmiForwardModelHandle(use_pca=True, use_lrad=False, lrad_second_order=False),
    priority_order=100,
)

# We modify things by adding new code. Not sure what the exact changes
# we want to make here is, but for this example we start with our
# existing StateElementFromClimatologyNh3, and then modify the
# creation to have a different initial guess


class MyNh3(StateElementFromClimatologyNh3):
    @classmethod
    def _setup_create(  # type: ignore[override]
        cls,
        pressure_list_fm: FullGridMappedArray,
        sid: StateElementIdentifier,
        retrieval_config: RetrievalConfiguration,
        sounding_metadata: SoundingMetadata,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        state_info: StateInfo,
        **kwargs: Any,
    ) -> OspSetupReturn | None:
        logger.opt(colors=True).info("<red>Note that we are using MyNh3</>")
        # This is copied from StateElementFromClimatologyNh3, with the idea
        # that we want to modify something here. You can certainly change
        # the logic, in the end we just need the initial value and initial
        # constraint
        if strategy is None:
            return None
        # We only use this if NH3 is in the error analysis
        # interferents or retrieval elements. Not sure of the exact
        # reason for this, but this is the logic used in muses-py in
        # states_initial_update.py and we duplicate this here.
        #
        # TODO Is the actually the right logic? Why should the initial
        # guess depend on it being an interferent or in the retrieval?
        if (
            StateElementIdentifier("NH3") not in strategy.error_analysis_interferents
            and StateElementIdentifier("NH3") not in strategy.retrieval_elements
        ):
            return None
        # Only have handling for CRIS, TES and AIRS.
        if (
            InstrumentIdentifier("CRIS") not in strategy.instrument_name
            and InstrumentIdentifier("TES") not in strategy.instrument_name
            and InstrumentIdentifier("AIRS") not in strategy.instrument_name
        ):
            return None
        surface_type = sounding_metadata.surface_type
        tsur = state_info[StateElementIdentifier("TSUR")].constraint_vector_fm[0]
        tatm0 = state_info[StateElementIdentifier("TATM")].constraint_vector_fm[0]
        nh3type: str | None = None
        for ins, sfunc in [
            (InstrumentIdentifier("CRIS"), cls.supplier_nh3_type_cris),
            (InstrumentIdentifier("TES"), cls.supplier_nh3_type_tes),
            (InstrumentIdentifier("AIRS"), cls.supplier_nh3_type_airs),
        ]:
            if ins in strategy.instrument_name:
                olist = [
                    observation_handle_set.observation(ins, None, None, None),
                ]
                rad = mpy_radiance_from_observation_list(olist, full_band=True)
                nh3type = sfunc(rad, tsur, tatm0, surface_type)
                break
        if nh3type is None:
            nh3type = "MOD"

        # ******************************************
        # My change, always use enhanced
        # ******************************************

        nh3type = "ENH"

        # End of my change

        clim_dir = retrieval_config.abs_dir("../OSP/Climatology/Climatology_files")
        # TODO Check if this is actually correct
        # Oddly the initial value comes from the prior file (so is_constraint is
        # True). Not sure if this is what was intended, but it is what muses_py
        # does in states_initial_update.py. We duplicate that here, but should
        # determine at some point if this is actually correct. Why even have the
        # non prior climatology if we aren't using it?
        value_fm, poltype = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            retrieval_config.input_file_helper,
            ind_type=nh3type,
            linear_interp=False,
        )
        constraint_vector_fm, _ = cls.read_climatology_2022(
            sid,
            pressure_list_fm,
            True,
            clim_dir,
            sounding_metadata,
            retrieval_config.input_file_helper,
            ind_type=nh3type,
            linear_interp=False,
        )
        create_kwargs = {}
        if poltype is not None:
            create_kwargs["poltype"] = poltype
        return OspSetupReturn(
            value_fm=value_fm,
            constraint_vector_fm=constraint_vector_fm,
            create_kwargs=create_kwargs,
        )


# We then register a creator for MyNh3, and set it to "cut in line" with
# the priority handle set. We set the priority order to 100, so it is a higher
# priority than any other creator we have registered

rs.state_element_handle_set.add_handle(
    StateElementWithCreateHandle(
        StateElementIdentifier("NH3"),
        MyNh3,
    ),
    priority_order=100,
)

if False:
    # If desired, capture each step so we can rerun this for debugging
    rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    rs.add_observer(rscap)

# Replace top level script_retrieval_ms with RetrievalStrategy (only matters
# if we are using py-retrieval instead of refractor-retrieve).
rs.register_with_muses_py()
