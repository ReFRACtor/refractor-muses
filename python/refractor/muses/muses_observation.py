from __future__ import annotations
from .observation_handle import ObservationHandle
from .muses_spectral_window import MusesSpectralWindow
from .retrieval_configuration import RetrievalConfiguration
from .tes_file import TesFile
from contextlib import contextmanager
import os
from pathlib import Path
import numpy as np
import refractor.framework as rf  # type: ignore
import abc
import copy
from loguru import logger
import pickle
import itertools
import collections.abc
import re
from typing import Iterator, Any, Generator, TypeVar
import typing
from .identifier import InstrumentIdentifier, StateElementIdentifier, FilterIdentifier

if typing.TYPE_CHECKING:
    from .current_state import CurrentState


class MeasurementId(collections.abc.Mapping):
    """py-retrieve uses a file called Measurement_ID.asc. This files
    contains information about the soundings we use. This is mostly
    just a standard keyword/value set, however there are a few
    complications:

    1. The names may be relative to the directory that the
       Measurement_ID.asc file is in, so we need to handle translating
       this to a full path since we aren't in general in the
       Measurement_ID.asc directory.

    2. There may be "associated" files that really logically should
       live in the Measurement_ID.asc file but don't because it is
       convenient to store them elsewhere - for example the
       omi_calibration_filename which comes from the strategy file.
       3. When reading the data, we often need to know the specific
       filters we will be working with, e.g., so we only read that
       data out of the sounding files.

    3. We also want to bring in the RetrievalConfiguration data, the
       separation between the two is pretty arbitrary and we tend to
       need both together.

    This class brings this stuff together. It is mostly just a dict
    mapping keyword to file or configuration value, but with these
    extra handling included.

    This class is an abstract interface, it is useful for testing to
    have a simple implementation that doesn't depend on the
    Measurement_ID.asc and strategy tables files (e.g., a hardcoded
    dict with the values).

    """

    @abc.abstractproperty
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        """
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def osp_abs_dir(self) -> Path | None:
        return None


class MeasurementIdDict(MeasurementId):
    """Implementation of MeasurementId that uses a dict"""

    def __init__(
        self,
        measurement_dict: dict,
        filter_list_dict: dict[InstrumentIdentifier, list[FilterIdentifier]],
        osp_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.measurement_dict = measurement_dict
        self._filter_list_dict = filter_list_dict
        self.osp_dir = Path(osp_dir).absolute() if osp_dir is not None else None

    @property
    def osp_abs_dir(self) -> Path | None:
        return self.osp_dir

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        """
        return self._filter_list_dict

    @filter_list_dict.setter
    def filter_list_dict(
        self, val: dict[InstrumentIdentifier, list[FilterIdentifier]]
    ) -> None:
        self._filter_list_dict = val

    def __getitem__(self, key: str) -> Any:
        return self.measurement_dict[key]

    def __iter__(self) -> Iterator[str]:
        return self.measurement_dict.__iter__()

    def __len__(self) -> int:
        return len(self.measurement_dict)


class MeasurementIdFile(MeasurementId):
    """Implementation of MeasurementId that uses the
    Measurement_ID.asc file."""

    def __init__(
        self,
        fname: str | os.PathLike[str],
        retrieval_config: RetrievalConfiguration,
        filter_list_dict: dict[InstrumentIdentifier, list[FilterIdentifier]],
    ) -> None:
        self.fname = Path(fname)
        self.base_dir = self.fname.parent.absolute()
        self._p = TesFile(self.fname, retrieval_config.input_file_monitor)
        self._filter_list_dict = filter_list_dict
        self._retrieval_config = retrieval_config

    @property
    def osp_abs_dir(self) -> Path | None:
        return self._retrieval_config.osp_abs_dir

    def __hash__(self) -> int:
        # We need a unique hash to separate MeasurementIds. I think just using the
        # absolute file name will do that.
        return hash(self.fname.absolute())

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        """
        return self._filter_list_dict

    @filter_list_dict.setter
    def filter_list_dict(
        self, val: dict[InstrumentIdentifier, list[FilterIdentifier]]
    ) -> None:
        self._filter_list_dict = val

    def __getitem__(self, key: str) -> Any:
        if key in self._p:
            return self._abs_dir(self._p[key])
        if key in self._retrieval_config:
            return self._retrieval_config[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(self._p, self._retrieval_config)

    def __len__(self) -> int:
        return len(self._p) + len(self._retrieval_config)

    def _abs_dir(self, v: Any) -> Any:
        # Don't try treating a list like a path.
        if isinstance(v, list):
            return v
        v = copy.copy(v)
        v = os.path.expandvars(os.path.expanduser(v))
        if re.match(r"^\.\./", v) or re.match(r"^\./", v):
            v = os.path.normpath(self.base_dir / v)
        return v


class MusesObservation(rf.ObservationSvImpBase, metaclass=abc.ABCMeta):
    """The Observation for MUSES is a standard ReFRACtor Observation,
    with a few extra pieces needed by the MUSES code.

    This class specifies the interface needed. Note like most of the
    time we use standard duck typing, so a MUSES observation doesn't
    actually need to inherit from this class if for whatever reason
    that isn't convenient. But is still useful to know that the
    interface is.

    The things added are:

    instrument_name - the name of the instrument the MusesObservation
        is for

    filter_data - metadata about the filters covered the
        MusesObservation

    sounding_desc - this is a dictionary with the instrument specific
        way of describing what sounding we are using. This is used in
        the product output files (so stuff in RetrievalOutput)

    spectral_window - an rf.Observation needs to supply the
        radiance/reflectance data to match against the forward
        model. Often this is a subset of the full instrument data
        (e.g., removing bad samples, applying microwindows). The
        rf.Observation class doesn't specify how this is done. The
        input data might have just already been subsetted, or we might
        apply a SpectralWindow to a larger set of data. For
        MusesObservation we always use a SpectralWindow.

    state_element_name_list - List of state elements if any that are
        used to deterimine radiance

    Right now, we require the spectral_window to be the more specific
    MusesSpectralWindow.  This is so we have handling for including
    bad samples, extending for RamanSioris, or doing a full band. If
    needed we can probably relax this requirement - we just need a way
    to handle these different cases. MusesSpectralWindow is a pretty
    general adapter around another SpectralWindow, so this probably
    isn't too much of a constraint requiring this.

    We have all the normal rf.Observation stuff, plus what is found in
    this class.

    """

    def __init__(
        self, coeff: np.ndarray, in_map: rf.StateMapping | None = None
    ) -> None:
        if in_map is None:
            super().__init__(coeff)
        else:
            super().__init__(coeff, in_map)

    @abc.abstractproperty
    def instrument_name(self) -> InstrumentIdentifier:
        """Name of instrument observation is for."""
        raise NotImplementedError()

    @abc.abstractproperty
    def filter_data(self) -> list[tuple[FilterIdentifier, int]]:
        """This returns a list of filter names and sizes. This is used
        as metadata in the py-retrieve structure called
        "radianceStep".

        Note this is similar but distinct from the filter_list_dict
        used in MeasurementId. That list corresponds to specific data
        read from a file. Often this is the same as the filter data
        used in "radianceStep", but in some cases py-retrieve wants to
        think of data as different filters even if it is read from one
        structure - so for example CrIS data gets separated into
        'CrIS-fsr-lw', 'CrIS-fsr-mw', 'CrIS-fsr-sw' even though the
        data is read into one array in read_noaa_cris_fsr. Individual
        classes can handle generating this filter_data however they
        like, there is no requirement that the number of filters is
        the same as the number of channels.

        This should return a list of pairs as a filter name and length
        of data (using the spectral_window).

        """
        raise NotImplementedError()

    @abc.abstractproperty
    def sounding_desc(self) -> dict[str, Any]:
        """Different types of instruments have different description
        of the sounding ID. This gets used in retrieval_l2_output for
        metadata.

        """
        raise NotImplementedError()

    @property
    def spectral_window(self) -> MusesSpectralWindow:
        """SpectralWindow to apply to the observation data."""
        raise NotImplementedError()

    @spectral_window.setter
    def spectral_window(self, val: MusesSpectralWindow) -> None:
        """Set the SpectralWindow to apply to the observation data."""
        raise NotImplementedError()

    def update_coeff_and_mapping(self, coeff: np.ndarray, mp: rf.StateMapping) -> None:
        """Update the objects coefficients and state mapping. Useful
        if we create observation before we have a CurrentState and/or
        StateVector - which we often do in unit tests.

        """
        # Default is no coefficients changing observation
        pass

    def state_element_name_list(self) -> list[StateElementIdentifier]:
        """List of state element names for this observation"""
        # Default is no state element changing observation
        return []

    def radiance_all_extended(
        self,
        skip_jacobian: bool = True,
        include_bad_sample: bool = False,
        full_band: bool = False,
        do_raman_ext: bool = False,
    ) -> rf.Spectrum:
        """Convenience function that changes the spectral_window
        (e.g., turn on bad samples), calls radiance_all, and then
        changes back.

        Normally we want just the radiance data, so the default is to
        skip the jacobian part. You can select that if you like by
        passing skip_jacobian=False

        """
        with self.modify_spectral_window(
            include_bad_sample=include_bad_sample,
            full_band=full_band,
            do_raman_ext=do_raman_ext,
        ):
            return self.radiance_all(skip_jacobian)

    @contextmanager
    def modify_spectral_window(
        self,
        include_bad_sample: bool = False,
        full_band: bool = False,
        do_raman_ext: bool = False,
    ) -> Generator[None, None, None]:
        """Convenience context that changes the spectral_window (e.g.,
        turn on bad samples), does something, and then changes back.

        """
        t1 = self.spectral_window.include_bad_sample
        t2 = self.spectral_window.full_band
        t3 = self.spectral_window.do_raman_ext
        try:
            self.spectral_window.include_bad_sample = include_bad_sample
            self.spectral_window.full_band = full_band
            self.spectral_window.do_raman_ext = do_raman_ext
            yield
        finally:
            self.spectral_window.include_bad_sample = t1
            self.spectral_window.full_band = t2
            self.spectral_window.do_raman_ext = t3

    @abc.abstractproperty
    def surface_altitude(self) -> rf.DoubleWithUnit:
        raise NotImplementedError()


class MusesObservationImp(MusesObservation):
    """Common behavior for each of the MusesObservation classes we
    have.

    Note that muses_py_dict is the old structure used in
    py-retrieve. We *only* use this internally, except for the
    creation of the old UIP that also depends on this (see
    MusesStrategyExecutor.uip_func).  We want to get to the point
    where the only use of the uip is to support the old py-retrieve
    forward models - however we are still cleaning a few spots.  But
    for now, we rely on having muses_py_dict.

    """

    def __init__(
        self,
        muses_py_dict: dict[str, Any],
        sdesc: dict[str, Any],
        num_channels: int = 1,
        coeff: np.ndarray | None = None,
        mp: None | rf.StateMapping = None,
    ) -> None:
        if coeff is None:
            super().__init__(np.array([]))
        else:
            if mp is None:
                raise RuntimeError("Both coeff and mp need to be None or not None")
            super().__init__(coeff, mp)
        self.muses_py_dict = muses_py_dict
        self._spectral_window = MusesSpectralWindow(None, None)
        self._num_channels = num_channels
        self._sounding_desc = sdesc
        self._filter_data_name: list[FilterIdentifier] = []
        self._filter_data_swin: None | MusesSpectralWindow = None

    def wn_and_sindex(self, sensor_index: int) -> tuple[list[np.ndarray], int]:
        """We have a couple of places where we need the complete
        frequency grid (so all spectral channels), and the sample
        index for one of the sensors in that full grid.

        This is pretty specific, but is used in a few muses-py calls
        (for ILS, for o3xsec.

        We put this into one function here, just so we don't end up
        with a bug where one place in the code does this differently

        """
        wn = [self.frequency_full(i) for i in range(self.num_channels)]
        wn_len = [len(w) for w in wn]
        wn = np.concatenate(wn)
        with self.modify_spectral_window(include_bad_sample=True, do_raman_ext=True):
            sindex = self.spectral_domain(sensor_index).sample_index - 1
        sindex += sum(wn_len[:sensor_index])
        return wn, sindex

    @property
    def cloud_pressure(self) -> rf.DoubleWithUnit:
        """Cloud pressure. I think all the instrument types handle
        this the same way, if not we can push this down to omi and
        tropomi only.

        """
        return rf.DoubleWithUnit(
            float(self.muses_py_dict["Cloud"]["CloudPressure"]), "hPa"
        )

    @property
    def cloud_fraction(self) -> float:
        """Cloud pressure. I think all the instrument types handle
        this the same way, if not we can push this down to omi and
        tropomi only.

        """
        return float(self.muses_py_dict["Cloud"]["CloudFraction"])

    @property
    def observation_table(self) -> dict[str, Any]:
        return self.muses_py_dict["Earth_Radiance"]["ObservationTable"]

    @property
    def wavelength_filter(self) -> dict[str, Any]:
        return self.muses_py_dict["Earth_Radiance"]["EarthWavelength_Filter"]

    @property
    def across_track(self) -> list[int]:
        res = []
        for i in range(self.num_channels):
            fname = self.filter_list[i]
            res.append(
                np.asarray(self.observation_table["XTRACK"])[
                    np.asarray(self.observation_table["Filter_Band_Name"]) == str(fname)
                ][0]
            )
        return res

    @property
    def earth_sun_distance(self) -> float:
        """Earth sun distance in meter"""
        return self.observation_table["EarthSunDistance"][0]

    def _avg_obs(self, nm: str, sensor_index: int) -> float:
        """Average values of the given name the the given sensor index
        (sometimes more than one entry in the table).

        """
        # Note this is defined in MusesObservationReflectance. If we need to generalize
        # this, we can try to come up with a more general way to handle this
        fname = self.filter_list[sensor_index]
        return np.mean(
            np.asarray(self.observation_table[nm])[
                np.asarray(self.observation_table["Filter_Band_Name"]) == str(fname)
            ]
        )

    @property
    def spacecraft_altitude(self) -> float:
        return np.mean(np.asarray(self.observation_table["SpacecraftAltitude"]))

    @property
    def solar_zenith(self) -> np.ndarray:
        return np.array(
            [
                float(self._avg_obs("SolarZenithAngle", i))
                for i in range(self.num_channels)
            ]
        )

    @property
    def scattering_angle(self) -> np.ndarray:
        return np.array(
            [
                float(self._avg_obs("ScatteringAngle", i))
                for i in range(self.num_channels)
            ]
        )

    @property
    def observation_zenith(self) -> np.ndarray:
        return np.array(
            [
                float(self._avg_obs("ViewingZenithAngle", i))
                for i in range(self.num_channels)
            ]
        )

    @property
    def relative_azimuth(self) -> np.ndarray:
        return np.array(
            [
                float(self._avg_obs("RelativeAzimuthAngle", i))
                for i in range(self.num_channels)
            ]
        )

    @property
    def latitude(self) -> np.ndarray:
        return np.array(
            [float(self._avg_obs("Latitude", i)) for i in range(self.num_channels)]
        )

    @property
    def surface_height(self) -> np.ndarray:
        return np.array(
            [float(self._avg_obs("TerrainHeight", i)) for i in range(self.num_channels)]
        )

    @property
    def longitude(self) -> np.ndarray:
        return np.array(
            [float(self._avg_obs("Longitude", i)) for i in range(self.num_channels)]
        )

    @property
    def filter_data(self) -> list[tuple[FilterIdentifier, int]]:
        if self._filter_data_swin is None:
            raise RuntimeError("Need to fill in self._filter_data_swin")
        res: list[tuple[FilterIdentifier, int]] = []
        sd = self.spectral_domain_all()
        for i, fltname in enumerate(self._filter_data_name):
            sz = self._filter_data_swin.apply(sd, i).data.shape[0]
            res.append((fltname, sz))
        return res

    def update_coeff_and_mapping(self, coeff: np.ndarray, mp: rf.StateMapping) -> None:
        """Update the objects coefficients and state mapping. Useful
        if we create observation before we have a CurrentState and/or
        StateVector - which we often do in unit tests.

        """
        # Despite the name "init", this really just sets the mapping and coefficient,
        # it doesn't redo any of the other class initialization. Note this comes from
        # the rf.SubStateVectorArray parent if you are looking for the code.
        self.init(coeff, mp)

    @property
    def sounding_desc(self) -> dict[str, Any]:
        """Different types of instruments have different description
        of the sounding ID. This gets used in retrieval_l2_output for
        metadata.

        """
        return self._sounding_desc

    def _v_num_channels(self) -> int:
        return self._num_channels

    def notify_update(self, sv: rf.StateVector) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        super().notify_update(sv)
        self._spec = [
            None,
        ] * self.num_channels

    @property
    def spectral_window(self) -> MusesSpectralWindow:
        return self._spectral_window

    @spectral_window.setter
    def spectral_window(self, val: MusesSpectralWindow) -> None:
        self._spectral_window = val

    def spectral_domain(self, sensor_index: int) -> rf.SpectralDomain:
        sd = self.spectral_domain_full(sensor_index)
        return self.spectral_window.apply(sd, sensor_index)

    def radiance(self, sensor_index: int, skip_jacobian: bool = False) -> rf.Spectrum:
        return self.spectral_window.apply(
            self.spectrum_full(sensor_index), sensor_index
        )

    def spectrum_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> rf.Spectrum:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        if sensor_index < 0 or sensor_index >= self.num_channels:
            raise RuntimeError("sensor_index out of range")
        sd = self.spectral_domain_full(sensor_index)
        sr = rf.SpectralRange(
            self.radiance_full(sensor_index, skip_jacobian=skip_jacobian),
            rf.Unit("sr^-1"),
            self.nesr_full(sensor_index),
        )
        return rf.Spectrum(sd, sr)

    def radiance_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> np.ndarray:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        raise NotImplementedError

    def spectral_domain_full(self, sensor_index: int) -> rf.SpectralDomain:
        """Spectral domain before we have removed bad samples or
        applied the microwindows."""
        # By convention, sample index starts with 1. This was from OCO-2, I'm not
        # sure if that necessarily makes sense here or not. But I think we have code
        # that depends on the 1 base.
        freq = self.frequency_full(sensor_index)
        sindex = np.array(list(range(len(freq)))) + 1
        return rf.SpectralDomain(freq, sindex, rf.Unit("nm"))

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        raise NotImplementedError

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        raise NotImplementedError

    def bad_sample_mask(self, sensor_index: int) -> np.ndarray:
        # Default way to find bad samples is to look for negative
        # NESR. Some of the the derived objects override this (e.g.,
        # Tropomi also check the solar model for negative values).
        return np.array(self.nesr_full(sensor_index) < 0)


class SimulatedObservation(MusesObservationImp):
    """This is a Observation based off of a underlying observation. We
    get the various pieces from the underlying observation, except we
    replace the radiance with other values (e.g, from a forward model
    run.

    """

    def __init__(
        self, obs: MusesObservationImp, replacement_spectrum: list[np.ndarray]
    ) -> None:
        # Note the muses_py_dict is needed here, although only for
        # generating a UIP. Right now we still need that functionality
        # in a few places.  The long term goal is to have this *only*
        # used by the old py-retrieve forward models, but we still are
        # working to that point. If we get there, it would be
        # reasonable to remove the muses_py_dict and just not have
        # SimulatedObservation work with the old forward models.
        super().__init__(
            obs.muses_py_dict, obs.sounding_desc, num_channels=obs.num_channels
        )
        self._obs: MusesObservationImp = copy.deepcopy(obs)
        # We only have replacement_spectrum where the current spectral
        # window is. We just pretend that all the pixels outside of
        # the spectral window are bad, because we don't have other
        # values.
        self._nesr_full = []
        self._rad_full = []
        self._spectral_window = copy.deepcopy(obs.spectral_window)
        for i in range(self.num_channels):
            gpt = obs.radiance(i).spectral_domain.sample_index - 1
            t = obs.nesr_full(i)
            t2 = np.full_like(t, -9999)
            t2[gpt] = t[gpt]
            self._nesr_full.append(t2)
            t = obs.radiance_full(i, skip_jacobian=True)
            t2 = np.full_like(t, -9999)
            t2[gpt] = replacement_spectrum[i]
            self._rad_full.append(t2)

        # Update bad pixel mask for our spectral window
        self.spectral_window.add_bad_sample_mask(self)

    @property
    def filter_list(self) -> list[FilterIdentifier]:
        return self._obs.filter_list

    @property
    def instrument_name(self) -> InstrumentIdentifier:
        return self._obs.instrument_name

    @property
    def cloud_pressure(self) -> rf.DoubleWithUnit:
        return self._obs.cloud_pressure

    @property
    def observation_table(self) -> dict[str, Any]:
        return self._obs.observation_table

    @property
    def across_track(self) -> list[int]:
        return self._obs.across_track

    @property
    def earth_sun_distance(self) -> float:
        return self._obs.earth_sun_distance

    @property
    def solar_zenith(self) -> np.ndarray:
        return self._obs.solar_zenith

    @property
    def observation_zenith(self) -> np.ndarray:
        return self._obs.observation_zenith

    @property
    def relative_azimuth(self) -> np.ndarray:
        return self._obs.relative_azimuth

    @property
    def latitude(self) -> np.ndarray:
        return self._obs.latitude

    @property
    def longitude(self) -> np.ndarray:
        return self._obs.longitude

    @property
    def filter_data(self) -> list[tuple[FilterIdentifier, int]]:
        return self._obs.filter_data

    def update_coeff_and_mapping(self, coeff: np.ndarray, mp: rf.StateMapping) -> None:
        # We don't do any updating here, the observation stays fixed
        pass

    @property
    def spectral_window(self) -> MusesSpectralWindow:
        return self._spectral_window

    @spectral_window.setter
    def spectral_window(self, val: MusesSpectralWindow) -> None:
        self._spectral_window = val

    def radiance_full(
        self, sensor_index: int, skip_jacobian: bool = False
    ) -> np.ndarray:
        """The full list of radiance, before we have removed bad
        samples or applied the microwindows.

        """
        return self._rad_full[sensor_index]

    def frequency_full(self, sensor_index: int) -> np.ndarray:
        """The full list of frequency, before we have removed bad
        samples or applied the microwindows.

        """
        return self._obs.frequency_full(sensor_index)

    def nesr_full(self, sensor_index: int) -> np.ndarray:
        """The full list of NESR, before we have removed bad samples
        or applied the microwindows.

        """
        return self._nesr_full[sensor_index]

    @property
    def surface_altitude(self) -> rf.DoubleWithUnit:
        return self._obs.surface_altitude


class SimulatedObservationHandle(ObservationHandle):
    """Just return the given observation always for the given
    instrument.  This is intended for testing, with for example a
    SimulatedObservation.

    """

    def __init__(
        self, instrument_name: InstrumentIdentifier, obs: MusesObservation
    ) -> None:
        self.instrument_name = instrument_name
        self.obs = obs

    def notify_update_target(self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_update_target")
        self.measurement_id = measurement_id

    def observation(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        **kwargs: Any,
    ) -> MusesObservation | None:
        if instrument_name != self.instrument_name:
            return None
        logger.debug(f"Creating observation using {self.__class__.__name__}")
        return copy.deepcopy(self.obs)


MusesObservationClassType = TypeVar("MusesObservationClassType", bound=MusesObservation)


class MusesObservationHandle(ObservationHandle):
    """A lot of our observation classes just map a name to a object of
    a specific class. This handles this generic construction.

    """

    def __init__(
        self,
        instrument_name: InstrumentIdentifier,
        obs_cls: type[MusesObservationClassType],
    ) -> None:
        self.instrument_name = instrument_name
        self.obs_cls = obs_cls
        # Keep the same observation around as long as the target doesn't
        # change - we just update the spectral windows.
        self.existing_obs: MusesObservation | None = None
        self.measurement_id: MeasurementId | None = None
        self.retrieval_config: RetrievalConfiguration | None = None

    def __getstate__(self) -> dict[str, Any]:
        # If we pickle, don't include the stashed obs
        attributes = self.__dict__.copy()
        del attributes["existing_obs"]
        return attributes

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self.existing_obs = None

    def notify_update_target(self, measurement_id: MeasurementId, retrieval_config: RetrievalConfiguration) -> None:
        # Need to read new data when the target changes
        logger.debug(f"Call to {self.__class__.__name__}::notify_update_target")
        self.existing_obs = None
        self.measurement_id = measurement_id
        self.retrieval_config = retrieval_config

    def observation(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        **kwargs: Any,
    ) -> MusesObservation | None:
        if instrument_name != self.instrument_name:
            return None
        if self.measurement_id is None or self.retrieval_config is None:
            raise RuntimeError("Need to call notify_update_target before observation")
        logger.debug(f"Creating observation using {self.obs_cls.__name__}")
        obs = self.obs_cls.create_from_id(
            self.measurement_id,
            self.existing_obs,
            current_state,
            spec_win,
            fm_sv,
            self.retrieval_config.input_file_monitor,
            osp_dir=self.measurement_id.osp_abs_dir,
            **kwargs,
        )
        if self.existing_obs is None:
            self.existing_obs = obs
        return obs


class MusesObservationHandlePickleSave(MusesObservationHandle):
    """It can take a surprising amount of time to read in the observation data.
    We are only talking about a few seconds, but since this is such a common thing
    in unit tests it is worth having a quicker way to do this.

    This handle is a simple adapter that saves the observation out as a pickle, and
    then the next time a observation gets read in a unit test we can read the pickle
    file."""

    def notify_update_target(self, measurement_id: MeasurementId, retrieval_config : RetrievalConfiguration) -> None:
        super().notify_update_target(measurement_id, retrieval_config)
        pname = measurement_id["run_dir"] / f"{self.instrument_name}_obs.pkl"
        if pname.exists():
            self.existing_obs = pickle.load(open(pname, "rb"))

    def observation(
        self,
        instrument_name: InstrumentIdentifier,
        current_state: CurrentState | None,
        spec_win: MusesSpectralWindow | None,
        fm_sv: rf.StateVector | None,
        **kwargs: Any,
    ) -> MusesObservation | None:
        if self.measurement_id is None:
            raise RuntimeError("Call notify_update_target first")
        might_save = self.existing_obs is None
        res = super().observation(
            instrument_name,
            current_state,
            spec_win,
            fm_sv,
            **kwargs,
        )
        if res is not None and might_save and self.existing_obs is not None:
            pname = self.measurement_id["run_dir"] / f"{self.instrument_name}_obs.pkl"
            pickle.dump(self.existing_obs, open(pname, "wb"))
        return res


__all__ = [
    "MusesObservation",
    "MusesObservationHandle",
    "MeasurementId",
    "MeasurementIdDict",
    "MeasurementIdFile",
    "SimulatedObservation",
    "SimulatedObservationHandle",
    "MusesObservationHandlePickleSave",
]
