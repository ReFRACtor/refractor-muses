import numpy as np
import re

# We must import all of these because otherwise pyhdf will complain about not finding some of the modules
from pyhdf.HDF import *  # type: ignore
from pyhdf.V import *  # type: ignore
from pyhdf.VS import *  # type: ignore
from pyhdf.SD import *  # type: ignore

import refractor.framework as rf  # type: ignore


class OmiLevel1File(object):
    def __init__(
        self,
        filename,
        along_track_index,
        across_track_indexes,
        filter_invalid_samples=True,
    ):
        # Need to open the HDF object to find vgroups from their names
        # Need to open SD to obtain the needed data
        self.hdf = HDF(filename)
        self.sd = SD(filename)

        # Interfaces into structure of the file
        self.vdata = self.hdf.vstart()
        self.vgroup = self.hdf.vgstart()

        self.along_track_index = along_track_index
        self.across_track_indexes = across_track_indexes

        self.filter_invalid_samples = filter_invalid_samples

        self._initialize_channels()

    def _initialize_channels(self):
        "Initalize which vgroups contain the per channel information"

        # We need to traverse all the vgroups to figure out which are the root ones
        # Assumption here is that the first group found is one of the root groups and
        # that all other root groups have the same class.
        # For OMI UV the root groups are the per channel information.
        # We need to figure this out because each of the channels have the same named
        # datasets making it useless to find vgroup.find since it will always just
        # return the first one.
        root_groups_class = None
        self._channel_refs = []
        self.channel_names = []

        ref = -1
        chan_idx = 0
        while chan_idx < len(self.across_track_indexes):
            try:
                ref = self.vgroup.getid(ref)
                vg = self.vgroup.attach(ref)
                if root_groups_class is None:
                    root_groups_class = vg._class
                if vg._class == root_groups_class:
                    self._channel_refs.append(ref)
                    self.channel_names.append(vg._name)
                    chan_idx += 1
                vg.detach()
            except HDF4Error:
                break

    def channel_group_ref(self, channel_index, group_name):
        chan_ref = self._channel_refs[channel_index]
        chan_vg = self.vgroup.attach(chan_ref)

        group_ref = None
        for tag, ref in chan_vg.tagrefs():
            if tag == HC.DFTAG_VG:
                group_vg = self.vgroup.attach(ref)
                if group_vg._name == group_name:
                    group_ref = ref
                    break
        if group_ref is None:
            raise LookupError(
                f"Could not find group {group_name} in channel group {chan_vg._name}"
            )

        chan_vg.detach()

        return group_ref

    def channel_data(self, channel_index, group_name, data_name):
        group_ref = self.channel_group_ref(channel_index, group_name)

        group_vg = self.vgroup.attach(group_ref)

        data = None
        for tag, ref in group_vg.tagrefs():
            if tag == HC.DFTAG_NDG:
                sds = self.sd.select(self.sd.reftoindex(ref))
                name, rank, dims, dtype, nattrs = sds.info()
                if name == data_name:
                    data = sds[:]
                    sds.endaccess()
                    break
                else:
                    sds.endaccess()

            elif tag == HC.DFTAG_VH:
                vd = self.vdata.attach(ref)
                nrecs, intmode, fields, size, name = vd.inquire()
                if name == data_name:
                    data = vd[:]

        if data is None:
            raise LookupError(f"Could not find data {data_name} for group {group_name}")

        return data

    def geolocation_data(self, channel_index, geo_name):
        geo_data = self.channel_data(channel_index, "Geolocation Fields", geo_name)

        xtrack_index = self.across_track_indexes[channel_index]

        if type(geo_data) is np.ndarray:
            if len(geo_data.shape) == 1:
                return float(geo_data[self.along_track_index])
            elif len(geo_data.shape) == 2:
                return float(geo_data[self.along_track_index, xtrack_index])
            else:
                raise Exception(f"Uknown data type shape: {geo_data.shape}")
        else:
            return geo_data[self.along_track_index][0]

    def swath_data(self, channel_index, data_name):
        swath_data = self.channel_data(channel_index, "Data Fields", data_name)

        xtrack_index = self.across_track_indexes[channel_index]

        if type(swath_data) is np.ndarray:
            if len(swath_data.shape) >= 3:
                return swath_data[self.along_track_index, xtrack_index, ...]
            elif len(swath_data.shape) == 2:
                return swath_data[self.along_track_index, ...]
            else:
                raise Exception(f"Uknown data type shape: {swatch_data.shape}")
        else:
            return swath_data[self.along_track_index]

    def number_all_sample(self, chan_index: int) -> int:
        "Number of all sample points without any quality filtering"

        # Find dimensions vdata to get number of samples for the band
        ref = self.vdata.find("nWavel:" + self.channel_names[chan_index])

        vd = self.vdata.attach(ref)
        num_samples = vd[0][0]
        vd.detach()

        return num_samples

    def _valid_samples(self, chan_index):
        "These are in the original data order, not the increasing order of sample_grid or _spectral_values"

        return np.where(self.swath_data(chan_index, "PixelQualityFlags") == 0)

    def number_valid_sample(self, chan_index: int):
        return len(self._valid_samples(chan_index)[0])

    def number_channel(self) -> int:
        return len(self.channel_names)

    def _spectral_order(self, chan_index: int) -> int:
        # UV-1 radiances and grid is given in reverse of increasing order
        if re.search("UV-1", self.channel_names[chan_index]):
            return -1
        else:
            return 1

    def _spectral_value(
        self, chan_index: int, dataset_prefix: str, units: str
    ) -> rf.SpectralRange:
        mantissa = self.swath_data(chan_index, f"{dataset_prefix}Mantissa").astype(
            float
        )
        precision_mantissa = self.swath_data(
            chan_index, f"{dataset_prefix}PrecisionMantissa"
        ).astype(float)
        exponent = self.swath_data(chan_index, f"{dataset_prefix}Exponent").astype(int)

        if self.filter_invalid_samples:
            # Ensure that we won't try to raise to a negative power by filtering valid
            # samples before running these computations
            valid_samples = self._valid_samples(chan_index)
            spectral_value = mantissa[valid_samples] * 10 ** exponent[valid_samples]
            uncertainty = (
                precision_mantissa[valid_samples] * 10 ** exponent[valid_samples]
            )
        else:
            spectral_value = mantissa * 10**exponent
            uncertainty = precision_mantissa * 10**exponent

        order = self._spectral_order(chan_index)
        return rf.SpectralRange(spectral_value[::order], units, uncertainty[::order])


class OmiLevel1RadianceFile(rf.Level1bSampleCoefficient, OmiLevel1File):
    def __init__(self, filename, along_track_index, across_track_indexes):
        # This call is essential so the class gets connected to its director class
        rf.Level1bSampleCoefficient.__init__(self)
        OmiLevel1File.__init__(
            self,
            filename,
            along_track_index,
            across_track_indexes,
            filter_invalid_samples=False,
        )

    # Methods needed by rf.Level1b

    def number_spectrometer(self) -> int:
        return self.number_channel()

    def number_sample(self, chan_index: int) -> int:
        return self.number_all_sample(chan_index)

    def bad_sample_mask(self, chan_index):
        return self.swath_data(chan_index, "PixelQualityFlags") != 0

    def latitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(self.geolocation_data(chan_index, "Latitude"), "deg")

    def longitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(self.geolocation_data(chan_index, "Longitude"), "deg")

    def sounding_zenith(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            self.geolocation_data(chan_index, "ViewingZenithAngle"), "deg"
        )

    def sounding_azimuth(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            self.geolocation_data(chan_index, "ViewingAzimuthAngle") % 360, "deg"
        )

    def solar_zenith(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            self.geolocation_data(chan_index, "SolarZenithAngle"), "deg"
        )

    def solar_azimuth(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            self.geolocation_data(chan_index, "SolarAzimuthAngle") % 360, "deg"
        )

    def stokes_coefficient(self, chan_index: int) -> np.ndarray:
        return np.array([1, 0, 0, 0], dtype=float)

    def altitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(
            self.geolocation_data(chan_index, "TerrainHeight"), "m"
        )

    def relative_velocity(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(0, "m/s")

    def time(self, chan_index: int) -> rf.Time:
        # Convert from seconds since 1993-01-01
        secs = self.geolocation_data(chan_index, "Time")
        return rf.Time.time_pgs(secs)

    def spectral_coefficient(self, chan_index: int) -> rf.ArrayWithUnit:
        wave_coeff = self.swath_data(chan_index, "WavelengthCoefficient")
        return rf.ArrayWithUnit(wave_coeff, "nm")

    def spectral_variable(self, chan_index: int) -> np.ndarray:
        wave_ref = self.swath_data(chan_index, "WavelengthReferenceColumn")[0]
        spec_vars = np.arange(0, self.number_sample(chan_index)) - wave_ref
        order = self._spectral_order(chan_index)
        return spec_vars[::order]

    def radiance(self, chan_index: int) -> rf.SpectralRange:
        return self._spectral_value(chan_index, "Radiance", "ph / nm / s / sr")

    def earth_sun_distance(self) -> rf.DoubleWithUnit:
        ref = self.vdata.find("EarthSunDistance")
        vd = self.vdata.attach(ref)
        return rf.DoubleWithUnit(vd[:][0][0], "m")


class OmiLevel1IrradianceFile(OmiLevel1File):
    def __init__(self, filename, across_track_indexes):
        # Only one value in the along track dimension for these files
        along_track_index = 0

        super().__init__(
            filename,
            along_track_index,
            across_track_indexes,
            filter_invalid_samples=True,
        )

    def irradiance(self, chan_index: int) -> rf.SpectralRange:
        return self._spectral_value(chan_index, "Irradiance", "ph / nm / s")

    def sample_grid(self, chan_index: int) -> rf.SpectralDomain:
        wave_coeff = self.swath_data(chan_index, "WavelengthCoefficient")
        wave_ref = self.swath_data(chan_index, "WavelengthReferenceColumn")[0]
        n_all_samp = self.number_all_sample(chan_index)

        wavelength = np.zeros(n_all_samp)
        for samp_idx in range(n_all_samp):
            for coeff_idx, coeff_val in enumerate(wave_coeff):
                wavelength[samp_idx] += coeff_val * (samp_idx - wave_ref) ** coeff_idx

        if self.filter_invalid_samples:
            valid_samples = self._valid_samples(chan_index)
            wavelength = wavelength[valid_samples]

        order = self._spectral_order(chan_index)
        return rf.SpectralDomain(rf.ArrayWithUnit(wavelength[::order], "nm"))


class OmiLevel1Reflectance(OmiLevel1RadianceFile):
    def __init__(
        self, l1b_filename, along_track_index, across_track_indexes, solar_models
    ):
        super().__init__(l1b_filename, along_track_index, across_track_indexes)
        self.solar_models = solar_models

    def radiance(self, chan_index: int) -> rf.SpectralRange:
        earth_shine_rad = super().radiance(chan_index)
        earth_shine_grid = super().sample_grid(chan_index)

        solar_irrad = self.solar_models[chan_index].solar_spectrum(earth_shine_grid)

        norm_rad = earth_shine_rad.data / solar_irrad.spectral_range.data
        norm_uncert = earth_shine_rad.uncertainty / solar_irrad.spectral_range.data

        return rf.SpectralRange(norm_rad, rf.Unit("sr^-1"), norm_uncert)
