from __future__ import annotations
from abc import ABC, abstractmethod
import datetime as dt
from enum import Enum
import os
from glob import glob
from pathlib import Path
import netCDF4 as ncdf
import h5py  # type: ignore
import numpy as np
import scipy
from scipy.io import readsav
from .misc import greatcircle
from .tes_file import TesFile
from loguru import logger
from typing import Any, cast, Iterator

# This is complicated enough that we have copied this over from muses-py, and edited things
# to work with muses. We will probably just leave this in place, the code is pretty clean.
# We rely on muses_py calls for the version 1 support. I don't think we'll run into this
# much in practice, so it isn't worth porting this over. We can revisit this if needed.


class CamelError(Exception):
    """Custom error to distinguish CAMEL-specific problems from Python errors"""

    pass


class CamelLandFlagError(CamelError):
    """Custom error specific to cases where CAMEL cannot find a close enough land grid cell"""

    pass


class UwisCamelOptions(Enum):
    V1_UWIS = "UWIS"
    V2_CAMEL = "CAMELv2"
    V2_CAMEL_CLIM = "CAMELv2-CLIM"

    @classmethod
    def emis_source_matches(cls, emis_src: str) -> bool:
        """Check if a given emissivity source is one of the recognized values"""
        our_values = [v.value for v in cls.__members__.values()]
        return emis_src in our_values

    @classmethod
    def emis_source_citation(cls, emis_src: str) -> str:
        if not cls.emis_source_matches(emis_src):
            raise CamelError(f"Unknown emission source: {emis_src}")

        if emis_src == cls.V1_UWIS.value:
            return (
                "University of Wisconsin MODIS Infrared Emissivity Database. See: Borbas et al., "
                "A high spectral resolution global land surface infrared emissivity database. "
                "Joint 2007 EUMETSAT Meteorological Satellite Conference and the 15th Satellite Meteorology & "
                "Oceanography Conference of the American Meteorological Society, Amsterdam, The Netherlands, "
                "24-28 September 2007"
            )
        elif emis_src == cls.V2_CAMEL.value:
            return (
                "The Combined ASTER MODIS Emissivity over Land (CAMEL) Database. See: Borbas et al. 2018, "
                "https://doi.org/10.3390/rs10040643"
            )
        elif emis_src == cls.V2_CAMEL_CLIM.value:
            return (
                "Climatology of the Combined ASTER MODIS Emissivity over Land (CAMEL) Database. See: Loveless et al. 2020, "
                "https://doi.org/10.3390/rs13010111"
            )
        else:
            return emis_src

    def get_native_hsr_length(self) -> int:
        if self == self.V1_UWIS:
            # This is the number of data points in the land matices read from the save file
            # by get_emis_uwis() as of 2022-01-20. If this is something that might change
            # (i.e. if someone is swapping in different save files) then this will need
            # to be gotten programmatically.
            return 416
        else:
            return AbstractCamelHsr.num_spectral_points


def get_emis_dispatcher(
    emis_type: str | UwisCamelOptions,
    i_latitude: float,
    i_longitude: float,
    i_altitude: float,  # unused currently, available for future
    i_oceanFlag: int,
    i_year: int,
    i_month: int,
    freq: np.ndarray,
    osp_dir: Path,
    camel_coef_dir: str | os.PathLike[str] | None = None,
    camel_lab_dir: str | os.PathLike[str] | None = None,
) -> dict:
    """Primary entry point for this module: gets the surface emissivity from the desired database.

    Parameters
    ----------
    emis_type
        Which source to use for the surface emissivity. Can be either an instance of the :py:class:`UwisCamelOptions`
        enum or one or the strings "UWIS", "CAMELv2", or "CAMELv2-CLIM". "UWIS" uses the version 1 University
        of Wisconsin surface emissivity. "CAMELv2" uses the updated surface emissivity database combining MODIS
        and ASTER data (see https://doi.org/10.3390/rs10040643). "CAMELv2-CLIM" uses the climatological version
        of the MODIS+ASTER emissivity (see https://doi.org/10.3390/rs13010111). Note that "CAMELv2" and
        "CAMELv2-CLIM" both require ``camel_coef_dir`` and ``camel_lab_dir`` be provided.

    i_latitude, i_longitude
        Latitude and longitude of the sounding for which emissivity is needed. South and west, respectively, should
        be given as negative values.

    i_altitude
        Altitude of the sounding for which emissivity is needed. Required for water soundings (to determine if
        ocean or freshwater).

    i_oceanFlag
        Indicates whether this sounding is a land (0) or ocean (1) sounding.

    i_year, i_month
        The year and month for which emissivity is needed, as integers.

    freq
        Frequency grid (in wavenumbers) to output the emissivity on. The emissivity will be interpolated from its
        native resolution to this grid.

    camel_coef_dir
        Path to the directory containing coefficients for the CAMEL emissivity. For ``emis_type == "CAMELv2"``, these
        have the pattern ``yyyy/CAM5K30CF_coef_yyyymm_V002.nc``, where ``yyyy`` is the year and ``mm`` the month. Note
        that this pattern indicates that the files are organized into subdirectories by year. These files can be
        downloaded from the LP DAAC at https://lpdaac.usgs.gov/products/cam5k30cfv002/ (as of 20 Jan 2022).

        For ``emis_type == "CAMELv2-CLIM", these have the pattern ``CAMEL_coef_climatology_mmMonth_V002.nc`` where
        ``mm`` is the month. These files were available at ftp://ftp.ssec.wisc.edu/pub/ICI/MEASURES_V002/climatology
        as of 1 Nov 2021, provided by Eva Borbas.

        This input is not required if ``emis_type == "UWIS"``

    camel_lab_dir
        Path to the directory containing the lab spectra eigenvector files for the CAMEL high spectral resolution
        database. These files have the pattern ``pchsr_vX.2.nc`` where ``X`` is a number between 8 and 12. These
        are available in the source code hosted on the LP DAAC at https://lpdaac.usgs.gov/products/cam5k30cfv002/
        under the "Documentation" tab (as of 20 Jan 2022). In the source code, these files are located in the
        ``Fortran/MEASURES_CAMEL_hsremis_lib_V002_v02r01_del_10092018/coef`` subdirectory.

        This input is not required if ``emis_type == "UWIS"``.


    Returns
    -------
    dict
        This dictionary contains the keys:

            * "instr_emis" - the emissivity on the ``freq`` frequency grid.
            * "native_emis" - the emissivity on the native frequency grid for the given ``emis_type``.
            * "native_wavenumber" - the native frequency grid corresponding to "native_emis".
            * "dist_to_tgt" - the distance in kilometers between the CAMEL grid cell from which the
              emissivity was taken and the given latitude & longitude. For ``emis_type == "UWIS"``, this
              will be a NaN.
    """
    # Frequency *may* have been passed as a tuple to get around the unhashability of numpy arrays for
    # caching, so convert it back
    if isinstance(freq, tuple):
        freq = np.array(freq)

    # Rely on the value checking of Enums to ensure that we have a valid emissivitiy type
    emis_type = UwisCamelOptions(emis_type)
    if emis_type in (
        UwisCamelOptions.V2_CAMEL,
        UwisCamelOptions.V2_CAMEL_CLIM,
    ) and (camel_coef_dir is None or camel_lab_dir is None):
        raise TypeError(
            '`camel_coef_dir` and `camel_lab_dir` are required if `emis_type` is "CAMELv2" or "CAMELv2-CLIM"'
        )

    if emis_type == UwisCamelOptions.V1_UWIS:
        native_emis = get_emis_uwis(
            i_latitude, i_longitude, i_year, i_month, osp_dir
        )  # need to update with altitude
    elif i_oceanFlag == 1:
        # The CAMEL databases do not include emissivity over ocean. This should be identical to how ocean/lake
        # emissivity is gotten in UWis (v1).
        native_emis = get_water_emis(i_altitude, osp_dir)
    elif emis_type == UwisCamelOptions.V2_CAMEL:
        try:
            assert camel_lab_dir is not None
            assert camel_coef_dir is not None
            camel_interface: AbstractCamelHsr = CamelTimeseriesHsr(
                camel_lab_dir, camel_coef_dir
            )
            native_emis = camel_interface.compute_emissivity_for_lat_lon(
                tgt_lat=i_latitude, tgt_lon=i_longitude, year=i_year, month=i_month
            )
        except CamelLandFlagError:
            logger.info(
                f"Could not find CAMEL emissivity near lat = {i_latitude}, lon = {i_longitude}, defaulting to ocean emissivity"
            )
            # This isn't quite ideal; `get_water_emis` differentiates between ocean and lake emissivity based on altitude
            # so if we have an elevated sounding over ocean, it might get treated as lake. However, if that's the case, we
            # are probably also treating a land sounding as water, so which water type we use isn't the largest concern.
            # Some test cases (e.g. test_targets/cris/20160818_009_42_05_5) fail with a CAMEL search radius of 10, so we
            # need this fallback. -JLL
            native_emis = get_water_emis(i_altitude, osp_dir)

    elif emis_type == UwisCamelOptions.V2_CAMEL_CLIM:
        try:
            assert camel_lab_dir is not None
            assert camel_coef_dir is not None
            camel_interface = CamelClimatologyHsr(camel_lab_dir, camel_coef_dir)
            native_emis = camel_interface.compute_emissivity_for_lat_lon(
                tgt_lat=i_latitude, tgt_lon=i_longitude, year=i_year, month=i_month
            )
        except CamelLandFlagError:
            logger.info(
                f"Could not find CAMEL emissivity near lat = {i_latitude}, lon = {i_latitude}, defaulting to ocean emissivity"
            )
            # See note above in the V2_CAMEL except block.
            native_emis = get_water_emis(i_altitude, osp_dir)
    else:
        raise NotImplementedError(
            f'Emissivity type "{emis_type.value}" has not been implemented'
        )

    instr_emis = scipy.interpolate.interp1d(
        native_emis["wavenumber"], native_emis["emissivity"], fill_value="extrapolate"
    )(freq)
    dist_to_tgt = native_emis.pop("camel_offset_distance_km", np.nan)
    ind = freq < np.amin(native_emis["wavenumber"])
    if ind.sum() > 0:
        instr_emis[ind] = native_emis["emissivity"][0]

    ind = instr_emis > 0.99
    if ind.sum() > 0:
        instr_emis[ind] = 0.99

    # Pad the native emissivity variables if needed so that land and ocean soundings will always
    # have the same native emissivity dimension. Otherwise py-combine will be very unhappy!
    if np.size(native_emis["emissivity"]) != np.size(native_emis["wavenumber"]):
        raise CamelError(
            "Native emissivity and its corresponding wavenumber are different size arrays!"
        )
    elif np.size(native_emis["emissivity"]) < emis_type.get_native_hsr_length():
        n = emis_type.get_native_hsr_length()
        m = np.size(native_emis["emissivity"])
        pad_length = n - m
        logger.info(
            f"Padding native emissivity with {pad_length} fill values to keep standard length of {n} for {emis_type.value}"
        )
        native_emis["emissivity"] = np.concatenate(
            [
                native_emis["emissivity"],
                np.full(pad_length, -999.0, dtype=native_emis["emissivity"].dtype),
            ]
        )
        native_emis["wavenumber"] = np.concatenate(
            [
                native_emis["wavenumber"],
                np.full(pad_length, -999.0, dtype=native_emis["wavenumber"].dtype),
            ]
        )

    return {
        "instr_emis": instr_emis,
        "native_emis": native_emis["emissivity"],
        "native_wavenumber": native_emis["wavenumber"],
        "dist_to_tgt": dist_to_tgt,
    }


# -------------------------------------------- #
# V2 CAMEL/CAMEL CLIMATOLOGY EMISSIVITY LOOKUP #
# -------------------------------------------- #


class CamelIndexTransform:
    """A class that handles converting 2D lat/lon or indices into the linear indices used by the CAMEL coefficient files

    To save disk space, the CAMEL coefficient files only store data from land grid cells
    (where there is actually emissivity information) and so unravels the 2D grid into a 1D
    vector of land cells. This class provides methods to convert between lat/lon and the
    linear indices in the coefficient file.

    Parameters
    ----------
    camel_ds:
        A handle to the netCDF coefficient file

    flag_var:
        The name of the variable in ``camel_ds`` that is the land/ocean flag.
        Note that we expect values of 0 to indicate water and >= 1 to indicate
        land.
    """

    def __init__(self, camel_ds: ncdf.Dataset, flag_var: str):
        logger.info("Instantiating CAMEL index transform")
        self._grid_lat = camel_ds["latitude"][:]
        self._grid_lon = camel_ds["longitude"][:]

        qual_flag = camel_ds[flag_var][:].astype(
            "int"
        )  # the climatology has a scale factor, so this becomes a float
        xx_land = qual_flag > 0
        qual_flag_cumulative = (
            np.cumsum(xx_land).reshape(xx_land.shape) - 1
        )  # adjust back to 0-based indices

        self._lin_indices = np.full(qual_flag.shape, -1)
        self._lin_indices[xx_land] = qual_flag_cumulative[xx_land]
        logger.info("Finished instantiating CAMEL index transform")

    def get_grid_indices(self, tgt_lat: float, tgt_lon: float) -> tuple[int, int]:
        """
        Calculate the 2D grid indices that correspond to a given latitude and longitude.

        Parameters
        ----------
        tgt_lat:
            The latitude to find the closest grid cell to (south is negative).

        tgt_lon:
            The longitude to find the closest grid cell to (west is negative).

        Returns
        -------
        ilat : int
            The index of the grid cell along the latitudinal dimension.

        ilon : int
            The index of the grid cell along the longitudinal dimension.
        """
        ilat = np.argmin(np.abs(self._grid_lat - tgt_lat))
        ilon = np.argmin(np.abs(self._grid_lon - tgt_lon))
        return int(ilat), int(ilon)

    def get_linear_index_from_grid_indices(self, ilat: int, ilon: int) -> int:
        """
        Calculate the linear index in the coefficient files that corresponds to given 2D grid indices

        Parameters
        ----------
        ilat
            The grid array index in the latitudinal dimension

        ilon
            The grid array index in the longitudinal dimension

        Returns
        -------
        idx : int
            The linear index in the coefficient file.

        Raises
        ------
        CamelError
            If the given grid indices point to an ocean grid cell, a :class:`CamelError` is
            raised.
        """
        idx = self._lin_indices[ilat, ilon]
        if idx < 0:
            raise CamelError(
                f"ilat = {ilat}, ilon = {ilon} is a water grid cell, no CAMEL data exists"
            )
        return idx

    def get_linear_index_from_lat_lon(
        self, tgt_lat: float, tgt_lon: float, fuzzy: bool = True
    ) -> tuple[int, float]:
        """
        Calculate the linear index in the coefficient file that corresponds to a given latitude and longitude

        Parameters
        ----------
        tgt_lat:
            The latitude to find the closest grid cell to (south is negative).

        tgt_lon:
            The longitude to find the closest grid cell to (west is negative).

        fuzzy:
            Whether to search for the nearest land grid cell (``True``, default) or require
            that the grid cell closest to the target lat/lon is a land grid cell (``False``).
            If this is ``True``, it will only search out to a limited radius, it will not search
            the entire globe.

        Returns
        -------
        idx : int
            The linear index in the coefficient file.

        gc_distance : float
            The great circle distance (in kilometers) between the target lat/lon and the
            selected CAMEL grid cell.

        Raises
        ------
        CamelError
            If the given grid indices point to an ocean grid cell, a :class:`CamelError` is
            raised.
        """
        if fuzzy:
            ilat, ilon = self.get_grid_indices_nearest_lat_lon(tgt_lat, tgt_lon)
        else:
            ilat, ilon = self.get_grid_indices(tgt_lat, tgt_lon)

        grid_lat = self._grid_lat[ilat]
        grid_lon = self._grid_lon[ilon]
        gc_distance = (
            greatcircle(tgt_lat, tgt_lon, grid_lat, grid_lon) * 1e-3
        )  # convert meters to kilometers
        idx = self.get_linear_index_from_grid_indices(ilat, ilon)
        return idx, gc_distance

    def get_grid_indices_nearest_lat_lon(
        self, tgt_lat: float, tgt_lon: float, search_grid_radius: int = 10
    ) -> tuple[int, int]:
        """
        Find the linear index for the CAMEL land grid cell closest to a given lat lon
        """
        ilat, ilon = self.get_grid_indices(tgt_lat, tgt_lon)
        if self._lin_indices[ilat, ilon] >= 0:
            # Simplest case - the index we would normally grab
            # is a valid one
            return ilat, ilon

        # If we didn't find a valid grid cell, we look at all the grid cells
        # out to the allowed radius, and find the closest one. I'm iterating
        # over the cells rather than doing a logical operation because this
        # makes it easier to wrap lat/lon when we're near the array edge.
        closest_sq_dist_deg = np.inf
        ilat_final: None | int = None
        ilon_final: None | int = None
        for jlat, jlon, _, _ in self.iter_subgrid_indices(
            ilat,
            ilon,
            search_grid_radius,
            search_grid_radius,
            self._grid_lat.size,
            self._grid_lon.size,
        ):
            if self._lin_indices[jlat, jlon] >= 0:
                sq_dist_deg = (self._grid_lat[jlat] - tgt_lat) ** 2 + (
                    self._grid_lon[jlon] - tgt_lon
                ) ** 2
                if sq_dist_deg < closest_sq_dist_deg:
                    closest_sq_dist_deg = sq_dist_deg
                    ilat_final = jlat
                    ilon_final = jlon

        # It's possible that we *still* haven't found a valid land pixel, if for
        # some reason we're trying to do a land retrieval in the middle of the Pacific,
        # for example. In which case, we should just stop.
        if ilat_final is None:
            raise CamelLandFlagError(
                f"No valid land grid cell found within {search_grid_radius} cells of lat = {tgt_lat}, lon = {tgt_lon}"
            )

        logger.info(
            f"Selected lat = {self._grid_lat[ilat_final]}, lon = {self._grid_lon[ilon_final]} for target lat = {tgt_lat}, lon = {tgt_lon}"
        )
        assert ilat_final is not None
        assert ilon_final is not None
        return ilat_final, ilon_final

    def iter_subgrid_indices(
        self,
        center_lon_idx: int,
        center_lat_idx: int,
        radius_lon: int,
        radius_lat: int,
        size_lon: int,
        size_lat: int,
    ) -> Iterator[tuple[int, int, int, int]]:
        """Index iterator to select a subset from an equirectangular
        lat/lon grid

        This will yield the indices for a rectangular subset of cells
        within a larger equirectangular (i.e. constant lat/lon across
        rows/columns) grid. If the subset runs into the edge of the
        parent grid, it will wrap. In the longitudinal direction, the
        index will just move around to the other side (i.e. if behaves
        as if the index goes N-2, N-1, 0, 1 for an N-long
        dimension). In the latitudinal direction, it assumes that we
        are going over the pole, and so the next grid will be the one
        at the same latitude but 180 deg away in longitude.

        Parameters
        ----------
        center_lon_idx, center_lat_idx
            The array indices in the longitude and latitude
            dimensions, respectively, of the parent array that define
            the center of the rectangular subset.

        radius_lon, radius_lat
            The number of grid cells away from the center one to
            traverse in the longitude and latitude dimensions. Giving
            2 and 1, respectively, for these will produce a 5x3
            subset.

        size_lon, size_lat
            The length of th parent array in the longitude and
            latitude dimensions.

        Returns
        -------
        iterator
            Returns four integers in each iteration: `i_grid`,
            `j_grid`, `i_sub`, and `j_sub`.  The `i_` values are
            longitude indices, `j_` are latitude indices. The `grid`
            indices index the parent grid, the `sub` ones index the
            subset. That is, assuming your arrays have longitude in
            the first dimension, `sub[i_sub, j_sub] = parent[i_grid,
            j_grid]`.

        """
        if center_lon_idx >= size_lon or center_lon_idx < 0:
            raise ValueError(
                "Center longitude index cannot be outside the range [0, size_lon)"
            )
        if center_lat_idx >= size_lat or center_lat_idx < 0:
            raise ValueError(
                "Center latitude index cannot be outside the range [0, size_lat)"
            )

        # In the loops:
        #   - i/j are internal indices that control how far from the center we are
        #   - i_sub/j_sub are the indices in the subset array we're looping over
        #   - i_grid/j_grid are the indices in the original (full) array we're subsetting
        for i, i_sub in zip(
            range(center_lon_idx - radius_lon, center_lon_idx + radius_lon + 1),
            range(2 * radius_lon + 1),
        ):
            # The longitude index is easy - since longitude wraps 360
            # -> 0, any time we are outside the range of allowed
            # indices, the modulus operator wraps us around like
            # crossing the date line
            i_grid = i % size_lon

            for j, j_sub in zip(
                range(center_lat_idx - radius_lat, center_lat_idx + radius_lat + 1),
                range(2 * radius_lat + 1),
            ):
                # Latitude is trickier to wrap, because we're going
                # over the poles, so we don't actually go from max
                # index to min index. Instead we need to get the box
                # over the pole, same latitude, but 180 deg.  opposite
                # longitude. Thus, we do want to repeat the 0 or
                # (size_lat - 1) latitude index when it wraps.
                #
                # Note that the way I'm finding the box 180 away in
                # longitude is an estimate; it might be a little off
                # depending on the integer rounding. However, for the
                # box nearest the pole, those boxes are small little
                # wedges, so the error shouldn't be too large, and if
                # we get far enough away from it for the rounding to
                # matter, we're so far from our target lat/lon that
                # the distance error is already pretty large.
                j_grid = center_lat_idx + j
                if j < 0:
                    j_grid = -j - 1
                    i_grid2 = (i_grid + size_lon // 2) % size_lon
                elif j >= size_lat:
                    j_grid = size_lat - (j - size_lat + 1)
                    i_grid2 = (i_grid + size_lon // 2) % size_lon
                else:
                    j_grid = j
                    i_grid2 = i_grid

                yield i_grid2, j_grid, i_sub, j_sub


class AbstractCamelHsr(ABC):
    """
    Top level class containing common functionality for both the yearly-resolved and climatological CAMEL HSR lookup

    There are two options for the CAMEL high spectra resolution (HSR) emissivity. The regular CAMEL
    database contains unique emissivity for each year and month through ~2016. The climatological database
    averages across years, thus providing values per month only. Calculating the HSR emissivity from each of
    these requires different code to to sum together the emissivity eigenvectors, yet the rest of the
    functionality is common to both. This class hold the common functionality; inherit from it and implement
    the :py:meth:`_compute_emissivity_from_pcs` method to provide a full implementation. You will also likely
    need to override the ``__init__`` method to take in the data files required by that particular implementation.


    Parameters
    ----------
    lab_data_dir
        Path to the directory containing the :file:`pchsr_vX.Y.nc` files which have the "eigenvalues" and
        eigenvectors of the lab spectra used to regenerate the HSR emissivity.

    coef_dir
        Path to the directory containing the coefficient files.
    """

    num_spectral_points = 417

    def __init__(
        self, lab_data_dir: str | os.PathLike[str], coef_dir: str | os.PathLike[str]
    ):
        logger.info(f"Instantiating {self.__class__.__name__}")
        self._lab_dir = Path(lab_data_dir)
        self._lab_data: dict[int, dict[str, np.ndarray]] = dict()
        self._coef_dir = Path(coef_dir)

    def _lab_file(self, lab_version: int) -> Path:
        """Return the path to a lab spec file for a given "version" """
        return self._lab_dir / f"pchsr_v{lab_version}.2.nc"

    def _load_lab_spectra(self, lab_version: int) -> dict:
        """Load the lab spectra for a specific "version". Caches the data to be faster on subsequent calls."""
        if lab_version in self._lab_data:
            return self._lab_data[lab_version]
        else:
            with ncdf.Dataset(self._lab_file(lab_version)) as ds:
                self._lab_data[lab_version] = {
                    "eigenvectors": ds["Eigenvectors"][:],
                    "eigenvalues": ds["Eigenvalues"][:],
                }

            return self._lab_data[lab_version]

    def _sum_eigens(self, lab_version: int, pc_coefficients: np.ndarray) -> np.ndarray:
        """Calculate the total HSR emissivity from the individual lab spectra eigenvectors

        Parameters
        ----------
        lab_version
            Which "version" of the lab data is required.

            ..note::
            I put "version" in quotes because it is the term used by the U. Wisconsin team,
            but I find it a little confusing. Each of the five files under the ``lab_data_dir``
            required by this class's ``__init__`` method is called one "version" of the lab
            spectra, despite the fact that all of them belong to the version 2 CAMEL product.
            Additionally, different grid cells will use different files depending on which one
            provides the best information for the land type in that grid cell.

            A better term for the five difference files would probably have been "collection"
            rather than "version," but I'll use "version" to be consistent with the CAMEL papers
            and documentation.

        pc_coefficients
            The array of coefficients to multiply the spectral eigenvectors by to reconstitute
            the high resolution spectrum. These will have been read from the coefficient files.

            Note that these arrays will *always* have fewer elements than there are eigenvectors
            in the lab files. For an array of N coefficients, only the first N eigenvectors will
            be summed.

        Returns
        -------
        emis : np.ndarray
            The HSR emissivity on the native frequency grid of the HSR database.
        """
        eigens = self._load_lab_spectra(lab_version)

        # Scale the eigenvectors by the coefficients, and add in the eigenvectors.
        # The tricky part is these are square matrices, so it's a little harder to confirm that
        # I've got the indexing correct. In `recon_hsriemis`, the first dimension
        # is the component and the second the wavenumber (high to low). However, that's
        # fortran ordering, so I'm guessing that I need to flip the dimensions in Python.
        npcs = pc_coefficients.size
        eigenvalues = eigens["eigenvalues"][:]
        eigenvectors = eigens["eigenvectors"][:, :npcs]

        pc_coefficients = np.broadcast_to(
            pc_coefficients.reshape(1, -1), eigenvectors.shape
        )
        emis = np.sum(eigenvectors * pc_coefficients, axis=1) + eigenvalues

        # Limit emis to 1. Matches what is done in `recon_hsriremis.f`, though
        # I'm curious about the units for emission such that it should be capped
        # at 1.
        emis = np.where(emis < 1, emis, 1)
        return emis

    @classmethod
    def get_hsr_wavenumbers(cls) -> np.ndarray:
        """Get the native frequency grid (in wavenumbers) of the HSR emissivities"""
        hsr_wavenums = 698 + np.arange(cls.num_spectral_points) * 5
        return hsr_wavenums.astype("double")

    @classmethod
    def get_hsr_wavelengths(cls) -> np.ndarray:
        """Get the native wavelength grid (in micrometers) of the HSR emissisivities"""
        wns = cls.get_hsr_wavenumbers()
        return 1 / wns * 1e6 / 1e2

    @abstractmethod
    def get_coef_file(self, year: int, month: int) -> Path:
        """
        Return the path to the coefficient netCDF file for a given year and month
        """
        pass

    @abstractmethod
    def _compute_emissivity_from_pcs(
        self, linear_index: int, year: int, month: int
    ) -> np.ndarray:
        """
        Compute the HSR emissivities for a given grid cell

        Parameters
        ----------
        linear_index
            The index (along the land grid cell dimension) in the coefficient files from
            which to read the coefficients. This can be obtained from a
            :class:`CamelIndexTransform` instance.

        Returns
        -------
        emis : np.ndarray
            The emissivities on the native frequency grid.
        """
        pass

    @abstractmethod
    def compute_emissivity_for_lat_lon(
        self, tgt_lat: float, tgt_lon: float, year: int, month: int, **kwargs: Any
    ) -> dict[str, Any]:
        pass

    def compute_emissivity(
        self,
        linear_index: int,
        year: int,
        month: int,
        spectral_index: str = "wavenumber",
    ) -> dict[str, Any]:
        """
        Compute the HSR emissivities for a given grid cell.

        Parameters
        ----------
        linear_index
            The index (along the land grid cell dimension) in the coefficient files from
            which to read the coefficients. This can be obtained from a
            :class:`CamelIndexTransform` instance.

        spectral_index
            Controls whether the coordinate for the spectrum are in wavenumbers (default)
            or micrometers. Any of the strings "wavenumber", "wn", "cm-1", or "cm1" will
            return wavenumbers, while the strings "wavelength", "wl", or "um" return
            micrometers.

        Returns
        -------
        emis : dict
            A dictionary with two keys: "emissivity" has the HSR emissivitity data, while
            the other will be either "wavenumber" or "wavelength" (depending on which units
            were requested with ``spectral_index``) and has the coordinates for the emissivity.
        """
        logger.info("Computing HSR emissivity from CAMEL database")
        wn_strs = ("wavenumber", "wn", "cm-1", "cm^-1")
        wl_strs = ("wavelength", "wl", "um")

        if spectral_index in wn_strs:
            x = self.get_hsr_wavenumbers()
            index_name = "wavenumber"
        elif spectral_index in wl_strs:
            x = self.get_hsr_wavelengths()
            index_name = "wavelength"
        else:
            wn_strs2 = '", "'.join(wn_strs)
            wl_strs2 = '", "'.join(wl_strs)
            allowed = f'"{wn_strs2}", "{wl_strs2}"'
            raise TypeError(f"Only allowed values for `spectral_index` are {allowed}")

        y = self._compute_emissivity_from_pcs(linear_index, year, month)
        return {index_name: x, "emissivity": y}

    def _emissivity_from_lat_lon_common(
        self,
        tgt_lat: float,
        tgt_lon: float,
        year: int,
        month: int,
        index_trans: CamelIndexTransform,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Compute the emissivity at a given latitude and longitude

        Parameters
        ----------
        tgt_lat
            The target latitude

        tgt_lon
            The target longitude

        index_trans
            An instance of :py:class:`CamelIndexTransform` that will be used to convert the given
            lat/lon into the linear index in the coefficient files.

        kwargs
            Additional keyword arguments to pass through to :py:meth:`compute_emissivity`.

        Returns
        -------
        emis : dict
            A dictionary with three keys: "emissivity" has the HSR emissivitity data,
            "camel_offset_distance_km" is the distance in kilometers between the target
            lat/lon and the CAMEL grid cell the emissivity is taken from, and the last
            will be either "wavenumber" or "wavelength" (depending on which units were
            requested with ``spectral_index``) and has the coordinates for the emissivity.


        """
        lin_idx, tgt_dist = index_trans.get_linear_index_from_lat_lon(tgt_lat, tgt_lon)
        emis = self.compute_emissivity(lin_idx, year, month, **kwargs)
        emis["camel_offset_distance_km"] = tgt_dist
        return emis


class CamelTimeseriesHsr(AbstractCamelHsr):
    """
    A concrete implementation of :py:class:`AbstractCamelHsr` that computes emissivities from the year-resolved CAMEL coefficients.

    Parameters
    ----------
    lab_data_dir
        Path to the directory containing the :file:`pchsr_vX.Y.nc` files which have the "eigenvalues" and
        eigenvectors of the lab spectra used to regenerate the HSR emissivity.

    coef_dir
        Path to the directory containing the coefficient files.
    """

    # I made these class variables to avoid rewriting the dates each place they're needed
    _FIRST_DATE = dt.date(2000, 4, 1)
    _LAST_DATE = dt.date(2016, 12, 1)

    def get_coef_file(self, year: int, month: int) -> Path:
        this_date = dt.date(year, month, 1)
        if this_date < self._FIRST_DATE:
            year = (
                self._FIRST_DATE.year
                if month >= self._FIRST_DATE.month
                else self._FIRST_DATE.year + 1
            )  # CAMEL only available from Apr 2000 on
            logger.warning(
                f"CAMEL data not available before {self._FIRST_DATE}, using data from {year:04d}-{month:02d}"
            )
        elif this_date > self._LAST_DATE:
            year = (
                self._LAST_DATE.year
                if month <= self._LAST_DATE.month
                else self._FIRST_DATE.year - 1
            )
            logger.warning(
                f"CAMEL data not available after {self._LAST_DATE}, using data from {year:04d}-{month:02d}"
            )
        return (
            self._coef_dir / f"{year:04d}/CAM5K30CF_coef_{year:04d}{month:02d}_V002.nc"
        )

    def compute_emissivity_for_lat_lon(
        self, tgt_lat: float, tgt_lon: float, year: int, month: int, **kwargs: Any
    ) -> dict[str, Any]:
        with ncdf.Dataset(self.get_coef_file(year, month)) as coef_ds:
            index_transform = CamelIndexTransform(coef_ds, "camel_qflag")

        return self._emissivity_from_lat_lon_common(
            tgt_lat=tgt_lat,
            tgt_lon=tgt_lon,
            year=year,
            month=month,
            index_trans=index_transform,
            **kwargs,
        )

    def _compute_emissivity_from_pcs(
        self, linear_index: int, year: int, month: int
    ) -> np.ndarray:
        with ncdf.Dataset(self.get_coef_file(year, month)) as coef_ds:
            npcs = coef_ds["pc_npcs"][linear_index].item()
            lab_version = coef_ds["pc_labvs"][linear_index].item()
            pc_coefs = coef_ds["pc_coefs"][linear_index, :npcs].filled(np.nan)

        return self._sum_eigens(lab_version, pc_coefs)


class CamelClimatologyHsr(AbstractCamelHsr):
    """
    A concrete implementation of :py:class:`AbstractCamelHsr` that computes emissivities from the climatological CAMEL coefficients.

    Parameters
    ----------
    lab_data_dir
        Path to the directory containing the :file:`pchsr_vX.Y.nc` files which have the "eigenvalues" and
        eigenvectors of the lab spectra used to regenerate the HSR emissivity.

    coef_ds
        An open handle to the appropriate netCDF dataset containing HSR coefficients for the desired month.
    """

    def get_coef_file(self, year: int, month: int) -> Path:
        """
        Return the path to the coefficient netCDF file for a given year and month
        """
        coef_file = f"CAMEL_coef_climatology_{month:02d}Month_V002.nc"
        return self._coef_dir / coef_file

    def compute_emissivity_for_lat_lon(
        self, tgt_lat: float, tgt_lon: float, year: int, month: int, **kwargs: Any
    ) -> dict[str, Any]:
        with ncdf.Dataset(self.get_coef_file(year, month)) as coef_ds:
            index_transform = CamelIndexTransform(coef_ds, "landflag")

        return self._emissivity_from_lat_lon_common(
            tgt_lat=tgt_lat,
            tgt_lon=tgt_lon,
            year=year,
            month=month,
            index_trans=index_transform,
            **kwargs,
        )

    def _compute_emissivity_from_pcs(
        self, linear_index: int, year: int, month: int, debug_info: dict | None = None
    ) -> np.ndarray:
        total_emis: np.ndarray | None = None
        with ncdf.Dataset(self.get_coef_file(year, month)) as coef_ds:
            # These are small and we need the whole array for each of them
            #
            # Unlike the timeseries CAMEL database, which only uses 1 set of lab eigenvectors
            # per grid cell, the climatology may combine multiple. Thus the variable of
            # coefficients has coefficients for every possible set of eigens, and we need to
            # be able to parse them apart.
            set_npcs = coef_ds["npcs_of_coef_set"][:].astype("int")
            set_indices = np.zeros(set_npcs.size + 1, dtype="int")
            set_indices[1:] = np.cumsum(set_npcs)
            set_labvs = coef_ds["labvs_of_coef_set"][:].astype("int")

            for iset, lab_version in enumerate(set_labvs):
                i1, i2 = set_indices[[iset, iset + 1]]
                set_coefs = coef_ds["pc_coefs"][linear_index, i1:i2]
                assert set_coefs.size == set_npcs[iset], (
                    f"Mismatch in number of coefficients for lab v{lab_version}, set index {iset}"
                )

                if np.all(set_coefs.mask):
                    # All masked (i.e. fill) values indicate that this lab set
                    # doesn't contribute to the requested grid cell
                    continue

                # Ensure that any fill values in the coefficients do not contribute
                # to the emissivity
                set_coefs = set_coefs.filled(0.0)
                set_weight = coef_ds["pc_coef_weights"][linear_index, iset]

                this_emis = self._sum_eigens(lab_version, set_coefs)
                if total_emis is None:
                    total_emis = set_weight * this_emis
                else:
                    total_emis += set_weight * this_emis

                if debug_info is not None:
                    lab_db_dict = debug_info.setdefault(
                        f"lab{lab_version}#{set_coefs.size}", dict()
                    )
                    lab_db_dict["emis"] = this_emis
                    lab_db_dict["weight"] = set_weight
        assert total_emis is not None
        return total_emis


def get_water_emis(surfaceAltitude: float, osp_dir: Path) -> dict[str, Any]:
    tes_emis_dir = osp_dir / "EMIS/EMIS_UWIS"
    if surfaceAltitude > 0.2:
        # Assume we're looking at freshwater when the altitude is above 200 m
        emissivity_file = tes_emis_dir / "Emissivity_Lake.asc"
        surface_type = "Lake"
    else:
        emissivity_file = tes_emis_dir / "Emissivity_Ocean.asc"
        surface_type = "Ocean"

    t = TesFile(emissivity_file)
    wavenumber = np.array(t.checked_table["Frequency"])
    emis = np.array(t.checked_table["Emissivity"])

    return {"wavenumber": wavenumber, "emissivity": emis, "surfaceType": surface_type}


# ------------------------------------------ #
# LEGACY (V1) U. WISCONSIN EMISSIVITY LOOKUP #
# ------------------------------------------ #


def get_emis_uwis(
    latin: float,
    lonin: float,
    year: int,
    month: int,
    osp_dir: Path,
    surfaceAltitude: None | np.ndarray = None,
    filenameSave: None | str = None,
    watertype: None | str = None,
    waterresult: None | dict[str, Any] = None,
    matrices: None | dict[str, np.ndarray] = None,
    filelats: None | np.ndarray = None,
    filelons: None | np.ndarray = None,
) -> dict[str, Any]:
    # IDL_LEGACY_NOTE: This function get_emis_uwis is the same as get_emis_uwis in TOOLS/EMIS_UWIS/get_emis_uwis.pro file.

    # only for TIR: 699 - 2774 cm-1

    FILLVALUE = np.float64(-9999.0)  # double
    bad = 0

    if filenameSave is None:
        filenameSave = ""

    if watertype is None:
        watertype = ""

    # Get the latitude and longitude from input parameters.
    lat = latin
    lon = lonin

    if lon > 180:
        lon = lon - 360.0

    # surface altitude in km  If EMIS is negative then sets to
    # freshwater EMIS if z>0.2 and set to ocean if z<0.2.

    tes_emis_dir = osp_dir / "EMIS/EMIS_UWIS/"

    # According to Eva Borjas, 2007 is the best year for this dataset
    if year >= 2013:
        year = 2007

    if filelats is None or filelons is None:
        with h5py.File(tes_emis_dir / "global_emis_inf10_location_small.nc", "r") as f:
            filelats = f["LAT"][:]
            filelons = f["LON"][:]

    lon_int = abs(filelons[0] - filelons[1])
    lat_int = abs(filelats[0] - filelats[1])

    filename = (
        str(tes_emis_dir)
        + "/global_emis_inf10*"
        + str("{0:04d}".format(int(year)))
        + "-"
        + str("{0:02d}".format(int(month)))
        + "*.nc"
    )

    # ../OSP/EMIS/EMIS_UWIS/global_emis_inf10_monthFilled_MYD11C3.A2007-06.041.nc

    fnames = glob(filename)
    if len(fnames) > 0:
        ncfile = fnames[0]
    else:
        # if not found, use 2007
        filename = (
            str(tes_emis_dir)
            + "/global_emis_inf10*"
            + str("{0:04d}".format(int(2007)))
            + "-"
            + str("{0:02d}".format(int(month)))
            + "*.nc"
        )
        ncfile = glob(filename)[0]

    found = 0
    if filenameSave == filename:
        found = 1

    if found == 0:
        #    double wavenumber(zdim=10)
        #    double emis_flag(ydim=7200, xdim=3600)
        #    double emis1(ydim=7200, xdim=3600)
        #    double emis2(ydim=7200, xdim=3600)
        #    double emis3(ydim=7200, xdim=3600)
        #    double emis4(ydim=7200, xdim=3600)
        #    double emis5(ydim=7200, xdim=3600)
        #    double emis6(ydim=7200, xdim=3600)
        #    double emis7(ydim=7200, xdim=3600)
        #    double emis8(ydim=7200, xdim=3600)
        #    double emis9(ydim=7200, xdim=3600)
        #    double emis10(ydim=7200, xdim=3600)

        # IDL_NOTE: In IDL it is assumed that the order of when the variables are written, they get assigned a number.
        #            ncdf_varget,fileID,0,wavenumber0
        #            ncdf_varget,fileID,1,emis_flag
        #            ncdf_varget,fileID,2,emis0
        #            ncdf_varget,fileID,3,emis1
        #            ncdf_varget,fileID,4,emis2
        #            ncdf_varget,fileID,5,emis3
        #            ncdf_varget,fileID,6,emis4
        #            ncdf_varget,fileID,7,emis5
        #            ncdf_varget,fileID,8,emis6
        #            ncdf_varget,fileID,9,emis7
        #            ncdf_varget,fileID,10,emis8
        #            ncdf_varget,fileID,11,emis9
        # The above IDL read statements and a print out of the Python code lead me to believe that this is the actual variables arrangements.

        # NetCDF variable id   Netcdf variable name  Python variable assignment.
        # 0                    'wavenumber'            wavenumber0
        # 1                    'emis_flag'             [Didn't read this variable.]
        # 2                    'emis1'                 emis0
        # 3                    'emis2'                 emis1
        # 4                    'emis3'                 emis2
        # 5                    'emis4'                 emis3
        # 6                    'emis5'                 emis4
        # 7                    'emis6'                 emis5
        # 8                    'emis7'                 emis6
        # 9                    'emis8'                 emis7
        # 10                   'emis9'                 emis8
        # 11                   'emis10'                emis9

        scale_factor_wvn = 1.0
        scale_factor_emis = 1.0

        wavenumber0, o_variable_attributes_dict = nc_read_variable(ncfile, "wavenumber")
        scale_factor_wvn = o_variable_attributes_dict["scale_factor"]

        emis0, o_variable_attributes_dict = nc_read_variable(ncfile, "emis1")
        scale_factor_emis = o_variable_attributes_dict["scale_factor"]

        emis1, _ = nc_read_variable(ncfile, "emis2")
        emis2, _ = nc_read_variable(ncfile, "emis3")
        emis3, _ = nc_read_variable(ncfile, "emis4")
        emis4, _ = nc_read_variable(ncfile, "emis5")
        emis5, _ = nc_read_variable(ncfile, "emis6")
        emis6, _ = nc_read_variable(ncfile, "emis7")
        emis7, _ = nc_read_variable(ncfile, "emis8")
        emis8, _ = nc_read_variable(ncfile, "emis9")
        emis9, _ = nc_read_variable(ncfile, "emis10")

        # Apply the scale factor to all applicable arrays.
        wavenumber0 = wavenumber0 * scale_factor_wvn
        FILLVALUE = FILLVALUE * scale_factor_emis
        emis0 = emis0 * scale_factor_emis
        emis1 = emis1 * scale_factor_emis
        emis2 = emis2 * scale_factor_emis
        emis3 = emis3 * scale_factor_emis
        emis4 = emis4 * scale_factor_emis
        emis5 = emis5 * scale_factor_emis
        emis6 = emis6 * scale_factor_emis
        emis7 = emis7 * scale_factor_emis
        emis8 = emis8 * scale_factor_emis
        emis9 = emis9 * scale_factor_emis
        filenameSave = filename
    # end if found == 0:

    lonind1 = np.where(np.min(abs(filelons - lon)) == abs(filelons - lon))[0]
    if len(lonind1) == 2:
        lonind = np.array(
            [
                np.mean(lonind1),
            ]
        )
    else:
        if filelons[lonind1] > lon:
            lonind = lonind1 - (filelons[lonind1] - lon) / lon_int
        else:
            if filelons[lonind1] < lon:
                lonind = lonind1 + (filelons[lonind1] - lon) / lon_int
            else:
                lonind = lonind1

    latind1 = np.where(min(abs(filelats - lat)) == abs(filelats - lat))[0]
    if len(latind1) == 2:
        latind = np.array(
            [
                np.mean(latind1),
            ]
        )
    else:
        if filelats[latind1] > lat:
            latind = latind1 + (filelats[latind1] - lat) / lat_int
        else:
            if filelats[latind1] < lat:
                latind = latind1 - (filelats[latind1] - lat) / lat_int
            else:
                latind = latind1

    # IDL_NOTE: EMIS0 Array[3600, 7200]
    # PYTHON_NOTE: emis0 (7200, 3600)
    emis0_interp = _bilinear(emis0.transpose(), latind[0], lonind[0], FILLVALUE)

    if np.all(emis0_interp > 0):
        emis1_interp = _bilinear(emis1.transpose(), latind[0], lonind[0])
        emis2_interp = _bilinear(emis2.transpose(), latind[0], lonind[0])
        emis3_interp = _bilinear(emis3.transpose(), latind[0], lonind[0])
        emis4_interp = _bilinear(emis4.transpose(), latind[0], lonind[0])
        emis5_interp = _bilinear(emis5.transpose(), latind[0], lonind[0])
        emis6_interp = _bilinear(emis6.transpose(), latind[0], lonind[0])
        emis7_interp = _bilinear(emis7.transpose(), latind[0], lonind[0])
        emis8_interp = _bilinear(emis8.transpose(), latind[0], lonind[0])
        emis9_interp = _bilinear(emis9.transpose(), latind[0], lonind[0])

        bfemis = np.flip(
            np.transpose(
                [
                    emis0_interp,
                    emis1_interp,
                    emis2_interp,
                    emis3_interp,
                    emis4_interp,
                    emis5_interp,
                    emis6_interp,
                    emis7_interp,
                    emis8_interp,
                    emis9_interp,
                ]
            ),
            1,
        )

        # if any of bfemis is 0 then redo this with 2007
        ind = np.where(bfemis < 0.10)[0]
        if ind.size > 0:
            bad = 1

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # calculate the PCA regression coefficients
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # save "D", PCM_NEW, PCU, PCM, wavenumber, all assuming on the 416
        # point grid.
        if matrices is None:
            filenamex = str(tes_emis_dir / "matrices.dat")
            logger.info(f"filenamex: {filenamex}")

            # check re-doing versus re-using
            fnames = glob(filenamex)
            if len(fnames) == 0:
                raise RuntimeError("Not implemented yet")
                # restore, tes_emis_dir+'pcs.dat'        ;eigenvectors (PC.U) and eigenvalues (PC.M) of the 123 selected laboratory spectra on HSR
                # restore, tes_emis_dir+'pcs_modres.dat' ;eigenvectors (PCU_NEW) and eigenvalues (PCM_NEW) of the 123 selected laboratory spectra on the ten hinge point resolution
                # restore, tes_emis_dir+'hsr_wavenum.dat';wavenumbers of the HSR
                # numemis=n_elements(bfemis)
                # wavenumber = hsr_wavenum
                # numwave = n_elements(wavenumber)
                # numpcs=6
                # A=PC_NEW(*,0:numpcs-1)
                # B=transpose(A)#A
                # D=invert(B)#transpose(A)
                # matrices = {D:D, PCM_NEW:PCM_NEW, PCU:PCU[*,0:numpcs-1], PCM:PCM, wavenumber:wavenumber}
                # save, matrices, filename = filenamex
                # ;print, 'redoing'
            else:
                matrices = _restore_matrices_variable(filenamex)
            # end else portion
        # end if matrices is None:

        coef = np.dot(bfemis - np.transpose(matrices["PCM_NEW"]), matrices["D"])

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % applying the coefs  to the first "npc"  PCs  on original resolution PC.U (416,numpcs)
        # % result: high spectral resolution  emissivity for each BF spectra: hsremis(416,numemis)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        hsremis = (
            np.ndarray.flatten(np.dot(matrices["PCU"].transpose(), coef.transpose()))
            + matrices["PCM"]
        )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %  cutting back emis > 1
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # if year not 2007 set year to 2007 and redo
        # if year 2007, set values greater than 1 to 1
        hsremis[hsremis > 1] = 1

        surfaceType = "Land"

        result = {
            "wavenumber": matrices["wavenumber"],
            "emissivity": hsremis,
            "surfaceType": surfaceType,
        }
    else:
        # these are going to be on a different frequency grid.  Hopefully
        # OK.  Wanted to keep on these grids as this is the TES grid which we
        # will interpolate to
        #
        # JLL 20 Jan 2022: in theory, this section could be replaced by the get_water_emis()
        # function above, as it copies this functionality. However, it looks like this does some
        # caching to avoid reloading the emissivity. Not sure how much that matters for speed, we
        # shouldn't be calling this very much in the same process.
        if surfaceAltitude is not None and surfaceAltitude.size > 0:
            raise RuntimeError("Water scene.  Must include altitude (in km) parameter")

        if surfaceAltitude is not None and surfaceAltitude[0] > 0.2:
            if watertype == "Lake":
                assert waterresult is not None
                result = waterresult
            else:
                watertype = "Lake"

                # fresh water
                t = TesFile(tes_emis_dir / "Emissivity_Lake.asc")

                wavenumber = np.asarray(t.checked_table["Frequency"])
                emis = np.asarray(t.checked_table["Emissivity"])

                surfaceType = "Lake"
                result = {
                    "wavenumber": wavenumber,
                    "emissivity": emis,
                    "surfaceType": surfaceType,
                }

                waterresult = result
        else:
            if watertype == "Ocean":
                assert waterresult is not None
                result = waterresult
            else:
                # ocean
                t = TesFile(tes_emis_dir / "Emissivity_Ocean.asc")

                wavenumber = np.asarray(t.checked_table["Frequency"])
                emis = np.asarray(t.checked_table["Emissivity"])

                surfaceType = "Ocean"
                result = {
                    "wavenumber": wavenumber,
                    "emissivity": emis,
                    "surfaceType": surfaceType,
                }

                waterresult = result
    # end else part of if np.all(emis0_interp > 0):

    indx = np.asarray(np.where(result["emissivity"] < 0))[0]
    if indx.size > 0:
        logger.warning("Emissivity less than 0")
        bad = 1

    indx = np.asarray(np.where(result["emissivity"] < 0.4))[0]
    if indx.size > 0:
        logger.warning("Emissivity less than 0.4")
        bad = 1

    # AT_LINE 274 TOOLS/EMIS_UWIS/get_emis_uwis.pro
    if bad == 1 and year != 2007:
        # redo with 2007, which is generally perfect
        result = get_emis_uwis(
            latin,
            lonin,
            2007,
            month,
            osp_dir,
            surfaceAltitude,
            filenameSave,
            watertype,
            waterresult,
            matrices,
            filelats,
            filelons,
        )
        return result
    elif bad == 1:
        result = get_emis_uwis(
            latin - 1,
            lonin,
            2007,
            month,
            osp_dir,
            surfaceAltitude,
            filenameSave,
            watertype,
            waterresult,
            matrices,
            filelats,
            filelons,
        )
        return result
    # end if bad == 1 and year != 2007:

    # For some reason, the wavenumber has datatype ">f8" instead of "float64".
    # This causes problems when we try to write it to the netCDF file, because
    # those functions don't know how to map >f8 to a netCDF datatype.
    result["wavenumber"] = result["wavenumber"].astype("float64")
    return result


def _bilinear(
    p_matrix: np.ndarray, latind: int, lonind: int, FILLVALUE: None | float = None
) -> np.ndarray:
    from .mpy import mpy_bilinear

    return mpy_bilinear(p_matrix, latind, lonind, FILLVALUE)


def nc_read_variable(
    i_filename: str, i_variable_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    # Read a specific variable into memory and returned the variable along with any variable attributes.
    try:
        nci = ncdf.Dataset(Path(i_filename).resolve(), mode="r")
        nci.set_auto_maskandscale(False)

        o_variable_attributes_dict = nci[i_variable_name].__dict__
        o_variable_data = nci[i_variable_name][:]
    finally:
        nci.close()

    return (o_variable_data, o_variable_attributes_dict)


def _restore_matrices_variable(i_filename: str) -> dict[str, np.ndarray]:
    # Function restores a binary file previously written out using IDL's SAVE program.
    # The functionality of this function is the same as the IDL's RESTORE program.

    # ** Structure <14d9858>, 5 tags, length=27184, data length=27184, refs=1:
    #   D               DOUBLE    Array[6, 10]
    #   PCM_NEW         DOUBLE    Array[10]
    #   PCU             DOUBLE    Array[416, 6]
    #   PCM             DOUBLE    Array[416]
    #   WAVENUMBER      DOUBLE    Array[416]
    # D_SIZE          = 6 * 10
    # PCM_NEW_SIZE    = 10
    # PCU_SIZE        = 416 * 6
    # PCM_SIZE        = 416
    # WAVENUMBER_SIZE = 416

    # The readsav() function returns an array of one element which is a dictionary.
    # We get the list of arrays dictionary with the 'matrices' key.

    my_recarray = cast(
        dict[int, dict[int, np.ndarray]], readsav(i_filename)["matrices"]
    )

    # These are the shapes of each element in the recarray, which should correspond to the 5 variables we expect to see:
    # Note that the dimensions are swapped versus IDL.
    #
    # my_recarray[0][0] (10, 6)
    # my_recarray[0][1] (10,)
    # my_recarray[0][2] (6, 416)
    # my_recarray[0][3] (416,)
    # my_recarray[0][4] (416,)

    o_matrices = {
        "D": my_recarray[0][0],
        "PCM_NEW": my_recarray[0][1],
        "PCU": my_recarray[0][2],
        "PCM": my_recarray[0][3],
        "wavenumber": my_recarray[0][4],
    }

    return o_matrices


__all__ = ["UwisCamelOptions", "get_emis_dispatcher"]
