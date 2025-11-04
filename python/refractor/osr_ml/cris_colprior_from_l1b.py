# type: ignore
"""
Title	: cris_colprior_from_l1b.py
To Run	: from cris_colprior_from_l1b import cris_colprior_from_l1b
Author	: Frank!
Date	: 20240525
Modf	: 20250204 - added x_cutoff to cris_colprior_from_l1b()
          20250210 - added full_interp keyword and functionality to cris_colprior_from_l1b()

"""

# Import modules
# =======================================
import numpy as np

from scipy.interpolate import NearestNDInterpolator, griddata, RegularGridInterpolator


# Functions and classes
# =======================================


def cris_colprior_from_l1b(
    l1b=None,
    l2muses=None,
    x_cutoff=9,
    near_loc=False,
    full_interp=False,
    prior_not_colprior=False,
    col_not_colprior=False,
    verbose=0,
):
    ##############
    # Find a L2 MUSES column prior for every L1b along_index and cross_index.
    # Missing values are filled by nearest-neightbor interpolation.
    #
    # Parameters
    # ---------
    # l1b: object/class; CrIS L1B data
    # l2muses: object/class; CrIS L2 MUSES data
    # x_cutoff: int; cutoff level for x (to avoid -999)
    # near_loc: boolean; pick the closest location to measurement
    # full_interp: boolean; instead of each granule individually, use all values
    # prior_not_colprior: boolean; use 67-level ret_prior instead of 5-level ret_colprior
    # col_not_colprior: boolean; use 5-level ret_col instead of 5-level ret_colprior
    # verbose (optional): boolean as integer; if 1 the routine prints the current file
    #
    # Returns
    # -------
    # prior_array_inter : ndarray; L2 MUSES column priors
    #                               [len(l1b.granule), 45, 30, 9, 5]
    ##############

    # Define which prior to use
    if col_not_colprior is False:
        if prior_not_colprior is False:
            _which_prior = np.copy(l2muses.ret_colprior)
        else:
            _which_prior = np.copy(l2muses.ret_prior[:, x_cutoff::])
    else:
        _which_prior = np.copy(l2muses.ret_col)

    # Find a L2 MUSES column prior for every L1B granule,
    # along_index, and cross_index
    if full_interp is False:
        if verbose == 1:
            print("Find column prior for granule")
        prior_array = np.zeros((len(l1b.granule), 45, 30, 9, len(_which_prior[0, :])))
        prior_array[:, :, :, :, :] = np.nan

        for i_g in range(0, len(l1b.granule)):
            for i_ai in range(0, 45):
                for i_ci in range(0, 30):
                    ind = np.where(
                        (l2muses.date == np.unique(l1b.date)[0])
                        & (l2muses.granule == np.unique(l1b.granule[i_g])[0])
                        & (l2muses.along_index == i_ai)
                        & (l2muses.cross_index == i_ci)
                    )[0]
                    if len(ind) > 0:
                        prior_array[i_g, i_ai, i_ci, :, :] = _which_prior[ind[0], :]

        # Perform nearest-neightbor interpolation to fill out all the
        # missing along_index and cross_index points
        if verbose == 1:
            print("Nearest-neightbor interpolation for granule")
        prior_array_inter = np.zeros_like(prior_array)
        prior_array_inter[:, :, :, :, :] = prior_array[:, :, :, :, :]

        for i_g in range(0, len(l1b.granule)):
            for i_l in range(0, len(_which_prior[0, :])):
                dummy_interp = prior_array[i_g, :, :, 0, i_l]
                if np.isfinite(np.nanmax(dummy_interp)):
                    mask = np.where(~np.isnan(dummy_interp))
                    interp = NearestNDInterpolator(
                        np.transpose(mask), dummy_interp[mask]
                    )
                    for i_ai in range(0, 45):
                        for i_ci in range(0, 30):
                            prior_array_inter[i_g, i_ai, i_ci, :, i_l] = interp(
                                i_ai, i_ci
                            )

        # Set invalid values to -999.99
        prior_array_inter[np.isnan(prior_array_inter)] = -999.99

    # Second (very slow, faster if ":" becomes "i_l" in line 94) method, if necessary
    if near_loc is True:
        n_obs = len(l1b.granule) * 45 * 30 * 9
        latitude_obs = np.reshape(l1b.latitude, (n_obs))
        longitude_obs = np.reshape(l1b.longitude, (n_obs))
        prior_array_inter = np.zeros((n_obs, len(_which_prior[0, :])))
        for i_o in range(0, n_obs):
            ind = np.argsort(
                (
                    np.absolute(latitude_obs[i_o] - l2muses.latitude) / 180.0
                    + np.absolute(longitude_obs[i_o] - l2muses.longitude) / 360.0
                )
            )
            for i_l in range(0, len(_which_prior[0, :])):
                ind2 = ind[np.where(_which_prior[ind, i_l] > 0)[0][0]]
                prior_array_inter[i_o, i_l] = _which_prior[ind2, i_l]

        # Reshape prior_array_inter
        prior_array_inter = np.reshape(
            prior_array_inter, (len(l1b.granule), 45, 30, 9, len(_which_prior[0, :]))
        )

        # Set invalid values to -999.99
        prior_array_inter[np.isnan(prior_array_inter)] = -999.99

    # Third method (pretty quick), if necessary
    if full_interp is True:
        n_obs = len(l1b.granule) * 45 * 30 * 9
        latitude_obs = np.reshape(l1b.latitude, (n_obs))
        longitude_obs = np.reshape(l1b.longitude, (n_obs))
        prior_array_inter = np.zeros(
            (len(l1b.granule), 45, 30, 9, len(_which_prior[0, :]))
        )
        prior_array_inter[:, :, :, :, :] = np.nan

        dlon = (np.max(l2muses.longitude) - np.min(l2muses.longitude)) / 0.25
        lon_new = np.linspace(
            np.min(l2muses.longitude), np.max(l2muses.longitude), int(dlon)
        )
        dlat = (np.max(l2muses.latitude) - np.min(l2muses.latitude)) / 0.25
        lat_new = np.linspace(
            np.min(l2muses.latitude), np.max(l2muses.latitude), int(dlat)
        )
        xi, yi = np.meshgrid(lon_new, lat_new, indexing="ij")

        for i_l in range(0, len(_which_prior[0, :])):
            ind_valid = np.where(
                (_which_prior[:, i_l] > -999.99) & (l2muses.quality == 1)
            )[0]
            x = l2muses.longitude[ind_valid]
            y = l2muses.latitude[ind_valid]
            zi = griddata(
                (x, y), _which_prior[ind_valid, i_l], (xi, yi), method="linear"
            )

            interpolating_function = RegularGridInterpolator(
                (lon_new, lat_new), zi, bounds_error=False, fill_value=None
            )
            prior_array_inter[:, :, :, :, i_l] = interpolating_function(
                (l1b.longitude[:, :, :, :], l1b.latitude[:, :, :, :])
            )

        # Set invalid values to -999.99
        prior_array_inter[np.isnan(prior_array_inter)] = -999.99

    # Return data
    return prior_array_inter
