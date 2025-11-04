"""
Title	: cris_pixel_corners.py
To Run	: from cris_pixel_corners import cris_pixel_corners
Author	: Frank!
Date	: 20250123
Modf	: 20250219 - added cris_pixel_corners()

"""

# Import modules
# =======================================
from __future__ import annotations
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from .cris_io import _ReadL1b

# Functions and classes
# =======================================


def cris_pixel_corners(l1b: _ReadL1b) -> tuple[np.ndarray, np.ndarray]:
    ##############
    # For a CrIS L1B object find the pixel corners based on each middle FOV.
    #
    # Parameters
    # ---------
    # l1b: object/class; CrIS L1B data
    #
    # Returns
    # -------
    # latitude_corners, longitude_corners : ndarrays; latitude and longitude corners for each pixel
    #                               [len(l1b.granule), 45, 30, 9, 4]
    ##############

    # Define some variables
    n_granule = len(l1b.granule)
    n_atrack = len(l1b.granule[0, :, 0, 0])
    n_xtrack = len(l1b.granule[0, 0, :, 0])
    n_fov = len(l1b.granule[0, 0, 0, :])

    latitude_corners = np.zeros((n_granule, n_atrack, n_xtrack, n_fov, 4))
    longitude_corners = np.zeros((n_granule, n_atrack, n_xtrack, n_fov, 4))
    latitude_corners[:, :, :, :, :] = np.nan
    longitude_corners[:, :, :, :, :] = np.nan

    # Loop through all dimensions and find pixel corners
    for i_granule in range(0, n_granule):
        for i_atrack in range(0, n_atrack):
            for i_xtrack in range(0, n_xtrack):
                for i_fov in range(0, n_fov):
                    latitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 0] = (
                        l1b.latitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.latitude[i_granule, i_atrack, i_xtrack, 6]
                            - l1b.latitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    longitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 0] = (
                        l1b.longitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.longitude[i_granule, i_atrack, i_xtrack, 6]
                            - l1b.longitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    latitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 1] = (
                        l1b.latitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.latitude[i_granule, i_atrack, i_xtrack, 0]
                            - l1b.latitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    longitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 1] = (
                        l1b.longitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.longitude[i_granule, i_atrack, i_xtrack, 0]
                            - l1b.longitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    latitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 2] = (
                        l1b.latitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.latitude[i_granule, i_atrack, i_xtrack, 2]
                            - l1b.latitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    longitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 2] = (
                        l1b.longitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.longitude[i_granule, i_atrack, i_xtrack, 2]
                            - l1b.longitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    latitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 3] = (
                        l1b.latitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.latitude[i_granule, i_atrack, i_xtrack, 8]
                            - l1b.latitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )
                    longitude_corners[i_granule, i_atrack, i_xtrack, i_fov, 3] = (
                        l1b.longitude[i_granule, i_atrack, i_xtrack, i_fov]
                        + 0.5
                        * (
                            l1b.longitude[i_granule, i_atrack, i_xtrack, 8]
                            - l1b.longitude[i_granule, i_atrack, i_xtrack, 4]
                        )
                    )

    return (latitude_corners, longitude_corners)
