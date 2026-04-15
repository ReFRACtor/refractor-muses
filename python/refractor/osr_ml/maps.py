"""
Title        :  maps.py
What is it   :  Routines to plot cartopy maps
Includes     :  map_data_1d()
Author        : Frank Werner
Date          : 20240722
Modf          : 20250124 - added pixels keyqord to map_data_1d()
                20250213 - fixed ax.add_feature(cfeature.BORDERS) call in
                           map_data_1d() by removing resolution keyword
                20250214 - added borders keyword to map_data_1d()
                20250217 - added log_norm keyword to map_data_1d()
                20250220 - changed 'pixels' to 'polygons' for style keyword in
                           map_data_1d(), added 'ellipse' option
                20250331 - added projection keyword to map_data_1d()
                20250606 - changed formatting of ax.coastlines() and ax.add_feature() in map_data_1d()


"""

# Import modules
# =======================================
from __future__ import annotations
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import cartopy.crs as ccrs  # type: ignore
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # type: ignore
import cartopy.feature as cfeature  # type: ignore

from .parula_cmap import parula_cmap

import warnings
from typing import Any

warnings.filterwarnings("ignore")


# Functions
# =======================================
def map_data_1d(
    ax: Any = None,
    projection: str = "PlateCarree",
    lon: Any = None,
    lat: Any = None,
    lon_pixels: Any = None,
    lat_pixels: Any = None,
    var: Any = None,
    mask: Any = None,
    cmap: Any = None,
    boundaries: Any = None,
    log_norm: bool = False,
    style: str = "dots",
    markersize: int = 1,
    coastlines_res: str = "10m",
    borders: bool = True,
    extent: list[float] = [-180, 180, -90, 90],
    bottom_labels: bool = True,
    top_labels: bool = True,
    left_labels: bool = True,
    right_labels: bool = True,
    xlocator: list[float] = [-135, -90, -45, 0, 45, 90, 135],
    ylocator: list[float] = [-90, -45, 0, 45, 90],
    grid_fs: int = 8,
    cbar_ticks: Any = None,
    cbar_fraction: float = 0.0188,
    cbar_pad: float = 0.08,
    cbar_orientation: str = "horizontal",
    cbar_aspect: int = 96,
    cbar_no_minorticks: bool = False,
    cbar_minor_nbins: Any = None,
    cbar_label: str = "",
    cbar_rotation: int = 0,
    cbar_labelpad: int = 5,
    cbar_fs: int | float = 12,
    cbar_do: bool = True,
    add_text: Any = None,
    add_text_fs: int | float = 12,
    fig_title: str = "",
    fig_title_pad: int | float = 0,
) -> None:
    ##############
    # Plots a map with 1d longitude, latitude, and data arrays.
    #
    # Parameters
    # ---------
    # ax: pyplot axis; the axis into which to plot the map
    # projection: string; map projection
    # lon, lat: ndarrays; longitude and latitude
    # var: ndarray; the variable to plot
    # mask (optional): ndarray; boolean as binary for masking data points
    # cmap (optional): pyplot colormap or string; color map to be used
    # boundaries (optional): ndarray; increments for the colormap
    # log_norm (optional): boolean; whether to use log-normal normalization
    # style (optional): string; 'dots' or 'interp' or 'grid_boxes' or 'polygons' or 'ellipse'
    # markersize (optional): integer or float; size of 'dots'
    # coastlines_res (optional): string; resolution for cartopy coastlines
    # borders (optional): boolean; whether to plot country borders
    # extent (optional): ndarray; extent of the map
    # bottom_labels, top_labels, left_labels, right_labels (optional): boolean;
    #                   whether to plot zonal and meridional labels
    # xlocator, ylocator (optional): ndarrays; locations of labels
    # grid_fs (optional): integer or float; font size of the labels (default for 12, 6)
    # cbar_ticks down to cbar_do (optional): integer, float, strings, and boolean;
    #                   settings for the colorbar (defaults for 12, 6)
    # add_text (optional): ndarray; position and string for additional text
    # add_text_fs (optional): integer or float; font size of add_text (default for 12, 6)
    # fig_title (optional): string; title of the figure
    # fig_title_pad (optional): integer or float; padding for the title of the figure
    #
    # Returns
    # -------
    # None
    ##############

    # Projection
    if projection == "PlateCarree":
        crs_proj = ccrs.PlateCarree()
    if projection == "Mollweide":
        crs_proj = ccrs.Mollweide()

    # Define mask
    if mask is None:
        mask = np.zeros_like(var)
    ind_plot = np.where(mask == 0)[0]

    # Define colormap
    if cmap is None:
        cmap = matplotlib.cm.Spectral_r  # type:ignore[attr-defined]
    if cmap == "Parula":
        cmap = parula_cmap()

    # Define boundaries and normalization
    if boundaries is None:
        boundaries = np.linspace(np.percentile(var, 1), np.percentile(var, 99), 41)
    if log_norm is False:
        norm: Any = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    else:
        norm = matplotlib.colors.LogNorm(vmin=boundaries[0], vmax=boundaries[-1])
        boundaries = None

    # Plot dots
    if style == "dots":
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        sm = plt.tripcolor(
            lon[ind_plot], lat[ind_plot], var2[ind_plot], cmap=cmap, norm=norm
        )
        for i in range(0, len(ind_plot)):
            c = cmap(norm(var[i]))
            plt.plot(
                lon[ind_plot[i]],
                lat[ind_plot[i]],
                "o",
                color=c,
                markersize=markersize,
                zorder=0,
            )

    # Plot unstructured triangular grid
    if style == "interp":
        sm = plt.tripcolor(
            lon[ind_plot], lat[ind_plot], var[ind_plot], cmap=cmap, norm=norm
        )

    # Plot grid boxes
    if style == "grid_boxes":
        xx, yy = np.meshgrid(lon, lat)
        sm = plt.pcolor(xx, yy, var, cmap=cmap, norm=norm)

    # Plot pixels
    if style == "polygons":
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        sm = plt.tripcolor(
            lon[ind_plot], lat[ind_plot], var2[ind_plot], cmap=cmap, norm=norm
        )
        for i in range(0, len(lat_pixels)):
            c = cmap(norm(var[i]))
            plt.fill(lon_pixels[i, :], lat_pixels[i, :], color=c, zorder=0)

    # Plot ellipse
    if style == "ellipse":
        var2 = np.zeros_like(var)
        var2[:] = np.nan
        sm = plt.tripcolor(
            lon[ind_plot], lat[ind_plot], var2[ind_plot], cmap=cmap, norm=norm
        )
        for i in range(0, len(lat_pixels)):
            c = cmap(norm(var[i]))

            A = np.stack(
                [
                    lon_pixels[i, :] ** 2,
                    lon_pixels[i, :] * lat_pixels[i, :],
                    lat_pixels[i, :] ** 2,
                    lon_pixels[i, :],
                    lat_pixels[i, :],
                ]
            ).T
            b = np.ones_like(lon_pixels[i, :])
            w = np.linalg.lstsq(A, b)[0].squeeze()
            xlin = np.linspace(np.min(lon_pixels[i, :]), np.max(lon_pixels[i, :]), 200)
            ylin = np.linspace(np.min(lat_pixels[i, :]), np.max(lat_pixels[i, :]), 200)
            X, Y = np.meshgrid(xlin, ylin)
            Z = w[0] * X**2 + w[1] * X * Y + w[2] * Y**2 + w[3] * X + w[4] * Y

            ax.contourf(X, Y, Z, [1, 2], colors=[c, "tab:white"])

    # Add coastlines and borders
    ax.coastlines(coastlines_res, color="black", linewidth=1, alpha=0.5)
    if borders is True:
        ax.add_feature(cfeature.BORDERS, color="black", linewidth=1, alpha=0.5)

    # Define the extent
    ax.set_extent(extent)

    # Plot gridlines
    gl = ax.gridlines(
        crs=crs_proj,
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.52,
        linestyle="--",
    )
    gl.bottom_labels = bottom_labels
    gl.top_labels = top_labels
    gl.left_labels = left_labels
    gl.right_labels = right_labels
    gl.xlocator = matplotlib.ticker.FixedLocator(xlocator)
    gl.ylocator = matplotlib.ticker.FixedLocator(ylocator)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": grid_fs, "color": "gray", "weight": "bold"}
    gl.ylabel_style = {"size": grid_fs, "color": "gray", "weight": "bold"}

    # Colorbar
    if cbar_do is True:
        cbar = plt.colorbar(
            sm,
            ticks=cbar_ticks,
            fraction=cbar_fraction,
            pad=cbar_pad,
            orientation=cbar_orientation,
            aspect=cbar_aspect,
            boundaries=boundaries,
        )

        cbar.ax.tick_params(labelsize=cbar_fs)
        if cbar_no_minorticks is True:
            cbar.minorticks_off()
        if cbar_minor_nbins is not None:
            if cbar_orientation == "horizontal":
                cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(n=cbar_minor_nbins))
        cbar.set_label(
            cbar_label, rotation=cbar_rotation, labelpad=cbar_labelpad, fontsize=cbar_fs
        )

    # Additional text
    if add_text is not None:
        plt.text(add_text[0], add_text[1], add_text[2], fontsize=add_text_fs)

    if fig_title != "":
        plt.title(fig_title, fontsize=cbar_fs, pad=fig_title_pad)

    # Return data
    return None
