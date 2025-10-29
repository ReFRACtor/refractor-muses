import numpy as np
import refractor.framework as rf


class BasicCloudProperties:
    def __init__(self, cloud_frac: float, cloud_pres: float, cloud_albedo: float = 0.8):
        self.cloud_frac = cloud_frac
        self.cloud_pres = cloud_pres
        self.cloud_alb = cloud_albedo

    def cloud_pres_level(self, pres_level_grid):
        # Assume input is in hPa and we want Pa
        return self.cloud_pres * 100

    def cloud_fraction(self):
        return rf.CloudFractionFromState(self.cloud_frac)

    def cloud_albedo(self, num_channels=1):
        albedo = np.zeros((num_channels, 1))
        albedo[:, 0] = self.cloud_alb

        which_retrieved = np.full((num_channels, 1), False, dtype=bool)
        band_reference = np.zeros(num_channels)
        band_reference[:] = 1000

        return rf.GroundLambertian(
            albedo,
            rf.ArrayWithUnit(band_reference, "nm"),
            ["Cloud"] * num_channels,
            rf.StateMappingAtIndexes(np.ravel(which_retrieved)),
        )


class MusesCloudProperties(BasicCloudProperties):
    def cloud_pres_level(self, pres_level_grid):
        # This approximates the MUSES calculation of layer pressures.
        # I've not checked how the layer pressures in rf_uip.ray_info.pbar
        # are actually calculated.
        if pres_level_grid[0] > pres_level_grid[-1]:
            raise ValueError(
                "Pressure level grid must be space-to-surface (i.e. decreasing)"
            )

        # It shouldn't actually matter if the layer grid is surf-to-space or space-to-surf, but
        # in the production code it is surf-to-space, so we be consistent.
        pres_layer_grid = np.flip(
            (pres_level_grid[:-1] + pres_level_grid[1:]) / 2, axis=0
        )
        ncloud_lay = np.count_nonzero(pres_layer_grid <= self.cloud_pres)
        if ncloud_lay + 1 < pres_level_grid.shape[0]:
            cloud_pres_hPa = (
                pres_level_grid[ncloud_lay] + pres_level_grid[ncloud_lay + 1]
            ) / 2
        else:
            cloud_pres_hPa = pres_level_grid[ncloud_lay]

        # assume we want the returned value in Pa
        return cloud_pres_hPa * 100
