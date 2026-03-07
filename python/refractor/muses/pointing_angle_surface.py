from __future__ import annotations
import refractor.framework as rf  # type: ignore
import typing
import math

if typing.TYPE_CHECKING:
    from .muses_refractive_index import MusesRefractiveIndex


class PointingAngleSurface:
    """This calculates the pointing angle at the surface for a given
    satellite pointing angle. This is mostly just geometry/earth
    curvature, but this includes refraction also. Not clear why this
    is done, the refraction is pretty small. But since we already have
    the code from py-retrieve to do this, not reason not to include
    this.

    Note that this replaces the earth ellipsoid with a sphere that has
    the same radius as the target location. Again this isn't a big
    effect, but be aware that this approximation is being done. I
    believe the OSS code makes the same approximation, so this is not
    a bad thing to I do. I believe the intent is to have the OSS code
    generate a result that matches the satellite pointing once it has
    traced from the surface to TOA.
    """

    def __init__(
        self,
        sat_altitude: rf.DoubleWithUnit,
        earth_radius: rf.DoubleWithUnit,
        p: rf.Pressure,
        alt: rf.Altitude,
        rindex: MusesRefractiveIndex,
    ) -> None:
        self.eradius = earth_radius.convert("m").value
        self.sat_altitude = sat_altitude.convert("m").value
        self.p = p
        self.alt = alt
        self.rindex = rindex

    def pointing_angle_surface(
        self,
        pointing_angle: rf.DoubleWithUnit,
    ) -> rf.DoubleWithUnit:
        agrid = (
            self.alt.altitude_grid(self.p, rf.Pressure.DECREASING_PRESSURE)
            .convert("m")
            .value.value
        )
        nlayers = agrid.shape[0] - 1
        ds_fix = 500.0

        # spherical snells law with n = 1

        sin_theta_u = (
            (self.sat_altitude + self.eradius)
            * math.sin(pointing_angle.convert("rad").value)
            / (agrid[-1] + self.eradius)
        )

        snells_constant = (agrid[-1] + self.eradius) * sin_theta_u

        cos_theta_u = math.sqrt(1.0 - sin_theta_u**2)

        for jj in reversed(range(0, nlayers)):  # go from top to bottom
            a_u = agrid[jj + 1]
            flag = 0
            while flag == 0:  # sub layer loop
                da = ds_fix * cos_theta_u
                # This while loop only exit if the following condition is true.
                if (a_u - da) < agrid[jj]:
                    da = a_u - agrid[jj]
                    flag = 1
                a_l = a_u - da
                n_l = self.rindex.refractive_index(rf.DoubleWithUnit(a_l, "m"))

                sin_theta_u = snells_constant / (a_l + self.eradius) / n_l
                cos_theta_u = math.sqrt(1 - sin_theta_u**2)
                a_u = a_l

        return rf.DoubleWithUnit(math.asin(sin_theta_u), "rad")


__all__ = [
    "PointingAngleSurface",
]
