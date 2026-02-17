from __future__ import annotations
import refractor.framework as rf  # type: ignore
from loguru import logger
import numpy as np
import subprocess
import os
from pathlib import Path
import typing
from typing import Self

if typing.TYPE_CHECKING:
    from .muses_observation import MusesObservation


class MusesRadiativeTransferVlidort(rf.RadiativeTransferImpBase):
    """This uses the VLIDORT cli that py-retrieve uses. This gives a forward
    model that is the same as the py-retrieve omi/tropomi forward model (with minor
    differences in calculation - the normal sort of round off differences).
    """

    def __init__(
        self,
        ground: rf.Ground,
        absorber: rf.Absorber,
        pressure: rf.Pressure,
        temperature: rf.Temperature,
        altitude: rf.Altitude,
        obs: MusesObservation,
        vlidort_dir: str | os.PathLike[str],
        vlidort_nstokes: int = 2,
        vlidort_nstreams: int = 4,
    ) -> None:
        super().__init__()
        self.ground = ground
        self.absorber = absorber
        self.pressure = pressure
        self.temperature = temperature
        self.altitude = altitude
        self.vlidort_dir = Path(vlidort_dir)
        self.obs = obs
        self.vlidort_nstokes = vlidort_nstokes
        self.vlidort_nstreams = vlidort_nstreams

    def clone(self) -> Self:
        return MusesRadiativeTransferVlidort(
            self.ground,
            self.absorber,
            self.pressure,
            self.temperature,
            self.altitude,
            self.vlidort_dir,
            self.obs,
            self.vlidort_nstokes,
            self.vlidort_nstreams,
        )

    def reflectance_ptr(
        self, sd: rf.SpectralDomain, spensor_index: int, skip_jacobian: bool
    ) -> rf.Spectrum:
        # Note, there is no way to actually skip the jacobians, so we ignore
        # that
        freq = sd.convert_wave("nm").value
        wn = sd.convert_wave("cm^-1").value

        vlidort_input_dir = self.vlidort_dir / "input"
        vlidort_input_dir.mkdir(parents=True, exist_ok=True)
        vlidort_output_dir = self.vlidort_dir / "output"
        vlidort_output_dir.mkdir(parents=True, exist_ok=True)

        with open(vlidort_input_dir / "config_rtm.asc", "w") as fh:
            print("'atm_lay.asc'", file=fh)
            print("'atm_lev.asc'", file=fh)
            print("'taug.asc'", file=fh)
            print("'surf_alb.asc'", file=fh)
            print("'vga.asc'", file=fh)
            print(f"{sd.data.shape[0]:>5d}", file=fh)
            print(f"{self.pressure.number_layer:>5d}", file=fh)

        # Write the Vga file
        elv = self.obs.surface_height[self.sensor_index]
        lat = self.obs.latitude[self.sensor_index]
        sza = self.obs.solar_zenith[self.sensor_index]
        # Clamp zen to be in range
        if sza <= 0.0:
            sza = 0.0001
        if sza >= 90.0:
            sza = 89.98
        vza = self.obs.observation_zenith[self.sensor_index]
        raz = self.obs.relative_azimuth[self.sensor_index]
        with open(vlidort_input_dir / "vga.asc", "w") as fh:
            print("ELEV,       DLAT,       SZA,        VZA,        RAZ", file=fh)
            print(
                f"{elv:12.4f} {lat:12.4f} {sza:12.4f} {vza:12.4f} {raz:12.4f}", file=fh
            )

        # Write surface albedo
        with open(vlidort_input_dir / "surf_alb.asc", "w") as fh:
            print(len(freq), file=fh)
            for wnv, freqv in zip(wn, freq):
                alb = self.ground.surface_parameter(wnv, self.sensor_index).value[0]
                print(f"{freqv:16.7f} {alb:16.7f}", file=fh)

        # Write atmosphere levels
        pgrid = self.pressure.pressure_grid(rf.Pressure.INCREASING_PRESSURE)
        plev = pgrid.convert("hPa").value.value
        tlev = self.temperature.temperature_grid(
            self.pressure, rf.Pressure.INCREASING_PRESSURE
        ).value.value
        hlev = [self.altitude[0].altitude(p).convert("m").value.value for p in pgrid]
        with open(vlidort_input_dir / "atm_lev.asc", "w") as fh:
            print(len(plev), file=fh)
            print(
                "Table Columns: Pres(mb), T(K), Altitude(m) (TOA to Surf each level)",
                file=fh,
            )
            for p, t, h in zip(plev, tlev, hlev):
                print(f"{p:16.8f} {t:16.5f} {h:16.5f}", file=fh)
        # Write atmosphere layers
        #
        # This isn't actually used when we pass in the taug, but it still needs
        # to be there to be read. Could probably change vlidort_cli to skip reading this,
        # but not worth the effort right now. This is in rtm/read_atm.f if we want to do
        # something here in the future.
        #
        # Put in dummy values, just to make it clear we aren't filling this in
        pbar = [
            -999.0,
        ] * self.pressure.number_layer
        tbar = [
            -999.0,
        ] * self.pressure.number_layer
        o3 = [
            -999.0,
        ] * self.pressure.number_layer
        with open(vlidort_input_dir / "atm_lay.asc", "w") as fh:
            print(self.pressure.number_layer, file=fh)
            print("Table Columns: Pres(mb), T(K), Column Density (molec/cm2)", file=fh)
            for i in range(self.pressure.number_layer):
                print(f"{pbar[i]:16.6f} {tbar[i]:16.7f} {o3[i]:16.7e}", file=fh)
        taug_val_v = []
        dod_dstate_v = []
        for wnv in wn:
            t = self.ocreator.absorber.optical_depth_each_layer(wnv, self.sensor_index)
            taug_val_v.append(t.value[:, 0])
            if not t.is_constant:
                dod_dstate_v.append(t.jacobian[:, 0, :])
        # For reference, this in nwn x nlay x nstate. For each wnv, it is
        # the value dodlay_dstate for each layer
        dod_dstate = np.array(dod_dstate_v) if len(dod_dstate_v) > 0 else None
        with open(vlidort_input_dir / "taug.asc", "w") as fh:
            print(f"{len(freq)} {taug_val_v[0].shape[0]}", file=fh)
            for i, (freqv, wnv) in enumerate(zip(freq, wn)):
                taugln = " ".join(f"{od:16.15e}" for od in taug_val_v[i])
                print(f"{i} {freqv:16.8f} {taugln}", file=fh)

        # Run VLIDORT CLI
        vlidort_command = [
            "vlidort_cli",
            "--input",
            f"{vlidort_input_dir}/",
            "--output",
            f"{vlidort_output_dir}/",
            "--nstokes",
            f"{self.vlidort_nstokes}",
            "--nstreams",
            f"{self.vlidort_nstreams}",
            "--od",
        ]
        logger.debug(f"\nRunning:\n{' '.join(vlidort_command)} ")
        subprocess.run(
            vlidort_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        # IWF = G * dI / dG, where I is a component of the stokes
        # vector (I, Q, U, V) and G is the gas optical depth (O3 in
        # our case)
        #
        # IWF also known as the normalized weighting function
        # The denormalized IWF: IWF_denorm = IWF / G

        # read result files from the RT model
        radiance_matrix = np.loadtxt(vlidort_output_dir / "Radiance.asc", skiprows=1)

        # Use the denormalized weighting function as provided by VLIDORT
        # this gives dvmr
        jacobian_o3_matrix = np.loadtxt(
            vlidort_output_dir / "IWF_denorm.asc", skiprows=1
        )

        jacobian_sf_matrix = np.loadtxt(vlidort_output_dir / "surf_WF.asc", skiprows=1)

        # Translate jacobian_sf_matrix to a jacobian relative to the state vector
        jacobian_sf_matrix = jacobian_sf_matrix[
            :, 1:
        ]  # First column is frequency, chop off
        if not self.ground.surface_parameter(wn[0], self.sensor_index).is_constant:
            jac_sf = np.concatenate(
                [
                    jacobian_sf_matrix[i, :][np.newaxis, :]
                    @ self.ground.surface_parameter(wnv, self.sensor_index).jacobian
                    for i, wnv in enumerate(wn)
                ]
            )
        else:
            jac_sf = None
        jac_tot = jac_sf

        # Translate jacobian_o3_matrix to a jacobian relative to the state vector.
        # jac_o3_matrix is drad_dodlay, it is nwn x nlay
        # For reference dod_dstate is in nwn x nlay x nstate. For each wnv, it is
        # the value dodlay_dstate for each layer
        #
        # We calculate drad_dstate, including just the absorber part of the jacobian
        if dod_dstate is not None:
            jacobian_o3_matrix = jacobian_o3_matrix[
                :, 1:
            ]  # First column is frequency, chop off
            jac_o3 = np.vstack(
                [
                    jacobian_o3_matrix[i, :] @ dod_dstate[i, :, :]
                    for i in range(jacobian_o3_matrix.shape[0])
                ]
            )
        else:
            jac_o3 = None
        # Combine to an overall jacobian
        if jac_o3 is not None:
            if jac_tot is None:
                jac_tot = jac_o3
            else:
                jac_tot += jac_o3
        if jac_tot is not None:
            # We only take I component here.
            rad = rf.ArrayAd_double_1(radiance_matrix[:, 1], jac_tot)
        else:
            rad = rf.ArrayAd_double_1(radiance_matrix[:, 1])
        return rf.Spectrum(sd, rf.SpectralRange(rad, rf.Unit("sr^-1")))

    def stokes(self, sd: rf.SpectralDomain, sensor_index: int) -> np.ndarray:
        raise NotImplementedError(
            """We don't have the logic in place for return the full
            stoke vector. Wouldn't be hard, just haven't had a reason
            to work through the logic"""
        )

    def stokes_and_jacobian(
        self, sd: rf.SpectralDomain, sensor_index: int
    ) -> rf.ArrayAd_double_2:
        raise NotImplementedError(
            """We don't have the logic in place for return the full
            stoke vector. Wouldn't be hard, just haven't had a reason
            to work through the logic"""
        )

    def desc(self) -> str:
        return "MusesRadiativeTransferVlidort"


__all__ = [
    "MusesRadiativeTransferVlidort",
]
