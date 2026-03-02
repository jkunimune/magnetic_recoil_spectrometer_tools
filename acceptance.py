"""
Contains the procedures for simulating the conversion foil and
aperture stage of the spectrometer.
"""

import typing as ty

import numpy as np
import numpy.typing as npt
from scipy import integrate

from util import rng, random_partition


########################################################################
# ==== FOIL SIMULATION

# ( [energy/J; n], [stopping_power/(J/m); n] )
SRXMData = tuple[npt.NDArray, npt.NDArray]
# ( [energy/J; n], [total_cross_section_density/(/m); n] )
CrossSectionTableTotal = tuple[npt.NDArray, npt.NDArray]

# ( [theta/rad; n], [energy/J; n] )
AngleEnergy = tuple[npt.NDArray, npt.NDArray]
# ( [theta/rad; n], [energy/J; n] )
AngleEnergyId = tuple[npt.NDArray, npt.NDArray, npt.NDArray]

# (phot_energy/J) -> cross_section_density/(/m)
CrossSectionFnTotal = ty.Callable[[float], float]
# (n_rays:int, phot_energy/J) -> electrons:AngleEnergy
CrossSectionFnGenrays = ty.Callable[[int, float], AngleEnergy]
CrossSection = tuple[CrossSectionFnTotal, CrossSectionFnGenrays]

Foil = tuple[SRXMData, list[CrossSection]]

ThetaPhiXYEnergyId = tuple[
    npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
]

RaysXAYBERelative = npt.NDArray


def srxm_attenuate(
    energy_in: npt.ArrayLike, depth_in: npt.ArrayLike, srxm: SRXMData, n_steps=1000
) -> npt.NDArray:
    """
    Calculate the slowing for charged particles with given `energy_in` and `depth_in`
    (depth until exiting material) using the slowing data table `srxm`.
    """
    if n_steps == 0:
        return energy_in
    else:
        range_curve = integrate.cumulative_trapezoid(x=srxm[0], y=1/srxm[1], initial=0)
        initial_range = np.interp(energy_in, srxm[0], range_curve)
        final_range = initial_range - depth_in
        return np.interp(final_range, range_curve, srxm[0])

def foil_trace(
    n_rays_incident: float,
    n_srxm_steps: int,
    phot_energy_in: float,
    foil_properties: Foil,
    foil_depth: float,
) -> AngleEnergyId:
    """
    Simulates the number, energy, and direction of the charged scattering products of
    `n_rays_incident` neutral rays with energy `phot_energy_in` through a foil with
    `foil_properties` that is `foil_depth` meters thick.

    Returns the angles, energies, and mechanism id of the produced charged rays. The
    length of this array divided by `n_rays_incident` is the raw pre-aperture efficiency
    of the system.
    """
    srxm, cross_sections = foil_properties
    cross_section_density_totals = [
        cross_section[0](phot_energy_in) for cross_section in cross_sections
    ]
    p_density = np.sum(cross_section_density_totals)
    n_rays_interacting = rng.binomial(
        n_rays_incident, 1 - np.exp(-p_density * foil_depth)
    )
    process_interaction_counts = random_partition(
        cross_section_density_totals, n_rays_interacting
    )

    elec_angles_out: list[npt.NDArray] = []
    elec_energies_out: list[npt.NDArray] = []
    for cross_section, count in zip(cross_sections, process_interaction_counts):
        elec_angle, elec_energy = cross_section[1](count, phot_energy_in)
        valid = elec_angle < np.pi / 2
        elec_angle = elec_angle[valid]
        elec_energy = elec_energy[valid]

        v = rng.random(elec_energy.size)
        ray_depth = (
            -1 / p_density * np.log(1 - v * (1 - np.exp(-p_density * foil_depth)))
        )
        elec_energy = srxm_attenuate(
            elec_energy,
            (foil_depth - ray_depth) / np.cos(elec_angle),
            srxm,
            n_srxm_steps,
        )
        valid = elec_energy > 0
        elec_angles_out.append(elec_angle[valid])
        elec_energies_out.append(elec_energy[valid])

    return (
        np.concatenate(elec_angles_out),
        np.concatenate(elec_energies_out),
        np.concatenate([np.repeat([i], x.size) for i, x in enumerate(elec_angles_out)]),
    )


def aperture(
    foil_electrons: AngleEnergyId,
    foil_radius: float,
    aperature_radius: float,
    aperature_offset: float,
    replication: int = 1,
) -> ThetaPhiXYEnergyId:
    """
    Accepts the output of `foil_trace` (`foil_electrons`), and using
    the given `foil_radius`, `aperature_radius`, and `aperature_offset`,
    filters the particles by whether they would pass through the aperture.

    Returns the full angle, energy, and position of the particles at the
    aperture plane.
    """
    angle_in, energy_in, id_in = foil_electrons
    id_out = np.array([])
    angle_out = np.array([])
    angle_phi_out = np.array([])
    pos_x_out = np.array([])
    pos_y_out = np.array([])
    energy_out = np.array([])

    for _ in range(replication):
        pos_phi_in = rng.random(foil_electrons[0].shape) * (2 * np.pi)
        angle_phi_in = rng.random(foil_electrons[0].shape) * (2 * np.pi)
        pos_r_in = foil_radius * (rng.random(foil_electrons[0].shape) ** 0.5)

        x, y = pos_r_in * np.cos(pos_phi_in), pos_r_in * np.sin(pos_phi_in)
        x += np.tan(angle_in) * np.cos(angle_phi_in) * aperature_offset
        y += np.tan(angle_in) * np.sin(angle_phi_in) * aperature_offset

        aperature_out_mask = (x * x + y * y) < (aperature_radius**2)

        id_out = np.concatenate([id_out, id_in[aperature_out_mask]])
        angle_out = np.concatenate([angle_out, angle_in[aperature_out_mask]])
        angle_phi_out = np.concatenate(
            [angle_phi_out, angle_phi_in[aperature_out_mask]]
        )
        pos_x_out = np.concatenate([pos_x_out, x[aperature_out_mask]])
        pos_y_out = np.concatenate([pos_y_out, y[aperature_out_mask]])
        energy_out = np.concatenate([energy_out, energy_in[aperature_out_mask]])

    return angle_out, angle_phi_out, pos_x_out, pos_y_out, energy_out, id_out


def rays_into_relative(
    rays: ThetaPhiXYEnergyId,
    center_energy: float,
) -> RaysXAYBERelative:
    """
    Converts `ThetaPhiXYEnergyId` to `RaysXAYBERelative`.
    - converts absolute energy (`joules`) to relative `delta_E for E = E_0 * (1 + delta_E)`.
    - converts angles to fractional vector off-beam-axis components.
    """
    th, ph, x, y, e, _ = rays
    return np.transpose(
        np.array(
            [
                x,
                np.sin(th) * np.cos(ph),
                y,
                np.sin(th) * np.sin(ph),
                1 - e / center_energy,
            ]
        )
    )


def relative_rays_into_cosyscript(rays: RaysXAYBERelative, color=0) -> str:
    """
    Converts `RaysXAYBERelative` into COSYScript source code (to be injected
    using the cosy module's substitution system.)
    """
    return "\n".join(
        [
            f"SR {x} {a} {y} {b} 0 {e} 0 0 {color};"
            for x, a, y, b, e in np.round(rays, decimals=10)
        ]
    )


def rays_to_cosyscript(
    rays: ThetaPhiXYEnergyId,
    center_energy: float,
    color=0,
) -> str:
    """
    Converts `aperture` output rays (`ThetaPhiXYEnergyId`) into COSYScript
    source code (to be injected using the cosy module's substitution system.)
    """
    return relative_rays_into_cosyscript(
        rays_into_relative(rays, center_energy),
        color,
    )
