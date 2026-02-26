import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from cross_sections import (
    gen_cross_section_compton,
    gen_cross_section_pairproduction,
    load_pairproduction_cross_section,
)
from acceptance import SRXMData, foil_trace, Foil, aperture
from physical_constants import mol, MeV, millimeter, centimeter, gram, classical_electron_radius, barn, micrometer

configurations = {
    "High-resolution": (15*MeV, 0.5*centimeter, 50*centimeter, 0.5*centimeter, 0.2*MeV, 0.015*MeV, 0.015*MeV),
    "High-efficiency": (15*MeV, 1.0*centimeter, 30*centimeter, 1.0*centimeter, 0.5*MeV, 0.15*MeV, 0.20*MeV),
    "Low-energy": (5*MeV, 1.0*centimeter, 30*centimeter, 1.0*centimeter, 0.5*MeV, 0.025*MeV, 0.025*MeV),
}

materials = {
    # "ideal Li": (3, 1.01, 0.5334*gram/centimeter**3, "lithium", False),
    # "Li": (3, 6.94, 0.5334*gram/centimeter**3, "lithium", True),
    "B": (5, 10.81, 2.35*gram/centimeter**3, "boron", True),
    "Si": (14, 28.09, 2.329*gram/centimeter**3, "silicon", True),
    "Fe": (26, 55.85, 7.874*gram/centimeter**3, "iron", True)
}

for configuration_name in configurations.keys():
    for material_name in materials.keys():
        E0, R_foil, d_aperture, R_aperture, ΔE, ΔE_geom_1, ΔE_geom_2 = configurations[configuration_name]
        Z, A, density, full_material_name, do_ranging = materials[material_name]

        print(material_name, configuration_name)

        number_density = density / A * mol  # [/m^3]

        with open(f"./_data/estar_{material_name.split()[-1]}.txt", "r", encoding="utf8") as srem:
            x_srem = np.array(
                [[float(y) for y in x.split(" ")[:2]] for x in srem.readlines()[9:]]
            )
            x_srem[:, 0] *= MeV  #  MeV  ->  J
            x_srem[:, 1] *= MeV / gram * centimeter**2  # MeV cm^2 / g -> J m^2 / kg
            x_srem[:, 1] *= density  #  J m^2 / kg  ->  J / m
            x_srem: SRXMData = x_srem[:, 0], x_srem[:, 1]

        x_crosssection_compton = gen_cross_section_compton(
            number_density * Z,
            np.linspace(2, 20, 50) * MeV,
            np.linspace(0.0, np.pi, 1000),
        )
        σ_tot, sample = x_crosssection_compton
        x_crosssection_pairproduction = gen_cross_section_pairproduction(
            Z,
            number_density,
            load_pairproduction_cross_section("./_data/pairprod_xsctn_medium.npz"),
        )
        x_foil: Foil = x_srem, [
            x_crosssection_compton,
            x_crosssection_pairproduction,
        ]

        if material_name == "B":
            print("Compton scattering:", x_crosssection_compton[0](16.7*MeV)/(number_density*Z)/barn, "b")
            print("Pair production:", x_crosssection_pairproduction[0](16.7*MeV)/number_density/barn, "b")

        δ_foil = ΔE/np.interp(E0 - 0.255*MeV, x_srem[0], x_srem[1])
        print(f"Thickness: {δ_foil/millimeter:.3f} mm")

        if not do_ranging:
            ΔE = 1e-6*MeV

        def dσ_dθγ(θγ):
            Ef = 1/(1/E0 + (1 - np.cos(θγ))/(.511*MeV))
            dσ_dΩγ = 1/2*classical_electron_radius**2*(Ef/E0)**2*(Ef/E0 + E0/Ef - np.sin(θγ)**2)
            return dσ_dΩγ*2*np.pi*np.sin(θγ)

        α = E0/(.511*MeV)
        σ_tot = 2 * np.pi * classical_electron_radius**2*((1 + α)/α**3*(2*α*(1 + α)/(1 + 2*α) - np.log(1 + 2*α)) + np.log(1 + 2*α)/(2*α) - (1 + 3*α) / (1 + 2*α)**2)
        θe_cutoff = R_aperture/d_aperture
        θγ_cutoff = 2*np.arctan(1/((1 + α)*np.tan(θe_cutoff)))
        σ_aperture = integrate.quad(dσ_dθγ, θγ_cutoff, np.pi)[0]
        expected_efficiency = number_density*Z*δ_foil*σ_aperture
        expected_spectrum = expected_efficiency/ΔE

        print(f"about {number_density*Z*δ_foil*σ_tot:.3g} of the photons should generate an electron")
        print(f"about {σ_aperture/σ_tot:.3g} of the electrons should go thru the aperture")

        def do_monte_carlo(
            gamma_energy: float,
            N: float,
        ):
            elec_angle, elec_energy, ids = foil_trace(
                n_rays_incident=N,
                n_srxm_steps=1000 if do_ranging else 0,
                phot_energy_in=gamma_energy,
                foil_properties=x_foil,
                foil_depth=δ_foil,
            )
            print(f"{elec_energy.size/N:.3g} of the photons generated an electron")

            t, p, x, y, en, i = aperture(
                (elec_angle, elec_energy, ids), R_foil, R_aperture, d_aperture
            )

            print(f"{en.size / elec_energy.size:.3g} of the electrons passed thru the aperture")

            if "efficiency" in configuration_name:
                bins = np.linspace(E0 - 2*MeV, E0, 101)
            else:
                bins = np.linspace(E0 - 1*MeV, E0, 101)
            N_PP, _ = np.histogram(en[i == 1], bins=bins)
            N_compton, _ = np.histogram(en[i == 0], bins=bins)

            return bins, N_PP, N_compton, en[i == 0].size

        N_γ = 1e4/expected_efficiency
        bins, N_PP, N_compton, N_total = do_monte_carlo(E0, N_γ)

        f = (N_PP + N_compton)/np.diff(bins)/N_γ

        E_max = E0 - 0.255*MeV
        x = np.linspace(bins[0], E_max, 301)
        y = np.where(
            x > E_max - ΔE,
            expected_spectrum*(1 - .4*np.exp((x - E_max)/ΔE_geom_1) - .6*np.exp((x - E_max)/ΔE_geom_2)),
            expected_spectrum*(.4*(np.exp((x - E_max + ΔE)/ΔE_geom_1) - np.exp((x - E_max)/ΔE_geom_1)) + .6*(np.exp((x - E_max + ΔE)/ΔE_geom_2) - np.exp((x - E_max)/ΔE_geom_2))),
        )

        plt.figure(facecolor="none", figsize=(3.5, 3.0))
        plt.locator_params(steps=[1, 2, 5, 10])
        plt.fill_between(np.repeat(bins, 2)[1:-1]/MeV, np.repeat(f, 2)/(MeV**-1), 0, color="C2", edgecolor="none")
        if do_ranging:
            plt.plot(
                np.array([E_max - ΔE, E_max - ΔE, E_max, E_max])/MeV,
                np.array([0, expected_spectrum, expected_spectrum, 0])/(MeV**-1),
                color="k", linewidth=0.8, linestyle="--",
            )
        plt.plot(x/MeV, y/(MeV**-1), color="k", linewidth=1.2, linestyle="-")
        plt.title(f"{configuration_name} config., {material_name} foil")
        plt.ylabel("Distribution (MeV⁻¹/photon)")
        plt.xlabel("Energy (MeV)")
        plt.ylim(0, None)
        plt.xlim(bins[0]/MeV, bins[-1]/MeV)
        print(f"Analytic efficiency: {expected_efficiency:.3g}")
        print(f"Monte-Carlo efficiency: {N_total/N_γ:.3g}")
        plt.tight_layout()
        if "ideal" not in material_name:
            plt.savefig(f"C:/Users/kunimune/Dropbox/MERGS/figures/spectrum {material_name} {configuration_name.lower()}.pdf")

        plt.show()
