R_FOIL = 0.015
L_DRIFT = 0.25
R_APERTURE = 0.015

import numpy as np
import matplotlib.pyplot as plt
from cross_sections import (
    gen_cross_section_compton,
    gen_cross_section_pairproduction,
    load_pairproduction_cross_section,
)
from acceptance import SRXMData, foil_trace, Foil, aperture
from physical_constants import mol, MeV,millimeter, centimeter

x_depth_mm = 0.5
Z = 16
x_density = 2.329085  # [g/cm^3]
x_atomic_weight = 28.085  # [amu | g/mol]
material_name = "si"

x_number_density = x_density * 1e6 / x_atomic_weight * mol  # [/m^3]

with open(f"./_data/estar_{material_name}.txt", "r", encoding="utf8") as srem:
    x_srem = np.array(
        [[float(y) for y in x.split(" ")[:2]] for x in srem.readlines()[9:]]
    )
    x_srem[:, 0] *= MeV  #  MeV  ->  J
    x_srem[:, 1] *= x_density * MeV / centimeter  #  MeV cm^2 / g  ->  J / m
    # x_srem[:, 1] *= 0  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_srem: SRXMData = x_srem[:, 0], x_srem[:, 1]

x_crosssection_compton = gen_cross_section_compton(
    x_number_density * Z,
    np.linspace(2, 20, 50) * MeV,
    np.linspace(0.0, np.pi, 1000),
)
x_crosssection_pairproduction = gen_cross_section_pairproduction(
    Z,
    x_number_density,
    load_pairproduction_cross_section("./_data/pairprod_xsctn_medium.npz"),
)
x_foil: Foil = x_srem, [
    x_crosssection_compton,
    x_crosssection_pairproduction,
]

def do_monte_carlo(
    gamma_energy: float, 
    N: int,        
    do_logs=False,
):
    elec_angle, elec_energy, ids = foil_trace(
        n_rays_incident=N,
        n_srxm_steps=10000,
        phot_energy_in=gamma_energy,
        foil_properties=x_foil,
        foil_depth=x_depth_mm * millimeter,
    )

    output_data = aperture(
        (elec_angle, elec_energy, ids), R_FOIL, R_APERTURE, L_DRIFT
    )
    t, p, x, y, en, i = output_data
    if do_logs:
        print(f"------ {material_name} {x_depth_mm}mm, gamma energy: {gamma_energy/MeV} MeV ------")
        print(f"pre-aperture efficiency: {elec_energy.size / N}")

        foil_area = 3.14159 * (R_FOIL / 1e-2) ** 2
        gamma_per_cm2_MW = 1.5e3

        print(f"post-aperture efficiency: {en.size / N}  [1/{N / en.size }]")
        print(f"electrons per megawatt: {(en.size / N) * gamma_per_cm2_MW * foil_area}")
        print(f"electrons @140MW: {(en.size / N) * gamma_per_cm2_MW * foil_area * 140}")
        print(
            f"post-aperture efficiency [compton only]: {en[i==0].size / N}  [1/{N / en[i==0].size }]"
        )
        print(
            f"electrons per megawatt [compton only]: {(en[i==0].size / N) * gamma_per_cm2_MW * foil_area}"
        )
        print(
            f"electrons @140MW [compton only]: {(en[i==0].size / N) * gamma_per_cm2_MW * foil_area * 140}"
        )

        print(
            f"[{np.min(en[i==0])/MeV} : {np.max(en[i==0])/MeV}] [{np.min(en[i==1])/MeV} : {np.max(en[i==1])/MeV}]"
        )

    bins = np.concatenate([np.linspace(0, 13.75, 50, endpoint=False), 14.75 - np.linspace(1, 0, 51)**2])*MeV
    N_PP, _ = np.histogram(en[i == 1], bins=bins)
    N_compton, _ = np.histogram(en[i == 0], bins=bins)

    return bins, N_PP, N_compton

E0 = 15*MeV
bins, N_PP, N_compton = do_monte_carlo(E0, int(1e9))

N_PP = N_PP/np.sum(N_PP)
N_compton = N_compton/np.sum(N_compton)
E = (bins[0:-1] + bins[1:])/2

f_compton = N_compton/np.diff(bins)
f_PP = N_PP/np.diff(bins)
i_start = np.nonzero(f_compton >= np.max(f_compton)/10)[0][0]
i_peak = np.argmax(f_compton)
i_end = np.nonzero(f_compton >= np.max(f_compton)/10)[0][-1]
start_half_max = np.interp(np.max(f_compton)/2, f_compton[i_start:i_peak + 1], E[i_start:i_peak + 1])
end_half_max = np.interp(np.max(f_compton)/2, f_compton[i_end:i_peak - 1:-1], E[i_end:i_peak - 1:-1])

print(f"Peak width = {(end_half_max - start_half_max)/MeV:.3f} MeV")
print(f"Signal-to-background = ratio * {np.max(f_compton)/np.max(f_PP):.3g}")

Z = np.linspace(1, 30)
compton_to_PP = 1/(0.1102*Z)
signal_to_background = np.max(f_compton)/np.max(f_PP)*compton_to_PP
plt.figure(facecolor="none", figsize=(3.5, 3.0))
plt.locator_params(steps=[1, 2, 5, 10])
plt.plot(Z, signal_to_background)
plt.grid()
plt.xlim(0, 30)
plt.ylim(0, 100)
plt.xlabel("Atomic number")
plt.ylabel("Peak height ratio")
plt.tight_layout()

ratios = np.linspace(0.1, 4, 51)
information_per_electron = np.empty_like(ratios)
for i, ratio in enumerate(ratios):
    N = 1/(1 + ratio)*N_PP + ratio/(1 + ratio)*N_compton
    p = N/np.diff(bins)
    dp_dE0 = -E/E0*np.gradient(p, E)
    dlogp_dE0 = dp_dE0/np.where(N > 0, p, np.nan)
    information_per_electron[i] = np.sum(dlogp_dE0**2*N, where=N > 0)

information_per_compton_electron = information_per_electron/(ratios/(1 + ratios))

plt.figure(facecolor="none", figsize=(3.5, 3.0))
plt.locator_params(steps=[1, 2, 5, 10])
plt.plot(ratios, information_per_compton_electron/(MeV**-2))
plt.grid()
plt.xlabel("Signal-to-PP ratio")
plt.ylabel("Information (nat/MeV²/electron)")
plt.xlim(0, 4)
plt.ylim(0, 1.05*np.max(information_per_compton_electron/(MeV**-2)))
plt.tight_layout()

plt.figure(facecolor="none", figsize=(3.5, 3.0))
plt.locator_params(steps=[1, 2, 5, 10])
p = (f_PP + f_compton)/2
plt.fill_between(E/MeV, p/(MeV**-1), 0)
plt.ylabel("Energy distribution (MeV⁻¹)")
plt.xlabel("Energy (MeV)")
plt.ylim(0, p.max()*1.05/(MeV**-1))
plt.xlim(0, 15)
plt.tight_layout()

plt.show()
