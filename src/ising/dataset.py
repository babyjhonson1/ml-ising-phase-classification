import numpy as np
import pandas as pd

from src.ising.simulation import random_spins, thermalize, metropolis_sweep
from src.ising.observables import magnetization, energy, nearest_neighbor_correlation, domain_wall_density

TC_2D_ISING = 2.269

def generate_observable_dataset(
    L=16,
    temperatures=None,
    samples_per_temperature=50,
    thermalization_sweeps=500,
    sampling_interval=20,
    J=1.0,
    seed=42,
):
    if temperatures is None:
        temperatures = np.linspace(0.2, 5.0, 50)

    np.random.seed(seed)

    records = []

    for T in temperatures:
        spins = random_spins(L=L)

        thermalize(spins, T, thermalization_sweeps, J,)

        abs_magnetizations = []
        energies = []
        correlations = []
        domain_walls = []

        for _ in range(samples_per_temperature):
            for _ in range(sampling_interval):
                metropolis_sweep(spins, T=T, J=J)

            abs_magnetizations.append(abs(magnetization(spins)))
            energies.append(energy(spins, J))
            correlations.append(nearest_neighbor_correlation(spins))
            domain_walls.append(domain_wall_density(spins))

        record = {
            "temperature": float(T),
            "abs_magnetization": float(np.mean(abs_magnetizations)),
            "energy": float(np.mean(energies)),
            "nearest_neighbor_correlation": float(np.mean(correlations)),
            "domain_wall_density": float(np.mean(domain_walls)),
            "phase_label": "ordered" if T < TC_2D_ISING else "disordered",
        }

        records.append(record)

    return pd.DataFrame(records)