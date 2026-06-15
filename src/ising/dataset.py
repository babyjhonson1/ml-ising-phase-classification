import numpy as np
from src.ising.simulation import random_spins, thermalize, metropolis_sweep

TC_2D_ISING = 2.269

def phase_label(T, tc=TC_2D_ISING):
    return 0 if T < tc else 1

def generate_snapshot_dataset(
    L=16,
    temperatures=None,
    samples_per_temperature=100,
    thermalization_sweeps=500,
    sampling_interval=20,
    J=1.0,
    seed=42,
):
    if temperatures is None:
        temperatures = np.linspace(1.0, 4.0, 60)

    np.random.seed(seed)

    X = []
    y = []
    T_values = []

    for temperature_index, T in enumerate(temperatures):
        spins = random_spins(L=L, seed=seed + temperature_index)

        thermalize(spins, T, thermalization_sweeps, J)

        for _ in range(samples_per_temperature):
            for _ in range(sampling_interval):
                metropolis_sweep(spins, T=T, J=J)

            X.append(spins.copy())
            y.append(phase_label(T))
            T_values.append(T)

    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int64)
    T_values = np.array(T_values, dtype=np.float32)

    return X, y, T_values