import numpy as np

def random_spins(L, seed=None):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(L, L))

def delta_energy_flip(spins, i, j, J=1.0):
    n_row, n_col = spins.shape
    s_right = spins[i, (j + 1) % n_col]
    s_left = spins[i, (j - 1) % n_col]
    s_down = spins[(i - 1) % n_row, j]
    s_up = spins[(i + 1) % n_row, j]

    return 2 * J * spins[i, j] * (s_down + s_left + s_right + s_up)

def metropolis_step(spins, T, J=1.0):
    n_row, n_col = spins.shape
    i = np.random.randint(0, n_row)
    j = np.random.randint(0, n_col)
    delta_E = delta_energy_flip(spins, i, j, J)
    if delta_E <= 0:
        spins[i, j] = -spins[i, j]
    else:
        prob = np.exp(-delta_E / T)
        eps = np.random.uniform()
        if eps <= prob:
            spins[i, j] = -spins[i, j]
    

def metropolis_sweep(spins, T, J=1.0):
    n_row, n_col = spins.shape
    for _ in range(n_row * n_col):
        metropolis_step(spins, T, J)

def thermalize(spins, T, n_sweeps, J=1.0):
    for _ in range(n_sweeps):
        metropolis_sweep(spins, T, J)
