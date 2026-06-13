import numpy as np

def magnetization(spins):
    return np.mean(spins)

def energy(spins, J=1.0):
    R = np.zeros_like(spins)
    D = np.zeros_like(spins)
    R[:, :-1] = spins[:, 1:]
    R[:, -1] = spins[:, 0]
    D[:-1, :] = spins[1:, :]
    D[-1, :] = spins[0, :]

    return -J * np.sum(spins * (R + D))

def nearest_neighbor_correlation(spins):
    R = np.zeros_like(spins)
    D = np.zeros_like(spins)
    R[:, :-1] = spins[:, 1:]
    R[:, -1] = spins[:, 0]
    D[:-1, :] = spins[1:, :]
    D[-1, :] = spins[0, :]

    n_row, n_col = spins.shape
    return np.sum(spins * (R + D)) / (2 * n_row * n_col)

def domain_wall_density(spins):
    R = np.zeros_like(spins)
    D = np.zeros_like(spins)
    R[:, :-1] = spins[:, 1:]
    R[:, -1] = spins[:, 0]
    D[:-1, :] = spins[1:, :]
    D[-1, :] = spins[0, :]

    right = np.sum(spins != R)
    down = np.sum(spins != D)
    n_row, n_col = spins.shape
    return (right + down) / (2 * n_row * n_col)
