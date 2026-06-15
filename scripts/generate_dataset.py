from pathlib import Path
import numpy as np
from src.ising.dataset import generate_snapshot_dataset


def main():
    output_path = Path("data/processed/ising_snapshots.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, T = generate_snapshot_dataset(
        L=16,
        temperatures=np.linspace(1.0, 4.0, 80),
        samples_per_temperature=50,
        thermalization_sweeps=500,
        sampling_interval=20,
        seed=42,
    )

    np.savez("data/processed/ising_snapshots.npz", X, y, T)

if __name__ == "__main__":
    main()