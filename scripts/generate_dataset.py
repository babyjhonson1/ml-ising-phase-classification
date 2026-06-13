from pathlib import Path
import numpy as np
from src.ising.dataset import generate_observable_dataset

def main():
    df = generate_observable_dataset(
        L=16,
        temperatures=np.linspace(0.5, 5.0, 50),
        samples_per_temperature=30,
        thermalization_sweeps=300,
        sampling_interval=10,
        seed=42,
    )

    df.to_csv("data/processed/ising_observables.csv", index=False)

    print(df.head())
    print()
    print(df.tail())
    print()

if __name__ == "__main__":
    main()