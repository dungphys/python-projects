import numpy as np
import pandas as pd

PARTICLE_CLASSES = ["photon", "electron", "muon", "pion"]

def generate_particle_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate particle detector measurements.
    Each class has distinct physics-inspired feature distributions.
    """
    rng = np.random.default_rng(seed)
    records = []

    samples_per_class = n_samples // len(PARTICLE_CLASSES)

    for label, particle in enumerate(PARTICLE_CLASSES):
        # Physics-inspired feature distributions per particle type
        if particle == "photon":
            energy     = rng.exponential(scale=50, size=samples_per_class)
            charge     = np.zeros(samples_per_class)
            hit_scale  = 0.8
        elif particle == "electron":
            energy     = rng.normal(loc=80, scale=20, size=samples_per_class)
            charge     = rng.choice([-1, 1], size=samples_per_class)
            hit_scale  = 1.2
        elif particle == "muon":
            energy     = rng.normal(loc=200, scale=50, size=samples_per_class)
            charge     = rng.choice([-1, 1], size=samples_per_class)
            hit_scale  = 0.5
        else:  # pion
            energy     = rng.gamma(shape=3, scale=30, size=samples_per_class)
            charge     = rng.choice([-1, 0, 1], size=samples_per_class)
            hit_scale  = 1.5

        momentum = rng.normal(loc=0, scale=energy[:, None] * 0.3,
                              size=(samples_per_class, 3))
        hits = rng.poisson(lam=hit_scale * 10, size=(samples_per_class, 4))

        df = pd.DataFrame({
            "energy":      energy,
            "momentum_x":  momentum[:, 0],
            "momentum_y":  momentum[:, 1],
            "momentum_z":  momentum[:, 2],
            "charge":      charge,
            "hit_0":       hits[:, 0],
            "hit_1":       hits[:, 1],
            "hit_2":       hits[:, 2],
            "hit_3":       hits[:, 3],
            "label":       label,
            "particle":    particle,
        })
        records.append(df)

    return pd.concat(records, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)