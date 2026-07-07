import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PhysicsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs derived physics features from raw detector measurements.

    Derived features:
    - transverse_momentum (pT): magnitude in x-y plane
    - total_momentum:           3D magnitude
    - pseudorapidity (eta):     beam-axis angle proxy
    - total_hits:               summed detector hits
    - hit_asymmetry:            detector imbalance (left vs right layers)
    - energy_momentum_ratio:    E/|p|  (approaches 1 for massless, >1 for massive)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X columns: energy, px, py, pz, charge, h0, h1, h2, h3
        energy = X[:, 0]
        px, py, pz = X[:, 1], X[:, 2], X[:, 3]
        h0, h1, h2, h3 = X[:, 5], X[:, 6], X[:, 7], X[:, 8]

        pT     = np.sqrt(px**2 + py**2)
        p_tot  = np.sqrt(px**2 + py**2 + pz**2)
        eta    = np.arctanh(np.clip(pz / (p_tot+1e-12), -0.9999, 0.9999))
        hits   = h0 + h1 + h2 + h3
        h_asym = (h0 + h1 - h2 - h3) / (hits + 1e-8)
        ep_rat = energy / (p_tot+1e-12)

        extra = np.stack([pT, p_tot, eta, hits, h_asym, ep_rat], axis=1)
        return np.hstack([X, extra])

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features else []
        return base + ["pT", "p_total", "pseudorapidity",
                       "total_hits", "hit_asymmetry", "energy_momentum_ratio"]