"""
Common Spatial Patterns (CSP) Implementation using Generalized Eigenvalue Solving
"""
import numpy as np
from scipy.linalg import eigh

class CommonSpatialPatterns:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        """Calculates CSP spatial filter matrix W."""
        covs_class0 = [np.cov(trial) for trial, label in zip(X, y) if label == 0]
        covs_class1 = [np.cov(trial) for trial, label in zip(X, y) if label == 1]
        
        R0 = np.mean(covs_class0, axis=0)
        R1 = np.mean(covs_class1, axis=0)
        
        # Generalized Eigenvalue Decomposition: R0 * w = lambda * (R0 + R1) * w
        evals, evecs = eigh(R0, R0 + R1)
        
        # Sort eigenvectors by magnitude
        ix = np.argsort(evals)[::-1]
        evecs = evecs[:, ix]
        
        # Select top and bottom m/2 spatial components
        half_m = self.n_components // 2
        self.filters_ = np.vstack([evecs[:, :half_m].T, evecs[:, -half_m:].T])
        return self

    def transform(self, X):
        """Projects EEG signals and extracts log-variance features."""
        features = []
        for trial in X:
            Z = np.dot(self.filters_, trial)  # Projected spatial signal
            var = np.var(Z, axis=1)
            log_var = np.log(var / np.sum(var))
            features.append(log_var)
        return np.array(features)
