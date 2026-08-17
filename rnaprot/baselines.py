"""
Baselines. The autoencoder is only interesting if it beats these.

Every model exposes .fit(R, P, cov) and .predict(R, cov) -> P_hat, so the same
cross-validation loop scores all of them on identical folds.

  MeanBaseline        predicts the training mean protein profile.
                      Defines R2 = 0. Any model below this is worse than useless.
  DesignBaseline      predicts from variety/treatment/time ONLY, no RNA at all.
                      This is the one people forget. If it matches your fancy
                      model, your model learned the experimental design, not
                      RNA-protein regulation.
  CognateBaseline     per protein, ridge on its own transcript (+ design).
                      The textbook biology answer.
  PCARidge            RNA -> k PCs -> ridge -> all proteins. Linear latent
                      space; the closest linear analogue of the autoencoder.
  PLSBaseline         partial least squares, i.e. supervised latent components
                      (same family as DIABLO / sparse PLS).
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class MeanBaseline:
    name = "mean"

    def fit(self, R, P, cov=None):
        self.mu_ = P.mean(axis=0)
        return self

    def predict(self, R, cov=None):
        return np.tile(self.mu_, (R.shape[0], 1))


class DesignBaseline:
    """Uses only the experimental covariates. The critical negative control."""
    name = "design_only"

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, R, P, cov=None):
        self.m_ = Ridge(alpha=self.alpha).fit(cov, P)
        return self

    def predict(self, R, cov=None):
        return self.m_.predict(cov)


class CognateBaseline:
    """Protein j predicted from its own transcript plus design covariates."""
    name = "cognate_ridge"

    def __init__(self, pairs, alpha=1.0, use_cov=True):
        # pairs: array of length n_prot with the RNA column index, -1 if unmapped
        self.pairs = np.asarray(pairs)
        self.alpha, self.use_cov = alpha, use_cov

    def fit(self, R, P, cov=None):
        n, k = P.shape
        self.models_ = []
        for j in range(k):
            g = self.pairs[j]
            x = R[:, [g]] if g >= 0 else np.zeros((n, 1), np.float32)
            X = np.hstack([x, cov]) if (self.use_cov and cov is not None) else x
            self.models_.append(Ridge(alpha=self.alpha).fit(X, P[:, j]))
        return self

    def predict(self, R, cov=None):
        n = R.shape[0]
        out = np.zeros((n, len(self.models_)), np.float32)
        for j, m in enumerate(self.models_):
            g = self.pairs[j]
            x = R[:, [g]] if g >= 0 else np.zeros((n, 1), np.float32)
            X = np.hstack([x, cov]) if (self.use_cov and cov is not None) else x
            out[:, j] = m.predict(X)
        return out


class PCARidge:
    name = "pca_ridge"

    def __init__(self, n_components=10, alpha=10.0, use_cov=True):
        self.k, self.alpha, self.use_cov = n_components, alpha, use_cov

    def fit(self, R, P, cov=None):
        k = min(self.k, R.shape[0] - 2, R.shape[1])
        self.pca_ = PCA(n_components=k).fit(R)
        Z = self.pca_.transform(R)
        if self.use_cov and cov is not None:
            Z = np.hstack([Z, cov])
        self.m_ = Ridge(alpha=self.alpha).fit(Z, P)
        return self

    def predict(self, R, cov=None):
        Z = self.pca_.transform(R)
        if self.use_cov and cov is not None:
            Z = np.hstack([Z, cov])
        return self.m_.predict(Z)


class PLSBaseline:
    name = "pls"

    def __init__(self, n_components=5):
        self.k = n_components

    def fit(self, R, P, cov=None):
        k = min(self.k, R.shape[0] - 2, R.shape[1])
        self.m_ = PLSRegression(n_components=k, scale=False).fit(R, P)
        return self

    def predict(self, R, cov=None):
        return self.m_.predict(R)
