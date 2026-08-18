"""
Condition-aligned multimodal autoencoder for UNPAIRED RNA-seq and proteomics.

Use this module when RNA-seq and protein measurements come from independent
biological samples, but share experimental conditions such as treatment and
 timepoint. Individual RNA and protein rows are never paired.

The cross-modal objective is condition-level:

    mean(Protein_hat_from_RNA | condition)
        ~ mean(Protein_observed | condition)

and the latent alignment objective is also condition-level:

    mean(z_RNA | condition) ~ mean(z_protein | condition)

This avoids inventing replicate pairings while still learning a shared latent
representation anchored by experimental conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Data container and loading
# ---------------------------------------------------------------------------

REQUIRED_META = {"treatment", "timepoint"}


def _condition_key(meta: pd.DataFrame, columns=("treatment", "timepoint")) -> np.ndarray:
    return meta.loc[:, list(columns)].astype(str).agg("|".join, axis=1).to_numpy()


def _check_meta(meta: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_META - set(meta.columns)
    if missing:
        raise ValueError(f"{label} metadata is missing columns: {sorted(missing)}")


@dataclass
class UnpairedOmicsData:
    """RNA and protein matrices with independent sample metadata.

    RNA and protein samples may have different IDs and different replicate names.
    The only cross-modal anchor is the experimental condition defined by
    ``condition_columns`` (default: treatment x timepoint).
    """

    rna: pd.DataFrame
    prot: pd.DataFrame
    rna_meta: pd.DataFrame
    prot_meta: pd.DataFrame
    cognate: Dict[str, str] = field(default_factory=dict)
    condition_columns: Tuple[str, ...] = ("treatment", "timepoint")

    def __post_init__(self):
        _check_meta(self.rna_meta, "RNA")
        _check_meta(self.prot_meta, "protein")
        if list(self.rna.index) != list(self.rna_meta.index):
            raise ValueError("RNA matrix and RNA metadata must share the same sample order")
        if list(self.prot.index) != list(self.prot_meta.index):
            raise ValueError("protein matrix and protein metadata must share the same sample order")
        for c in self.condition_columns:
            if c not in self.rna_meta or c not in self.prot_meta:
                raise ValueError(f"condition column {c!r} must exist in both metadata tables")
        if len(self.shared_conditions) < 2:
            raise ValueError("fewer than two experimental conditions are shared across modalities")

    @property
    def n_rna_samples(self) -> int:
        return self.rna.shape[0]

    @property
    def n_prot_samples(self) -> int:
        return self.prot.shape[0]

    @property
    def rna_conditions(self) -> np.ndarray:
        return _condition_key(self.rna_meta, self.condition_columns)

    @property
    def prot_conditions(self) -> np.ndarray:
        return _condition_key(self.prot_meta, self.condition_columns)

    @property
    def shared_conditions(self) -> List[str]:
        r = set(self.rna_conditions)
        p = set(self.prot_conditions)
        # preserve the RNA metadata order for deterministic notebooks
        return [c for c in pd.unique(self.rna_conditions) if c in r and c in p]

    def describe(self) -> str:
        rc = pd.Series(self.rna_conditions).value_counts().sort_index()
        pc = pd.Series(self.prot_conditions).value_counts().sort_index()
        return (
            f"RNA: {self.n_rna_samples} samples x {self.rna.shape[1]:,} genes\n"
            f"Protein: {self.n_prot_samples} samples x {self.prot.shape[1]:,} proteins\n"
            f"Shared conditions: {len(self.shared_conditions)}\n"
            f"RNA replicates/condition: {rc.min()}-{rc.max()}\n"
            f"Protein replicates/condition: {pc.min()}-{pc.max()}\n"
            f"Cognate pairs mapped: {len(self.cognate):,}"
        )


def load_unpaired_from_csv(
    rna_path,
    prot_path,
    rna_meta_path,
    prot_meta_path=None,
    mapping_path=None,
    variety_label=None,
    features_are_rows=True,
    condition_columns=("treatment", "timepoint"),
) -> UnpairedOmicsData:
    """Load independent RNA and protein sample sets.

    ``rna_meta_path`` and ``prot_meta_path`` may point to the same metadata file
    when both matrices use the same *labels* for condition annotation. That does
    not create biological pairing: this loader subsets each modality separately
    and downstream code never aligns rows by RNA/protein sample ID.
    """

    rna = pd.read_csv(rna_path, index_col=0)
    prot = pd.read_csv(prot_path, index_col=0)
    if features_are_rows:
        rna, prot = rna.T, prot.T

    rna_meta = pd.read_csv(rna_meta_path, index_col=0)
    prot_meta = pd.read_csv(prot_meta_path or rna_meta_path, index_col=0)

    if variety_label is not None:
        rna_meta = rna_meta.copy()
        prot_meta = prot_meta.copy()
        rna_meta["variety"] = variety_label
        prot_meta["variety"] = variety_label

    rna_ids = [s for s in rna_meta.index if s in rna.index]
    prot_ids = [s for s in prot_meta.index if s in prot.index]
    if not rna_ids:
        raise ValueError("no RNA sample IDs in RNA metadata are present in the RNA matrix")
    if not prot_ids:
        raise ValueError("no protein sample IDs in protein metadata are present in the protein matrix")
    if len(rna_ids) < len(rna_meta):
        print(f"[load] RNA: using {len(rna_ids)}/{len(rna_meta)} metadata samples present in matrix")
    if len(prot_ids) < len(prot_meta):
        print(f"[load] protein: using {len(prot_ids)}/{len(prot_meta)} metadata samples present in matrix")

    cognate = {}
    if mapping_path is not None and Path(mapping_path).exists():
        mp = pd.read_csv(mapping_path)
        if mp.shape[1] < 2:
            raise ValueError("mapping file must have at least two columns: protein_id, gene_id")
        cognate = dict(zip(mp.iloc[:, 0].astype(str), mp.iloc[:, 1].astype(str)))

    return UnpairedOmicsData(
        rna=rna.loc[rna_ids],
        prot=prot.loc[prot_ids],
        rna_meta=rna_meta.loc[rna_ids],
        prot_meta=prot_meta.loc[prot_ids],
        cognate=cognate,
        condition_columns=tuple(condition_columns),
    )


# ---------------------------------------------------------------------------
# Leak-free preprocessing; fitted separately in each modality
# ---------------------------------------------------------------------------

class UnpairedPreprocessor:
    def __init__(
        self,
        n_rna=1500,
        n_prot=800,
        rna_mode="counts",
        prot_mode="intensity",
        min_cpm=1.0,
        min_frac=0.5,
        max_missing_frac=0.3,
        required_proteins=None,
    ):
        self.n_rna = n_rna
        self.n_prot = n_prot
        self.rna_mode = rna_mode
        self.prot_mode = prot_mode
        self.min_cpm = min_cpm
        self.min_frac = min_frac
        self.max_missing_frac = max_missing_frac
        self.required_proteins = tuple(map(str, required_proteins or ()))
        self.fitted_ = False

    def _rna_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.rna_mode == "counts":
            lib = X.sum(axis=1).replace(0, np.nan)
            cpm = X.div(lib, axis=0) * 1e6
            return np.log2(cpm + 1.0)
        return X.astype(float)

    def _prot_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.prot_mode == "intensity":
            return np.log2(X.replace(0, np.nan) + 1.0)
        return X.astype(float)

    def fit(self, data: UnpairedOmicsData, rna_train_idx, prot_train_idx):
        rna_train_idx = np.asarray(rna_train_idx, int)
        prot_train_idx = np.asarray(prot_train_idx, int)

        rna_all = self._rna_transform(data.rna)
        prot_all = self._prot_transform(data.prot)
        rna = rna_all.iloc[rna_train_idx]
        prot = prot_all.iloc[prot_train_idx]

        if self.rna_mode == "counts":
            raw = data.rna.iloc[rna_train_idx]
            lib = raw.sum(axis=1).replace(0, np.nan)
            cpm = raw.div(lib, axis=0) * 1e6
            keep_rna = (cpm >= self.min_cpm).mean(axis=0) >= self.min_frac
        else:
            keep_rna = rna.notna().mean(axis=0) >= self.min_frac
        rna = rna.loc[:, keep_rna[keep_rna].index].fillna(0.0)

        keep_prot = prot.isna().mean(axis=0) <= self.max_missing_frac
        prot = prot.loc[:, keep_prot[keep_prot].index]
        self.prot_impute_ = prot.min(axis=0) - 1.0
        prot_i = prot.fillna(self.prot_impute_)

        self.rna_features_ = (
            rna.var(axis=0).sort_values(ascending=False).head(self.n_rna).index
        )
        top_prot = list(prot_i.var(axis=0).sort_values(ascending=False).head(self.n_prot).index)
        # Optional fixed targets are used by permutation tests so the null is
        # evaluated on exactly the same proteins as the observed analysis.
        for name in self.required_proteins:
            if name in prot_i.columns and name not in top_prot:
                top_prot.append(name)
        self.prot_features_ = pd.Index(top_prot)
        if len(self.rna_features_) == 0 or len(self.prot_features_) == 0:
            raise ValueError("preprocessing removed all RNA or protein features")

        rr = rna.loc[:, self.rna_features_]
        pp = prot_i.loc[:, self.prot_features_]
        self.rna_mu_ = rr.mean(axis=0)
        self.rna_sd_ = rr.std(axis=0).replace(0, 1.0).fillna(1.0)
        self.prot_mu_ = pp.mean(axis=0)
        self.prot_sd_ = pp.std(axis=0).replace(0, 1.0).fillna(1.0)
        self.fitted_ = True
        return self

    def transform(self, data: UnpairedOmicsData):
        if not self.fitted_:
            raise RuntimeError("call .fit() first")
        rna = self._rna_transform(data.rna).reindex(columns=self.rna_features_).fillna(0.0)
        prot = self._prot_transform(data.prot).reindex(columns=self.prot_features_)
        prot = prot.fillna(self.prot_impute_.reindex(self.prot_features_))
        R = (rna - self.rna_mu_) / self.rna_sd_
        P = (prot - self.prot_mu_) / self.prot_sd_
        return R.to_numpy(np.float32), P.to_numpy(np.float32)

    def inverse_protein(self, P, protein_names=None):
        """Back-transform standardised protein values to the transformed log scale."""
        P = np.asarray(P, float)
        names = list(self.prot_features_ if protein_names is None else protein_names)
        mu = self.prot_mu_.reindex(names).to_numpy(float)
        sd = self.prot_sd_.reindex(names).to_numpy(float)
        return P * sd + mu


# ---------------------------------------------------------------------------
# Condition-aware splits and aggregation
# ---------------------------------------------------------------------------

def leave_condition_out(data: UnpairedOmicsData):
    """Hold out one complete treatment x timepoint cell in BOTH modalities."""
    rc = data.rna_conditions
    pc = data.prot_conditions
    splits = []
    for cond in data.shared_conditions:
        rte = np.where(rc == cond)[0]
        pte = np.where(pc == cond)[0]
        rtr = np.where(rc != cond)[0]
        ptr = np.where(pc != cond)[0]
        if len(rte) and len(pte):
            splits.append(
                dict(
                    held_condition=cond,
                    rna_train=rtr,
                    rna_test=rte,
                    prot_train=ptr,
                    prot_test=pte,
                )
            )
    return splits


def leave_treatment_out(data: UnpairedOmicsData):
    """Hold out one treatment in BOTH modalities; a strong extrapolation test."""
    rt = data.rna_meta["treatment"].astype(str).to_numpy()
    pt = data.prot_meta["treatment"].astype(str).to_numpy()
    common = [t for t in pd.unique(rt) if t in set(pt)]
    splits = []
    for t in common:
        splits.append(
            dict(
                held_condition=f"treatment={t}",
                rna_train=np.where(rt != t)[0],
                rna_test=np.where(rt == t)[0],
                prot_train=np.where(pt != t)[0],
                prot_test=np.where(pt == t)[0],
            )
        )
    return splits


def _shared_conditions_for_indices(data, rna_idx, prot_idx):
    rc = data.rna_conditions[np.asarray(rna_idx, int)]
    pc = data.prot_conditions[np.asarray(prot_idx, int)]
    pset = set(pc)
    return [c for c in pd.unique(rc) if c in pset]


def _means_by_condition(X, conditions, indices, ordered_conditions):
    indices = np.asarray(indices, int)
    conditions = np.asarray(conditions)
    out = []
    for c in ordered_conditions:
        ii = indices[conditions[indices] == c]
        if len(ii) == 0:
            raise ValueError(f"condition {c!r} has no samples in requested subset")
        out.append(np.nanmean(X[ii], axis=0))
    return np.asarray(out, np.float32)


def condition_mean_matrices(data, R, P, rna_idx, prot_idx, conditions=None):
    conditions = conditions or _shared_conditions_for_indices(data, rna_idx, prot_idx)
    XR = _means_by_condition(R, data.rna_conditions, rna_idx, conditions)
    YP = _means_by_condition(P, data.prot_conditions, prot_idx, conditions)
    return XR, YP, list(conditions)


def _condition_metadata(data: UnpairedOmicsData) -> pd.DataFrame:
    """One row per shared condition, used only for design-only baselines/plots."""
    rows = []
    for cond in data.shared_conditions:
        hit = np.where(data.rna_conditions == cond)[0][0]
        row = data.rna_meta.iloc[hit].copy()
        row["condition"] = cond
        rows.append(row)
    return pd.DataFrame(rows).set_index("condition")


def condition_design_matrix(data: UnpairedOmicsData) -> pd.DataFrame:
    """Main-effect treatment/timepoint dummy matrix for all known conditions.

    The coding schema uses the experimental design labels only, never held-out
    protein values, so constructing the column schema before CV is safe.
    """
    m = _condition_metadata(data)
    return pd.get_dummies(
        m[["treatment", "timepoint"]].astype(str),
        drop_first=True, dtype=float
    )


# ---------------------------------------------------------------------------
# Neural model
# ---------------------------------------------------------------------------

def _mlp(sizes, dropout=0.1):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.SiLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class ConditionAlignedAE(nn.Module):
    """Two autoencoders whose latent spaces are aligned by condition centroids."""

    def __init__(self, n_rna, n_prot, hidden=(64, 16), latent=6, dropout=0.1):
        super().__init__()
        self.enc_r = _mlp([n_rna, *hidden, latent], dropout)
        self.enc_p = _mlp([n_prot, *hidden, latent], dropout)
        self.dec_r = _mlp([latent, *reversed(hidden), n_rna], dropout)
        self.dec_p = _mlp([latent, *reversed(hidden), n_prot], dropout)

    def encode_rna(self, rna):
        return self.enc_r(rna)

    def encode_prot(self, prot):
        return self.enc_p(prot)

    def predict_protein_from_rna(self, rna):
        return self.dec_p(self.enc_r(rna))


def _torch_condition_losses(
    model,
    R,
    P,
    rna_cond,
    prot_cond,
    weights=(0.5, 0.5, 2.0, 0.5),
):
    """Reconstruction + condition-mean cross-modal + centroid alignment loss."""
    z_r = model.encode_rna(R)
    z_p = model.encode_prot(P)
    r_hat = model.dec_r(z_r)
    p_hat = model.dec_p(z_p)
    p_from_r = model.dec_p(z_r)

    mse = nn.functional.mse_loss
    l_r = mse(r_hat, R)
    l_p = mse(p_hat, P)

    shared = [c for c in pd.unique(rna_cond) if c in set(prot_cond)]
    cross_terms = []
    align_terms = []
    for c in shared:
        ri = torch.as_tensor(np.where(np.asarray(rna_cond) == c)[0], device=R.device)
        pi = torch.as_tensor(np.where(np.asarray(prot_cond) == c)[0], device=P.device)
        pred_centroid = p_from_r.index_select(0, ri).mean(dim=0)
        obs_centroid = P.index_select(0, pi).mean(dim=0)
        cross_terms.append(mse(pred_centroid, obs_centroid))
        zr_centroid = z_r.index_select(0, ri).mean(dim=0)
        zp_centroid = z_p.index_select(0, pi).mean(dim=0)
        align_terms.append(mse(zr_centroid, zp_centroid))

    if not cross_terms:
        raise ValueError("training subset contains no conditions represented in both modalities")
    l_cross = torch.stack(cross_terms).mean()
    l_align = torch.stack(align_terms).mean()
    total = weights[0] * l_r + weights[1] * l_p + weights[2] * l_cross + weights[3] * l_align
    parts = dict(
        rna=float(l_r.detach()),
        protein=float(l_p.detach()),
        condition_cross=float(l_cross.detach()),
        centroid_align=float(l_align.detach()),
    )
    return total, parts


class ConditionAlignedAERegressor:
    name = "condition_aligned_ae"

    def __init__(
        self,
        latent=6,
        hidden=(64, 16),
        dropout=0.1,
        lr=1e-3,
        weight_decay=3e-4,
        epochs=500,
        patience=80,
        weights=(0.5, 0.5, 2.0, 0.5),
        seed=0,
        device="cpu",
        verbose=False,
    ):
        self.latent = latent
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.weights = tuple(weights)
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def _inner_condition_split(self, rna_cond, prot_cond):
        shared = [c for c in pd.unique(rna_cond) if c in set(prot_cond)]
        if len(shared) < 3:
            return None
        rng = np.random.default_rng(self.seed)
        val_cond = shared[int(rng.integers(len(shared)))]
        return val_cond

    def fit(self, R, P, rna_conditions, prot_conditions):
        R = np.asarray(R, np.float32)
        P = np.asarray(P, np.float32)
        rna_conditions = np.asarray(rna_conditions).astype(str)
        prot_conditions = np.asarray(prot_conditions).astype(str)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(self.device)

        val_cond = self._inner_condition_split(rna_conditions, prot_conditions)
        if val_cond is None:
            r_train = np.arange(len(R)); p_train = np.arange(len(P))
            r_val = np.array([], int); p_val = np.array([], int)
        else:
            r_val = np.where(rna_conditions == val_cond)[0]
            p_val = np.where(prot_conditions == val_cond)[0]
            r_train = np.where(rna_conditions != val_cond)[0]
            p_train = np.where(prot_conditions != val_cond)[0]

        t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev)
        Rt, Pt = t(R), t(P)
        self.model_ = ConditionAlignedAE(
            R.shape[1], P.shape[1], hidden=self.hidden, latent=self.latent, dropout=self.dropout
        ).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, self.epochs))

        best = np.inf
        best_state = None
        bad = 0
        self.history_ = []
        for ep in range(self.epochs):
            self.model_.train()
            loss, parts = _torch_condition_losses(
                self.model_, Rt[r_train], Pt[p_train],
                rna_conditions[r_train], prot_conditions[p_train], self.weights
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
            opt.step()
            sched.step()

            self.model_.eval()
            with torch.no_grad():
                if len(r_val) and len(p_val):
                    pred = self.model_.predict_protein_from_rna(Rt[r_val]).mean(dim=0)
                    obs = Pt[p_val].mean(dim=0)
                    vloss = nn.functional.mse_loss(pred, obs).item()
                else:
                    vloss = parts["condition_cross"]
            self.history_.append(vloss)
            if vloss < best - 1e-6:
                best = vloss
                bad = 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_ = float(best)
        self.epochs_run_ = ep + 1
        if self.verbose:
            print(f"    [condition-AE] epochs={self.epochs_run_} val condition-MSE={self.best_val_:.4f}")
        return self

    @torch.no_grad()
    def predict_samples(self, R):
        self.model_.eval()
        dev = next(self.model_.parameters()).device
        Rt = torch.tensor(np.asarray(R, np.float32), device=dev)
        return self.model_.predict_protein_from_rna(Rt).cpu().numpy()

    def predict_condition(self, R):
        return self.predict_samples(R).mean(axis=0, keepdims=True)

    @torch.no_grad()
    def encode_rna(self, R):
        self.model_.eval()
        dev = next(self.model_.parameters()).device
        Rt = torch.tensor(np.asarray(R, np.float32), device=dev)
        return self.model_.encode_rna(Rt).cpu().numpy()

    @torch.no_grad()
    def encode_protein(self, P):
        self.model_.eval()
        dev = next(self.model_.parameters()).device
        Pt = torch.tensor(np.asarray(P, np.float32), device=dev)
        return self.model_.encode_prot(Pt).cpu().numpy()


# ---------------------------------------------------------------------------
# Condition-level baselines
# ---------------------------------------------------------------------------

def _mean_predict(Ytr):
    return Ytr.mean(axis=0, keepdims=True)


def _design_predict(Ytr, train_conditions, test_conditions, design_df, alpha=5.0):
    Xtr = design_df.loc[list(train_conditions)].to_numpy(float)
    Xte = design_df.loc[list(test_conditions)].to_numpy(float)
    return Ridge(alpha=alpha, fit_intercept=True).fit(Xtr, Ytr).predict(Xte)


def _pca_ridge_predict(Xtr, Ytr, Xte, n_components=3, alpha=10.0):
    k = max(1, min(int(n_components), len(Xtr) - 1, Xtr.shape[1]))
    pca = PCA(n_components=k, random_state=0).fit(Xtr)
    return Ridge(alpha=alpha).fit(pca.transform(Xtr), Ytr).predict(pca.transform(Xte))


def _pls_predict(Xtr, Ytr, Xte, n_components=2):
    k = max(1, min(int(n_components), len(Xtr) - 1, Xtr.shape[1], Ytr.shape[1]))
    return PLSRegression(n_components=k, scale=False, max_iter=1000).fit(Xtr, Ytr).predict(Xte)


def _loo_select_components(X, Y, method, max_components, alpha=10.0):
    """Choose latent dimension using only the OUTER training conditions.

    With eight total experimental cells, a fixed 3-component PCA model is
    relatively flexible. This inner leave-one-condition-out loop chooses the
    smallest-dimensional model supported by the outer-training data.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = len(X)
    max_k = max(1, min(int(max_components), n - 2, X.shape[1], Y.shape[1]))
    candidates = list(range(1, max_k + 1))
    scores = []
    for k in candidates:
        losses = []
        for i in range(n):
            tr = np.arange(n) != i
            te = ~tr
            if method == "pca":
                pred = _pca_ridge_predict(X[tr], Y[tr], X[te], n_components=k, alpha=alpha)
            elif method == "pls":
                pred = _pls_predict(X[tr], Y[tr], X[te], n_components=k)
            else:
                raise ValueError(f"unknown method {method!r}")
            losses.append(float(np.mean((Y[te] - pred) ** 2)))
        scores.append(np.mean(losses))
    # Stable tie break: prefer the simpler model.
    best = min(range(len(candidates)), key=lambda i: (scores[i], candidates[i]))
    return candidates[best], pd.DataFrame({"n_components": candidates, "inner_mse": scores})


def _design_residual_predict(
    Xtr, Ytr, Xte, train_conditions, test_conditions, design_df,
    method="pca", n_components=2, design_alpha=1e-6, ridge_alpha=10.0,
):
    """Predict protein variation *beyond* treatment/timepoint main effects.

    Both modalities are first residualised against the design using the OUTER
    training conditions only. A PCA+ridge or PLS model then predicts protein
    residuals from RNA residuals, and the protein design prediction is added
    back for the held-out condition.
    """
    Dtr = design_df.loc[list(train_conditions)].to_numpy(float)
    Dte = design_df.loc[list(test_conditions)].to_numpy(float)
    x_design = Ridge(alpha=design_alpha, fit_intercept=True).fit(Dtr, Xtr)
    y_design = Ridge(alpha=design_alpha, fit_intercept=True).fit(Dtr, Ytr)
    Xtr_res = Xtr - x_design.predict(Dtr)
    Xte_res = Xte - x_design.predict(Dte)
    Ytr_res = Ytr - y_design.predict(Dtr)
    y_base = y_design.predict(Dte)
    if method == "pca":
        delta = _pca_ridge_predict(
            Xtr_res, Ytr_res, Xte_res, n_components=n_components, alpha=ridge_alpha
        )
    elif method == "pls":
        delta = _pls_predict(Xtr_res, Ytr_res, Xte_res, n_components=n_components)
    else:
        raise ValueError(f"unknown method {method!r}")
    return y_base + delta, Xtr_res, Ytr_res


def _cognate_ridge_predict_unfiltered(
    data, pre, split, Ytr, train_conditions, test_conditions, protein_names,
    alpha=5.0, fallback=None,
):
    """Cognate transcript -> protein baseline without top-N RNA selection.

    Reviewer-facing change: cognate transcripts are *not* required to survive
    the generic top-variance RNA filter. Every mapped cognate gene present in
    the RNA matrix is tested. The RNA transform is sample-wise (e.g. CPM/log2);
    scaling is learned from outer-training condition means only.

    Returns prediction plus booleans indicating whether each protein had a
    mapped/present cognate transcript and whether that transcript varied across
    the training conditions.
    """
    if fallback is None:
        fallback = np.tile(Ytr.mean(axis=0), (len(test_conditions), 1))
    pred = np.asarray(fallback, float).copy()
    mapped_present = np.zeros(len(protein_names), dtype=bool)
    variable = np.zeros(len(protein_names), dtype=bool)

    rna_log = pre._rna_transform(data.rna)
    rna_columns = {str(c): c for c in rna_log.columns}
    genes = []
    protein_gene = []
    for p in protein_names:
        g = data.cognate.get(str(p))
        g = None if g is None else str(g)
        protein_gene.append(g)
        if g is not None and g in rna_columns and g not in genes:
            genes.append(g)
    if not genes:
        return pred, mapped_present, variable

    gene_cols = [rna_columns[g] for g in genes]
    Xall = rna_log.loc[:, gene_cols].to_numpy(float)
    Xtr = _means_by_condition(
        Xall, data.rna_conditions, split["rna_train"], train_conditions
    )
    Xte = _means_by_condition(
        Xall, data.rna_conditions, split["rna_test"], test_conditions
    )
    mu = np.nanmean(Xtr, axis=0)
    sd_raw = np.nanstd(Xtr, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd_raw) & (sd_raw > 1e-8), sd_raw, 1.0)
    Xtr = np.nan_to_num((Xtr - mu) / sd, nan=0.0)
    Xte = np.nan_to_num((Xte - mu) / sd, nan=0.0)
    gene_idx = {g: i for i, g in enumerate(genes)}

    for j, g in enumerate(protein_gene):
        if g is None or g not in gene_idx:
            continue
        i = gene_idx[g]
        mapped_present[j] = True
        variable[j] = bool(np.isfinite(sd_raw[i]) and sd_raw[i] > 1e-8)
        model = Ridge(alpha=alpha).fit(Xtr[:, [i]], Ytr[:, j])
        pred[:, j] = model.predict(Xte[:, [i]])
    return pred, mapped_present, variable

# ---------------------------------------------------------------------------
# Cross-validation and metrics
# ---------------------------------------------------------------------------

def _fold_preprocessors(data, splits, pre_kwargs, fixed_protein_names=None):
    pres, psets = [], []
    kwargs = dict(pre_kwargs or {})
    if fixed_protein_names is not None:
        kwargs["required_proteins"] = list(map(str, fixed_protein_names))
    for s in splits:
        pre = UnpairedPreprocessor(**kwargs).fit(
            data, s["rna_train"], s["prot_train"]
        )
        pres.append(pre)
        psets.append(set(map(str, pre.prot_features_)))
    common = set.intersection(*psets) if psets else set()
    if fixed_protein_names is None:
        proteins = [str(p) for p in pres[0].prot_features_ if str(p) in common]
    else:
        requested = list(map(str, fixed_protein_names))
        missing = [p for p in requested if p not in common]
        if missing:
            raise ValueError(
                f"{len(missing)} fixed protein targets do not survive preprocessing in every fold"
            )
        proteins = requested
    return pres, proteins


def run_unpaired_cv(
    data: UnpairedOmicsData,
    splits=None,
    pre_kwargs=None,
    ae_kwargs=None,
    pca_components=3,
    pls_components=2,
    tune_components=True,
    models=None,
    fixed_protein_names=None,
    seed=0,
    verbose=True,
):
    """Leak-free condition-held-out evaluation for unpaired multi-omics.

    Reviewer-facing changes:
      * cognate ridge bypasses top-N RNA feature selection;
      * PCA/PLS dimensions can be selected by inner condition-level CV;
      * pca1_ridge diagnoses a dominant one-dimensional/global-response axis;
      * design-residualised PCA/PLS test whether RNA adds information beyond
        treatment/timepoint main effects;
      * per-condition relative R2 is computed across proteins;
      * fold-specific training-mean baselines are retained for leverage analyses.
    """
    splits = leave_condition_out(data) if splits is None else splits
    pre_kwargs = pre_kwargs or {}
    ae_kwargs = ae_kwargs or {}
    default_models = [
        "mean", "design_only", "cognate_ridge", "pca1_ridge",
        "pca_ridge", "pls", "design_resid_pca_ridge",
        "design_resid_pls", "condition_aligned_ae",
    ]
    model_names = default_models if models is None else list(models)
    unknown = set(model_names) - set(default_models)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")

    pres, protein_names = _fold_preprocessors(
        data, splits, pre_kwargs, fixed_protein_names=fixed_protein_names
    )
    if not protein_names:
        raise ValueError("no protein target survives preprocessing in every fold")
    design = condition_design_matrix(data)
    nprot = len(protein_names)
    sse = {m: np.zeros(nprot, float) for m in model_names}
    sst = np.zeros(nprot, float)
    pred_rows = {m: [] for m in model_names}
    true_rows, baseline_rows, fold_rows, conditions_out = [], [], [], []
    cognate_used = np.zeros(nprot, int)
    cognate_variable = np.zeros(nprot, int)
    component_rows = []

    for fold, (s, pre) in enumerate(zip(splits, pres)):
        R, Pfull = pre.transform(data)
        pcol = {str(p): i for i, p in enumerate(pre.prot_features_)}
        take = [pcol[p] for p in protein_names]
        P = Pfull[:, take]
        train_conds = _shared_conditions_for_indices(data, s["rna_train"], s["prot_train"])
        test_conds = _shared_conditions_for_indices(data, s["rna_test"], s["prot_test"])
        Xtr, Ytr, train_conds = condition_mean_matrices(
            data, R, P, s["rna_train"], s["prot_train"], train_conds
        )
        Xte, Yte, test_conds = condition_mean_matrices(
            data, R, P, s["rna_test"], s["prot_test"], test_conds
        )

        if tune_components:
            pca_k, pca_tune = _loo_select_components(
                Xtr, Ytr, "pca", max_components=pca_components, alpha=10.0
            )
            pls_k, pls_tune = _loo_select_components(
                Xtr, Ytr, "pls", max_components=pls_components
            )
        else:
            pca_k, pls_k = int(pca_components), int(pls_components)
            pca_tune = pls_tune = None
        component_rows.append(dict(
            fold=fold, held=s["held_condition"], pca_components=pca_k, pls_components=pls_k
        ))

        pred_std = {}
        mean_pred = np.tile(_mean_predict(Ytr), (len(test_conds), 1))
        if "mean" in model_names:
            pred_std["mean"] = mean_pred
        if "design_only" in model_names:
            pred_std["design_only"] = _design_predict(Ytr, train_conds, test_conds, design)
        if "pca1_ridge" in model_names:
            pred_std["pca1_ridge"] = _pca_ridge_predict(Xtr, Ytr, Xte, n_components=1)
        if "pca_ridge" in model_names:
            pred_std["pca_ridge"] = _pca_ridge_predict(
                Xtr, Ytr, Xte, n_components=pca_k
            )
        if "pls" in model_names:
            pred_std["pls"] = _pls_predict(Xtr, Ytr, Xte, n_components=pls_k)
        if "cognate_ridge" in model_names:
            cpred, cavail, cvar = _cognate_ridge_predict_unfiltered(
                data, pre, s, Ytr, train_conds, test_conds, protein_names,
                fallback=mean_pred,
            )
            pred_std["cognate_ridge"] = cpred
            cognate_used += cavail.astype(int)
            cognate_variable += cvar.astype(int)
        if "design_resid_pca_ridge" in model_names or "design_resid_pls" in model_names:
            # Fit design residualisation on outer-training conditions only.
            Dtr = design.loc[list(train_conds)].to_numpy(float)
            Dte = design.loc[list(test_conds)].to_numpy(float)
            x_design = Ridge(alpha=1e-6, fit_intercept=True).fit(Dtr, Xtr)
            y_design = Ridge(alpha=1e-6, fit_intercept=True).fit(Dtr, Ytr)
            Xtr_res = Xtr - x_design.predict(Dtr)
            Xte_res = Xte - x_design.predict(Dte)
            Ytr_res = Ytr - y_design.predict(Dtr)
            Ybase = y_design.predict(Dte)
            if tune_components:
                rpca_k, _ = _loo_select_components(
                    Xtr_res, Ytr_res, "pca", max_components=pca_components, alpha=10.0
                )
                rpls_k, _ = _loo_select_components(
                    Xtr_res, Ytr_res, "pls", max_components=pls_components
                )
            else:
                rpca_k, rpls_k = pca_k, pls_k
            component_rows[-1]["design_resid_pca_components"] = rpca_k
            component_rows[-1]["design_resid_pls_components"] = rpls_k
            if "design_resid_pca_ridge" in model_names:
                pred_std["design_resid_pca_ridge"] = Ybase + _pca_ridge_predict(
                    Xtr_res, Ytr_res, Xte_res, n_components=rpca_k
                )
            if "design_resid_pls" in model_names:
                pred_std["design_resid_pls"] = Ybase + _pls_predict(
                    Xtr_res, Ytr_res, Xte_res, n_components=rpls_k
                )
        if "condition_aligned_ae" in model_names:
            ae = ConditionAlignedAERegressor(seed=seed + fold, **ae_kwargs)
            ae.fit(
                R[s["rna_train"]], P[s["prot_train"]],
                data.rna_conditions[s["rna_train"]], data.prot_conditions[s["prot_train"]]
            )
            ae_sample_pred = ae.predict_samples(R[s["rna_test"]])
            rtest_cond = data.rna_conditions[s["rna_test"]]
            pred_std["condition_aligned_ae"] = np.asarray([
                ae_sample_pred[rtest_cond == c].mean(axis=0) for c in test_conds
            ])

        Yte_log = pre.inverse_protein(Yte, protein_names)
        Ytr_log = pre.inverse_protein(Ytr, protein_names)
        baseline = Ytr_log.mean(axis=0)
        pred_log = {}
        for name in model_names:
            if name == "mean":
                pred_log[name] = np.tile(baseline, (len(test_conds), 1))
            else:
                pred_log[name] = pre.inverse_protein(pred_std[name], protein_names)

        for row_i, cond in enumerate(test_conds):
            y = Yte_log[row_i]
            true_rows.append(y)
            baseline_rows.append(baseline.copy())
            conditions_out.append(cond)
            sst += (y - baseline) ** 2
            denom_total = float(np.sum((y - baseline) ** 2))
            for name in model_names:
                pp = pred_log[name][row_i]
                pred_rows[name].append(pp)
                err2 = (y - pp) ** 2
                sse[name] += err2
                rel_r2 = np.nan if denom_total <= 1e-12 else 1.0 - float(np.sum(err2)) / denom_total
                fold_rows.append(dict(
                    fold=fold,
                    held=s["held_condition"],
                    condition=cond,
                    model=name,
                    n_test_conditions=len(test_conds),
                    condition_relative_R2=rel_r2,
                    rmse_log=float(np.sqrt(np.mean(err2))),
                    pca_components=pca_k,
                    pls_components=pls_k,
                ))

        if verbose:
            recent = [r for r in fold_rows if r["fold"] == fold]
            best = min(recent, key=lambda x: x["rmse_log"])
            print(
                f"  fold {fold + 1}/{len(splits)} held={s['held_condition']}: "
                f"best={best['model']} RMSE(log)={best['rmse_log']:.3f}; "
                f"PCA k={pca_k}, PLS k={pls_k}"
            )

    oof = dict(
        conditions=conditions_out,
        proteins=protein_names,
        true=np.asarray(true_rows),
        baseline=np.asarray(baseline_rows),
        pred={m: np.asarray(v) for m, v in pred_rows.items()},
        sse=sse,
        sst=sst,
        cognate_used_folds=cognate_used,
        cognate_variable_folds=cognate_variable,
        n_folds=len(splits),
        component_selection=pd.DataFrame(component_rows),
    )
    return pd.DataFrame(fold_rows), oof


def oof_r2_table(oof):
    """Per-protein pooled OOF R2 using each fold's training-mean baseline."""
    y = np.asarray(oof["true"], float)
    b = np.asarray(oof.get("baseline"), float) if "baseline" in oof else None
    if b is not None and b.shape == y.shape:
        denom = np.sum((y - b) ** 2, axis=0)
        denom = np.where(denom == 0, np.nan, denom)
        return pd.DataFrame(
            {name: 1.0 - np.sum((y - np.asarray(pred)) ** 2, axis=0) / denom
             for name, pred in oof["pred"].items()},
            index=oof["proteins"],
        )
    denom = np.where(oof["sst"] == 0, np.nan, oof["sst"])
    return pd.DataFrame(
        {name: 1.0 - oof["sse"][name] / denom for name in oof["pred"]},
        index=oof["proteins"],
    )


def cognate_coverage_table(oof):
    """Show whether the cognate baseline actually used a transcript."""
    n = int(oof.get("n_folds", len(oof.get("conditions", []))))
    used = np.asarray(oof.get("cognate_used_folds", np.zeros(len(oof["proteins"]))), int)
    var = np.asarray(oof.get("cognate_variable_folds", np.zeros(len(oof["proteins"]))), int)
    return pd.DataFrame({
        "protein": oof["proteins"],
        "cognate_present_folds": used,
        "cognate_variable_folds": var,
        "fraction_folds_cognate_present": used / max(n, 1),
        "fraction_folds_cognate_variable": var / max(n, 1),
    }).set_index("protein")


def oof_condition_metrics(oof):
    """Per-held-condition RMSE and multivariate relative R2 across proteins."""
    y = np.asarray(oof["true"], float)
    b = np.asarray(oof["baseline"], float)
    rows = []
    for name, pred in oof["pred"].items():
        p = np.asarray(pred, float)
        for i, cond in enumerate(oof["conditions"]):
            sse = float(np.sum((y[i] - p[i]) ** 2))
            sst = float(np.sum((y[i] - b[i]) ** 2))
            rows.append(dict(
                condition=cond, model=name,
                condition_relative_R2=np.nan if sst <= 1e-12 else 1.0 - sse / sst,
                rmse=float(np.sqrt(np.mean((y[i] - p[i]) ** 2))),
            ))
    return pd.DataFrame(rows)


def condition_score_sensitivity(oof):
    """How much does the pooled median R2 change if one scored condition is removed?

    This is a *metric leverage* diagnostic: models are not retrained. It directly
    answers whether a condition such as T1|t3 disproportionately drives the
    headline pooled R2.
    """
    y = np.asarray(oof["true"], float)
    b = np.asarray(oof["baseline"], float)
    conds = np.asarray(oof["conditions"], str)
    full = oof_r2_table(oof).median(axis=0)
    rows = []
    for omit in pd.unique(conds):
        keep = conds != omit
        denom = np.sum((y[keep] - b[keep]) ** 2, axis=0)
        denom = np.where(denom == 0, np.nan, denom)
        for name, pred in oof["pred"].items():
            p = np.asarray(pred, float)
            r2 = 1.0 - np.sum((y[keep] - p[keep]) ** 2, axis=0) / denom
            med = float(np.nanmedian(r2))
            rows.append(dict(
                omitted_condition=omit, model=name,
                median_R2_without_condition=med,
                full_median_R2=float(full[name]),
                delta_median_R2=med - float(full[name]),
            ))
    return pd.DataFrame(rows)


def oof_condition_rmse(oof):
    rows = []
    y = np.asarray(oof["true"])
    for name, p in oof["pred"].items():
        err = np.asarray(p) - y
        for i, c in enumerate(oof["conditions"]):
            rows.append(
                dict(condition=c, model=name, rmse=float(np.sqrt(np.mean(err[i] ** 2))))
            )
    return pd.DataFrame(rows)


def condition_residuals(oof, model="condition_aligned_ae"):
    """Observed - predicted protein condition means; descriptive, not replicate residuals."""
    D = np.asarray(oof["true"]) - np.asarray(oof["pred"][model])
    return pd.DataFrame(D, index=oof["conditions"], columns=oof["proteins"])


def permutation_null(
    data,
    n_perm=100,
    seed=0,
    models=None,
    reference_proteins=None,
    progress=True,
    **cv_kwargs,
):
    """Condition-label permutation null for cross-modal predictability.

    Protein condition labels are permuted as complete cells, preserving all
    protein replicates within each cell. The same protein targets can be fixed to
    the observed OOF set via ``reference_proteins`` so observed and null median
    R2 values are directly comparable.

    This tests whether the observed RNA/protein CONDITION correspondence is
    stronger than random correspondence. It does *not* establish causality from
    RNA to protein; a shared biological severity axis can still be real signal.
    """
    rng = np.random.default_rng(seed)
    out = []
    original = data.prot_meta.copy()
    unique = list(pd.unique(data.prot_conditions))
    cond_meta = _condition_metadata(data)
    for i in range(int(n_perm)):
        d = copy.copy(data)
        pm = original.copy()
        perm = rng.permutation(unique)
        mapping = dict(zip(unique, perm))
        new_rows = []
        for old in data.prot_conditions:
            vals = cond_meta.loc[mapping.get(old, old)]
            new_rows.append([vals[c] for c in data.condition_columns])
        for j, c in enumerate(data.condition_columns):
            pm[c] = [r[j] for r in new_rows]
        d.prot_meta = pm
        _, oof = run_unpaired_cv(
            d, verbose=False, models=models, fixed_protein_names=reference_proteins, **cv_kwargs
        )
        med = oof_r2_table(oof).median(axis=0)
        row = {"perm": i}
        row.update({str(k): float(v) for k, v in med.items()})
        out.append(row)
        if progress and ((i + 1) % max(1, min(10, n_perm)) == 0 or i + 1 == n_perm):
            print(f"  permutation {i + 1}/{n_perm}")
    return pd.DataFrame(out).set_index("perm")


def permutation_pvalue_table(observed_oof, null_df):
    """One-sided empirical p-values for observed median R2 > permutation null."""
    obs = oof_r2_table(observed_oof).median(axis=0)
    rows = []
    for model in null_df.columns:
        vals = pd.to_numeric(null_df[model], errors="coerce").dropna().to_numpy(float)
        if model not in obs or len(vals) == 0:
            continue
        observed = float(obs[model])
        p = (1.0 + float(np.sum(vals >= observed))) / (len(vals) + 1.0)
        rows.append(dict(
            model=model, observed_median_R2=observed,
            null_mean=float(np.mean(vals)),
            null_median=float(np.median(vals)),
            null_q95=float(np.quantile(vals, 0.95)),
            empirical_p=p, n_perm=len(vals),
        ))
    return pd.DataFrame(rows).sort_values("empirical_p")


# ---------------------------------------------------------------------------
# Full-data fit and latent summaries
# ---------------------------------------------------------------------------

def fit_full_condition_ae(data, pre_kwargs=None, ae_kwargs=None):
    pre_kwargs = pre_kwargs or {}
    ae_kwargs = ae_kwargs or {}
    pre = UnpairedPreprocessor(**pre_kwargs).fit(
        data,
        np.arange(data.n_rna_samples),
        np.arange(data.n_prot_samples),
    )
    R, P = pre.transform(data)
    ae = ConditionAlignedAERegressor(**ae_kwargs).fit(
        R, P, data.rna_conditions, data.prot_conditions
    )
    return pre, ae, R, P


def latent_coordinates(data, ae, R, P):
    zr = ae.encode_rna(R)
    zp = ae.encode_protein(P)
    r = pd.DataFrame(zr, index=data.rna.index, columns=[f"z{i+1}" for i in range(zr.shape[1])])
    p = pd.DataFrame(zp, index=data.prot.index, columns=r.columns)
    r["modality"] = "RNA"; p["modality"] = "Protein"
    r["condition"] = data.rna_conditions; p["condition"] = data.prot_conditions
    return pd.concat([r, p], axis=0)


def latent_centroid_distances(data, ae, R, P):
    zr = ae.encode_rna(R)
    zp = ae.encode_protein(P)
    rows = []
    for c in data.shared_conditions:
        a = zr[data.rna_conditions == c].mean(axis=0)
        b = zp[data.prot_conditions == c].mean(axis=0)
        rows.append(dict(condition=c, latent_centroid_distance=float(np.linalg.norm(a - b))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic unpaired demo
# ---------------------------------------------------------------------------

def simulate_unpaired(
    n_rna_features=400,
    n_proteins=120,
    n_rep_rna=3,
    n_rep_prot=3,
    latent=4,
    seed=0,
):
    """Create independent RNA/protein replicates sharing condition-level biology."""
    rng = np.random.default_rng(seed)
    treatments = ["T0", "T1"]
    times = ["t0", "t1", "t2", "t3"]
    conds = [(t, tm) for tm in times for t in treatments]
    C = len(conds)

    condition_z = rng.normal(size=(C, latent))
    # add structured treatment/time effects so extrapolation is not pure noise
    for i, (t, tm) in enumerate(conds):
        condition_z[i, 0] += 1.2 * (t == "T1")
        condition_z[i, 1] += 0.5 * times.index(tm)

    Wr = rng.normal(scale=0.8, size=(latent, n_rna_features))
    Wp = rng.normal(scale=0.8, size=(latent, n_proteins))
    rna_rows, prot_rows = [], []
    rmeta, pmeta = [], []
    rids, pids = [], []
    for ci, (t, tm) in enumerate(conds):
        for r in range(n_rep_rna):
            z = condition_z[ci] + rng.normal(scale=0.45, size=latent)
            x = z @ Wr + rng.normal(scale=0.7, size=n_rna_features)
            # positive pseudo-counts; loader can treat demo as logged values
            rna_rows.append(x)
            sid = f"R_{t}_{tm}_{r}"
            rids.append(sid)
            rmeta.append(dict(treatment=t, timepoint=tm, replicate=f"r{r}", variety="demo"))
        for r in range(n_rep_prot):
            z = condition_z[ci] + rng.normal(scale=0.50, size=latent)
            y = z @ Wp + rng.normal(scale=0.75, size=n_proteins)
            prot_rows.append(y)
            sid = f"P_{t}_{tm}_{r}"
            pids.append(sid)
            pmeta.append(dict(treatment=t, timepoint=tm, replicate=f"p{r}", variety="demo"))

    genes = [f"gene{i:04d}" for i in range(n_rna_features)]
    prots = [f"prot{i:04d}" for i in range(n_proteins)]
    cognate = {p: genes[i % n_rna_features] for i, p in enumerate(prots)}
    return UnpairedOmicsData(
        rna=pd.DataFrame(rna_rows, index=rids, columns=genes),
        prot=pd.DataFrame(prot_rows, index=pids, columns=prots),
        rna_meta=pd.DataFrame(rmeta, index=rids),
        prot_meta=pd.DataFrame(pmeta, index=pids),
        cognate=cognate,
    )
