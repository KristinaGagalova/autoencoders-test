"""
Data loading, leak-free preprocessing, and design-aware cross-validation splits.

Design assumed here (edit CONFIG in run_analysis.py if yours differs):
    2 varieties x 2 treatments x 4 timepoints x 3 replicates = 48 samples
    RNA  : ~60,000 genes    (samples x genes after loading)
    PROT : ~6,000 proteins  (samples x proteins after loading)

The single most important idea in this module: every filtering / scaling
decision is FIT ON TRAINING SAMPLES ONLY. With n = 48 and p = 66,000, selecting
"the 2,000 most variable genes" on the full matrix before cross-validation
leaks test information and can invent double-digit R2 out of pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
@dataclass
class OmicsData:
    """Matched RNA / protein matrices plus the experimental design table."""

    rna: pd.DataFrame            # samples x genes
    prot: pd.DataFrame           # samples x proteins
    meta: pd.DataFrame           # samples x [variety, treatment, timepoint, replicate]
    cognate: dict = field(default_factory=dict)   # protein_id -> gene_id

    def __post_init__(self):
        assert list(self.rna.index) == list(self.prot.index) == list(self.meta.index), \
            "RNA, protein and metadata must share the same sample order"

    @property
    def n_samples(self) -> int:
        return self.rna.shape[0]

    def describe(self) -> str:
        m = self.meta
        return (
            f"{self.n_samples} samples | {self.rna.shape[1]:,} genes | "
            f"{self.prot.shape[1]:,} proteins\n"
            f"  varieties : {sorted(m['variety'].unique())}\n"
            f"  treatments: {sorted(m['treatment'].unique())}\n"
            f"  timepoints: {sorted(m['timepoint'].unique())}\n"
            f"  replicates: {sorted(m['replicate'].unique())}\n"
            f"  cognate pairs mapped: {len(self.cognate):,}"
        )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_from_csv(rna_path, prot_path, meta_path, mapping_path=None,
                  features_are_rows=True) -> OmicsData:
    """
    Expected file layout (the usual output of featureCounts / MaxQuant / DIA-NN):

      rna_path  : rows = genes,    columns = sample IDs   (raw counts or TPM)
      prot_path : rows = proteins, columns = sample IDs   (LFQ / iBAQ intensities)
      meta_path : rows = sample IDs, columns =
                  variety, treatment, timepoint, replicate
      mapping_path (optional): two columns protein_id, gene_id
                  -> defines the *cognate* pair used for the discordance analysis
    """
    rna = pd.read_csv(rna_path, index_col=0)
    prot = pd.read_csv(prot_path, index_col=0)
    if features_are_rows:
        rna, prot = rna.T, prot.T

    meta = pd.read_csv(meta_path, index_col=0)
    required = {"variety", "treatment", "timepoint", "replicate"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    common = [s for s in meta.index if s in rna.index and s in prot.index]
    if len(common) < len(meta):
        print(f"[load] warning: using {len(common)}/{len(meta)} samples present in all files")

    cognate = {}
    if mapping_path is not None:
        mp = pd.read_csv(mapping_path)
        cognate = dict(zip(mp.iloc[:, 0], mp.iloc[:, 1]))

    return OmicsData(rna.loc[common], prot.loc[common], meta.loc[common], cognate)


# --------------------------------------------------------------------------
# Preprocessing  (fit on train only)
# --------------------------------------------------------------------------
class OmicsPreprocessor:
    """
    Filter -> transform -> select -> standardise, with all parameters learned
    from the training samples only.

    rna_mode  : 'counts' applies CPM + log2(x+1); 'logged' assumes you already
                did DESeq2/limma-voom style normalisation.
    prot_mode : 'intensity' applies log2(x+1); 'logged' assumes already logged.
    n_rna / n_prot : number of most-variable features kept. With n = 48 samples,
                keeping more than a few thousand genes buys nothing but variance.
    """

    def __init__(self, n_rna=3000, n_prot=2000, rna_mode="counts",
                 prot_mode="intensity", min_cpm=1.0, min_frac=0.5,
                 max_missing_frac=0.3):
        self.n_rna, self.n_prot = n_rna, n_prot
        self.rna_mode, self.prot_mode = rna_mode, prot_mode
        self.min_cpm, self.min_frac = min_cpm, min_frac
        self.max_missing_frac = max_missing_frac
        self.fitted_ = False

    # ---- internal transforms -------------------------------------------
    def _rna_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.rna_mode == "counts":
            lib = X.sum(axis=1).replace(0, np.nan)
            cpm = X.div(lib, axis=0) * 1e6
            return np.log2(cpm + 1.0)
        return X.astype(float)

    def _prot_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.prot_mode == "intensity":
            Xp = X.replace(0, np.nan)
            return np.log2(Xp + 1.0)
        return X.astype(float)

    # ---- fit -------------------------------------------------------------
    def fit(self, data: OmicsData, train_idx: np.ndarray):
        rna = self._rna_transform(data.rna).iloc[train_idx]
        prot = self._prot_transform(data.prot).iloc[train_idx]

        # 1. expression / detection filters
        if self.rna_mode == "counts":
            lib = data.rna.iloc[train_idx].sum(axis=1)
            cpm = data.rna.iloc[train_idx].div(lib, axis=0) * 1e6
            keep_rna = (cpm >= self.min_cpm).mean(axis=0) >= self.min_frac
        else:
            keep_rna = rna.notna().mean(axis=0) >= self.min_frac
        rna = rna.loc[:, keep_rna[keep_rna].index]

        keep_prot = prot.isna().mean(axis=0) <= self.max_missing_frac
        prot = prot.loc[:, keep_prot[keep_prot].index]

        # 2. protein imputation: per-protein half-minimum (left-censored MS values)
        self.prot_impute_ = (prot.min(axis=0) - 1.0)

        prot_i = prot.fillna(self.prot_impute_)
        rna = rna.fillna(0.0)

        # 3. variance-based selection (training variance only)
        self.rna_features_ = (rna.var(axis=0)
                              .sort_values(ascending=False)
                              .head(self.n_rna).index)
        self.prot_features_ = (prot_i.var(axis=0)
                               .sort_values(ascending=False)
                               .head(self.n_prot).index)

        # 4. standardisation
        r = rna[self.rna_features_]
        p = prot_i[self.prot_features_]
        self.rna_mu_, self.rna_sd_ = r.mean(axis=0), r.std(axis=0).replace(0, 1.0)
        self.prot_mu_, self.prot_sd_ = p.mean(axis=0), p.std(axis=0).replace(0, 1.0)
        self.fitted_ = True
        return self

    # ---- transform -------------------------------------------------------
    def transform(self, data: OmicsData):
        if not self.fitted_:
            raise RuntimeError("call .fit() first")
        rna = self._rna_transform(data.rna).reindex(columns=self.rna_features_).fillna(0.0)
        prot = self._prot_transform(data.prot).reindex(columns=self.prot_features_)
        prot = prot.fillna(self.prot_impute_.reindex(self.prot_features_))
        R = (rna - self.rna_mu_) / self.rna_sd_
        P = (prot - self.prot_mu_) / self.prot_sd_
        return R.to_numpy(np.float32), P.to_numpy(np.float32)


# --------------------------------------------------------------------------
# Design matrices
# --------------------------------------------------------------------------
def design_matrix(meta: pd.DataFrame, terms=("variety", "treatment", "timepoint"),
                  interactions=(("treatment", "timepoint"),), intercept=True):
    """Dummy-coded design matrix; returns (X, column_names)."""
    parts, names = [], []
    if intercept:
        parts.append(np.ones((len(meta), 1)))
        names.append("intercept")
    cols = {}
    for t in terms:
        d = pd.get_dummies(meta[t].astype(str), prefix=t, drop_first=True).astype(float)
        cols[t] = d
        parts.append(d.to_numpy())
        names += list(d.columns)
    for a, b in interactions:
        da, db = cols[a], cols[b]
        for ca in da.columns:
            for cb in db.columns:
                parts.append((da[ca] * db[cb]).to_numpy().reshape(-1, 1))
                names.append(f"{ca}:{cb}")
    return np.hstack(parts).astype(np.float64), names


def covariate_matrix(meta: pd.DataFrame) -> np.ndarray:
    """Compact numeric covariates fed to the neural network alongside RNA."""
    X, _ = design_matrix(meta, terms=("variety", "treatment", "timepoint"),
                         interactions=(), intercept=False)
    return X.astype(np.float32)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def _groups(meta, keys):
    return meta[list(keys)].astype(str).agg("|".join, axis=1).to_numpy()


def leave_condition_out(meta: pd.DataFrame):
    """
    Hold out all 3 replicates of one variety x treatment x timepoint cell.
    16 folds. This is the default honest split: replicates of the same cell are
    near-duplicates, so splitting them across train/test leaks the answer.
    """
    g = _groups(meta, ("variety", "treatment", "timepoint"))
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def leave_variety_out(meta: pd.DataFrame):
    """Train on one variety, predict the other. The strongest generalisation test."""
    g = meta["variety"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def leave_timepoint_out(meta: pd.DataFrame):
    """Temporal extrapolation: can the model predict an unseen timepoint?"""
    g = meta["timepoint"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def naive_random(meta: pd.DataFrame, k=8, seed=0):
    """
    Deliberately WRONG split, kept as a diagnostic. Run it to see how much
    performance replicate leakage manufactures; report the gap in your methods.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(meta))
    return [(np.setdiff1d(np.arange(len(meta)), f), np.sort(f))
            for f in np.array_split(idx, k)]


SPLIT_SCHEMES = {
    "leave_condition_out": leave_condition_out,
    "leave_variety_out": leave_variety_out,
    "leave_timepoint_out": leave_timepoint_out,
    "naive_random": naive_random,
}


# --------------------------------------------------------------------------
# Temporal lag
# --------------------------------------------------------------------------
def lag_pairs(meta: pd.DataFrame, lag=1):
    """
    Index pairs (i_rna, j_prot) matching RNA at time t to protein at time t+lag
    within the same variety x treatment x replicate series.

    Used to test the regulatory-lag hypothesis: does RNA_t predict protein_{t+1}
    better than it predicts protein_t?
    """
    m = meta.reset_index(drop=True)
    order = {v: i for i, v in enumerate(sorted(m["timepoint"].unique()))}
    key = m[["variety", "treatment", "replicate"]].astype(str).agg("|".join, axis=1)
    pos = {(k, order[t]): i for i, (k, t) in enumerate(zip(key, m["timepoint"]))}
    pairs = [(i, pos[(k, order[t] + lag)])
             for i, (k, t) in enumerate(zip(key, m["timepoint"]))
             if (k, order[t] + lag) in pos]
    return np.array(pairs, dtype=int)
