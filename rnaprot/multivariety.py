"""
Per-variety analysis for designs where varieties have DIFFERENT samples AND
DIFFERENT gene/protein sets (separate assemblies or annotations).

Why this module exists
----------------------
The original pipeline assumed one shared feature space across all 48 samples,
which allowed `leave_variety_out` (train on A, predict B) as the strongest
generalisation test. If varieties do not share a gene vocabulary, that test is
impossible: there is no common input space for the model to transfer through.

The replacement is a two-stage design:

  Stage 1  Fit each variety independently.  n = 24, features = that variety's own.
  Stage 2  Compare the RESULTS across varieties at the level of biology
           (orthologous genes), not model weights.

Stage 2 is still a real validation. "The same orthologous genes show
post-transcriptional regulation in both varieties" is a reproducibility claim,
and arguably a stronger one than cross-prediction, because it does not depend on
the two assemblies being numerically comparable.

Sample size warning
-------------------
n = 24 per variety, not 48. Everything must shrink accordingly:
  latent   12  ->  6
  n_rna  3000  ->  1500
  n_prot 1500  ->  800
  folds    16  ->  8  (treatment x timepoint, 3 replicates held out together)
At n = 24 the autoencoder is very unlikely to beat PCA+ridge. Run it, report
the comparison honestly, and expect the discordance analysis to carry the paper.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import OmicsData, load_from_csv


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_variety(rna_path, prot_path, meta_path, mapping_path=None,
                 variety_label=None, features_are_rows=True) -> OmicsData:
    """
    Load ONE variety from its own set of files.

    Each variety has its own gene and protein universe, so nothing is shared or
    reindexed here. If `meta_path` lacks a `variety` column (likely, since the
    file only contains one variety) it is filled in from `variety_label`.
    """
    data = load_from_csv(rna_path, prot_path, meta_path, mapping_path,
                         features_are_rows=features_are_rows)
    if "variety" not in data.meta.columns:
        if variety_label is None:
            raise ValueError("metadata has no 'variety' column and no "
                             "variety_label was supplied")
        data.meta = data.meta.assign(variety=variety_label)
    return data


def split_by_variety(data: OmicsData, drop_absent_features=True) -> dict:
    """
    Split an already-loaded combined object into one OmicsData per variety.

    Use this if your files are combined but the feature sets are effectively
    variety-specific (i.e. genes present in one variety are NaN or all-zero in
    the other). With drop_absent_features=True, each variety keeps only the
    features it actually measures.
    """
    out = {}
    for v, idx in data.meta.groupby("variety").groups.items():
        idx = list(idx)
        rna, prot = data.rna.loc[idx], data.prot.loc[idx]
        if drop_absent_features:
            rna = rna.loc[:, (rna.notna() & (rna != 0)).any(axis=0)]
            prot = prot.loc[:, prot.notna().any(axis=0)]
        cog = {p: g for p, g in data.cognate.items()
               if p in prot.columns and g in rna.columns}
        out[str(v)] = OmicsData(rna, prot, data.meta.loc[idx], cog)
    return out


# --------------------------------------------------------------------------
# Splits appropriate to a single variety (n = 24)
# --------------------------------------------------------------------------
def leave_condition_out_single(meta: pd.DataFrame):
    """
    8 folds: hold out all 3 replicates of one treatment x timepoint cell.
    The single-variety equivalent of the original 16-fold scheme.
    """
    g = meta[["treatment", "timepoint"]].astype(str).agg("|".join, axis=1).to_numpy()
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def leave_timepoint_out_single(meta: pd.DataFrame):
    """4 folds. Temporal extrapolation within one variety."""
    g = meta["timepoint"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def leave_treatment_out_single(meta: pd.DataFrame):
    """
    2 folds: train on control, predict treated (and vice versa).
    The closest available substitute for the lost leave_variety_out test --
    it asks whether the RNA->protein map learned under one condition still
    holds under the other. A failure here IS the biological result:
    treatment altered the mapping.
    """
    g = meta["treatment"].astype(str).to_numpy()
    idx = np.arange(len(meta))
    return [(idx[g != u], idx[g == u]) for u in pd.unique(g)]


def naive_random_single(meta: pd.DataFrame, k=6, seed=0):
    """Deliberately leaky split, kept as a diagnostic only."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(meta))
    return [(np.setdiff1d(np.arange(len(meta)), f), np.sort(f))
            for f in np.array_split(idx, k)]


SPLIT_SCHEMES_SINGLE = {
    "leave_condition_out": leave_condition_out_single,
    "leave_timepoint_out": leave_timepoint_out_single,
    "leave_treatment_out": leave_treatment_out_single,
    "naive_random": naive_random_single,
}


# --------------------------------------------------------------------------
# Config appropriate to n = 24
# --------------------------------------------------------------------------
CONFIG_SINGLE_VARIETY = dict(
    n_rna=1500,
    n_prot=800,
    latent=6,
    hidden=(64, 16),
    epochs=400,
    dropout=0.20,
    weight_decay=3e-3,
    recon_weight=0.5,
    cross_weight=3.0,
    align_weight=0.5,
)


# --------------------------------------------------------------------------
# Design handling for a single variety
# --------------------------------------------------------------------------
def design_matrix_single(meta: pd.DataFrame, intercept=True):
    """
    Design matrix without the (now constant) variety term.

    Calling the original design_matrix on single-variety data is not an error --
    get_dummies(drop_first=True) yields zero columns for a constant factor -- but
    this version is explicit about it and keeps the term names clean.
    """
    parts, names = [], []
    if intercept:
        parts.append(np.ones((len(meta), 1)))
        names.append("intercept")
    cols = {}
    for t in ("treatment", "timepoint"):
        d = pd.get_dummies(meta[t].astype(str), prefix=t, drop_first=True).astype(float)
        cols[t] = d
        parts.append(d.to_numpy())
        names += list(d.columns)
    for ca in cols["treatment"].columns:
        for cb in cols["timepoint"].columns:
            parts.append((cols["treatment"][ca] * cols["timepoint"][cb])
                         .to_numpy().reshape(-1, 1))
            names.append(f"{ca}:{cb}")
    return np.hstack(parts).astype(np.float64), names


def covariate_matrix_single(meta: pd.DataFrame, interactions=True) -> np.ndarray:
    """
    Treatment + timepoint dummies for a single variety.

    With interactions=True (default) a 2x4 design gives 7 columns:
    1 treatment + 3 timepoint + 3 treatment:timepoint. The docstring
    previously claimed 4 columns, which described the no-interaction case --
    and rnaprot.data.covariate_matrix really does build the 4-column version.
    Passing two different covariate sets to the CV model and the full-data
    model made their latent spaces non-comparable, so this must now be chosen
    explicitly and threaded through run_cv via its covariate_fn argument.
    """
    if interactions:
        X, _ = design_matrix_single(meta, intercept=False)
    else:
        parts = [pd.get_dummies(meta[t].astype(str), prefix=t,
                                drop_first=True).astype(float).to_numpy()
                 for t in ("treatment", "timepoint")]
        X = np.hstack(parts)
    return X.astype(np.float32)


def covariate_matrix_none(meta: pd.DataFrame):
    """
    No design covariates at all -- the model sees RNA only.

    Use this for the discordance analysis. If treatment and timepoint are
    inputs to the protein predictor, the residual D = observed - predicted is
    "protein not explained by RNA *and* design", and then testing whether D
    depends on treatment is asking a question the model was already handed the
    answer to. An RNA-only predictor makes D mean what the analysis claims it
    means.
    """
    return None


# --------------------------------------------------------------------------
# Stage 2: cross-variety comparison of RESULTS
# --------------------------------------------------------------------------
def rna_condition_means(data: OmicsData) -> pd.DataFrame:
    """
    Per-gene RNA profile, mean within each treatment x timepoint condition.

    Used by load_ortholog_map() to break multi-way RBH ties: two loci can be
    indistinguishable at the sequence level (identical protein, or not yet
    diverged) and still be regulated differently, which condition-mean RNA
    can see and sequence alignment cannot.
    """
    cond = data.meta["treatment"].astype(str) + "|" + data.meta["timepoint"].astype(str)
    return data.rna.groupby(cond.to_numpy()).mean()  # conditions x genes


def load_ortholog_map(path, col_a=0, col_b=1, rna_a=None, rna_b=None,
                      min_shared_conditions=3) -> pd.DataFrame:
    """
    Table linking variety A gene/protein IDs to variety B IDs.

    Sources, in rough order of preference:
      - a published ortholog table for your species pair
      - OrthoFinder / OrthoMCL run on the two proteomes
      - reciprocal best BLAST/DIAMOND hits (RBH)
      - shared locus IDs, if both assemblies were annotated against a common
        reference (in which case the "different genes" are really just
        presence/absence, and this map is trivial)

    RBH in particular is frequently many-to-many, not 1:1: a gene family with
    near-identical paralogs can have several equally-good hits, and sequence
    identity alone cannot say which one is "the" ortholog -- there may be
    nothing at the sequence level distinguishing them. The old version of
    this function returned every (id_a, id_b) pair verbatim, which silently
    pseudoreplicates a tied id_a's downstream statistics once per candidate
    (a gene with 100 RBH hits contributed 100 near-identical rows to
    compare_discordance's Fisher/Spearman tests). This version instead
    collapses each id_a to exactly one id_b.

    If `rna_a` / `rna_b` (condition-mean RNA matrices, conditions x genes --
    see `rna_condition_means()`) are given, multi-way ties are broken with
    expression: keep whichever id_b candidate has the most correlated RNA
    profile across the conditions both varieties share. That is independent
    information sequence alignment cannot see -- two loci can encode
    identical proteins and still be regulated differently. Without rna_a/
    rna_b, ties are resolved by keeping the first candidate and a warning is
    printed; correctness then depends on the input already being close to
    1:1.

    Every row is tagged `resolution`:
      unambiguous             -- exactly one candidate
      expression_resolved_tie -- >1 candidate, broken by RNA correlation
      unresolved_tie          -- >1 candidate, kept the first (no rna_a/
                                 rna_b given, or no usable expression profile)
    plus `n_candidates` and `best_corr` (NaN unless expression-resolved), so
    a downstream comparison can be reported once on the full table and once
    restricted to `unambiguous` only, as a sensitivity check on how much the
    tie-breaking is actually doing.

    Only ties on the id_a side are resolved to a single id_b. The same id_b
    can still legitimately appear against more than one id_a afterward (e.g.
    two id_a paralogs both best-matching the same id_b) -- that is not an
    error, it is what asymmetric gene-family expansion between the two
    assemblies looks like, so it is left as-is rather than force-fit into a
    strict 1:1 map.
    """
    m = pd.read_csv(path)
    raw = pd.DataFrame({"id_a": m.iloc[:, col_a].astype(str),
                        "id_b": m.iloc[:, col_b].astype(str)}).dropna()

    shared_cond = None
    if rna_a is not None and rna_b is not None:
        shared_cond = sorted(set(rna_a.index) & set(rna_b.index))
        if len(shared_cond) < min_shared_conditions:
            print(f"[load_ortholog_map] only {len(shared_cond)} conditions shared "
                 f"between rna_a/rna_b -- too few to correlate on, falling back "
                 f"to first-candidate for ties")
            shared_cond = None

    rows = []
    n_unresolved = 0
    for id_a, grp in raw.groupby("id_a"):
        cands = grp["id_b"].unique().tolist()
        if len(cands) == 1:
            rows.append((id_a, cands[0], 1, "unambiguous", np.nan))
            continue

        best_c, best_r = None, -np.inf
        if shared_cond is not None and id_a in rna_a.columns:
            a_prof = rna_a.loc[shared_cond, id_a].to_numpy(float)
            if a_prof.std() > 0:
                for c in cands:
                    if c not in rna_b.columns:
                        continue
                    b_prof = rna_b.loc[shared_cond, c].to_numpy(float)
                    if b_prof.std() == 0:
                        continue
                    r = np.corrcoef(a_prof, b_prof)[0, 1]
                    if r > best_r:
                        best_c, best_r = c, r

        if best_c is not None:
            rows.append((id_a, best_c, len(cands), "expression_resolved_tie", best_r))
        else:
            n_unresolved += 1
            rows.append((id_a, cands[0], len(cands), "unresolved_tie", np.nan))

    if n_unresolved:
        print(f"[load_ortholog_map] {n_unresolved:,} multi-way ties kept their "
             f"first candidate (no rna_a/rna_b given, or no usable expression "
             f"profile) -- treat these rows as lower confidence, see the "
             f"'resolution' column")

    return pd.DataFrame(rows, columns=["id_a", "id_b", "n_candidates",
                                       "resolution", "best_corr"])


def compare_discordance(disc_a: pd.DataFrame, disc_b: pd.DataFrame,
                        ortho: pd.DataFrame, alpha=0.05) -> dict:
    """
    Is post-transcriptional regulation reproducible across varieties?

    Joins the two per-variety discordance tables through the ortholog map and
    asks three questions:
      1. Do the significant sets overlap more than chance? (Fisher exact)
      2. Do the effect sizes correlate? (Spearman on F statistics)
      3. Which orthologs are conserved vs variety-specific?

    Returns a dict with a merged table and summary statistics.
    """
    a = disc_a.copy(); a.index = a.index.astype(str)
    b = disc_b.copy(); b.index = b.index.astype(str)

    merged = (ortho
              .join(a.add_suffix("_a"), on="id_a", how="inner")
              .join(b.add_suffix("_b"), on="id_b", how="inner"))
    if merged.empty:
        raise ValueError("no orthologs matched both discordance tables -- "
                         "check that the ID formats in the map match the "
                         "protein IDs used in each analysis")

    sig_a = merged["q_value_a"] < alpha
    sig_b = merged["q_value_b"] < alpha

    table = np.array([[int((sig_a & sig_b).sum()), int((sig_a & ~sig_b).sum())],
                      [int((~sig_a & sig_b).sum()), int((~sig_a & ~sig_b).sum())]])
    odds, p_fisher = stats.fisher_exact(table)

    ok = np.isfinite(merged["F_stat_a"]) & np.isfinite(merged["F_stat_b"])
    rho = stats.spearmanr(merged.loc[ok, "F_stat_a"],
                          merged.loc[ok, "F_stat_b"])

    merged["class"] = np.select(
        [sig_a & sig_b, sig_a & ~sig_b, ~sig_a & sig_b],
        ["conserved", "variety_A_specific", "variety_B_specific"],
        default="neither")

    return dict(
        merged=merged,
        n_orthologs=len(merged),
        n_conserved=int((sig_a & sig_b).sum()),
        n_a_only=int((sig_a & ~sig_b).sum()),
        n_b_only=int((~sig_a & sig_b).sum()),
        contingency=table,
        fisher_odds=float(odds),
        fisher_p=float(p_fisher),
        spearman_rho=float(rho.statistic),
        spearman_p=float(rho.pvalue),
    )


def compare_latent_factors(anova_a: pd.DataFrame, anova_b: pd.DataFrame) -> pd.DataFrame:
    """
    Do the two varieties organise their variation along the same design axes?

    Compares the per-term variance-explained profiles. This needs no ortholog
    map at all, because it compares design annotations rather than genes.
    """
    terms = [c for c in anova_a.columns if c in anova_b.columns and c != "total_R2"]
    rows = []
    for t in terms:
        rows.append(dict(term=t,
                         max_var_explained_A=float(anova_a[t].max()),
                         max_var_explained_B=float(anova_b[t].max()),
                         mean_var_explained_A=float(anova_a[t].mean()),
                         mean_var_explained_B=float(anova_b[t].mean())))
    return pd.DataFrame(rows)


def compare_lag(lag_a: pd.DataFrame, lag_b: pd.DataFrame,
                ortho: pd.DataFrame) -> dict:
    """Is regulatory lag assigned consistently to orthologous genes?"""
    a = lag_a.copy(); a.index = a.index.astype(str)
    b = lag_b.copy(); b.index = b.index.astype(str)
    merged = (ortho
              .join(a[["lag_preference", "lag_class"]].add_suffix("_a"), on="id_a", how="inner")
              .join(b[["lag_preference", "lag_class"]].add_suffix("_b"), on="id_b", how="inner"))
    if merged.empty:
        return dict(merged=merged, n=0)
    ok = np.isfinite(merged["lag_preference_a"]) & np.isfinite(merged["lag_preference_b"])
    rho = stats.spearmanr(merged.loc[ok, "lag_preference_a"],
                          merged.loc[ok, "lag_preference_b"])
    agree = (merged["lag_class_a"] == merged["lag_class_b"]).mean()
    return dict(merged=merged, n=len(merged),
                class_agreement=float(agree),
                spearman_rho=float(rho.statistic),
                spearman_p=float(rho.pvalue))
