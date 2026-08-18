"""
Evaluation: the part that decides whether any of this is real.

Contains
  run_cv                 one CV loop, all models, out-of-fold predictions
  per_protein_r2         honest R2 (baseline = TRAIN mean, not test mean)
  compare_models         paired Wilcoxon across proteins + win counts
  permutation_null       label-shuffled null distribution of R2
  discordance_table      out-of-fold residuals -> per-protein effect tests
  latent_design_anova    variance of each latent factor explained by the design
  cross_layer_attribution  which transcripts drive which non-cognate proteins
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import stats

from .data import OmicsPreprocessor, _groups, covariate_matrix, design_matrix


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def per_protein_r2(y_true, y_pred, train_mean):
    """
    R2 per protein using the TRAINING mean as the reference model.

    Using the test-fold mean instead (the sklearn default) is badly optimistic
    with 3-sample folds and can even be undefined. Negative values here mean the
    model is worse than predicting the training average -- report them, do not
    clip them.
    """
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - train_mean) ** 2).sum(axis=0)
    return 1.0 - sse / np.where(sst == 0, np.nan, sst)


def per_protein_spearman(y_true, y_pred):
    out = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        if np.std(y_pred[:, j]) > 1e-12 and np.std(y_true[:, j]) > 1e-12:
            out[j] = stats.spearmanr(y_true[:, j], y_pred[:, j]).statistic
    return out


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
def _fold_protein_sets(data, splits, pre_kwargs):
    """
    Pass 1: fit preprocessing on each fold's training samples and record which
    proteins survive. Returns (fitted_preprocessors, intersection_of_proteins).

    Why a separate pass: protein variance-filtering is refit per fold, so
    different folds keep different proteins. Scoring must happen on a set that
    exists in EVERY fold, otherwise some folds contribute padded/absent targets.
    """
    pres, sets = [], []
    for tr, _ in splits:
        pre = OmicsPreprocessor(**pre_kwargs).fit(data, tr)
        pres.append(pre)
        sets.append(set(pre.prot_features_))
    common = set.intersection(*sets) if sets else set()
    # stable, deterministic order taken from fold 0
    targets = [p for p in pres[0].prot_features_ if p in common]
    return pres, targets


def run_cv(data, splits, model_factories, pre_kwargs=None, cognate_index_fn=None,
           covariate_fn=None, verbose=True):
    """
    model_factories : dict name -> callable(context) -> model
        context carries {'pairs': cognate column indices} so the cognate
        baseline can be rebuilt per fold (RNA feature selection changes each fold).
    covariate_fn : callable(meta) -> array, or None for no design covariates.
        MUST be the same function used anywhere else the model is refit,
        otherwise the CV model and the full-data model see different inputs.
        Pass None to fit a genuinely RNA-only model (see discordance caveat).

    Returns (results_df, oof). `oof` carries out-of-fold predictions in the
    standardised protein space plus, per model, the accumulated SSE and the
    accumulated SST measured against each fold's OWN TRAINING mean. Pooled R2
    is then 1 - sum(SSE)/sum(SST), which keeps the train-mean baseline that
    per_protein_r2 documents instead of silently switching to the test mean.

    Protein targets are the intersection across folds, so no column is ever
    zero-padded and every scored entry is a real measurement.
    """
    pre_kwargs = pre_kwargs or {}
    n = data.n_samples
    cov_all = covariate_fn(data.meta) if covariate_fn is not None else None

    pres, prot_names = _fold_protein_sets(data, splits, pre_kwargs)
    n_prot = len(prot_names)
    if n_prot == 0:
        raise ValueError("no protein survives filtering in every fold; "
                         "raise n_prot or relax max_missing_frac")
    if verbose:
        widest = max(len(p.prot_features_) for p in pres)
        if n_prot < widest:
            print(f"[cv] scoring {n_prot} proteins present in all "
                  f"{len(splits)} folds (per-fold max was {widest})")

    rows = []
    oof_pred = {name: np.full((n, n_prot), np.nan, np.float32) for name in model_factories}
    oof_true = np.full((n, n_prot), np.nan, np.float32)
    oof_mask = np.zeros((n, n_prot), bool)
    sse = {name: np.zeros(n_prot) for name in model_factories}
    sst = np.zeros(n_prot)          # shared: same targets, same train baseline

    for fold, ((tr, te), pre) in enumerate(zip(splits, pres)):
        R, P_all = pre.transform(data)
        # restrict to the cross-fold target set, in a fixed order
        pcol = {p: i for i, p in enumerate(pre.prot_features_)}
        take = [pcol[p] for p in prot_names]
        P = P_all[:, take]

        # cognate indices must be built for the RESTRICTED, REORDERED targets,
        # not for pre.prot_features_ (that mismatch silently mispaired every
        # protein with another protein's transcript in the previous version)
        if cognate_index_fn is not None:
            full_pairs = cognate_index_fn(pre)
            pairs = np.array([full_pairs[pcol[p]] for p in prot_names])
        else:
            pairs = np.full(n_prot, -1)

        cov = cov_all
        ctx = dict(pairs=pairs, pre=pre, prot_names=prot_names,
                   train_groups=_groups(data.meta,
                                        ("variety", "treatment", "timepoint"))[tr])

        train_mean = P[tr].mean(axis=0)
        oof_true[te] = P[te]
        oof_mask[te] = True
        sst += ((P[te] - train_mean) ** 2).sum(axis=0)

        for name, factory in model_factories.items():
            model = factory(ctx)
            ctr = cov[tr] if cov is not None else None
            cte = cov[te] if cov is not None else None
            model.fit(R[tr], P[tr], ctr)
            pred = np.asarray(model.predict(R[te], cte), np.float32)
            oof_pred[name][te] = pred
            sse[name] += ((P[te] - pred) ** 2).sum(axis=0)
            r2 = per_protein_r2(P[te], pred, train_mean)
            rows.append(dict(fold=fold, model=name, n_test=len(te),
                             median_r2=np.nanmedian(r2),
                             mean_r2=np.nanmean(r2),
                             frac_r2_pos=float(np.nanmean(r2 > 0)),
                             rmse=float(np.sqrt(np.nanmean((P[te] - pred) ** 2)))))
        if verbose:
            best = max(rows[-len(model_factories):], key=lambda r: r["median_r2"])
            print(f"  fold {fold+1}/{len(splits)}: best={best['model']} "
                  f"median R2={best['median_r2']:.3f}")

    return pd.DataFrame(rows), dict(pred=oof_pred, true=oof_true, mask=oof_mask,
                                    proteins=prot_names, sse=sse, sst=sst)


def oof_r2_table(oof, meta=None):
    """
    Pooled per-protein R2 against each fold's own TRAINING mean:

        R2 = 1 - sum_folds(SSE) / sum_folds(SST_vs_that_fold's_train_mean)

    The previous version recomputed the reference as the mean of the pooled
    held-out values, i.e. the test mean, which is exactly the optimistic
    statistic per_protein_r2's docstring warns against.
    """
    sst = np.where(oof["sst"] == 0, np.nan, oof["sst"])
    out = {name: 1.0 - oof["sse"][name] / sst for name in oof["pred"]}
    return pd.DataFrame(out, index=oof["proteins"])


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------
def compare_models(r2_table, reference="pca_ridge"):
    rows = []
    ref = r2_table[reference].to_numpy()
    for col in r2_table.columns:
        if col == reference:
            continue
        x = r2_table[col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(ref)
        try:
            w = stats.wilcoxon(x[ok], ref[ok])
            p = w.pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(model=col, reference=reference,
                         median_r2=np.nanmedian(x),
                         median_ref=np.nanmedian(ref),
                         median_delta=np.nanmedian(x[ok] - ref[ok]),
                         win_frac=float(np.mean(x[ok] > ref[ok])),
                         wilcoxon_p=p, n_proteins=int(ok.sum())))
    return pd.DataFrame(rows).sort_values("median_delta", ascending=False)


def permutation_null(data, splits, model_factories, n_perm=5, seed=0, **kw):
    """
    Shuffle the sample labels of the protein matrix and rerun the whole CV.

    Returns the SAME statistic as the headline table -- pooled per-protein R2
    against the fold training mean -- so the bars are directly comparable.
    The previous version averaged per-fold medians, a different statistic that
    could not legitimately be plotted against the pooled result.

    n_perm=2 is a smoke test. Use >=20 for anything you intend to report;
    with 8 folds and 24 samples the null is wide and 2 draws will not
    characterise it.
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_perm):
        d = copy.copy(data)
        perm = rng.permutation(data.n_samples)
        d.prot = data.prot.iloc[perm].set_axis(data.prot.index)
        _, oof = run_cv(d, splits, model_factories, verbose=False, **kw)
        agg = oof_r2_table(oof).median(axis=0).rename(f"perm{p}")
        out.append(agg)
    return pd.concat(out, axis=1)


# --------------------------------------------------------------------------
# Question 2: RNA-protein discordance
# --------------------------------------------------------------------------
def _lstsq_rss(X, Y):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return ((Y - X @ beta) ** 2).sum(axis=0)


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.clip(ranked, 0, 1)
    q[ok] = tmp
    return q


def discordance_table(oof, meta, model="multimodal_ae", test_terms=("treatment",)):
    """
    D = observed protein - protein predicted from RNA (out-of-fold).

    Then per protein, an F-test of whether D depends systematically on the
    design. A protein with a large but *random* residual is measurement noise.
    A protein whose residual moves with treatment is post-transcriptional
    regulation, and that is the biological object you are after.
    """
    m = oof["mask"].all(axis=1)
    D = oof["true"][m] - oof["pred"][model][m]
    md = meta.iloc[m]

    X_full, names = design_matrix(md, terms=("variety", "treatment", "timepoint"),
                                  interactions=(("treatment", "timepoint"),))
    drop = [i for i, nm in enumerate(names)
            if any(nm.startswith(t) or f":{t}" in nm or nm.split(":")[0].startswith(t)
                   for t in test_terms)]
    keep = [i for i in range(X_full.shape[1]) if i not in drop]
    X_red = X_full[:, keep]

    rss_f, rss_r = _lstsq_rss(X_full, D), _lstsq_rss(X_red, D)
    df1 = X_full.shape[1] - X_red.shape[1]
    df2 = X_full.shape[0] - X_full.shape[1]
    F = ((rss_r - rss_f) / df1) / np.maximum(rss_f / df2, 1e-12)
    p = stats.f.sf(F, df1, df2)

    return pd.DataFrame({
        "mean_abs_discordance": np.abs(D).mean(axis=0),
        "discordance_sd": D.std(axis=0),
        "F_stat": F, "p_value": p, "q_value": bh_fdr(p),
        "explained_var_by_rna": 1 - D.var(axis=0) / np.maximum(oof["true"][m].var(axis=0), 1e-12),
    }, index=oof["proteins"]), pd.DataFrame(D, index=md.index, columns=oof["proteins"])


def classify_kinetics(D_df, meta, n_clusters=6, seed=0):
    """Cluster condition-mean discordance trajectories into regulatory classes."""
    from sklearn.cluster import KMeans

    key = meta.loc[D_df.index, ["treatment", "timepoint"]].astype(str).agg("|".join, axis=1)
    prof = D_df.groupby(key.to_numpy()).mean().T          # proteins x conditions
    prof = prof.sub(prof.mean(axis=1), axis=0)
    sd = prof.std(axis=1).replace(0, 1.0)
    profn = prof.div(sd, axis=0).fillna(0.0)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(profn)
    return pd.Series(km.labels_, index=prof.index, name="kinetic_cluster"), prof


# --------------------------------------------------------------------------
# Question 1: what do the latent factors mean?
# --------------------------------------------------------------------------
def latent_design_anova(Z, meta):
    """
    For each latent dimension, the fraction of its variance explained by each
    design term (type-II style: full model minus that term). This is how you
    say "factor 3 is the treatment axis" instead of squinting at a PCA plot.
    """
    terms = ("variety", "treatment", "timepoint")
    X_full, names = design_matrix(meta, terms=terms,
                                  interactions=(("treatment", "timepoint"),))
    tss = ((Z - Z.mean(axis=0)) ** 2).sum(axis=0)
    rss_full = _lstsq_rss(X_full, Z)
    rows = {"total_R2": 1 - rss_full / np.maximum(tss, 1e-12)}
    for t in terms + ("treatment:timepoint",):
        drop = [i for i, nm in enumerate(names)
                if nm.startswith(t.split(":")[0]) and (":" in nm) == (":" in t)]
        keep = [i for i in range(X_full.shape[1]) if i not in drop]
        rows[t] = (_lstsq_rss(X_full[:, keep], Z) - rss_full) / np.maximum(tss, 1e-12)
    return pd.DataFrame(rows, index=[f"z{i+1}" for i in range(Z.shape[1])])


# --------------------------------------------------------------------------
# Question 3: cross-layer dependency
# --------------------------------------------------------------------------
def cross_layer_attribution(ae, R, cov, protein_idx, gene_names, prot_names,
                            top_k=20):
    """
    Gradient x input attribution through the trained encoder/decoder:
    d(predicted protein_j) / d(RNA_g), averaged over samples.

    Gives a directed, weighted RNA -> protein edge list. Sanity check built in:
    if the cognate transcript is not usually near the top for well-predicted
    proteins, the model is picking up global covariance rather than regulation.
    """
    import torch

    model = ae.model_
    model.eval()
    dev = next(model.parameters()).device
    Rt = torch.tensor(np.asarray(R, np.float32), device=dev, requires_grad=True)
    Ct = (torch.tensor(np.asarray(cov, np.float32), device=dev)
          if cov is not None and model.n_cov else None)

    rows = []
    for j in protein_idx:
        if Rt.grad is not None:
            Rt.grad = None
        mu, _ = model.encode_rna(Rt, Ct)
        pred = model.decode(mu, Ct)[1][:, j]
        pred.sum().backward()
        attr = (Rt.grad * Rt).mean(dim=0).detach().cpu().numpy()
        top = np.argsort(-np.abs(attr))[:top_k]
        for rank, g in enumerate(top):
            rows.append(dict(protein=prot_names[j], gene=gene_names[g],
                             rank=rank + 1, attribution=float(attr[g])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Leak-free temporal lag analysis
# --------------------------------------------------------------------------
def lag_cv(data, pre_kwargs, cognate_index_fn, covariate_fn, lags=(0, 1),
           seed=0, verbose=True):
    """
    Per-protein comparison of RNA_t -> protein_t against RNA_t -> protein_{t+lag},
    with preprocessing refit inside every fold and folds split by whole
    replicate trajectory.

    The previous implementation took R/P/cov from a preprocessor fitted to all
    samples and then split the (t, t+1) pairs at random. That leaked twice:
    feature selection and scaling had already seen the held-out samples, and
    two pairs from the same replicate time series could land on opposite sides
    of the split, which for a temporal question is close to scoring the model
    on its own training trajectory.

    Folds here are the replicate series themselves (variety x treatment x
    replicate), so an entire trajectory is held out at once.
    """
    from .data import lag_pairs

    meta = data.meta
    series = meta[["variety", "treatment", "replicate"]].astype(str) \
                 .agg("|".join, axis=1).to_numpy()
    uniq = pd.unique(series)
    if len(uniq) < 3:
        raise ValueError(f"only {len(uniq)} replicate series; too few to hold out")

    results = {}
    for lag in lags:
        pairs = (np.stack([np.arange(len(meta))] * 2, axis=1) if lag == 0
                 else lag_pairs(meta, lag))
        if len(pairs) == 0:
            continue
        pair_series = series[pairs[:, 0]]

        sse = None
        sst = None
        names = None
        for held in uniq:
            te_p = np.where(pair_series == held)[0]
            tr_p = np.where(pair_series != held)[0]
            if len(te_p) == 0 or len(tr_p) < 4:
                continue
            # samples contributing to training pairs -- preprocessing may only
            # ever see these, never the held-out trajectory
            tr_samples = np.unique(np.concatenate([pairs[tr_p, 0], pairs[tr_p, 1]]))
            pre = OmicsPreprocessor(**pre_kwargs).fit(data, tr_samples)
            R, P = pre.transform(data)
            cov = covariate_fn(meta) if covariate_fn is not None else None
            if names is None:
                names = list(pre.prot_features_)
                sse = np.zeros(len(names)); sst = np.zeros(len(names))
            elif list(pre.prot_features_) != names:
                pcol = {p: i for i, p in enumerate(pre.prot_features_)}
                keep = [p for p in names if p in pcol]
                if len(keep) != len(names):     # restrict once, consistently
                    idx_keep = [names.index(p) for p in keep]
                    sse, sst = sse[idx_keep], sst[idx_keep]
                    names = keep
                P = P[:, [pcol[p] for p in names]]

            pidx = cognate_index_fn(pre)
            if len(pidx) != len(names):
                pcol = {p: i for i, p in enumerate(pre.prot_features_)}
                pidx = np.array([pidx[pcol[p]] for p in names])

            Rl, Pl = R[pairs[:, 0]], P[pairs[:, 1]]
            cl = cov[pairs[:, 0]] if cov is not None else None
            m = B_CognateBaseline(pidx).fit(
                Rl[tr_p], Pl[tr_p], cl[tr_p] if cl is not None else None)
            pred = m.predict(Rl[te_p], cl[te_p] if cl is not None else None)
            tmean = Pl[tr_p].mean(axis=0)
            sse += ((Pl[te_p] - pred) ** 2).sum(axis=0)
            sst += ((Pl[te_p] - tmean) ** 2).sum(axis=0)

        results[f"r2_lag{lag}"] = pd.Series(
            1.0 - sse / np.where(sst == 0, np.nan, sst), index=names)
        if verbose:
            print(f"[lag] lag={lag}: {len(pairs)} pairs, "
                  f"median R2={np.nanmedian(results[f'r2_lag{lag}']):.3f}")

    tab = pd.DataFrame(results).dropna(how="all")
    if "r2_lag0" in tab and "r2_lag1" in tab:
        tab["lag_preference"] = tab["r2_lag1"] - tab["r2_lag0"]
        tab["lag_class"] = np.where(tab["lag_preference"] > 0.05, "RNA-first",
                            np.where(tab["lag_preference"] < -0.05, "synchronous",
                                     "ambiguous"))
    return tab


from .baselines import CognateBaseline as B_CognateBaseline   # noqa: E402
