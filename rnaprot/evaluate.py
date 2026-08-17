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
def run_cv(data, splits, model_factories, pre_kwargs=None, cognate_index_fn=None,
           verbose=True):
    """
    model_factories : dict name -> callable(context) -> model
        context carries {'pairs': cognate column indices} so the cognate
        baseline can be rebuilt per fold (feature selection changes each fold).

    Returns (results_df, oof) where oof[name] is an n_samples x n_proteins
    matrix of out-of-fold predictions in the *standardised* protein space,
    together with the protein names actually scored.
    """
    pre_kwargs = pre_kwargs or {}
    n = data.n_samples
    rows = []
    oof_pred, oof_true, oof_mask, prot_names = {}, None, None, None

    for fold, (tr, te) in enumerate(splits):
        pre = OmicsPreprocessor(**pre_kwargs).fit(data, tr)
        R, P = pre.transform(data)
        cov = covariate_matrix(data.meta)

        if prot_names is None:
            prot_names = list(pre.prot_features_)
            oof_true = np.full((n, len(prot_names)), np.nan, np.float32)
            oof_mask = np.zeros((n, len(prot_names)), bool)
            for name in model_factories:
                oof_pred[name] = np.full((n, len(prot_names)), np.nan, np.float32)
        elif list(pre.prot_features_) != prot_names:
            # keep folds comparable: score the intersection defined by fold 0
            keep = [i for i, p in enumerate(pre.prot_features_) if p in set(prot_names)]
            idx = {p: i for i, p in enumerate(pre.prot_features_)}
            P = P[:, [idx[p] for p in prot_names if p in idx]]
            missing = [p for p in prot_names if p not in idx]
            if missing:
                pad = np.zeros((P.shape[0], len(missing)), np.float32)
                P = np.hstack([P, pad])
                order = [p for p in prot_names if p in idx] + missing
                P = P[:, [order.index(p) for p in prot_names]]

        pairs = cognate_index_fn(pre) if cognate_index_fn else np.full(P.shape[1], -1)
        ctx = dict(pairs=pairs, pre=pre,
                   train_groups=_groups(data.meta, ("variety", "treatment", "timepoint"))[tr])

        train_mean = P[tr].mean(axis=0)
        oof_true[te] = P[te]
        oof_mask[te] = True

        for name, factory in model_factories.items():
            model = factory(ctx)
            model.fit(R[tr], P[tr], cov[tr])
            pred = np.asarray(model.predict(R[te], cov[te]), np.float32)
            oof_pred[name][te] = pred
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

    return pd.DataFrame(rows), dict(pred=oof_pred, true=oof_true,
                                    mask=oof_mask, proteins=prot_names)


def oof_r2_table(oof, meta):
    """Per-protein R2 computed once over all out-of-fold predictions."""
    m = oof["mask"].all(axis=1)
    Y = oof["true"][m]
    ref = Y.mean(axis=0)
    out = {}
    for name, Phat in oof["pred"].items():
        out[name] = per_protein_r2(Y, Phat[m], ref)
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
    Any structure surviving this is fitting artefact. With n = 48 this is the
    single most convincing control you can put in a paper.
    """
    rng = np.random.default_rng(seed)
    out = []
    for p in range(n_perm):
        d = copy.copy(data)
        perm = rng.permutation(data.n_samples)
        d.prot = data.prot.iloc[perm].set_axis(data.prot.index)
        res, _ = run_cv(d, splits, model_factories, verbose=False, **kw)
        agg = res.groupby("model")["median_r2"].mean().rename(f"perm{p}")
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
