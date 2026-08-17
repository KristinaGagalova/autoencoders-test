#!/usr/bin/env python
"""
End-to-end analysis driver.

    python run_analysis.py --demo                # synthetic data, known truth
    python run_analysis.py --rna rna.csv --prot prot.csv --meta meta.csv \
                           --mapping protein_to_gene.csv --out results/

What it produces in --out:

    cv_folds.csv                per-fold, per-model performance
    per_protein_r2_<split>.csv  per-protein R2 for every model
    model_comparison.csv        paired tests: does the AE beat the baselines?
    permutation_null.csv        label-shuffled null R2
    discordance.csv             per-protein RNA->protein residual + F-test + FDR
    discordance_matrix.csv      sample x protein residuals (feeds clustering)
    kinetic_clusters.csv        regulatory classes from residual trajectories
    latent_factors.csv          latent coordinates per sample
    latent_design_anova.csv     what each latent factor encodes
    cross_layer_edges.csv       directed RNA -> protein attributions
    lag_comparison.csv          RNA_t -> protein_t  vs  RNA_t -> protein_t+1
    figures/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rnaprot import baselines as B
from rnaprot.data import (SPLIT_SCHEMES, OmicsPreprocessor, covariate_matrix,
                          lag_pairs, load_from_csv)
from rnaprot.evaluate import (classify_kinetics, compare_models,
                              cross_layer_attribution, discordance_table,
                              latent_design_anova, oof_r2_table,
                              permutation_null, run_cv)
from rnaprot.models import AERegressor
from rnaprot.simulate import simulate

CONFIG = dict(
    n_rna=3000,          # genes kept after variance filtering (of your ~60k)
    n_prot=1500,         # proteins kept (of your ~6k); raise once it runs
    latent=12,           # latent dimensions. n=48 -> do not exceed ~16
    hidden=(128, 32),
    epochs=400,
    dropout=0.15,
    weight_decay=1e-3,
    recon_weight=0.5,    # weight on RNA->RNA and protein->protein reconstruction
    cross_weight=3.0,    # weight on the RNA -> protein path (the one you care about)
    align_weight=0.5,    # weight on ||z_rna - z_protein||
)


# --------------------------------------------------------------------------
def build_models(cfg, variational=False):
    def ae_factory(ctx):
        return AERegressor(latent=cfg["latent"], hidden=cfg["hidden"],
                           dropout=cfg["dropout"], epochs=cfg["epochs"],
                           weight_decay=cfg["weight_decay"],
                           weights=(cfg["recon_weight"], cfg["recon_weight"],
                                    cfg["cross_weight"], cfg["align_weight"]),
                           groups=ctx.get("train_groups"),
                           variational=variational)
    return {
        "mean": lambda ctx: B.MeanBaseline(),
        "design_only": lambda ctx: B.DesignBaseline(),
        "cognate_ridge": lambda ctx: B.CognateBaseline(ctx["pairs"]),
        "pca_ridge": lambda ctx: B.PCARidge(n_components=10),
        "pls": lambda ctx: B.PLSBaseline(n_components=5),
        "multimodal_ae": ae_factory,
    }


def make_cognate_index_fn(cognate):
    """Map each retained protein to the column index of its transcript."""
    def fn(pre: OmicsPreprocessor):
        gpos = {g: i for i, g in enumerate(pre.rna_features_)}
        return np.array([gpos.get(cognate.get(p, None), -1)
                         for p in pre.prot_features_])
    return fn


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--rna"), ap.add_argument("--prot"), ap.add_argument("--meta")
    ap.add_argument("--mapping", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", nargs="+",
                    default=["leave_condition_out", "leave_variety_out", "naive_random"])
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--variational", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer epochs/features")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(n_rna=1200, n_prot=600, epochs=120)

    # ---- data ----------------------------------------------------------
    truth = None
    if args.demo:
        data, truth = simulate(seed=0)
        truth.to_csv(out / "simulation_truth.csv")
    else:
        data = load_from_csv(args.rna, args.prot, args.meta, args.mapping)
    print(data.describe(), "\n")

    pre_kwargs = dict(n_rna=cfg["n_rna"], n_prot=cfg["n_prot"])
    models = build_models(cfg, args.variational)
    cog_fn = make_cognate_index_fn(data.cognate)

    # ---- cross-validation over several splitting schemes ---------------
    all_folds, r2_tables, oofs = [], {}, {}
    for scheme in args.splits:
        print(f"[cv] {scheme}")
        splits = SPLIT_SCHEMES[scheme](data.meta)
        folds, oof = run_cv(data, splits, models, pre_kwargs, cog_fn)
        folds["split_scheme"] = scheme
        all_folds.append(folds)
        r2 = oof_r2_table(oof, data.meta)
        r2.to_csv(out / f"per_protein_r2_{scheme}.csv")
        r2_tables[scheme], oofs[scheme] = r2, oof

    pd.concat(all_folds).to_csv(out / "cv_folds.csv", index=False)

    if truth is not None:
        cls = (r2_tables[args.splits[0]].join(truth["truth_class"])
               .groupby("truth_class").median(numeric_only=True))
        cls["n"] = truth["truth_class"].value_counts()
        cls.to_csv(out / "r2_by_truth_class.csv")
        print("\n[demo] pooled out-of-fold median R2 by planted class")
        print(cls.round(3).to_string(), "\n")

    main_scheme = args.splits[0]
    comp = compare_models(r2_tables[main_scheme], reference="pca_ridge")
    comp.to_csv(out / "model_comparison.csv", index=False)
    print("\n[compare] vs pca_ridge on", main_scheme)
    print(comp.to_string(index=False), "\n")

    # ---- permutation null ----------------------------------------------
    if args.n_perm > 0:
        print(f"[null] {args.n_perm} label permutations")
        null = permutation_null(data, SPLIT_SCHEMES[main_scheme](data.meta), models,
                                n_perm=args.n_perm, pre_kwargs=pre_kwargs,
                                cognate_index_fn=cog_fn)
        null.to_csv(out / "permutation_null.csv")
        print(null.round(3).to_string(), "\n")

    # ---- discordance ----------------------------------------------------
    disc, D = discordance_table(oofs[main_scheme], data.meta, model="multimodal_ae")
    if truth is not None:
        disc = disc.join(truth["truth_class"])
    disc.sort_values("q_value").to_csv(out / "discordance.csv")
    D.to_csv(out / "discordance_matrix.csv")

    clusters, prof = classify_kinetics(D, data.meta)
    clusters.to_frame().join(disc).to_csv(out / "kinetic_clusters.csv")
    n_sig = int((disc["q_value"] < 0.05).sum())
    print(f"[discordance] {n_sig}/{len(disc)} proteins with treatment-dependent "
          f"residuals at FDR 5%\n")

    # ---- refit on all data for interpretation --------------------------
    pre = OmicsPreprocessor(**pre_kwargs).fit(data, np.arange(data.n_samples))
    R, P = pre.transform(data)
    cov = covariate_matrix(data.meta)
    ae = AERegressor(latent=cfg["latent"], hidden=cfg["hidden"], dropout=cfg["dropout"],
                     epochs=cfg["epochs"], weight_decay=cfg["weight_decay"],
                     weights=(cfg["recon_weight"], cfg["recon_weight"],
                              cfg["cross_weight"], cfg["align_weight"]),
                     variational=args.variational,
                     groups=data.meta[["variety", "treatment", "timepoint"]]
                            .astype(str).agg("|".join, axis=1).to_numpy(),
                     verbose=True).fit(R, P, cov)

    Z = ae.encode(R, cov, "rna")
    pd.DataFrame(Z, index=data.meta.index,
                 columns=[f"z{i+1}" for i in range(Z.shape[1])]) \
      .join(data.meta).to_csv(out / "latent_factors.csv")
    anova = latent_design_anova(Z, data.meta)
    anova.to_csv(out / "latent_design_anova.csv")
    print("[latent] variance explained per factor")
    print(anova.round(3).to_string(), "\n")

    # ---- cross-layer edges ---------------------------------------------
    best = r2_tables[main_scheme]["multimodal_ae"].sort_values(ascending=False)
    pos = {p: i for i, p in enumerate(pre.prot_features_)}
    idx = [pos[p] for p in best.head(30).index if p in pos]
    edges = cross_layer_attribution(ae, R, cov, idx, list(pre.rna_features_),
                                    list(pre.prot_features_))
    edges["is_cognate"] = [data.cognate.get(p) == g
                           for p, g in zip(edges["protein"], edges["gene"])]
    edges.to_csv(out / "cross_layer_edges.csv", index=False)

    # ---- temporal lag: per protein, does RNA_t explain protein_{t+1} better? ----
    lag_r2 = {}
    for lag in (0, 1):
        pairs = (np.stack([np.arange(data.n_samples)] * 2, axis=1) if lag == 0
                 else lag_pairs(data.meta, lag))
        Rl, Pl, cl = R[pairs[:, 0]], P[pairs[:, 1]], cov[pairs[:, 0]]
        npair = len(pairs)
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(npair), 4)
        acc = []
        for f in folds:
            tr = np.setdiff1d(np.arange(npair), f)
            m = B.CognateBaseline(cog_fn(pre)).fit(Rl[tr], Pl[tr], cl[tr])
            from rnaprot.evaluate import per_protein_r2
            acc.append(per_protein_r2(Pl[f], m.predict(Rl[f], cl[f]), Pl[tr].mean(axis=0)))
        lag_r2[f"r2_lag{lag}"] = np.nanmean(acc, axis=0)
    lag_tab = pd.DataFrame(lag_r2, index=list(pre.prot_features_))
    lag_tab["lag_preference"] = lag_tab["r2_lag1"] - lag_tab["r2_lag0"]
    lag_tab["class"] = np.where(lag_tab["lag_preference"] > 0.05, "RNA-first / protein-later",
                       np.where(lag_tab["lag_preference"] < -0.05, "synchronous", "ambiguous"))
    if truth is not None:
        lag_tab = lag_tab.join(truth["truth_class"])
    lag_tab.sort_values("lag_preference", ascending=False).to_csv(out / "lag_atlas.csv")
    print("[lag] regulatory lag atlas:",
          lag_tab["class"].value_counts().to_dict(), "\n")

    # ---- figures --------------------------------------------------------
    make_figures(out, r2_tables, anova, disc, prof, clusters, ae, truth)

    json.dump(cfg, open(out / "config_used.json", "w"), indent=2)
    print(f"done -> {out.resolve()}")


# --------------------------------------------------------------------------
def make_figures(out, r2_tables, anova, disc, prof, clusters, ae, truth):
    fig_dir = out / "figures"

    # 1. model comparison per split scheme
    schemes = list(r2_tables)
    fig, axes = plt.subplots(1, len(schemes), figsize=(5 * len(schemes), 4.2),
                             sharey=True, squeeze=False)
    for ax, s in zip(axes[0], schemes):
        t = r2_tables[s]
        ax.boxplot([t[c].dropna().clip(-1, 1) for c in t.columns],
                   tick_labels=list(t.columns), showfliers=False)
        ax.axhline(0, color="crimson", lw=1, ls="--")
        ax.set_title(s), ax.tick_params(axis="x", rotation=45)
    axes[0][0].set_ylabel("per-protein $R^2$ (out-of-fold)")
    fig.suptitle("Does the autoencoder beat simpler models? "
                 "(red line = predicting the training mean)")
    fig.tight_layout(), fig.savefig(fig_dir / "fig1_model_comparison.png", dpi=150)
    plt.close(fig)

    # 2. leakage demonstration
    if "naive_random" in r2_tables and "leave_condition_out" in r2_tables:
        fig, ax = plt.subplots(figsize=(5, 4))
        for s, c in (("naive_random", "tab:orange"), ("leave_condition_out", "tab:blue")):
            v = r2_tables[s]["multimodal_ae"].dropna().clip(-1, 1)
            ax.hist(v, bins=40, alpha=0.6, label=s, color=c)
        ax.set_xlabel("per-protein $R^2$"), ax.legend()
        ax.set_title("Replicate leakage inflates performance")
        fig.tight_layout(), fig.savefig(fig_dir / "fig2_leakage.png", dpi=150)
        plt.close(fig)

    # 3. latent factor / design heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = [c for c in anova.columns if c != "total_R2"]
    im = ax.imshow(anova[cols].to_numpy(), aspect="auto", cmap="viridis",
                   vmin=0, vmax=max(0.05, float(anova[cols].to_numpy().max())))
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(anova)), anova.index)
    ax.set_title("Variance of each latent factor explained by design")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(), fig.savefig(fig_dir / "fig3_latent_design.png", dpi=150)
    plt.close(fig)

    # 4. discordance volcano
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = disc["mean_abs_discordance"]
    y = -np.log10(np.clip(disc["p_value"], 1e-300, 1))
    if "truth_class" in disc:
        for cls, g in disc.groupby("truth_class"):
            ax.scatter(g["mean_abs_discordance"],
                       -np.log10(np.clip(g["p_value"], 1e-300, 1)),
                       s=8, alpha=0.6, label=cls)
        ax.legend(fontsize=7)
    else:
        ax.scatter(x, y, s=8, alpha=0.5)
    ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=1)
    ax.set_xlabel("mean |RNA-protein discordance|")
    ax.set_ylabel(r"$-\log_{10}$ p (treatment effect on residual)")
    ax.set_title("Where RNA stops explaining protein")
    fig.tight_layout(), fig.savefig(fig_dir / "fig4_discordance_volcano.png", dpi=150)
    plt.close(fig)

    # 5. kinetic clusters
    k = clusters.nunique()
    fig, axes = plt.subplots(1, k, figsize=(2.4 * k, 2.8), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        sub = prof.loc[clusters[clusters == i].index]
        ax.plot(sub.T.to_numpy(), color="grey", alpha=0.06, lw=0.6)
        ax.plot(sub.mean(axis=0).to_numpy(), color="crimson", lw=2)
        ax.set_title(f"cluster {i} (n={len(sub)})", fontsize=8)
        ax.set_xticks(range(prof.shape[1]), prof.columns, rotation=90, fontsize=6)
    fig.suptitle("Discordance trajectories = regulatory classes")
    fig.tight_layout(), fig.savefig(fig_dir / "fig5_kinetic_clusters.png", dpi=150)
    plt.close(fig)

    # 6. training curve
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ae.history_)
    ax.set_xlabel("epoch"), ax.set_ylabel("validation cross-modal MSE")
    ax.set_title("Autoencoder training (early stopped)")
    fig.tight_layout(), fig.savefig(fig_dir / "fig6_training.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
