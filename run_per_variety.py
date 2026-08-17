#!/usr/bin/env python
"""
Run the full analysis SEPARATELY for each variety, then compare results.

Use this when varieties have different samples AND different gene/protein sets.

    # two varieties, separate files each
    python run_per_variety.py \
        --variety A --rna A_rna.csv --prot A_prot.csv --meta A_meta.csv --mapping A_map.csv \
        --variety B --rna B_rna.csv --prot B_prot.csv --meta B_meta.csv --mapping B_map.csv \
        --ortholog orthologs_A_to_B.csv \
        --out results_per_variety

    # synthetic check first
    python run_per_variety.py --demo --out results_per_variety

Outputs, per variety, under <out>/<variety>/ :
    cv_folds.csv, model_comparison.csv, per_protein_r2.csv,
    discordance.csv, kinetic_clusters.csv, lag_atlas.csv,
    latent_factors.csv, latent_design_anova.csv, figures/

Plus, at the top level, the cross-variety comparison:
    cross_variety_discordance.csv   ortholog-level conserved / specific calls
    cross_variety_summary.json      Fisher and Spearman statistics
    figures/cross_variety.png
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
from rnaprot.data import OmicsPreprocessor, lag_pairs
from rnaprot.evaluate import (classify_kinetics, compare_models,
                              discordance_table, latent_design_anova,
                              oof_r2_table, per_protein_r2, permutation_null,
                              run_cv)
from rnaprot.models import AERegressor
from rnaprot.multivariety import (CONFIG_SINGLE_VARIETY, SPLIT_SCHEMES_SINGLE,
                                  compare_discordance, compare_lag,
                                  compare_latent_factors,
                                  covariate_matrix_single, load_ortholog_map,
                                  load_variety)


# --------------------------------------------------------------------------
def make_cognate_index_fn(cognate):
    def fn(pre):
        gpos = {g: i for i, g in enumerate(pre.rna_features_)}
        return np.array([gpos.get(cognate.get(p), -1) for p in pre.prot_features_])
    return fn


def build_models(cfg, groups_getter):
    def ae_factory(ctx):
        return AERegressor(latent=cfg["latent"], hidden=cfg["hidden"],
                           dropout=cfg["dropout"], epochs=cfg["epochs"],
                           weight_decay=cfg["weight_decay"],
                           weights=(cfg["recon_weight"], cfg["recon_weight"],
                                    cfg["cross_weight"], cfg["align_weight"]),
                           groups=ctx.get("train_groups"))
    return {
        "mean":          lambda ctx: B.MeanBaseline(),
        "design_only":   lambda ctx: B.DesignBaseline(),
        "cognate_ridge": lambda ctx: B.CognateBaseline(ctx["pairs"]),
        "pca_ridge":     lambda ctx: B.PCARidge(n_components=6),   # n=24 -> fewer PCs
        "pls":           lambda ctx: B.PLSBaseline(n_components=4),
        "multimodal_ae": ae_factory,
    }


# --------------------------------------------------------------------------
def analyse_one_variety(data, label, cfg, out_dir, splits_wanted, n_perm,
                        truth=None):
    """Full pipeline for a single variety. Returns a dict of result tables."""
    out = Path(out_dir) / label
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nVARIETY {label}\n{'='*70}")
    print(f"{data.n_samples} samples | {data.rna.shape[1]:,} genes | "
          f"{data.prot.shape[1]:,} proteins")
    if data.n_samples > 30:
        print(f"  note: expected ~24 samples for one variety, got {data.n_samples}")

    cog_fn = make_cognate_index_fn(data.cognate)
    pre_kw = dict(n_rna=cfg["n_rna"], n_prot=cfg["n_prot"])
    models = build_models(cfg, None)

    # ---- cross-validation ------------------------------------------------
    r2_tables, oofs, all_folds = {}, {}, []
    for scheme in splits_wanted:
        print(f"\n[cv] {scheme}")
        splits = SPLIT_SCHEMES_SINGLE[scheme](data.meta)
        folds, oof = run_cv(data, splits, models, pre_kw, cog_fn)
        folds["split_scheme"] = scheme
        all_folds.append(folds)
        r2 = oof_r2_table(oof, data.meta)
        r2.to_csv(out / f"per_protein_r2_{scheme}.csv")
        r2_tables[scheme], oofs[scheme] = r2, oof

    pd.concat(all_folds).to_csv(out / "cv_folds.csv", index=False)
    main = splits_wanted[0]

    comp = compare_models(r2_tables[main], reference="pca_ridge")
    comp.to_csv(out / "model_comparison.csv", index=False)
    print(f"\n[compare] vs pca_ridge ({main})")
    print(comp.to_string(index=False))

    # ---- permutation null -------------------------------------------------
    if n_perm > 0:
        print(f"\n[null] {n_perm} permutations")
        null = permutation_null(data, SPLIT_SCHEMES_SINGLE[main](data.meta), models,
                                n_perm=n_perm, pre_kwargs=pre_kw,
                                cognate_index_fn=cog_fn)
        null.to_csv(out / "permutation_null.csv")
        print(null.round(3).to_string())

    # ---- discordance ------------------------------------------------------
    disc, D = discordance_table(oofs[main], data.meta, model="multimodal_ae")
    if truth is not None:
        disc = disc.join(truth["truth_class"])
    disc.sort_values("q_value").to_csv(out / "discordance.csv")
    D.to_csv(out / "discordance_matrix.csv")
    n_sig = int((disc["q_value"] < 0.05).sum())
    print(f"\n[discordance] {n_sig}/{len(disc)} proteins at FDR 5%")

    clusters, prof = classify_kinetics(D, data.meta, n_clusters=5)
    clusters.to_frame().join(disc).to_csv(out / "kinetic_clusters.csv")

    # ---- refit on all samples for interpretation -------------------------
    pre = OmicsPreprocessor(**pre_kw).fit(data, np.arange(data.n_samples))
    R, P = pre.transform(data)
    cov = covariate_matrix_single(data.meta)
    ae = AERegressor(latent=cfg["latent"], hidden=cfg["hidden"],
                     dropout=cfg["dropout"], epochs=cfg["epochs"],
                     weight_decay=cfg["weight_decay"],
                     weights=(cfg["recon_weight"], cfg["recon_weight"],
                              cfg["cross_weight"], cfg["align_weight"]),
                     groups=data.meta[["treatment", "timepoint"]]
                            .astype(str).agg("|".join, axis=1).to_numpy(),
                     verbose=True).fit(R, P, cov)

    Z = ae.encode(R, cov, "rna")
    pd.DataFrame(Z, index=data.meta.index,
                 columns=[f"z{i+1}" for i in range(Z.shape[1])]) \
      .join(data.meta).to_csv(out / "latent_factors.csv")

    # variety term is constant here, so annotate against treatment/time only
    meta_for_anova = data.meta.assign(variety="const")
    anova = latent_design_anova(Z, meta_for_anova).drop(columns=["variety"],
                                                        errors="ignore")
    anova.to_csv(out / "latent_design_anova.csv")
    print("\n[latent] variance explained")
    print(anova.round(3).to_string())

    # ---- lag atlas --------------------------------------------------------
    pairs_idx = cog_fn(pre)
    lag_r2 = {}
    for lag in (0, 1):
        pr = (np.stack([np.arange(data.n_samples)] * 2, axis=1) if lag == 0
              else lag_pairs(data.meta, lag))
        Rl, Pl, cl = R[pr[:, 0]], P[pr[:, 1]], cov[pr[:, 0]]
        rng = np.random.default_rng(0)
        acc = []
        for f in np.array_split(rng.permutation(len(pr)), 4):
            tr = np.setdiff1d(np.arange(len(pr)), f)
            m = B.CognateBaseline(pairs_idx).fit(Rl[tr], Pl[tr], cl[tr])
            acc.append(per_protein_r2(Pl[f], m.predict(Rl[f], cl[f]),
                                      Pl[tr].mean(axis=0)))
        lag_r2[f"r2_lag{lag}"] = np.nanmean(acc, axis=0)
    lag_tab = pd.DataFrame(lag_r2, index=list(pre.prot_features_))
    lag_tab["lag_preference"] = lag_tab["r2_lag1"] - lag_tab["r2_lag0"]
    lag_tab["lag_class"] = np.where(lag_tab["lag_preference"] > 0.05, "RNA-first",
                            np.where(lag_tab["lag_preference"] < -0.05, "synchronous",
                                     "ambiguous"))
    lag_tab.sort_values("lag_preference", ascending=False).to_csv(out / "lag_atlas.csv")
    print(f"\n[lag] {lag_tab['lag_class'].value_counts().to_dict()}")

    _variety_figures(out, label, r2_tables, disc, prof, clusters, anova, ae, lag_tab)

    return dict(r2=r2_tables[main], disc=disc, anova=anova, lag=lag_tab,
                clusters=clusters, comparison=comp)


# --------------------------------------------------------------------------
def _variety_figures(out, label, r2_tables, disc, prof, clusters, anova, ae, lag_tab):
    fd = out / "figures"
    order = ["mean", "design_only", "cognate_ridge", "pca_ridge", "pls", "multimodal_ae"]

    schemes = list(r2_tables)
    fig, axes = plt.subplots(1, len(schemes), figsize=(5 * len(schemes), 4.2),
                             sharey=True, squeeze=False)
    for ax, s in zip(axes[0], schemes):
        t = r2_tables[s]
        cols = [c for c in order if c in t]
        ax.boxplot([t[c].dropna().clip(-1, 1) for c in cols],
                   tick_labels=cols, showfliers=False)
        ax.axhline(0, color="crimson", lw=1, ls="--")
        ax.set_title(s); ax.tick_params(axis="x", rotation=30)
    axes[0][0].set_ylabel("per-protein $R^2$")
    fig.suptitle(f"Variety {label}: model comparison (n=24)")
    fig.tight_layout(); fig.savefig(fd / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(disc["mean_abs_discordance"],
               -np.log10(np.clip(disc["p_value"], 1e-300, 1)),
               s=8, alpha=0.5, color="steelblue")
    ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=1)
    ax.set_xlabel("mean |discordance|"); ax.set_ylabel(r"$-\log_{10}$ p")
    ax.set_title(f"Variety {label}: RNA-protein discordance")
    fig.tight_layout(); fig.savefig(fd / "discordance_volcano.png", dpi=150); plt.close(fig)

    k = clusters.nunique()
    fig, axes = plt.subplots(1, k, figsize=(2.6 * k, 3.0), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        sub = prof.loc[clusters[clusters == i].index]
        ax.plot(sub.T.to_numpy(), color="grey", alpha=0.07, lw=0.6)
        ax.plot(sub.mean(axis=0).to_numpy(), color="crimson", lw=2)
        ax.set_title(f"cluster {i} (n={len(sub)})", fontsize=8)
        ax.set_xticks(range(prof.shape[1]))
        ax.set_xticklabels(prof.columns, rotation=90, fontsize=6)
    fig.suptitle(f"Variety {label}: discordance kinetics")
    fig.tight_layout(); fig.savefig(fd / "kinetic_clusters.png", dpi=150); plt.close(fig)

    cols = [c for c in anova.columns if c != "total_R2"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(anova[cols].to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(anova))); ax.set_yticklabels(anova.index)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Variety {label}: latent factor annotation")
    fig.tight_layout(); fig.savefig(fd / "latent_design.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(ae.history_, color="steelblue")
    ax.set_xlabel("epoch"); ax.set_ylabel("val cross-modal MSE")
    ax.set_title(f"Variety {label}: training")
    fig.tight_layout(); fig.savefig(fd / "training.png", dpi=150); plt.close(fig)


# --------------------------------------------------------------------------
def cross_variety_report(results, ortho_path, out_dir, labels):
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    a, b = labels[0], labels[1]

    summary = {}

    # latent comparison needs no ortholog map
    lat = compare_latent_factors(results[a]["anova"], results[b]["anova"])
    lat.to_csv(out / "cross_variety_latent.csv", index=False)
    print("\n[cross] latent factor annotation, both varieties")
    print(lat.round(3).to_string(index=False))

    if ortho_path is None:
        print("\n[cross] no --ortholog map supplied; skipping gene-level comparison.")
        print("        Latent-level comparison above is still valid.")
        json.dump(summary, open(out / "cross_variety_summary.json", "w"), indent=2)
        return

    ortho = load_ortholog_map(ortho_path)
    print(f"\n[cross] ortholog map: {len(ortho):,} pairs")

    res = compare_discordance(results[a]["disc"], results[b]["disc"], ortho)
    res["merged"].to_csv(out / "cross_variety_discordance.csv", index=False)
    summary["discordance"] = {k: v for k, v in res.items()
                              if k not in ("merged", "contingency")}
    summary["discordance"]["contingency"] = res["contingency"].tolist()

    print(f"  orthologs matched     : {res['n_orthologs']:,}")
    print(f"  conserved (both sig)  : {res['n_conserved']:,}")
    print(f"  {a}-specific          : {res['n_a_only']:,}")
    print(f"  {b}-specific          : {res['n_b_only']:,}")
    print(f"  Fisher odds={res['fisher_odds']:.2f}  p={res['fisher_p']:.3g}")
    print(f"  Spearman rho={res['spearman_rho']:.3f}  p={res['spearman_p']:.3g}")

    lagres = compare_lag(results[a]["lag"], results[b]["lag"], ortho)
    if lagres["n"]:
        lagres["merged"].to_csv(out / "cross_variety_lag.csv", index=False)
        summary["lag"] = {k: v for k, v in lagres.items() if k != "merged"}
        print(f"  lag class agreement   : {lagres['class_agreement']:.1%}")

    m = res["merged"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(np.log10(np.clip(m["F_stat_a"], 1e-3, None)),
               np.log10(np.clip(m["F_stat_b"], 1e-3, None)),
               s=8, alpha=0.4, color="steelblue")
    ax.set_xlabel(f"log10 F, variety {a}"); ax.set_ylabel(f"log10 F, variety {b}")
    ax.set_title(f"Discordance effect sizes\nSpearman rho = {res['spearman_rho']:.2f}")

    ax = axes[1]
    counts = m["class"].value_counts()
    ax.bar(counts.index, counts.values, color=["seagreen", "steelblue", "orange", "lightgray"][:len(counts)])
    ax.set_ylabel("orthologs"); ax.tick_params(axis="x", rotation=20)
    ax.set_title("Conserved vs variety-specific regulation")
    fig.tight_layout(); fig.savefig(out / "figures" / "cross_variety.png", dpi=150)
    plt.close(fig)

    json.dump(summary, open(out / "cross_variety_summary.json", "w"), indent=2)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--variety", action="append", default=[])
    ap.add_argument("--rna", action="append", default=[])
    ap.add_argument("--prot", action="append", default=[])
    ap.add_argument("--meta", action="append", default=[])
    ap.add_argument("--mapping", action="append", default=[])
    ap.add_argument("--ortholog", default=None)
    ap.add_argument("--out", default="results_per_variety")
    ap.add_argument("--splits", nargs="+",
                    default=["leave_condition_out", "leave_treatment_out"])
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIG_SINGLE_VARIETY)
    if args.quick:
        cfg.update(n_rna=800, n_prot=400, epochs=150)

    datasets, truths = {}, {}
    if args.demo:
        from rnaprot.simulate import simulate
        for i, lab in enumerate(["A", "B"]):
            d, t = simulate(n_genes=4000, n_prot=800, n_var=1, seed=i)
            d.meta = d.meta.assign(variety=lab)
            # give each variety its own gene/protein namespace
            d.rna.columns = [f"{lab}_{g}" for g in d.rna.columns]
            d.prot.columns = [f"{lab}_{p}" for p in d.prot.columns]
            d.cognate = {f"{lab}_{p}": f"{lab}_{g}" for p, g in d.cognate.items()}
            t.index = [f"{lab}_{p}" for p in t.index]
            datasets[lab], truths[lab] = d, t
        # trivial ortholog map: protN in A <-> protN in B
        common = [p.split("_", 1)[1] for p in datasets["A"].prot.columns][:600]
        pd.DataFrame({"id_a": [f"A_{p}" for p in common],
                      "id_b": [f"B_{p}" for p in common]}) \
          .to_csv(Path(args.out) / "demo_orthologs.csv", index=False) \
          if Path(args.out).exists() else None
        Path(args.out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id_a": [f"A_{p}" for p in common],
                      "id_b": [f"B_{p}" for p in common]}) \
          .to_csv(Path(args.out) / "demo_orthologs.csv", index=False)
        args.ortholog = str(Path(args.out) / "demo_orthologs.csv")
    else:
        n = len(args.variety)
        if not (n == len(args.rna) == len(args.prot) == len(args.meta)):
            raise SystemExit("supply --variety/--rna/--prot/--meta once per variety")
        maps = args.mapping + [None] * (n - len(args.mapping))
        for i in range(n):
            datasets[args.variety[i]] = load_variety(
                args.rna[i], args.prot[i], args.meta[i], maps[i],
                variety_label=args.variety[i])
            truths[args.variety[i]] = None

    results = {}
    for lab, d in datasets.items():
        results[lab] = analyse_one_variety(d, lab, cfg, args.out, args.splits,
                                           args.n_perm, truths.get(lab))

    labels = list(datasets)
    if len(labels) >= 2:
        cross_variety_report(results, args.ortholog, args.out, labels)

    print(f"\ndone -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
