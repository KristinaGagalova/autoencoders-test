"""
Synthetic RNA + protein time-course with KNOWN ground truth.

Purpose: this is not a toy. Run your whole pipeline on this first. Because you
know which proteins are RNA-explainable, which are lagged, and which respond to
treatment without any RNA signal, you can measure whether your method recovers
the truth *before* you trust it on real data. If the pipeline cannot find
planted signal at n = 48, it will not find real signal at n = 48 either.

(column `truth_class` of the returned table):

  cognate_direct     protein_t tracks its own RNA_t              linear; PCA/ridge wins
  cognate_lagged     protein_t tracks its own RNA_{t-1}          lag analysis should win
  trans_regulated    driven by a *different* gene's RNA          cross-layer attribution
  nonlinear_cognate  saturating tanh response to its own RNA     where an AE can beat linear
  interaction        needs TWO transcripts multiplied together   where an AE can beat linear
  protein_only       treatment hits protein, RNA is flat         big discordance, R2 ~ 0
  noise              nothing                                     null / FDR calibration

Seven planted classes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import OmicsData


def simulate(n_genes=8000, n_prot=1500, n_var=2, n_treat=2, n_time=4, n_rep=3,
             seed=0, effect=1.6, noise=0.9):
    rng = np.random.default_rng(seed)

    meta = pd.DataFrame(
        [(f"V{v}", f"T{t}", f"t{k}", f"r{r}")
         for v in range(n_var) for t in range(n_treat)
         for k in range(n_time) for r in range(n_rep)],
        columns=["variety", "treatment", "timepoint", "replicate"],
    )
    meta.index = [f"S{i:03d}" for i in range(len(meta))]
    n = len(meta)

    tnum = meta["timepoint"].str[1:].astype(int).to_numpy()
    treat = (meta["treatment"] == "T1").astype(float).to_numpy()
    var = (meta["variety"] == "V1").astype(float).to_numpy()

    # ---- latent biological programs -------------------------------------
    # F1 treatment response rising with time, F2 circadian-ish, F3 variety
    F = np.stack([
        treat * (tnum / (n_time - 1)),
        np.sin(2 * np.pi * tnum / n_time),
        var,
        rng.normal(0, 0.3, n),                      # nuisance batch-like factor
    ], axis=1)

    # ---- RNA -------------------------------------------------------------
    base = rng.normal(6, 2, n_genes)
    responsive = rng.random(n_genes) < 0.30           # genes that actually respond
    Wr = rng.normal(0, 1, (F.shape[1], n_genes)) * responsive
    RNA_log = base + F @ Wr * effect + rng.normal(0, noise, (n, n_genes))
    RNA_log = np.clip(RNA_log, 0, None)

    # ---- assign protein classes -----------------------------------------
    classes = np.array(["cognate_direct"] * n_prot, dtype=object)
    q = rng.random(n_prot)
    classes[q > 0.22] = "cognate_lagged"
    classes[q > 0.38] = "trans_regulated"
    classes[q > 0.52] = "nonlinear_cognate"   # saturating RNA -> protein
    classes[q > 0.66] = "interaction"         # needs TWO transcripts multiplied
    classes[q > 0.78] = "protein_only"
    classes[q > 0.90] = "noise"

    # proteins are wired to *responsive* transcripts: a protein whose transcript
    # never moves carries no recoverable RNA -> protein relationship by construction
    resp_idx = np.where(responsive)[0]
    cognate_gene = rng.choice(resp_idx, n_prot, replace=n_prot > len(resp_idx))
    driver_gene = rng.choice(resp_idx, n_prot, replace=True)
    partner_gene = rng.choice(resp_idx, n_prot, replace=True)

    # lagged RNA: value of the same replicate series one timepoint earlier
    prev = np.arange(n)
    key = meta[["variety", "treatment", "replicate"]].agg("|".join, axis=1).to_numpy()
    pos = {(k, t): i for i, (k, t) in enumerate(zip(key, tnum))}
    for i, (k, t) in enumerate(zip(key, tnum)):
        prev[i] = pos.get((k, t - 1), i)
    RNA_prev = RNA_log[prev]

    P = np.zeros((n, n_prot))
    for j, c in enumerate(classes):
        g = cognate_gene[j]
        if c == "cognate_direct":
            P[:, j] = 4 + 0.8 * RNA_log[:, g]
        elif c == "cognate_lagged":
            P[:, j] = 4 + 0.8 * RNA_prev[:, g]
        elif c == "trans_regulated":
            P[:, j] = 4 + 0.7 * RNA_log[:, driver_gene[j]] + 0.1 * RNA_log[:, g]
        elif c == "nonlinear_cognate":
            x = RNA_log[:, g] - RNA_log[:, g].mean()
            P[:, j] = 8 + 2.5 * np.tanh(1.5 * x)          # saturating response
        elif c == "interaction":
            a = RNA_log[:, g] - RNA_log[:, g].mean()
            b = RNA_log[:, partner_gene[j]] - RNA_log[:, partner_gene[j]].mean()
            P[:, j] = 8 + 0.8 * a * b                     # needs a product term
        elif c == "protein_only":
            P[:, j] = 8 + effect * treat * (tnum / (n_time - 1)) * 2.0
        else:
            P[:, j] = 8.0
    P = P + rng.normal(0, noise * 0.8, (n, n_prot))

    genes = [f"gene{i:05d}" for i in range(n_genes)]
    prots = [f"prot{j:05d}" for j in range(n_prot)]

    rna_df = pd.DataFrame(np.round(2 ** RNA_log), index=meta.index, columns=genes)  # counts-like
    prot_df = pd.DataFrame(2 ** P, index=meta.index, columns=prots)                 # intensity-like

    cognate = {prots[j]: genes[cognate_gene[j]] for j in range(n_prot)}
    truth = pd.DataFrame({
        "protein": prots,
        "truth_class": classes,
        "cognate_gene": [genes[g] for g in cognate_gene],
        "driver_gene": [genes[d] if c == "trans_regulated" else ""
                        for d, c in zip(driver_gene, classes)],
        "partner_gene": [genes[d] if c == "interaction" else ""
                         for d, c in zip(partner_gene, classes)],
    }).set_index("protein")

    return OmicsData(rna_df, prot_df, meta, cognate), truth
