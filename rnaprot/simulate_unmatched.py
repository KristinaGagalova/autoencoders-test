"""
Synthetic data for the UNMATCHED design.

RNA and protein are measured on DIFFERENT plants drawn from the same
population. This simulator makes that explicit: it draws two independent sets
of plants per condition, so there is no sample-level correspondence to
exploit, only a shared underlying condition-level truth.

Planted gene classes (column `truth_class`):

  concordant     transcript and protein both respond, same direction
  RNA_only       transcript responds, protein does not      -> post-transcriptional
  protein_only   protein responds, transcript does not      -> post-transcriptional
  opposite       both respond, opposite directions          -> post-transcriptional
  lagged         protein reproduces the transcript response one timepoint later
  none           neither responds                           -> null / FDR calibration

If the gene-level pipeline cannot separate these, it will not find real
biology either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_unmatched(n_genes=4000, n_prot=1200, n_time=4, n_rep=3,
                       variety="V1", seed=0, effect=1.4,
                       rna_noise=0.55, prot_noise=0.75):
    rng = np.random.default_rng(seed)
    tps = [f"t{k}" for k in range(n_time)]

    def make_meta(prefix):
        rows = [(tr, t, f"r{r}") for tr in ("T0", "T1")
                for t in tps for r in range(n_rep)]
        m = pd.DataFrame(rows, columns=["treatment", "timepoint", "replicate"])
        m["variety"] = variety
        m.index = [f"{prefix}{i+1:03d}" for i in range(len(m))]
        return m

    # two INDEPENDENT sets of plants -- this is the whole point
    rna_meta = make_meta("R")
    prot_meta = make_meta("P")

    genes = [f"gene{i:05d}" for i in range(n_genes)]
    prots = [f"prot{j:05d}" for j in range(n_prot)]
    cognate_gene = rng.choice(n_genes, n_prot, replace=False)
    cognate = {prots[j]: genes[cognate_gene[j]] for j in range(n_prot)}

    # ---- planted classes ------------------------------------------------
    q = rng.random(n_prot)
    cls = np.array(["concordant"] * n_prot, dtype=object)
    cls[q > 0.22] = "RNA_only"
    cls[q > 0.40] = "protein_only"
    cls[q > 0.55] = "opposite"
    cls[q > 0.66] = "lagged"
    cls[q > 0.78] = "none"

    # ---- true condition-level effect trajectories -----------------------
    # ramp with time; each responding gene gets its own amplitude
    ramp = np.arange(n_time) / (n_time - 1)
    amp = rng.normal(0, 1, n_prot) * effect
    amp = np.sign(amp) * (np.abs(amp) + 0.6)         # keep away from zero

    rna_eff = np.zeros((n_prot, n_time))
    prot_eff = np.zeros((n_prot, n_time))
    for j, c in enumerate(cls):
        a = amp[j]
        if c == "concordant":
            rna_eff[j] = a * ramp
            prot_eff[j] = 0.8 * a * ramp
        elif c == "RNA_only":
            rna_eff[j] = a * ramp
        elif c == "protein_only":
            prot_eff[j] = a * ramp
        elif c == "opposite":
            rna_eff[j] = a * ramp
            prot_eff[j] = -0.8 * a * ramp
        elif c == "lagged":
            rna_eff[j] = a * ramp
            prot_eff[j, 1:] = 0.8 * a * ramp[:-1]     # protein follows by one

    # non-cognate genes: some respond too, so the transcriptome is realistic
    other = np.setdiff1d(np.arange(n_genes), cognate_gene)
    other_resp = rng.random(len(other)) < 0.25
    other_amp = rng.normal(0, effect, len(other)) * other_resp

    # ---- build sample matrices ------------------------------------------
    def build(meta, base_n, eff_rows, eff_index, noise, extra=None):
        n = len(meta)
        base = rng.normal(9, 1.8, base_n)
        M = np.tile(base, (n, 1)) + rng.normal(0, noise, (n, base_n))
        treated = (meta["treatment"] == "T1").to_numpy()
        tidx = meta["timepoint"].map({t: i for i, t in enumerate(tps)}).to_numpy()
        for row, col in enumerate(eff_index):
            M[treated, col] += eff_rows[row][tidx[treated]]
        if extra is not None:
            cols, amps = extra
            for k, col in enumerate(cols):
                M[treated, col] += amps[k] * ramp[tidx[treated]]
        return M

    rna_M = build(rna_meta, n_genes, rna_eff, cognate_gene, rna_noise,
                  extra=(other, other_amp))
    prot_M = build(prot_meta, n_prot, prot_eff, np.arange(n_prot), prot_noise)

    rna = pd.DataFrame(rna_M, index=rna_meta.index, columns=genes)
    prot = pd.DataFrame(prot_M, index=prot_meta.index, columns=prots)

    truth = pd.DataFrame({
        "protein": prots, "cognate_gene": [genes[g] for g in cognate_gene],
        "truth_class": cls, "amplitude": amp,
    }).set_index("protein")

    return dict(rna=rna, rna_meta=rna_meta, prot=prot, prot_meta=prot_meta,
                cognate=cognate, truth=truth)
