# RNA → protein multimodal autoencoder for small factorial omics designs

A analysis pipeline for matched transcriptome + proteome time-course data with a
small sample size (n ≈ 48) and a very large feature space (~60,000 genes,
~6,000 proteins).

The pipeline is built around one question — **where, when and why does RNA stop
explaining protein?** — and around one methodological commitment: *every claim
the autoencoder makes must survive comparison against a simpler model and
against a permutation null.*

---

## Install & run

```bash
pip install torch scikit-learn pandas numpy scipy matplotlib

# 1. sanity check on synthetic data with known ground truth
python run_analysis.py --demo --out results_demo

# 2. your data
python run_analysis.py \
    --rna  rna_counts.csv \
    --prot protein_lfq.csv \
    --meta sample_metadata.csv \
    --mapping protein_to_gene.csv \
    --out results_real
```

Input format:

| file | rows | columns |
|---|---|---|
| `--rna` | genes | sample IDs (raw counts, or set `rna_mode='logged'`) |
| `--prot` | proteins | sample IDs (LFQ / iBAQ intensities) |
| `--meta` | sample IDs | `variety, treatment, timepoint, replicate` |
| `--mapping` | — | `protein_id, gene_id` (defines the cognate pair) |

**Run `--demo` first.** It plants seven known classes of protein behaviour into
synthetic data and tells you whether the pipeline recovers them. If a method
cannot find planted signal at n = 48, it will not find real signal at n = 48.

---

## The model

```
    RNA  ──► encoder_R ──┐
                          ├──► z (shared latent, 8–16 dims) ──┬──► decoder_R ──► RNA_hat
 protein  ──► encoder_P ──┘                                   └──► decoder_P ──► protein_hat
```

Four loss terms, each mapping to one of the project questions:

| term | what it buys you |
|---|---|
| `recon_rna`, `recon_prot` | z retains structure in each layer |
| **`cross`** | protein predicted from the RNA encoder **only** — its residuals *are* the gene-level discordance score |
| `align` | ‖z_RNA − z_protein‖², the neural analogue of MOFA's shared factors |

At test time only the RNA path is used: `RNA → encoder_R → z → decoder_P → protein_hat`.

Sizing: `3000 → 128 → 32 → z(12)`. Dropout, weight decay, early stopping and a
tiny latent are not decoration — they are the only thing between you and a model
that memorises 48 samples and generalises to nothing.

---

## The three rules this pipeline enforces

**1. Feature selection happens inside the CV loop.**
Picking "the 2,000 most variable genes" on the full matrix before
cross-validating leaks test information and can manufacture double-digit R²
from pure noise. `OmicsPreprocessor.fit()` only ever sees training samples.

**2. Replicates never straddle a split.**
Three replicates of the same variety × treatment × timepoint cell are near
duplicates. The default split (`leave_condition_out`) holds out all three
together. `naive_random` is included deliberately so you can *measure* the
inflation and report it.

| scheme | question it answers |
|---|---|
| `leave_condition_out` | honest default, 16 folds |
| `leave_variety_out` | train on variety A, predict B — conserved vs variety-specific regulation |
| `leave_timepoint_out` | temporal extrapolation |
| `naive_random` | diagnostic only — shows how much leakage buys |

**3. R² is measured against the training mean, not the test mean.**
With 3-sample test folds the sklearn default is badly optimistic and sometimes
undefined. Negative values mean *worse than predicting the training average* —
they are reported, not clipped. Pooled out-of-fold R² is the headline number;
per-fold R² on 3 samples is close to meaningless.

---

## What the demo revealed (n = 48, 8,000 genes, 1,500 proteins)

### The autoencoder does not automatically win

Pooled out-of-fold median R², `leave_condition_out`:

| planted class | mean | design only | cognate ridge | PCA+ridge | PLS | **multimodal AE** |
|---|---|---|---|---|---|---|
| cognate_direct (linear) | 0.00 | 0.38 | 0.41 | **0.50** | 0.51 | 0.35 |
| nonlinear_cognate (tanh) | 0.00 | 0.37 | 0.40 | **0.48** | 0.49 | 0.34 |
| trans_regulated | 0.00 | 0.36 | 0.36 | **0.45** | 0.46 | 0.33 |
| **interaction (needs a product term)** | 0.00 | −0.18 | −0.20 | −0.20 | −0.18 | **−0.06** |
| protein_only | 0.00 | 0.35 | 0.40 | **0.71** | 0.71 | 0.54 |
| noise | 0.00 | −0.13 | −0.14 | −0.11 | −0.09 | **−0.08** |

Read this honestly. **On linear and mildly nonlinear biology, PCA + ridge beats
the autoencoder** — the AE pays a variance penalty at n = 48 and gains nothing.
The AE only pulls ahead where the relationship genuinely needs interaction terms
between transcripts, and on the noise class where its shrinkage protects it.

The practical conclusion: run the autoencoder *as a hypothesis test*, not as a
foregone conclusion. If it does not beat `pca_ridge` on your data, that is a
publishable finding — the RNA → protein map in your system is essentially
linear — and it saves you from a reviewer making the point for you.

Watch `design_only` hardest. It uses no RNA at all. If it matches your AE, the
model learned your experimental design, not RNA–protein regulation.

Under label permutation every model collapses to R² ≈ −0.03 to −0.09, confirming
none of the above is fitting artefact.

### The discordance test is the strongest result

Residual `D = observed protein − protein predicted from RNA` (out-of-fold), then
a per-protein F-test for treatment dependence, BH-corrected:

| planted class | detected at FDR 5% |
|---|---|
| **protein_only** (true post-transcriptional regulation) | **85%** |
| noise (should be null) | 3% |
| cognate_direct / lagged / trans | 4–6% |
| interaction / nonlinear | 11% |

85% power with a well-calibrated 3% false-positive rate. This — not prediction
accuracy — is what I would build the paper on.

The 11% on the interaction and nonlinear classes is a real caveat: **model
misspecification leaks into the discordance score.** A protein the model simply
cannot fit will show structured residuals that look like post-transcriptional
regulation. Mitigate it by checking that your significant hits are not
concentrated in the proteins your model fits worst.

Clustering the residual trajectories recovers the planted class cleanly —
one cluster captured 190/190 `protein_only` proteins.

### The regulatory lag atlas works

Per protein, compare RNA_t → protein_t against RNA_t → protein_{t+1}:

| planted class | % preferring lag-1 |
|---|---|
| **cognate_lagged** | **88%** |
| cognate_direct | 6% |
| nonlinear_cognate | 11% |
| noise | 46% (chance) |

Clean separation. Caveat: `protein_only` also prefers lag-1 (91%), because a
monotone temporal trend is trivially better predicted one step ahead. Filter
lag hits against the discordance table before interpreting them.

### Cross-layer attribution failed, and you should know why

Gradient × input attribution through the trained encoder recovered the true
cognate transcript as the rank-1 driver **0% of the time**, and anywhere in the
top 20 only 3% of the time.

This is a genuine limitation, not a bug. A 12-dimensional bottleneck compresses
3,000 transcripts into a handful of factors; gradients with respect to
individual genes then reflect *principal-component loadings*, not regulatory
wiring. Correlated transcripts share credit arbitrarily.

If cross-layer dependency (your question 3) matters to you, **do not use
autoencoder attribution for it.** Use instead:

- sparse elastic net / stability selection per protein over **module
  eigengenes** (WGCNA-style), not individual transcripts;
- or a group-lasso penalty on the encoder input layer so the model is forced to
  select transcripts;
- and validate any edge by whether it replicates in the held-out variety.

---

## Outputs

| file | answers |
|---|---|
| `cv_folds.csv`, `model_comparison.csv`, `fig1` | Is the AE worth it? Paired Wilcoxon vs every baseline |
| `permutation_null.csv` | Is any of it above chance? |
| `fig2_leakage.png` | How much did the naive split inflate performance? |
| `latent_design_anova.csv`, `fig3` | **Q1 shared axes** — variance of each latent factor explained by variety / treatment / time / interaction |
| `latent_factors.csv` | Per-sample latent coordinates, ready for plotting |
| `discordance.csv`, `fig4` | **Q2 discordance** — per-protein residual, F-test, FDR |
| `discordance_matrix.csv`, `kinetic_clusters.csv`, `fig5` | Regulatory classes from residual trajectories |
| `lag_atlas.csv` | Regulatory lag per protein: RNA-first / synchronous / ambiguous |
| `cross_layer_edges.csv` | **Q3 cross-layer** — directed attributions, with the cognate-recovery sanity check above |
| `r2_by_truth_class.csv` | Demo only: recovery of planted classes |

---

## Suggested order of work

1. `--demo` — confirm the pipeline recovers planted signal.
2. Your data with `--splits leave_condition_out naive_random`. Compare. Report the gap.
3. If `multimodal_ae` does not beat `pca_ridge`, keep the linear model and put the
   discordance analysis at the centre of the paper.
4. If it does beat it, rerun with `--splits leave_variety_out` and show the
   advantage survives across varieties. That is the result worth defending.
5. Only then consider Stage 4 (ESM-2 embeddings) and Stage 5 (graph attention).

## Tuning notes for n = 48

| parameter | value | why |
|---|---|---|
| `n_rna` | 3,000 | 60K genes collapse to a few thousand informative ones immediately |
| `n_prot` | 1,500–6,000 | variance filtering enriches for signal; report which you used |
| `latent` | 8–16 | above ~16 the model starts memorising conditions |
| `dropout` | 0.10–0.25 | |
| `cross_weight` | 3.0 | pushes capacity toward the RNA → protein path you care about |
| `--variational` | optional | a VAE adds a smoothness prior; it did not improve prediction here |

The framing I would use in a paper: *statistically controlled multi-omics plus
interpretable machine learning to discover temporal RNA–protein regulatory
relationships* — not "we applied deep learning to omics data".
