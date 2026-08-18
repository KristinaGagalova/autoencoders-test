# Unpaired RNA-seq / Proteomics Extension

This is an add-on for the repository:

`KristinaGagalova/autoencoders-test`

It is intended for experiments where RNA-seq and proteomics were measured from **different biological samples**, while sharing experimental conditions such as treatment and timepoint.

## Files

- `rna_protein_unpaired_per_variety.ipynb` — notebook based on the structure of `rna_protein_per_variety.ipynb`.
- `rnaprot/unpaired.py` — data loading, leak-aware preprocessing, condition-aligned autoencoder, condition-level baselines, CV, residuals, permutation control, and synthetic demo.

Copy both into the root of the existing repository so that the new module is located at:

```text
rnaprot/unpaired.py
```

## Required metadata

RNA and protein may use completely different sample IDs. Each modality needs a metadata table with at least:

```text
sample_id,treatment,timepoint,replicate,variety
RNA_001,T0,t0,r1,Norin
RNA_002,T0,t0,r2,Norin
...
```

and independently:

```text
sample_id,treatment,timepoint,replicate,variety
PROT_101,T0,t0,p1,Norin
PROT_102,T0,t0,p2,Norin
...
```

The `replicate` labels do **not** need to correspond between modalities.

If the existing repository metadata correctly annotates both matrices, the same metadata path can be supplied to `rna_meta` and `prot_meta`; downstream code still does not pair RNA and protein rows.

## Model

The original paired objective contains terms equivalent to:

```text
RNA_i -> Protein_i
z_RNA_i ~ z_Protein_i
```

Those objectives are invalid when sample `i` is not the same biological material in both assays.

The unpaired model instead uses:

```text
mean(predicted protein from RNA | condition)
    ~ mean(observed protein | condition)

mean(z_RNA | condition)
    ~ mean(z_Protein | condition)
```

where a condition is `treatment x timepoint` within a variety.

The complete loss is:

```text
L = w_rna_reconstruction * L_rna_reconstruction
  + w_protein_reconstruction * L_protein_reconstruction
  + w_condition_cross * L_condition_mean_RNA_to_protein
  + w_centroid_align * L_condition_latent_alignment
```

No arbitrary RNA/protein replicate pairing is generated.

## Cross-validation

The primary split is leave-one-condition-out. A treatment x timepoint cell is held out from **both** RNA and proteomics.

The model is trained on the remaining conditions, receives the RNA samples from the unseen condition, and predicts the expected protein condition mean. Prediction is compared with the mean of the independent held-out proteomics replicates.

Preprocessing is fitted within each training fold.

## Baselines

The notebook evaluates:

- mean protein profile;
- design-only ridge regression;
- condition-level cognate RNA ridge regression;
- PCA + ridge;
- PLS;
- condition-aligned autoencoder.

PCA/ridge and PLS are also fitted to **condition means**, so they do not require artificial replicate pairing.

## Interpretation

The out-of-fold residual is:

```text
observed protein condition mean - predicted protein condition mean
```

It is a **condition-level RNA-protein residual**, not an individual-sample discordance score.

With only eight treatment x timepoint cells per variety, use these residuals primarily for ranking and hypothesis generation unless additional independent conditions or experiments support formal inference.

Replicate-specific temporal lag analysis is intentionally not implemented here because independent RNA and protein samples do not define a valid `RNA_rep_i(t) -> Protein_rep_i(t+1)` trajectory.


# Reviewer-addressed unpaired RNA/protein analysis

This bundle is designed to replace the current `rnaprot/unpaired.py` and to run through `rna_protein_unpaired_reviewer_addressed.ipynb`.

## What changed

### 1. Cognate ridge no longer depends on top-N RNA variance filtering

Previously, `cognate_ridge` only used a cognate transcript if that transcript happened to be among `N_RNA` top-variable RNA features in the fold. Otherwise it silently fell back to the protein training mean.

The new `_cognate_ridge_predict_unfiltered()` path:

- uses the protein→gene mapping directly;
- uses every mapped cognate gene present in the RNA matrix;
- applies the same sample-wise RNA transform (e.g. CPM + log2);
- learns cognate-gene scaling using outer-training condition means only;
- reports per-protein fold coverage with `cognate_coverage_table()`.

This makes `cognate_ridge` a real biological baseline rather than mostly a mean-model fallback.

### 2. PCA and PLS dimensionality is nested inside outer CV

`_loo_select_components()` chooses PCA/PLS component count using leave-one-condition-out validation among the outer-training conditions only. The held-out outer condition never selects the component count.

The selected dimensions are stored in:

```python
oof["component_selection"]
```

### 3. Added controls for a shared low-rank/design response

New models:

- `pca1_ridge`: one RNA principal component only. If this is almost as good as the tuned PCA model, a single global response axis carries much of the predictive signal.
- `design_resid_pca_ridge`: removes treatment/timepoint main effects from RNA and protein using outer-training conditions, then predicts protein residuals from RNA residuals.
- `design_resid_pls`: analogous PLS control.

These do not prove or disprove causality, but they distinguish broad experimental-design structure from cross-modal information beyond the main effects.

### 4. Added per-condition R2 and leverage diagnostics

`oof_condition_metrics()` reports per-held-condition RMSE and relative R2 across proteins.

`condition_score_sensitivity()` recomputes the headline median per-protein R2 after removing one scored condition at a time. This is a metric-leverage diagnostic for conditions such as `T1|t3`.

### 5. Strengthened permutation null

`permutation_null()` now:

- permutes complete protein condition labels, keeping protein replicates together;
- accepts a `models` subset so linear nulls can be run cheaply;
- accepts `reference_proteins`, allowing the null to be evaluated on exactly the same protein targets as the observed analysis;
- returns a permutation-by-model table.

`permutation_pvalue_table()` calculates one-sided empirical p-values for observed median R2 exceeding the null.

A significant permutation test supports stronger-than-random RNA/protein **condition correspondence**. It does not establish causal RNA→protein regulation.

## Recommended final run

In the notebook configuration:

```python
TUNE_COMPONENTS = True
RUN_PERMUTATIONS = True
N_PERM = 200
```

For a more stable tail probability, use 500–1000 linear permutations if compute allows. The AE permutation test is separated because it is much more expensive:

```python
RUN_AE_PERMUTATIONS = True
N_PERM_AE = 50  # exploratory; increase for final inference if feasible
```

## Files

- `rnaprot/unpaired.py` — replacement module.
- `rna_protein_unpaired_reviewer_addressed.ipynb` — notebook with diagnostics and interpretation guidance.
- `reviewer_changes.patch` — unified diff against the current GitHub `rnaprot/unpaired.py` fetched on 2026-08-18.
