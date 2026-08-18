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
