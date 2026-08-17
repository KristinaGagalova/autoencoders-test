# Per-variety analysis — when varieties don't share genes

## What changed and why

Your design is two **independent experiments**, not one:

```
Variety A: 2 treatments x 4 timepoints x 3 reps = 24 samples, gene set A
Variety B: 2 treatments x 4 timepoints x 3 reps = 24 samples, gene set B
```

Different samples, different gene/protein universes. That breaks one thing I
told you earlier: **`leave_variety_out` is no longer possible.** Training on A
and predicting B needs a shared input space; there isn't one. Ignore that
recommendation from the earlier README.

The replacement is a two-stage design:

| stage | what happens | needs orthologs? |
|---|---|---|
| 1 | Fit each variety independently, n = 24 | no |
| 2 | Compare the *results* across varieties | yes, for gene-level |

Stage 2 is still real validation — arguably better than cross-prediction.
"The same orthologous genes show post-transcriptional regulation in both
varieties" is a reproducibility claim that doesn't require the two assemblies to
be numerically comparable.

---

## Running it

```bash
python run_per_variety.py \
    --variety A --rna A_rna.csv --prot A_prot.csv --meta A_meta.csv --mapping A_map.csv \
    --variety B --rna B_rna.csv --prot B_prot.csv --meta B_meta.csv --mapping B_map.csv \
    --ortholog orthologs_A_to_B.csv \
    --out results_per_variety
```

Repeat the four flags once per variety, in matching order. Sanity-check first:

```bash
python run_per_variety.py --demo --quick --out results_demo_pv
```

The `--ortholog` flag is optional. Without it you still get both per-variety
analyses and the latent-factor comparison; you lose only the gene-level
conserved/specific calls.

### Ortholog file format

```csv
id_a,id_b
A_AT1G01010,B_Cult2g00123
A_AT1G01030,B_Cult2g00456
```

IDs must match the **protein IDs** used in each variety's protein matrix. Get
them from OrthoFinder on the two proteomes, reciprocal best BLAST, or a
published ortholog table. If both assemblies were annotated against a common
reference, the map is trivial and your "different genes" are really just
presence/absence.

---

## What changes at n = 24

Half the samples means everything shrinks:

| parameter | 48 samples | 24 samples | reason |
|---|---|---|---|
| `latent` | 12 | **6** | above ~8 the model memorises the 8 design cells |
| `n_rna` | 3000 | **1500** | |
| `n_prot` | 1500 | **800** | |
| `hidden` | (128, 32) | **(64, 16)** | |
| `weight_decay` | 1e-3 | **3e-3** | more shrinkage |
| PCA components | 10 | **6** | |
| PLS components | 5 | **4** | |
| CV folds | 16 | **8** | treatment x timepoint cells |

These are already the defaults in `CONFIG_SINGLE_VARIETY`.

**Expect the autoencoder to lose.** At n = 48 it already only won on
interaction-type relationships. At n = 24 the variance penalty roughly doubles.
Run it, report the comparison honestly, and build the paper on the discordance
analysis.

---

## New split schemes

`leave_variety_out` is gone. Available within a variety:

| scheme | folds | what it tests |
|---|---|---|
| `leave_condition_out` | 8 | honest default — all 3 reps of a treatment x timepoint cell held out |
| `leave_treatment_out` | 2 | **train on control, predict treated.** The closest substitute for the lost cross-variety test |
| `leave_timepoint_out` | 4 | temporal extrapolation |
| `naive_random` | 6 | diagnostic only — shows replicate leakage |

`leave_treatment_out` is worth running deliberately. It asks whether the
RNA→protein map learned under control still holds under treatment. **A failure
here is the biological result**, not a modelling failure: it means treatment
altered the mapping, which was one of your original questions.

---

## Cross-variety outputs

At the top level of `--out`:

| file | what it answers |
|---|---|
| `cross_variety_discordance.csv` | per-ortholog: conserved / A-specific / B-specific |
| `cross_variety_lag.csv` | do orthologs get the same lag class in both varieties? |
| `cross_variety_latent.csv` | do both varieties organise variation along the same design axes? (no orthologs needed) |
| `cross_variety_summary.json` | Fisher exact on significant-set overlap, Spearman on effect sizes |
| `figures/cross_variety.png` | F-statistic scatter + conserved/specific bar chart |

**How to read the summary:**

- **Fisher p < 0.05 with odds > 1** — the same genes are discordant in both
  varieties more often than chance. Strong reproducibility claim.
- **Spearman rho on F statistics** — graded version of the same thing; robust to
  where you set the FDR threshold.
- **Fisher p not significant** — either genuinely variety-specific regulation, or
  you're underpowered at n = 24 per variety. Report the effect size (odds ratio)
  alongside the p-value; don't over-interpret a null.

The demo run illustrates the negative case: it simulates the two varieties with
different random seeds, so there is no shared biology by construction, and the
comparison correctly returns Fisher p = 0.40, rho = 0.10. That is what "no
conservation" looks like.

---

## Adapting the notebook

The notebook still works — point it at one variety at a time:

```python
# Section 1 CONFIG
RNA_CSV  = "A_rna.csv"
PROT_CSV = "A_prot.csv"
META_CSV = "A_meta.csv"      # 24 rows
MAPPING_CSV = "A_map.csv"

N_RNA, N_PROT = 1500, 800
LATENT, HIDDEN = 6, (64, 16)
WEIGHT_DECAY = 3e-3
SPLIT_SCHEME = "leave_condition_out"
```

Then change one import so the splits are the single-variety versions:

```python
# Section 4, replace the SPLIT_SCHEMES import
from rnaprot.multivariety import SPLIT_SCHEMES_SINGLE as SPLIT_SCHEMES
from rnaprot.multivariety import covariate_matrix_single as covariate_matrix
```

Run it once per variety into different `OUT_DIR`s, then do the cross-variety
comparison in a final cell:

```python
from rnaprot.multivariety import load_ortholog_map, compare_discordance

disc_a = pd.read_csv("results_A/discordance.csv", index_col=0)
disc_b = pd.read_csv("results_B/discordance.csv", index_col=0)
ortho  = load_ortholog_map("orthologs_A_to_B.csv")

res = compare_discordance(disc_a, disc_b, ortho)
print(f"conserved: {res['n_conserved']}, A-only: {res['n_a_only']}, "
      f"B-only: {res['n_b_only']}")
print(f"Fisher p = {res['fisher_p']:.3g}, Spearman rho = {res['spearman_rho']:.3f}")
```

---

## Normalisation note

Normalise each variety **separately**. Library sizes, dispersion estimates and
variance filters should all be computed within a variety, because the two
assemblies have different gene counts and the CPM denominators aren't
comparable. The pipeline does this automatically when you feed it separate
files — just don't concatenate the count matrices beforehand.

---

## Framing for the paper

> Because the two varieties were annotated against different assemblies, models
> were fitted independently within each variety (n = 24) and compared at the
> level of orthologous genes. Cross-validation held out all three replicates of
> each treatment x timepoint cell. Post-transcriptional regulation was called
> from out-of-fold RNA→protein residuals, and orthologs significant in both
> varieties were classified as conserved.

That is a cleaner and more defensible design than pooling two incompatible
feature spaces would have been.
