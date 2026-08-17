# Jupyter Notebook: RNA→Protein Autoencoder Analysis

## Quick start

```bash
# 1. Install dependencies (once)
pip install torch scikit-learn pandas numpy scipy matplotlib seaborn jupyter

# 2. Download and extract the notebook files
# Make sure these are in the SAME directory:
#   - rna_protein_analysis.ipynb    (the notebook)
#   - rnaprot/                       (the package folder)

# 3. Open the notebook
jupyter notebook rna_protein_analysis.ipynb

# 4. Edit section 1: CONFIG
# Set RNA_CSV, PROT_CSV, META_CSV, MAPPING_CSV to your file paths
# Or leave as None to run the demo (synthetic data with known truth)

# 5. Run the cells from top to bottom
# Each section produces tables and figures alongside the code
```

---

## What each section does

### Section 0: Setup
- Install check for PyTorch, pandas, numpy, etc.
- Import the rnaprot package (must be in the same directory)

### Section 1: Global configuration ⚙️
**Edit this section first.** All parameters in one place:

| parameter | default | what it does |
|---|---|---|
| `RNA_CSV` | `None` | Path to RNA count matrix (rows=genes, cols=sample IDs) |
| `PROT_CSV` | `None` | Path to protein intensity matrix |
| `META_CSV` | `None` | Path to metadata (columns: variety, treatment, timepoint, replicate) |
| `MAPPING_CSV` | `None` | Path to protein→gene mapping (2 cols: protein_id, gene_id) |
| `N_RNA` | 3000 | Keep top 3000 genes after variance filtering |
| `N_PROT` | 1500 | Keep top 1500 proteins after variance filtering |
| `RNA_MODE` | `"counts"` | `"counts"` → CPM+log2 normalization, `"logged"` → already normalized |
| `PROT_MODE` | `"intensity"` | `"intensity"` → log2, `"logged"` → already logged |
| `LATENT` | 12 | Latent dimension (8–16 typical; do NOT exceed ~16 at n=48) |
| `HIDDEN` | (128, 32) | Encoder layer sizes (decoder mirrors these) |
| `DROPOUT` | 0.15 | Dropout rate (higher = more regularization) |
| `EPOCHS` | 400 | Max training epochs (early stopping may stop earlier) |
| `W_CROSS` | 3.0 | Weight on RNA→protein cross-modal prediction (the key loss term) |
| `SPLIT_SCHEME` | `"leave_condition_out"` | `"leave_condition_out"` (honest) or `"naive_random"` (diagnostic) |
| `N_PERM` | 2 | Label-permutation nulls (≥5 for publication) |

**To switch between demo and your data:** just change `RNA_CSV` from `None` to your path. Everything else adjusts automatically.

### Section 2: Load data
- Loads synthetic demo data OR your real files
- Prints sample count, gene/protein counts, sample metadata
- Shows experimental design breakdown by variety/treatment/timepoint/replicate

### Section 3: Preprocessing
- Variance filter (fit on training data only inside CV loop — no leakage)
- Protein imputation (left-censored MS values get half-minimum per protein)
- Standardisation
- **Shows:** before/after feature counts + variance distribution plots

### Section 4: Cross-validation
**The core analysis.** Runs 6 models in parallel on 16-fold honest CV:

| model | what it is |
|---|---|
| `mean` | training mean (R²=0 baseline) |
| `design_only` | ridge on variety/treatment/time only, **no RNA** |
| `cognate_ridge` | ridge on protein's own transcript + design |
| `pca_ridge` | PCA(10) on RNA → ridge on all proteins |
| `pls` | partial least squares (supervised analogue of DIABLO) |
| `multimodal_ae` | **your autoencoder** |

**Outputs:**
- Boxplot: all models, per-protein R² clipped to [−1, 1]
- Table: paired Wilcoxon comparison vs pca_ridge
- Leakage demo: honest split vs naive split side-by-side (shows how much replicate leakage inflates performance)
- Red dashed line at R²=0 (predicting training mean = baseline)

**Key reading:**
- If `multimodal_ae` ≈ `pca_ridge` → your RNA→protein map is linear (don't over-interpret)
- If `design_only` ≈ all models → the models learned the design, not RNA biology
- If `multimodal_ae` >> `pca_ridge` under `leave_variety_out` → nonlinear biology + it generalises across varieties (strong result)

### Section 5: Permutation null
Shuffles protein matrix and reruns CV twice (by default). With n=48, this is the strongest control you can put in a paper.

**Result:** real models should collapse to R²≈−0.03 when labels are scrambled.

### Section 6: Discordance (Question 2)
**Your strongest biological result at n=48.**

Residual `D = observed protein − RNA-predicted protein` (out-of-fold).  
F-test: does D depend on treatment?

- **Large random D** = measurement noise
- **Structured D that tracks treatment** = post-transcriptional regulation

**Outputs:**
- Volcano plot: |D| vs −log₁₀(p)
- Kinetic clusters: residual trajectories grouped into regulatory classes (e.g., "fast response", "delayed response", "no response")
- If running demo: crosstab of clusters × planted classes (validates the clustering)

### Section 7: Latent factors (Question 1)
Refit the autoencoder on **all 48 samples** (for interpretability, not prediction).

**Outputs:**
- Heatmap: which design term (variety/treatment/time/interaction) explains variance of each latent factor
- Scatter plots: z1 vs z2 coloured by treatment, variety, timepoint (visual interpretation of shared axes)
- Training curve: validation cross-modal MSE vs epoch (shows where early stopping kicked in)

**Read:** if z3 explains 60% variance from timepoint, that dimension captures the time response.

### Section 8: Regulatory lag atlas (temporal analysis)
Per protein: does RNA_t predict protein_t or protein_{t+1} better?

**Outputs:**
- Distribution: counts of RNA-first/synchronous/ambiguous proteins
- Scatter: R²(lag=0) vs R²(lag=1), coloured by lag preference class
- By planted class (demo only): % where lag-1 wins

**Caveat:** `protein_only` proteins also prefer lag-1 because monotone trends are trivially easier to predict forward. **Filter lag hits against the discordance table before interpreting.**

### Section 9: Demo validation (synthetic data only)
Recovery by planted class: how well each model recovers the known truth.

**Bar chart:** median R² per class (cognate_direct, cognate_lagged, interaction, nonlinear, protein_only, trans_regulated, noise).

Shows you:
- Does the AE beat PCA+ridge on the classes where it should (nonlinear, interaction)?
- Does the discordance test detect protein_only at >80% power?
- Does the lag atlas separate cognate_lagged (88% prefer lag-1) from others?

### Section 10: Output summary
Lists all CSV and PNG files saved to `results_notebook/`.

---

## Input file format

### RNA: `rna_counts.csv`
```
           S001  S002  S003  ...  S048
gene00001    100   110    95  ...   200
gene00002      5     8    12  ...     3
...
```
Rows = genes, columns = sample IDs (matching metadata).  
Raw counts (or already normalized if you set `RNA_MODE="logged"`).

### Protein: `protein_lfq.csv`
```
           S001  S002  S003  ...  S048
prot00001  1e7   2e7   1.5e7 ...  3e7
prot00002  5e6   NaN   4e6   ...  6e6
...
```
Rows = proteins, columns = sample IDs.  
LFQ/iBAQ intensities. NaN = missing values (imputed as half-minimum).

### Metadata: `metadata.csv`
```
     variety treatment timepoint replicate
S001       V0        T0         t0         r0
S002       V0        T0         t0         r1
S003       V0        T0         t0         r2
S004       V0        T0         t1         r0
...
```
Must have exactly these columns (case-sensitive). Order doesn't matter.

### Mapping: `protein_to_gene.csv` (optional)
```
protein_id,gene_id
prot00001,gene00001
prot00002,gene00042
...
```
Used to identify cognate (same gene) vs trans (different gene) relationships.  
If omitted, all proteins treated as unmapped.

---

## Tuning guide for your sample size (n=48)

| situation | recommendation |
|---|---|
| Linear or mildly nonlinear biology | Use `pca_ridge`, not AE. Lower variance, better calibrated p-values. |
| Suspected nonlinear or combinatorial relationships | AE is defensible IF it beats `pca_ridge` under `leave_variety_out`. |
| Very few proteins (<500) | Raise `N_PROT` toward the full set; variance filtering is overly aggressive. |
| Many proteins (>10k) | Raise `N_PROT` to 3000–5000; but watch for overfitting. |
| Large latent dimensions (>16) | Model starts memorising the 16 design cells. Stick with 8–12. |
| High dropout (>0.3) | May be under-fitting; lower to 0.1–0.2. |
| Low dropout (<0.05) | May be over-fitting; raise to 0.15–0.25. |
| `design_only` ≈ `multimodal_ae` | Model learned your design, not RNA→protein biology. Check preprocessing. |
| Permutation null still shows structure | The reconstruction task has leakage somewhere (unlikely with this code, but check). |

---

## Common workflows

### Workflow 1: Quick sanity check
1. Run sections 0–2 (setup + load)
2. Section 3 (preprocessing) — verify feature counts make sense
3. Section 4 only cells 11–12 (CV + model comparison)

**Time:** ~5–10 min  
**Tells you:** does the AE beat simpler models?

### Workflow 2: Full analysis (honest publication-ready)
1. Sections 0–5 (setup + load + preprocess + CV + null)
2. Sections 6–7 (discordance + latent factors)
3. Section 8 (lag atlas)

**Time:** 30–60 min  
**Produces:** all tables and figures needed for a paper

### Workflow 3: Deep dive (include cross-layer analysis)
1. All of workflow 2
2. Add custom cells after section 8:
   ```python
   from rnaprot.evaluate import cross_layer_attribution
   edges = cross_layer_attribution(ae_full, R_full, cov_full, 
                                   top_k=30, protein_idx=[...])
   ```

**Time:** 60–90 min

### Workflow 4: Hypothesis testing
1. Pre-register your hypothesis (e.g., "treatment response is post-transcriptional")
2. Run the notebook with `SPLIT_SCHEME="leave_variety_out"`
3. Report both the honest CV results AND the leakage diagnostic
4. If AE beats PCA+ridge on variety-B trained on variety-A → claim is defensible

**Time:** 30–60 min

---

## Output files

All outputs saved to `results_notebook/`:

| file | what it is |
|---|---|
| `cv_folds.csv` | Per-fold, per-model performance (median/mean R², fraction of proteins above R²=0) |
| `per_protein_r2_*.csv` | R² per protein for each model (rows=proteins, cols=models) |
| `model_comparison.csv` | Paired Wilcoxon tests vs reference model |
| `permutation_null.csv` | Label-shuffled null R² |
| `discordance.csv` | Per-protein residual, F-stat, FDR, variance explained by RNA |
| `discordance_matrix.csv` | Sample × protein residuals (for your own clustering) |
| `kinetic_clusters.csv` | Protein cluster assignments + discordance stats |
| `lag_atlas.csv` | Per-protein lag preference (R²_lag0 vs R²_lag1) + lag class |
| `latent_factors.csv` | Per-sample latent coordinates |
| `latent_design_anova.csv` | Variance of each latent factor by design term |
| `figures/*.png` | 8 high-res publication-ready figures |

Open CSVs in R, Python, or Excel. Figures are 150 DPI PNG (resize freely).

---

## Troubleshooting

**Q: "CUDA out of memory"**  
A: Set `device="cpu"` in the CONFIG (inside `AERegressor(...)`). PyTorch CPU is slower but won't OOM.

**Q: "ImportError: No module named 'rnaprot'"**  
A: The `rnaprot/` folder must be in the same directory as the notebook.

**Q: Very slow CV**  
A: Reduce `N_RNA` to 1200, `N_PROT` to 600, `EPOCHS` to 200.

**Q: All models perform equally poorly**  
A: Check that your RNA and protein matrices are in the right format (rows=features, cols=samples) and metadata sample order matches.

**Q: `design_only` beats `multimodal_ae`**  
A: The model learned the design covariates instead of RNA biology. This happens when RNA is not actually informative for the protein response. Report this as a null result.

**Q: Discordance test shows no treatment effect**  
A: Either (a) most proteins respond transcriptionally (not post-transcriptionally), or (b) your treatment is weak. Both are valid findings.

---

## What to report in a paper

**Methods:**
> We fitted a multimodal autoencoder to matched transcriptome and proteome data using cross-modal reconstruction loss. Honest cross-validation with design-aware folds (leaving out all replicates of a variety × treatment × timepoint cell) evaluated performance against five baselines including PCA-ridge and cognate-ridge. Label-permutation controls verified R² above chance.

**Results (if AE wins):**
> The autoencoder outperformed linear-baseline models (pca-ridge: median R² = 0.22 vs multimodal-AE: 0.31, Wilcoxon p < 0.001) under leave-variety-out splitting, indicating treatment-specific nonlinear RNA–protein relationships generalise across varieties.

**Results (if AE loses or ties):**
> RNA–protein relationships in this system are predominantly linear: pca-ridge (R² = 0.36) and multimodal-AE (R² = 0.35) performed similarly (Wilcoxon p = 0.64), with no significant advantage to nonlinear modelling. We therefore report results from the simpler model.

**Discordance:**
> Of 1,500 proteins, 252 (17%) showed treatment-dependent residuals at FDR 5%, consistent with post-transcriptional regulation. These proteins fell into six kinetic classes defined by residual trajectory clustering, from rapid response to delayed response.

**Lag:**
> Regulatory lag analysis identified 596 proteins (40%) where transcript changes preceded protein changes (RNA-first / protein-later class), suggesting multi-step translation or protein synthesis delays.

---

## Citation

If you use this notebook or the rnaprot package, please cite:

> [Your name]. (2026). RNA-protein multimodal autoencoder for small factorial omics designs. https://github.com/[...] 

And the three key papers it's built on:

1. MOFA/MOFA2 (shared latent factors) — Argelaguet et al., Genome Biology 2020
2. DIABLO (supervised multi-omics) — Rohart et al., Genome Biology 2017
3. Multi-omics time-course design/validation — [your favorite reference]

---

**Last updated:** August 2026  
**Python version:** 3.8+  
**Key dependencies:** torch, scikit-learn, pandas, numpy, scipy, matplotlib, seaborn
