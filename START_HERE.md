# RNA → Protein Multimodal Autoencoder Analysis

**Complete toolkit for small-sample, high-dimensional transcriptomics + proteomics integration at n ≈ 48 samples.**

---

## 📦 What's included

```
outputs/
├── rna_protein_analysis.ipynb    ← Jupyter notebook (run this!)
├── rnaprot/                       ← Python package (must be in same folder)
│   ├── data.py                    (loading, preprocessing, CV splits)
│   ├── models.py                  (multimodal AE in PyTorch)
│   ├── baselines.py               (5 baseline models)
│   ├── evaluate.py                (metrics, permutation null, discordance)
│   ├── simulate.py                (synthetic data with known ground truth)
│   └── __init__.py
├── rnaprot_project.zip            ← Everything packaged for download
├── NOTEBOOK_GUIDE.md              ← Detailed guide to the notebook
├── README.md                       ← Technical documentation
└── START_HERE.md                  ← You are here
```

---

## 🚀 Quick start (5 minutes)

### Option A: Run on synthetic demo data (no files needed)

```bash
# 1. Install (once)
pip install torch scikit-learn pandas numpy scipy matplotlib seaborn jupyter

# 2. Open the notebook
cd /path/to/outputs
jupyter notebook rna_protein_analysis.ipynb

# 3. Run all cells from top to bottom
#    Leave RNA_CSV = None in section 1
#    The notebook will generate synthetic data with 7 known protein classes
```

### Option B: Run on your real data

```bash
# 1. Prepare your files in the same directory as the notebook:
#    - rna_counts.csv       (rows=genes, cols=samples)
#    - protein_lfq.csv      (rows=proteins, cols=samples)
#    - metadata.csv         (cols: variety, treatment, timepoint, replicate)
#    - protein_to_gene.csv  (optional: cols: protein_id, gene_id)

# 2. Open the notebook and edit section 1:
RNA_CSV = "rna_counts.csv"
PROT_CSV = "protein_lfq.csv"
META_CSV = "metadata.csv"
MAPPING_CSV = "protein_to_gene.csv"

# 3. Run all cells
```

---

## 📖 Read this next

| Document | when to read | time |
|---|---|---|
| **NOTEBOOK_GUIDE.md** | You're about to run the notebook | 10 min |
| **README.md** | You want to understand the methods | 15 min |
| **rnaprot/**  code | You want to modify or extend the analysis | 30 min |

---

## 🎯 What does this do?

**Three core questions about RNA → protein relationships:**

1. **Shared axes of variation** — Which transcriptional programs show up in both RNA and protein?
   - Output: latent factor heatmap + design ANOVA

2. **RNA-protein discordance** — Where does RNA fail to explain protein (post-transcriptional regulation)?
   - Output: volcano plot + kinetic clusters + regulatory classes

3. **Cross-layer dependency** — Which transcripts predict which non-cognate proteins?
   - Output: directed attribution network (with caveats, see README.md)

**Bonus:** regulatory lag atlas — which proteins respond with a delay?

---

## 💡 Key features

✅ **Honest cross-validation** — replicates never leak across train/test splits  
✅ **No feature selection leakage** — filtering fit on training samples only  
✅ **Permutation nulls** — label-shuffled controls prove results are real  
✅ **Comparison baselines** — AE results benchmarked against PCA, ridge, PLS, cognate-ridge  
✅ **Demo data with ground truth** — validate the pipeline before trusting it on real data  
✅ **Design-aware splits** — exploit your factorial structure (variety × treatment × time × replicate)  
✅ **Publication-ready figures** — 150 DPI, vectorisable, labeled  

---

## ⚠️ Important: Read this before running

### Sample size: n = 48
With 3 replicates × 2 varieties × 2 treatments × 4 timepoints, your **effective sample size for parameter learning is very small**. This design is powerful for hypothesis testing (the replication structure) but limited for fitting complex models.

**Implications:**
- The autoencoder **does not automatically win** against simpler models (it shouldn't — simpler is often better at n=48)
- Deep learning is only justified if your biology is nonlinear or combinatorial
- Permutation controls are essential (included in the notebook)
- Always use `leave_variety_out` split to test generalization across varieties
- Report the `naive_random` split gap to show you didn't leak information

### The discordance test is your strongest result
Protein residuals D = observed − RNA-predicted show which genes are regulated post-transcriptionally. This is **much more likely to be reproducible** than raw prediction R² at n=48.

### Report both positive and null results
- If AE ties PCA+ridge: "RNA-protein relationships are linear; we report simpler model"
- If lag atlas is empty: "No evidence for regulatory delay; treatment response is synchronous"
- If design_only ≈ AE: "Model learned the design, not RNA biology; suggests weak transcriptional response"

All of these are valid findings.

---

## 📊 Example workflow

```
1. Load demo data (5 min)
   → Confirm pipeline recovers planted signal

2. Load your real data (5 min)
   → Check filtering: feature counts make sense?

3. Run leave_condition_out CV (15 min)
   → Are models beating baselines?

4. Run leave_variety_out CV (15 min)
   → Does AE generalise across varieties?

5. Run permutation nulls (10 min)
   → Is the signal above chance?

6. Discordance analysis (10 min)
   → Which proteins are post-transcriptionally regulated?

7. Lag atlas (5 min)
   → Are there regulatory delays?

→ Total: ~60 minutes for full analysis
```

---

## 🔧 System requirements

- **Python** 3.8+
- **Memory** 8 GB RAM (16+ GB recommended)
- **GPU** optional (CPU is fine for n=48)
- **Storage** 500 MB for code + data + outputs
- **Time** 30–90 min per run (depending on CV folds + permutations)

---

## ✅ Validation checklist before publishing

- [ ] Ran demo first; pipeline recovers planted classes at >80% power?
- [ ] Ran `leave_variety_out` split; results generalise across varieties?
- [ ] Reported `naive_random` split gap (how much did replicates leak)?
- [ ] Permutation nulls are clearly below real R² (your signal isn't chance)?
- [ ] `design_only` model does NOT match your fancy model?
- [ ] Discordance hits are NOT concentrated in worst-fit proteins?
- [ ] If claiming regulatory lag: filtered lag hits against discordance table?
- [ ] If claiming nonlinear biology: AE beats PCA+ridge at FDR-corrected significance?
- [ ] Methods section specifies split scheme and early-stopping criterion?

---

## 🤔 Frequently asked questions

**Q: My AE performs worse than PCA+ridge. Should I worry?**  
A: No. At n=48, simpler models are often better. Report this finding: "Linear latent space (PCA) explains RNA-protein relationships without benefit from nonlinear modelling."

**Q: How do I know if the discordance test is real?**  
A: Cross-filter against kinetic clustering. Proteins with FDR<5% should cluster into a group with structured residuals (not random noise). See fig5_kinetic_clusters.png.

**Q: Can I use this for n=200 samples?**  
A: Yes, with modifications. Raise `LATENT` to 16–32, `N_PROT` to 3000–6000, `EPOCHS` to 600. Start with the same CONFIG and adjust up if you have evidence of over-regularization (training curve plateaus early).

**Q: The notebook is very slow — how do I speed it up?**  
A: 
- Reduce `N_RNA` to 1200, `N_PROT` to 600
- Reduce `EPOCHS` to 200
- Reduce `N_PERM` to 1 (or 0)
- Use `leave_condition_out` (16 folds) instead of `leave_variety_out` (2 folds)

**Q: Can I add my own models?**  
A: Yes. Add a new class to `rnaprot/baselines.py` with `.fit(R, P, cov)` and `.predict(R, cov)` methods. The CV loop will automatically include it.

**Q: What if I have unpaired RNA/protein (different samples)?**  
A: This pipeline requires matched samples. You'll need a different approach (e.g., imputation, or model proteins as a function of bulk RNA statistics).

---

## 📚 Further reading

**For methodological depth:**
- README.md (this package)
- MOFA2 paper: Argelaguet et al. Genome Biology 2020 (latent factors)
- DIABLO paper: Rohart et al. Genome Biology 2017 (multi-omics integration)

**For autoencoder design:**
- "An Introduction to Variational Autoencoders" — Kingma & Welling
- Chapter 13 of Raschka's "Machine Learning with PyTorch" (autoencoder architectures)

**For small-sample ML:**
- Hastie et al. "The Elements of Statistical Learning" (why simplicity wins at small n)
- "High-Dimensional Statistics: A Non-Asymptotic Viewpoint" — Wainwright (theory)

---

## 📬 Support

**If the notebook crashes:**
1. Check the error message in the notebook cell output
2. Search NOTEBOOK_GUIDE.md "Troubleshooting" section
3. Verify your input files match the expected format
4. Try reducing `N_RNA` and `N_PROT` (features × samples imbalance is the most common issue)

**If you get bad results:**
1. Run the demo first — does it recover planted signal?
2. Check `cv_folds.csv` — are models separating at all on individual folds?
3. Look at permutation nulls — are they clearly below your real R²?
4. Verify sample order matches across RNA, protein, and metadata CSVs

---

## 📄 Citation

If this toolkit is useful, please cite:

```bibtex
@software{rnaprot2026,
  title={RNA-protein multimodal autoencoder for small factorial omics designs},
  author={[Your name]},
  url={https://github.com/[...]},
  year={2026}
}
```

And the foundational papers:
- Argelaguet et al. (2020) MOFA2
- Rohart et al. (2017) DIABLO
- Raschka et al. (2022) ML with PyTorch book

---

**Ready to go?** Open the notebook: `jupyter notebook rna_protein_analysis.ipynb`

Then read **NOTEBOOK_GUIDE.md** for detailed section-by-section guidance.

Good luck! 🚀
