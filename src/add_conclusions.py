import json
import uuid

PATH = "rna_protein_unpaired_per_variety.ipynb"

with open(PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

last = cells[-1]
assert last["cell_type"] == "markdown" and "Interpretation checklist" in "".join(last["source"]), \
    "expected notebook to end with the '## 10 -- Interpretation checklist' cell"

conclusions_md = """## 11 — Results and conclusions

**Condition-level RNA→protein correspondence is real.** Every model beats
its permutation null at p<0.05 in both varieties except `pls` in norin
(p=0.11) — including both design-residualized controls, so this is not an
artifact of having few independent conditions.

**A single gene's own transcript barely predicts its own protein.**
`cognate_ridge` median R² is ~0.00 in norin, 0.07 in cadenza, despite the
matched transcript being present and variable in essentially every fold
(`fraction_folds_cognate_present` = 1.0, ~95–96% variable). Protein state is
predicted far better by the *shared axis of variation* across many genes
(`pca_ridge`: median R² 0.33 norin / 0.36 cadenza) than by gene-specific
cognate signal — consistent with post-transcriptional regulation being
noisy per-gene while condition-level shifts are coordinated transcriptome-wide.

**The autoencoder underperforms simple PCA→ridge.** `condition_aligned_ae`
median R² is 0.08 (norin) / 0.21 (cadenza) vs. 0.33 / 0.36 for `pca_ridge`.
With only a handful of independent conditions, a 1–3 component linear model
has less to overfit than the network — the added AE complexity is not
paying for itself here.

**RNA carries information beyond experimental design, but not uniformly.**
`design_resid_pca_ridge` / `design_resid_pls` keep positive median R²
(0.14–0.22) after removing treatment/timepoint main effects, but their
*mean* R² is strongly negative (down to −1.4 in cadenza) — a minority of
folds are predicted very badly, so this residual signal should be reported
as real-but-unstable, not uniformly good.

**No single condition drives the headline result.** Leave-one-condition-out
sensitivity shifts median R² by only ~0.005–0.007 on average; the most
influential single condition/model combination moves it by up to 0.14
(norin) / 0.20 (cadenza) — worth a closer look if leaning on that specific
number, but not evidence the whole result rests on one condition.

**Cross-variety concordance (via RBH homolog matching) is weak but real, and
gets stronger when trusted more.** Of 81,761 resolved norin↔cadenza
ortholog pairs (96.2% unambiguous 1:1, 3.8% multi-way ties resolved by
RNA-expression correlation), only 45 top-variance proteins had a
`condition_aligned_ae` R² in both varieties. Their R² values correlate at
Spearman ρ = 0.275 on the full set, rising to **0.33** restricted to
unambiguous orthologs only — a modest positive signal that gene
predictability from RNA is somewhat conserved between varieties,
strengthened (not diluted) by restricting to confidently-matched orthologs.
n=45 is small, so this should be read as suggestive rather than conclusive.

**Overall:** the transcriptome→proteome condition-level link holds up under
permutation testing and is not carried by a single condition, but it comes
from broad multivariate structure rather than gene-by-gene cognate
prediction, a simple linear model captures it better than the autoencoder
tested here, and there is a modest, ortholog-confidence-dependent echo of
that predictability across the two wheat varieties.
"""

new_cell = {
    "cell_type": "markdown",
    "id": uuid.uuid4().hex[:8],
    "metadata": {},
    "source": conclusions_md.splitlines(keepends=True),
}

cells.append(new_cell)

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("appended conclusions cell, id =", new_cell["id"], "total cells now:", len(cells))
