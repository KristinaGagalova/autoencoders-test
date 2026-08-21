import json

PATH = "rna_protein_unpaired_per_variety.ipynb"

with open(PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# --- cell 4: point ORTHOLOG_CSV at the RBH homolog map ---
c4 = cells[4]
assert c4["cell_type"] == "code"
src4 = "".join(c4["source"])
assert "ORTHOLOG_CSV = None" in src4, "expected cell 4 to contain 'ORTHOLOG_CSV = None'"
src4 = src4.replace(
    '# Optional ortholog table: columns id_a, id_b\nORTHOLOG_CSV = None',
    '# Optional ortholog table: RBH homolog map between varieties (query,target),\n'
    '# resolved to id_a/id_b by load_ortholog_map() below.\n'
    'ORTHOLOG_CSV = "data/norin_cadenza_rbh-clean.csv"',
)
assert 'ORTHOLOG_CSV = "data/norin_cadenza_rbh-clean.csv"' in src4
c4["source"] = src4.splitlines(keepends=True)
c4["outputs"] = []
c4["execution_count"] = None

# --- cell 32: use load_ortholog_map() (RNA-expression tie-breaking) instead
#     of a naive id_a/id_b merge, matching rna_protein_per_variety.ipynb /
#     rna_protein_umap_shared_space.ipynb ---
c32 = cells[32]
assert c32["cell_type"] == "code"
old_src32 = "".join(c32["source"])
assert 'if ORTHOLOG_CSV is not None and len(R2_TABLES) >= 2:' in old_src32

new_src32 = '''from rnaprot.multivariety import load_ortholog_map

if ORTHOLOG_CSV is not None and len(R2_TABLES) >= 2:
    labs = list(R2_TABLES)[:2]
    a, b = labs

    def _rna_condition_means(d):
        # Same definition as rnaprot.multivariety.rna_condition_means(), but
        # UnpairedOmicsData exposes rna_meta (not meta), so this is inlined
        # rather than reusing that helper directly.
        cond = d.rna_meta["treatment"].astype(str) + "|" + d.rna_meta["timepoint"].astype(str)
        return d.rna.groupby(cond.to_numpy()).mean()

    # RBH is frequently many-to-many; load_ortholog_map() collapses each id_a
    # to one id_b, breaking multi-way ties with condition-mean RNA expression
    # correlation instead of silently pseudoreplicating tied candidates.
    rna_cm_a = _rna_condition_means(DATA[a])
    rna_cm_b = _rna_condition_means(DATA[b])
    ortho = load_ortholog_map(ORTHOLOG_CSV, rna_a=rna_cm_a, rna_b=rna_cm_b)
    ortho.to_csv(OUT_DIR / "ortholog_map.csv", index=False)

    n_o = len(ortho)
    n_tied = int((ortho["n_candidates"] > 1).sum())
    resolved = ortho.loc[ortho.resolution == "expression_resolved_tie", "best_corr"]
    print(f"ortholog map: {n_o:,} pairs "
          f"({(ortho.resolution == 'unambiguous').sum():,} unambiguous, "
          f"{len(resolved):,} tie-resolved by expression"
          + (f" [median best_corr={resolved.median():.2f}]" if len(resolved) else "")
          + f", {(ortho.resolution == 'unresolved_tie').sum():,} unresolved)")
    tie_frac = n_tied / n_o if n_o else 0.0
    print(f"  {tie_frac:.1%} of matched genes hit a multi-way RBH tie")

    ra = R2_TABLES[a]["condition_aligned_ae"].rename("R2_a").rename_axis("id_a").reset_index()
    rb = R2_TABLES[b]["condition_aligned_ae"].rename("R2_b").rename_axis("id_b").reset_index()
    comp = ortho.merge(ra, on="id_a", how="inner").merge(rb, on="id_b", how="inner")
    print(f"orthologs with R\\u00b2 in both varieties: {len(comp)}")
    display(comp.head())
    if len(comp) >= 3:
        rho = comp[["R2_a", "R2_b"]].corr(method="spearman").iloc[0, 1]
        print("Spearman R\\u00b2 concordance (full ortholog map):", rho)

    # sensitivity check: does restricting to unambiguous RBH hits change the
    # headline concordance? if not, the expression tie-break wasn't doing
    # much work; if it does, that belongs explicitly in the methods.
    unamb = ortho[ortho.resolution == "unambiguous"]
    comp_unamb = unamb.merge(ra, on="id_a", how="inner").merge(rb, on="id_b", how="inner")
    if len(comp_unamb) >= 3:
        rho_u = comp_unamb[["R2_a", "R2_b"]].corr(method="spearman").iloc[0, 1]
        print(f"  sensitivity check, unambiguous_rbh only: n={len(comp_unamb):,} | "
              f"Spearman rho={rho_u:.3f}")

    comp.to_csv(OUT_DIR / "cross_variety_r2_concordance.csv", index=False)
else:
    print("No ortholog table configured; per-variety analysis is complete.")
'''

c32["source"] = new_src32.splitlines(keepends=True)
c32["outputs"] = []
c32["execution_count"] = None

# --- clear stale outputs/exec counts on every other code cell so the
#     notebook reflects one clean top-to-bottom run, not the old
#     out-of-order interactive session (45, 61, 47, 48, ... 62 KeyboardInterrupt) ---
for c in cells:
    if c["cell_type"] == "code" and c is not c4 and c is not c32:
        c["outputs"] = []
        c["execution_count"] = None

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("patched", PATH)
