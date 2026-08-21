import json, sys
nb = json.load(open('rna_protein_unpaired_per_variety.ipynb'))
idxs = [int(x) for x in sys.argv[1:]]
for i in idxs:
    c = nb['cells'][i]
    ct = c['cell_type']
    print('===== CELL %d (%s) exec=%s =====' % (i, ct, c.get('execution_count')))
    print(''.join(c['source']))
    print()
