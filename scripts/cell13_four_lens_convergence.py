# ============================================================================
# scripts/cell13_four_lens_convergence.py
# Cell 13 — four-lens convergence + Table A18
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 20
# (160 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 13 - PHASE 8: FOUR-LENS CONVERGENCE - lens_convergence.csv + TABLE A18
# (run AFTER Cells 9, 10, 11, 12)
# Unifies the four independent attribution lenses per (zone, KPI) cell:
#   sobol_drivers  - stability-filtered driver set D(Yk)        (Cell 9 / A14)
#   causal_parents - dual-algorithm consensus parents           (Cell 10 / A16)
#   shap_top3      - 3 variables with largest mean|SHAP| share  (Cell 11 / A17)
#   symbolic_vars  - >=2-of-3 GP consensus variables            (Cell 12)
# Writes tables/lens_convergence.csv (pipe-separated sets, one row per cell)
# and table_A18_lens_convergence.csv (the appendix table: per-lens sets,
# pairwise Jaccard overlaps, n-lens agreement per variable).
# Read-only w.r.t. the producing cells; fails loudly on missing inputs.
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import itertools
import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
cs = pd.read_csv(BASE + '/tables/causal_summary.csv')
sh = pd.read_csv(BASE + '/tables/shap_attribution.csv')
sv = pd.read_csv(BASE + '/tables/symbolic_variables.csv')
dfc = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
CITY = dfc.groupby('climate_zone').city.first().to_dict()

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
KPIS = ['EUI', 'IDD', 'TL']
EXCLUDED = [('0B', 'IDD'), ('2B', 'TL'), ('4B', 'TL')]
CELLS = [(z, k) for z in ZONES for k in KPIS if (z, k) not in EXCLUDED]
assert len(CELLS) == 21, f'expected 21 cells, got {len(CELLS)}'

LENS_ORDER = ['sobol_drivers', 'causal_parents', 'shap_top3', 'symbolic_vars']

# ---- assemble the four sets per cell ----
sobv = sob[~sob.input.str.startswith('GROUP')]
sob_map, shap_map, sym_map = {}, {}, {}
for (z, k) in CELLS:
    sob_map[(z, k)] = set(sobv[(sobv.zone == z) & (sobv.kpi == k) &
                               (sobv.in_driver_set == True)].input)
    shap_map[(z, k)] = set(sh[(sh.zone == z) & (sh.kpi == k) &
                             (sh.is_top3 == True)].input)
    sym_map[(z, k)] = set(sv[(sv.zone == z) & (sv.kpi == k) &
                             (sv.in_symbolic_lens == True)].input)
cau_map = {}
for _, r in cs.iterrows():
    cau_map[(r.zone, r.kpi)] = set(
        x.strip() for x in str(r.causal_parents).split(';')
        if x.strip() and str(r.causal_parents) != 'nan')

missing = [(z, k, L) for (z, k) in CELLS for L, m in
           [('sobol', sob_map), ('causal', cau_map), ('shap', shap_map),
            ('symbolic', sym_map)] if (z, k) not in m]
assert not missing, f'missing lens data (are all producing cells done?): {missing}'
assert all(len(m[(z, k)]) > 0 for m in (sob_map, shap_map, sym_map)
           for (z, k) in CELLS), 'an input lens is empty (unexpected)'
# 0B EUI causal = empty set is a REAL result (A16): keep it, flagged in text
print('0B EUI causal set is empty (known result - kept, Jaccard vs it = 0.0)')

# ---- lens_convergence.csv (input for Fig 10) ----
rows = []
for (z, k) in CELLS:
    rows.append({'zone': z, 'kpi': k,
                 'sobol_drivers': '|'.join(sorted(sob_map[(z, k)])),
                 'causal_parents': '|'.join(sorted(cau_map[(z, k)])),
                 'shap_top3': '|'.join(sorted(shap_map[(z, k)])),
                 'symbolic_vars': '|'.join(sorted(sym_map[(z, k)]))})
    s, c, h, y = (sob_map[(z, k)], cau_map[(z, k)],
                  shap_map[(z, k)], sym_map[(z, k)])
    def jac(a, b):
        u = a | b
        return len(a & b) / len(u) if u else np.nan
    for (na, a), (nb, b) in itertools.combinations(
            [('sobol', s), ('causal', c), ('shap', h), ('symbolic', y)], 2):
        rows[-1][f'J_{na}_{nb}'] = round(jac(a, b), 3)
lc = pd.DataFrame(rows)
lc.to_csv(BASE + '/tables/lens_convergence.csv', index=False)
print(f'lens_convergence.csv written ({len(lc)} rows, 21 modeled cells)')

# ---- Table A18 ----
NAME = {'pcm_type_wall': 'PCM type (wall)', 'thickness_wall_level': 'PCM thickness (wall)',
        'melting_point_wall_level': 'PCM melt point (wall)', 'position_wall': 'PCM position (wall)',
        'pcm_type_roof': 'PCM type (roof)', 'thickness_roof_level': 'PCM thickness (roof)',
        'melting_point_roof_level': 'PCM melt point (roof)', 'position_roof': 'PCM position (roof)'}
INPUTS = list(NAME.keys())

def fmt(s):
    return '; '.join(NAME[v] for v in sorted(s)) if s else '\u2014 (none detected)'

a18 = pd.DataFrame([{
    'Zone': z, 'City': CITY[z], 'KPI': k,
    'Sobol drivers D(Yk)': fmt(sob_map[(z, k)]),
    'Causal parents': fmt(cau_map[(z, k)]),
    'SHAP top-3': fmt(shap_map[(z, k)]),
    'Symbolic consensus': fmt(sym_map[(z, k)]),
    'J S-C': rows[CELLS.index((z, k))]['J_sobol_causal'],
    'J S-SH': rows[CELLS.index((z, k))]['J_sobol_shap'],
    'J S-Y': rows[CELLS.index((z, k))]['J_sobol_symbolic'],
    'J C-SH': rows[CELLS.index((z, k))]['J_causal_shap'],
    'J C-Y': rows[CELLS.index((z, k))]['J_causal_symbolic'],
    'J SH-Y': rows[CELLS.index((z, k))]['J_shap_symbolic'],
} for (z, k) in CELLS])
a18.to_csv(BASE + '/tables/table_A18_lens_convergence.csv', index=False)

# ---- diagnostics for the Section 3.4 text ----
print('\n' + '=' * 76)
print('DIAGNOSTICS (for Section 3.4)')
print('=' * 76)
jpairs = [('J_sobol_causal', 'Sobol-Causal'), ('J_sobol_shap', 'Sobol-SHAP'),
          ('J_sobol_symbolic', 'Sobol-Symbolic'), ('J_causal_shap', 'Causal-SHAP'),
          ('J_causal_symbolic', 'Causal-Symbolic'), ('J_shap_symbolic', 'SHAP-Symbolic')]
print('mean pairwise Jaccard over the 21 modeled cells:')
for c, lab in jpairs:
    v = lc[c]
    print(f'  {lab:16s} mean={v.mean():.3f} median={v.median():.2f} '
          f'min={v.min():.2f} max={v.max():.2f}')
allj = np.concatenate([lc[c].values for c, _ in jpairs])
print(f'  ALL 126 pairwise values: mean={np.nanmean(allj):.3f} median={np.median(allj):.2f}')

print('\nvariables named by >=3 of the 4 lenses (count of cells, of 21):')
M = np.zeros((len(INPUTS), len(CELLS)), dtype=int)
for j, (z, k) in enumerate(CELLS):
    sets = [sob_map[(z, k)], cau_map[(z, k)], shap_map[(z, k)], sym_map[(z, k)]]
    for i, v in enumerate(INPUTS):
        M[i, j] = sum(v in s for s in sets)
for i, v in enumerate(INPUTS):
    n3 = int((M[i] >= 3).sum())
    print(f'  {NAME[v]:24s} {n3:2d}  (lens-histogram: '
          f'{dict(zip(*np.unique(M[i], return_counts=True)))})')

print('\nlens set sizes (mean per lens, of 8 variables):')
for lab, m in [('sobol', sob_map), ('causal', cau_map), ('shap', shap_map),
               ('symbolic', sym_map)]:
    sizes = [len(m[(z, k)]) for (z, k) in CELLS]
    print(f'  {lab:9s} mean={np.mean(sizes):.2f} min={min(sizes)} max={max(sizes)}')

print('\n' + '=' * 76)
print('Table A18 - four-lens convergence per cell (21 rows) '
      '-> tables/table_A18_lens_convergence.csv')
print('=' * 76)
print(a18.to_string(index=False, max_colwidth=60))

print('\nCaption A18:')
print('Table A18. Cross-method convergence per zone-KPI cell: the four '
      'independent attribution lenses - the stability-filtered Sobol driver '
      'set D(Yk), the dual-algorithm consensus causal parents, the top-3 '
      'SHAP variables, and the symbolic-regression consensus variables - '
      'with the pairwise Jaccard overlap of every lens pair. The symbolic '
      'lens operates on the raw simulation outputs (no surrogate), and the '
      'causal lens is empty for 0B EUI (Section 3.3).')
print('\nDONE 13 - paste this output back.')
