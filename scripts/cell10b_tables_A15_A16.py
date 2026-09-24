# ============================================================================
# scripts/cell10b_tables_A15_A16.py
# Cell 10b — Tables A15 + A16 from causal outputs
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 15
# (137 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 10b — TABLES A15 + A16 from Cell 10 outputs   (run AFTER Cell 10)
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
ce = pd.read_csv(BASE + '/tables/causal_edges.csv')
cs = pd.read_csv(BASE + '/tables/causal_summary.csv')
shd = pd.read_csv(BASE + '/tables/causal_shd.csv')
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
dfc = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
CITY = dfc.groupby('climate_zone').city.first().to_dict()

VARS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
        'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
NAME = {'pcm_type_wall': 'PCM type (wall)', 'thickness_wall_level': 'PCM thickness (wall)',
        'melting_point_wall_level': 'PCM melt point (wall)', 'position_wall': 'PCM position (wall)',
        'pcm_type_roof': 'PCM type (roof)', 'thickness_roof_level': 'PCM thickness (roof)',
        'melting_point_roof_level': 'PCM melt point (roof)', 'position_roof': 'PCM position (roof)'}
ZONES = sorted(ce.zone.unique())
assert len(ce) == 168, f'expected 168 rows, got {len(ce)}'
assert ce.groupby(['zone', 'kpi']).size().eq(8).all(), 'cell without 8 variables'

# ---------------- diagnostics quoted in the manuscript ----------------
print('=' * 76)
print('DIAGNOSTICS (for Section 3.3)')
print('=' * 76)
for a in ['pc', 'ges', 'notears']:
    f = ce[f'{a}_freq']
    print(f'{a:8s}: max edge freq = {f.max():.3f} | edges >= 0.90: {int((f >= 0.90).sum())} of 168')
nest_bad = []
for _, r in cs.iterrows():
    c = set(x.strip() for x in str(r.causal_parents).split(';') if x.strip() and str(r.causal_parents) != 'nan')
    s = set(sob[(sob.zone == r.zone) & (sob.kpi == r.kpi) & (sob.in_driver_set == True)].input)
    if not c <= s:
        nest_bad.append((r.zone, r.kpi, sorted(c - s)))
print('Nesting (consensus causal subset of Sobol drivers D(Yk)): '
      f'{"ALL 21 CELLS NESTED" if not nest_bad else nest_bad}')
pv = ce[ce.causal_parent == True].input.value_counts()
print('consensus parents per variable (of 21 cells):')
for v in VARS:
    print(f'  {NAME[v]:24s} {int(pv.get(v, 0)):2d}')

# ---------------- Table A15 ----------------
a15 = ce.copy()
a15['zo'] = a15.zone.map({z: i for i, z in enumerate(ZONES)})
a15['ko'] = a15.kpi.map({'EUI': 0, 'IDD': 1, 'TL': 2})
a15['vo'] = a15.input.map({v: i for i, v in enumerate(VARS)})
a15 = a15.sort_values(['zo', 'ko', 'vo']).drop(columns=['zo', 'ko', 'vo'])
out15 = pd.DataFrame({
    'Zone': a15.zone, 'City': a15.zone.map(CITY), 'KPI': a15.kpi,
    'Variable': a15.input.map(NAME),
    'PC freq': a15.pc_freq.round(3), 'GES freq': a15.ges_freq.round(3),
    'NOTEARS freq': a15.notears_freq.round(3), 'Mean freq': a15.mean_freq.round(3),
    'n algos >= 0.90': a15.n_algos_agree.astype(int),
    'Causal parent': np.where(a15.causal_parent == True, 'yes', ''),
})

# ذخیره در درایو و محیط محلی
out15.to_csv(BASE + '/tables/table_A15_causal_edge_freq.csv', index=False)
out15.to_csv('table_A15_causal_edge_freq.csv', index=False)

print('\n' + '=' * 76)
print('Table A15 - bootstrap edge-retention frequencies (168 rows) '
      '-> tables/table_A15_causal_edge_freq.csv')
print('=' * 76)
print(out15.head(16).to_string(index=False))
print('... (16 of 168 rows shown)')

try:
    files.download('table_A15_causal_edge_freq.csv')
except Exception as e:
    print('دانلود خودکار انجام نشد (فایل در گوگل درایو ذخیره شده است):', e)

# ---------------- Table A16 ----------------
cs2 = cs.copy()
cs2['zo'] = cs2.zone.map({z: i for i, z in enumerate(ZONES)})
cs2['ko'] = cs2.kpi.map({'EUI': 0, 'IDD': 1, 'TL': 2})
cs2 = cs2.sort_values(['zo', 'ko']).drop(columns=['zo', 'ko'])
par_txt = cs2.causal_parents.fillna('').str.strip()
par_txt = par_txt.replace('', '— (none detected)')
out16 = pd.DataFrame({
    'Zone': cs2.zone, 'City': cs2.zone.map(CITY), 'KPI': cs2.kpi,
    'Consensus causal parents': par_txt,
    'n causal': cs2.n_causal.astype(int),
    'n Sobol drivers': cs2.n_sobol_drivers.astype(int),
    'Jaccard': cs2.jaccard.round(2),
    'Spearman rho': cs2.spearman_rho.round(2),
    'Mean SHD': cs2.shd_mean.round(2),
})

# ذخیره در درایو و محیط محلی
out16.to_csv(BASE + '/tables/table_A16_cross_lens.csv', index=False)
out16.to_csv('table_A16_cross_lens.csv', index=False)

print('\n' + '=' * 76)
print('Table A16 - cross-lens summary (21 rows) -> tables/table_A16_cross_lens.csv')
print('=' * 76)
print(out16.to_string(index=False))
j = cs2.jaccard.dropna()
rho = cs2.spearman_rho.dropna()
print(f'\nOverall: Jaccard mean={j.mean():.3f} median={j.median():.2f} | '
      f'rho mean={rho.mean():.3f} median={rho.median():.2f}')
print('SHD between algorithm skeletons (mean over cells):')
print(shd.pivot_table(index='algo_a', columns='algo_b', values='shd', aggfunc='mean').round(2).to_string())

try:
    files.download('table_A16_cross_lens.csv')
except Exception as e:
    print('دانلود خودکار انجام نشد (فایل در گوگل درایو ذخیره شده است):', e)

print('\nCaption A15:')
print('Table A15. Bootstrap edge-retention frequencies for each candidate '
      'input-to-KPI edge in the 21 modeled zone-KPI cells: fraction of 500 '
      'row-resampled refits in which PC, GES, and NOTEARS each retained the '
      'edge, their mean, the number of algorithm families reaching the 90% '
      'threshold, and consensus causal-parent status (>=2 of 3 families at '
      '>=0.90).')
print('\nCaption A16:')
print('Table A16. Cross-lens agreement per zone-KPI cell: consensus causal '
      'parents; size of the stability-filtered Sobol driver set D(Yk) '
      '(Table A14); Jaccard overlap of the two sets; Spearman rank correlation '
      'between mean causal edge frequency and the Sobol bootstrap-mean S_T over '
      'the eight inputs; and mean structural Hamming distance between '
      'algorithm skeletons.')
print('\nDONE 10b - paste this output back.')
