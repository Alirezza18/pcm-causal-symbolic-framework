# ============================================================================
# scripts/cell09b_tables_A13_A14.py
# Cell 9b — Tables A13 + A14 from sobol_indices.csv
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 10
# (92 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 9b — TABLES A13 + A14 from tables/sobol_indices.csv   (run AFTER Cell 9)
# A13: per-variable indices for the 21 modeled cells (mu*, S1, ST, CI,
#      bootstrap retain_frac, driver-set flag). 168 rows = 21 x 8.
# A14: driver-set matrix D(Yk) - 21 rows x 8 boolean flags.
# If your session restarted, run this FIRST (it remounts Drive) and then 9c/9d.
# ============================================================================
# ---- mount Drive if this is a fresh session (no-op if already mounted) ----
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import pandas as pd
import numpy as np
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
bm  = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

VARS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
        'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
SHORT = {'pcm_type_wall': 'PCM type (wall)', 'thickness_wall_level': 'PCM thickness (wall)',
         'melting_point_wall_level': 'PCM melt point (wall)', 'position_wall': 'PCM position (wall)',
         'pcm_type_roof': 'PCM type (roof)', 'thickness_roof_level': 'PCM thickness (roof)',
         'melting_point_roof_level': 'PCM melt point (roof)', 'position_roof': 'PCM position (roof)'}

vars_only = sob[~sob.input.str.startswith('GROUP')].copy()

# sanity gates - stop if the underlying run is not what 3.1 promises
assert len(vars_only) == 21 * 8, f'expected 168 per-variable rows, got {len(vars_only)}'
assert vars_only.groupby(['zone', 'kpi']).size().eq(8).all(), 'cell without 8 variables'
assert set(vars_only.adopted) <= set(bm.best_model), 'adopted model not in benchmark'

# ---------------- Table A13 ----------------
a13 = vars_only.copy()
a13['morris_mu_star'] = a13['morris_mu_star'].round(3)
a13['S1'] = a13['S1'].round(3)
a13['ST'] = a13['ST'].round(3)
a13['ST_conf'] = a13['ST_conf'].round(3)
a13['ST_boot_mean'] = a13['ST_boot_mean'].round(3)
a13['retain_frac'] = a13['retain_frac'].round(2)
a13['Variable'] = a13['input'].map(SHORT)
a13['Driver'] = np.where(a13['in_driver_set'].fillna(False).astype(bool), 'yes', '')
a13 = a13[['zone', 'city', 'kpi', 'adopted', 'Variable', 'morris_mu_star',
           'S1', 'ST', 'ST_conf', 'ST_boot_mean', 'retain_frac', 'Driver']]
a13.columns = ['Zone', 'City', 'KPI', 'Adopted model', 'Variable', 'Morris mu*',
               'S_i', 'S_Ti', 'S_Ti 95% CI', 'S_Ti boot mean', 'Retain frac', 'Driver']
a13.to_csv(BASE + '/tables/table_A13_sobol_per_variable.csv', index=False)
a13.to_csv('table_A13_sobol_per_variable.csv', index=False)
print(f'Table A13 saved: {len(a13)} rows (21 modeled cells x 8 variables)')

# ---------------- Table A14 ----------------
a14 = (vars_only.assign(flag=vars_only.in_driver_set.fillna(False).astype(bool))
       .pivot_table(index=['zone', 'city', 'kpi'], columns='input', values='flag', aggfunc='first')
       .reindex(columns=VARS)
       .astype(int))
a14.columns = [SHORT[c] for c in a14.columns]
a14['n_drivers'] = a14.sum(axis=1)
a14 = a14.reset_index()
a14.to_csv(BASE + '/tables/table_A14_driver_matrix.csv', index=False)
a14.to_csv('table_A14_driver_matrix.csv', index=False)
print(f'Table A14 saved: {len(a14)} rows x 8 driver flags')

print('\nTable A13. Per-variable sensitivity indices for the 21 modeled zone-KPI combinations: '
      'Morris mu* screening ranking, first-order (S_i) and total-order (S_Ti) Sobol indices with 95% CI, '
      'bootstrap stability (mean and fraction of 500 row-resampled refits with S_Ti >= 0.05), and '
      'stability-filtered driver-set membership (S_Ti >= 0.05 in >= 90% of bootstraps). '
      'Tabriz TL is absent (constant target, Section 3.1); Bandar Abbas IDD and Kashan TL are excluded '
      'as degenerate targets (Section 3.1).')
print('\nTable A14. Stability-filtered driver sets D(Yk) per zone-KPI combination '
      '(1 = variable retained as driver).')

print('\n--- A13 preview (first 10 rows) ---')
print(a13.head(10).to_string(index=False))
print('\n--- A14 (full) ---')
print(a14.to_string(index=False))

print('\n--- TOP-3 variables by ST per cell (for the 3.2 text) ---')
top3 = (vars_only.sort_values('ST', ascending=False)
        .groupby(['zone', 'city', 'kpi'])['input'].apply(lambda s: ', '.join(s.head(3))))
for (zone, city, kpi), t in top3.items():
    print(f'  {zone} {city:11s} {kpi:3s}: {t}')

for f in ['table_A13_sobol_per_variable.csv', 'table_A14_driver_matrix.csv']:
    files.download(f)
print('DONE 9b - paste this output back.')
