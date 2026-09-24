# ============================================================================
# scripts/cell19c_table_A26_companion.py
# Cell 19c — Table A26: held-out benchmark companion
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 43
# (95 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 19c - TABLE A26: HELD-OUT BENCHMARK COMPANION (run after 19b; no E+)
# Creates Table A26 summarizing surrogate performance & UQ metrics per cell.
# ============================================================================
import os
import numpy as np
import pandas as pd
from google.colab import files

try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

BASE = '/content/drive/MyDrive/PCM_Study'

f_ho = BASE + '/tables/holdout_validation.csv'
f_uq = BASE + '/tables/uq_conformal.csv'
f_vd = BASE + '/tables/uq_vardec.csv'

# Load holdout dataset and normalize join keys
if os.path.exists(f_ho):
    ho = pd.read_csv(f_ho)
    ZC = {
        'BandarAbbas': '0B', 'Bushehr': '1B', 'Kashan': '2B', 'Rasht': '3A',
        'Tehran': '3B', 'Hamedan': '4A', 'Tabriz': '4B', 'Ardebil': '5C'
    }
    # Check if 'zone' has valid zone codes; if not, fall back to mapping 'city'
    key = 'zone' if 'zone' in ho.columns and ho['zone'].astype(str).str.match(r'^\d[A-Z]$').all() else 'city'
    if key == 'city' and 'city' in ho.columns:
        ho['zone'] = ho['city'].map(ZC)

    if 'kpi' in ho.columns:
        ho['kpi'] = ho['kpi'].astype(str).str.upper()
else:
    ho = pd.DataFrame()

uq = pd.read_csv(f_uq) if os.path.exists(f_uq) else pd.DataFrame()
vd = pd.read_csv(f_vd) if os.path.exists(f_vd) else pd.DataFrame()

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']

# Normalize KPI in UQ datasets if available
for df in [uq, vd]:
    if not df.empty and 'kpi' in df.columns:
        df['kpi'] = df['kpi'].astype(str).str.upper()

if not ho.empty and {'zone', 'kpi'}.issubset(ho.columns):
    ho = ho.set_index(['zone', 'kpi'])
if not uq.empty and {'zone', 'kpi'}.issubset(uq.columns):
    uq = uq.set_index(['zone', 'kpi'])
if not vd.empty and {'zone', 'kpi'}.issubset(vd.columns):
    vd = vd.set_index(['zone', 'kpi'])

rows = []
index_source = uq.index if not uq.empty else [(z, k) for z in ZONES for k in ['EUI', 'IDD', 'TL']]

for (z, k) in sorted(index_source, key=lambda i: (ZONES.index(i[0]) if i[0] in ZONES else 99, i[1])):
    r = {'Zone': z, 'KPI': k}

    if not ho.empty and (z, k) in ho.index:
        h = ho.loc[(z, k)]
        r['Adopted'] = h.get('adopted', 'n/a')
        r['Holdout R2'] = round(float(h['holdout_r2']), 3) if pd.notna(h.get('holdout_r2')) else 'n/a'
        r['Holdout RMSE'] = round(float(h['holdout_rmse']), 4) if pd.notna(h.get('holdout_rmse')) else 'n/a'
        r['n_test'] = int(h['n_test']) if pd.notna(h.get('n_test')) else 'n/a'
    else:
        r.update({'Adopted': 'n/a', 'Holdout R2': 'n/a', 'Holdout RMSE': 'n/a', 'n_test': 'n/a'})

    if not uq.empty and (z, k) in uq.index:
        u = uq.loc[(z, k)]
        r['Conformal coverage'] = round(float(u['coverage']), 3) if pd.notna(u.get('coverage')) else 'n/a'
        r['PI width (eng.)'] = round(float(u['width']), 3) if pd.notna(u.get('width')) else 'n/a'
        r['PI width (norm.)'] = round(float(u['width_norm']), 3) if pd.notna(u.get('width_norm')) else 'n/a'

    if not vd.empty and (z, k) in vd.index:
        v = vd.loc[(z, k)]
        r['f_model'] = round(float(v['f_model']), 2) if pd.notna(v.get('f_model')) else 'n/a'

    rows.append(r)

A26 = pd.DataFrame(rows)
csv_path = BASE + '/tables/table_A26_holdout_benchmark.csv'
A26.to_csv(csv_path, index=False)

print('=' * 76)
print(f'Table A26 - surrogate quality chain ({len(A26)} cells) -> {csv_path}')
print('=' * 76)
print(A26.to_string(index=False))
files.download(csv_path)

print('\nDONE 19c - paste this output back.')
