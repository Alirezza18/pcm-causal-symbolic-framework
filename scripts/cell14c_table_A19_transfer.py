# ============================================================================
# scripts/cell14c_table_A19_transfer.py
# Cell 14c — Table A19: cross-climate transfer matrix
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 24
# (87 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 14c - TABLE A19: CROSS-CLIMATE TRANSFER MATRIX (run AFTER Cell 14a)
# A19: shape-transfer R2 of every donor surrogate on every target zone
#      (8 donors x 8 targets x 3 KPIs = 24 rows). Excluded pairs (Section
#      3.1: 0B IDD; 2B TL, 4B TL) appear as em-dashes, matching Fig 11.
# Read-only w.r.t. the producing cell; writes one formatted appendix CSV.
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
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITY = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
        '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
KPIS = ['EUI', 'IDD', 'TL']
EXCL_ZONE = {'EUI': [], 'IDD': ['0B'], 'TL': ['2B', '4B']}

frames = []
for k in KPIS:
    df = pd.read_csv(f'{BASE}/tables/transfer_shape_{k}.csv', index_col=0)
    dm = df.reindex(index=ZONES, columns=ZONES)
    for z in EXCL_ZONE[k]:
        assert dm.loc[z].isna().all() and dm[z].isna().all(), \
            f'{k}: exclusion pattern broken at {z} (expect all-NaN row+col)'
    for z in ZONES:
        if z in EXCL_ZONE[k]:
            continue
        n_exp = 8 - len(EXCL_ZONE[k])
        assert dm.loc[z].notna().sum() == n_exp, \
            f'{k}: row {z} must have exactly {n_exp} values'
    out = pd.DataFrame({'Donor': ZONES,
                        'Donor city': [CITY[z] for z in ZONES],
                        'KPI': k})
    for t in ZONES:                       # columns = target zones
        out[f'{t} ({CITY[t]})'] = dm[t].round(3).values
    frames.append(out)

a19 = pd.concat(frames, ignore_index=True)
n_nan = int(a19.isna().sum().sum())
# IDD: row+col 0B = 15 slots; TL: rows+cols of 2B and 4B = 28 slots -> 43
assert len(a19) == 24
assert n_nan == 43, f'expected 43 excluded slots (IDD 15 + TL 28), got {n_nan}'
a19_out = a19.fillna('\u2014')                 # em-dash in the printed/exported table

# ============================ SAVE & DOWNLOAD FIX ============================
tables_dir = BASE + '/tables'
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, 'table_A19_transfer_matrix.csv')

a19_out.to_csv(csv_path, index=False)

print('=' * 76)
print('Table A19 - cross-climate shape-transfer R2 (24 rows: 8 donors x 3 KPIs,')
print('43 excluded slots shown as em-dashes)')
print(f'-> {csv_path}')
print('=' * 76)
print(a19_out.head(16).to_string(index=False))
print('... (16 of 24 rows shown)')

# دانلود مستقیم و بدون خطا از مسیر گوگل درایو
files.download(csv_path)

print('\nOff-diagonal summary (for the text; NaNs excluded):')
for k in KPIS:
    dm = pd.read_csv(f'{BASE}/tables/transfer_shape_{k}.csv',
                     index_col=0).reindex(index=ZONES, columns=ZONES).values
    off = dm[~np.eye(8, dtype=bool)]
    print(f'  {k}: mean={np.nanmean(off):.3f} min={np.nanmin(off):.3f} '
          f'max={np.nanmax(off):.3f} (n={int(np.isfinite(off).sum())})')
print('\nCaption A19:')
print('Table A19. Shape-transfer performance of every donor surrogate on '
      'every target climate zone, per KPI: R2 of the zone-row surrogate '
      'evaluated on the zone-column designs after level recalibration. '
      'Diagonal entries are the within-zone hold-out references (Table A12); '
      'em-dashes mark the combinations excluded in Section 3.1; the '
      'regime grouping and per-panel means are shown in Figure 11.')
print('\nDONE 14c - paste this output back.')
