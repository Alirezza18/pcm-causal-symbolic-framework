# ============================================================================
# scripts/cell09c_table2_grouped_st.py
# Cell 9c — Table 2 (main text): grouped total-order ST
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 11
# (68 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 9c — TABLE 2 (main text): grouped total-order ST, wall vs roof
# Runs after Cell 9 (and 9b). Saves tables/table2_grouped_st.csv
# For complementary groups, ST_wall + ST_roof - 1 = wall x roof interaction
# share (the variance not attributable to either assembly alone).
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

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITY  = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}

g = sob[sob.input.str.startswith('GROUP')].copy()
assert g.groupby(['zone', 'kpi']).size().eq(2).all(), 'expected wall+roof groups for every cell'

piv = g.pivot_table(index='zone', columns=['kpi', 'input'], values='ST').round(3)
piv.columns = [f'{kpi}_{grp.replace("GROUP_", "")}' for kpi, grp in piv.columns]

for kpi in ['EUI', 'IDD', 'TL']:
    piv[f'{kpi}_interaction'] = (piv[f'{kpi}_wall'] + piv[f'{kpi}_roof'] - 1).round(3)

order = [f'{k}_{p}' for k in ['EUI', 'IDD', 'TL'] for p in ['wall', 'roof', 'interaction']]
piv = piv.reindex(ZONES)
piv.insert(0, 'City', [CITY[z] for z in piv.index])
piv.index.name = 'Zone'
piv = piv[order]

# the 3 excluded combos (Cell 9 skips) legitimately appear as NaN - keep them
# visible for honesty, same discipline as Table A11
EXCL = {('0B', 'IDD'): 'degenerate (3.1)', ('2B', 'TL'): 'degenerate (3.1)',
        ('4B', 'TL'): 'constant target (3.1)'}
missing = []
for kpi in ['EUI', 'IDD', 'TL']:
    for z in ZONES:
        if pd.isna(piv.loc[z, f'{kpi}_wall']):
            if (z, kpi) not in EXCL:
                raise ValueError(f'unexpected NaN for {z} {kpi} - not a known exclusion')
            missing.append((z, kpi))
piv.to_csv(BASE + '/tables/table2_grouped_st.csv')
piv.to_csv('table2_grouped_st.csv')
print('Table 2 (main text) - grouped total-order Sobol ST by climate zone')
print(piv.to_string(na_rep='—'))
print('\nNote: interaction share = ST_wall + ST_roof - 1 (variance not attributable')
print('to either assembly alone); negative values indicate near-additive assemblies.')
if missing:
    print('\nExcluded combinations (shown as "—", see Section 3.1):')
    for z, kpi in missing:
        print(f'  {z} {kpi}: {EXCL[(z, kpi)]}')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write('CELL 9c | Table 2 grouped ST saved\n')
files.download('table2_grouped_st.csv')
print('DONE 9c - paste this output back.')
