# ============================================================================
# scripts/cell15c_table_A20_knees.py
# Cell 15c — Table A20: knee designs + single-objective optima
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 27
# (147 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 15c - TABLE A20: KNEE DESIGNS + SINGLE-OBJECTIVE OPTIMA (after 15a)
# A20: per zone - the knee design (Pareto point nearest the ideal point in
# min-max normalized space) and the three single-objective optima, in RAW
# engineering units, with the multi-objective price of each optimum spelled
# out: how much EUI / IDD / TL the knee pays relative to each pure optimum
# (the "cost of balance"), and how much each pure optimum pays on the other
# two KPIs relative to the knee. Read-only w.r.t. Cell 15a outputs; writes
# one formatted appendix CSV.
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
fr = pd.read_csv(BASE + '/tables/pareto_front.csv')
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')
dfc = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
CITY = dfc.groupby('climate_zone').city.first().to_dict()

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
KPIS = ['EUI', 'IDD', 'TL']
IN_LABEL = {'pcm_type_wall': 'W-type', 'thickness_wall_level': 'W-thk',
            'melting_point_wall_level': 'W-melt', 'position_wall': 'W-pos',
            'pcm_type_roof': 'R-type', 'thickness_roof_level': 'R-thk',
            'melting_point_roof_level': 'R-melt', 'position_roof': 'R-pos'}
INPUTS = list(IN_LABEL)
NAME = {'pcm_type_wall': 'PCM type (wall)', 'thickness_wall_level': 'PCM thickness (wall)',
        'melting_point_wall_level': 'PCM melt point (wall)', 'position_wall': 'PCM position (wall)',
        'pcm_type_roof': 'PCM type (roof)', 'thickness_roof_level': 'PCM thickness (roof)',
        'melting_point_roof_level': 'PCM melt point (roof)', 'position_roof': 'PCM position (roof)'}
VAR_ORDER = ['PCM type (wall)', 'PCM thickness (wall)', 'PCM melt point (wall)',
             'PCM position (wall)', 'PCM type (roof)', 'PCM thickness (roof)',
             'PCM melt point (roof)', 'PCM position (roof)']

KNEE = kr[kr.type == 'knee'].set_index('zone')
OPTS = kr[kr.type.str.startswith('min_')]

# ---- text diagnostics ----
print('=' * 76)
print('DIAGNOSTICS (for Section 3.6)')
print('=' * 76)
# variable frequency among knee designs across zones
freq = {NAME[c]: 0 for c in INPUTS}
for z in ZONES:
    if z not in KNEE.index:
        continue
    row = KNEE.loc[z]
    # non-zero-varied variables only: compare to zone grid minima
    sub = fr[fr.zone == z]
    for c in INPUTS:
        if sub[c].nunique() > 1 and row[c] != sub[c].min():
            freq[NAME[c]] += 1
        elif sub[c].nunique() == 1:
            freq[NAME[c]] += 0
print('knee-design variable activity (zones in which the variable is not at '
      'its grid minimum):')
for v, n in freq.items():
    print(f'  {v:24s} {n}/8')

# cost of balance: knee vs each pure optimum, in each KPI
print('\ncost of balance (knee pays this % more than the pure optimum):')
print(f'  {"zone":5s} ' + ' '.join(f'{k:>10s}' for k in KPIS))
for z in ZONES:
    if z not in KNEE.index:
        continue
    cells = []
    for k in KPIS:
        o = OPTS[(OPTS.zone == z) & (OPTS.type == f'min_{k}')]
        if len(o) == 0 or k not in KNEE.loc[z].index or pd.isna(KNEE.loc[z][k]):
            cells.append('        --')
            continue
        ov = float(o.iloc[0][k]); kv = float(KNEE.loc[z][k])
        pct = 100 * (kv - ov) / ov if ov != 0 else np.nan
        cells.append(f'{pct:9.1f}%')
    print(f'  {z:5s} ' + ' '.join(cells))

print('\npure-optimum price (each optimum pays this % more on the other KPIs '
      'vs the knee):')
for z in ZONES:
    if z not in KNEE.index:
        continue
    parts = []
    for k in KPIS:
        o = OPTS[(OPTS.zone == z) & (OPTS.type == f'min_{k}')]
        if len(o) == 0:
            continue
        others = [kk for kk in KPIS if kk != k and kk in o.iloc[0].index
                  and not pd.isna(o.iloc[0][kk]) and not pd.isna(KNEE.loc[z][kk])]
        for kk in others:
            ov, kv = float(o.iloc[0][kk]), float(KNEE.loc[z][kk])
            if kv != 0:
                parts.append(f'{k}-opt pays +{100*(ov-kv)/kv:.0f}% {kk}')
    print(f'  {z:5s} ' + (' | '.join(parts) if parts else '(single KPI)'))

# ---- Table A20 ----
rows = []
for z in ZONES:
    if z not in KNEE.index:
        continue
    sub = fr[fr.zone == z]
    ok = [k for k in KPIS if k in KNEE.loc[z].index
          and not pd.isna(KNEE.loc[z][k])]
    recs = [('knee (multi-objective)', KNEE.loc[z])] + \
           [(f'min {k}', OPTS[(OPTS.zone == z) & (OPTS.type == f'min_{k}')].iloc[0])
            for k in KPIS
            if len(OPTS[(OPTS.zone == z) & (OPTS.type == f'min_{k}')])]
    for label, r in recs:
        row = {'Zone': z, 'City': CITY.get(z, r['city']), 'Design': label}
        for v in VAR_ORDER:
            col = [c for c in INPUTS if NAME[c] == v][0]
            row[v] = int(r[col])
        for k in ok:
            row[k] = round(float(r[k]), 1)
        rows.append(row)
A20 = pd.DataFrame(rows)
A20 = A20.sort_values(['Zone', 'Design'],
                      key=lambda s: s.map({z: i for i, z in enumerate(ZONES)})
                      if s.name == 'Zone' else s)
csv_path = BASE + '/tables/table_A20_pareto_knee.csv'
A20.to_csv(csv_path, index=False)
print('\n' + '=' * 76)
print(f'Table A20 - knee designs and single-objective optima '
      f'({len(A20)} rows) -> {csv_path}')
print('=' * 76)
print(A20.head(12).to_string(index=False))
print('...')

files.download(csv_path)
print('\nCaption A20:')
print('Table A20. Knee designs and single-objective optima per climate zone, '
      'in engineering units. For each zone: the knee design - the Pareto point '
      'minimizing the Euclidean distance to the ideal point in min-max '
      'normalized KPI space - and the pure minima of each KPI over the full '
      'level grid, with all eight design variables and the three resulting '
      'KPI values. Variable levels are the level-coded integers of the '
      'Section 3.1 design space; the three excluded combinations are absent.')
print('\nDONE 15c - paste this output back.')
