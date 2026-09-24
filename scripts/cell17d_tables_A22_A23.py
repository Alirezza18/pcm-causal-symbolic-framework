# ============================================================================
# scripts/cell17d_tables_A22_A23.py
# Cell 17d — Tables A22 + A23 for Section 3.8
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 36
# (131 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 17d - TABLES A22 + A23: APPENDIX FOR SECTION 3.8 (run AFTER 17a/17c)
# Pure formatting of already-saved tables; no model calls.
#
#   A22 - NOMOGRAPH CURVE ATLAS (all 8 zones x 8 variables x modeled KPIs):
#     the complete marginal-effect curves behind Fig 14. Fig 14 shows four
#     representative zones; this table is the full atlas a reader needs to
#     check any claim ("the V-shape holds in every climate") zone by zone.
#     Each row: zone, KPI, variable, level, % change of the KPI vs the
#     all-median design. A 'span' column (max - min of the curve) ranks the
#     variable's leverage in that cell.
#   A23 - DOMINANT-ELEMENT ELECTION LEDGER:
#     the vote behind Table 6's 'Dominant' column, shown in full: for each
#     zone, the SHAP rank-1 variable of every modeled cell (A17), the vote
#     count per family, and the tie-break (total SHAP share) where the vote
#     split. Makes the Table 6 rule auditable instead of asserted.
#
# Reads: tables/nomograph_curves.csv (17a) + shap_attribution.csv (Cell 11)
# Saves: tables/table_A22_nomograph_atlas.csv + table_A23_dominant_election.csv
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
cur = pd.read_csv(BASE + '/tables/nomograph_curves.csv')
sh = pd.read_csv(BASE + '/tables/shap_attribution.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
VARS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
        'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
FAMILY = {'pcm_type_wall': 'wall PCM type', 'thickness_wall_level': 'wall thickness',
          'melting_point_wall_level': 'wall melt point', 'position_wall': 'wall position',
          'pcm_type_roof': 'roof PCM type', 'thickness_roof_level': 'roof thickness',
          'melting_point_roof_level': 'roof melt point', 'position_roof': 'roof position'}
KORD = {'EUI': 0, 'IDD': 1, 'TL': 2}

# ================= A22: full nomograph atlas =================
a = cur.copy()
a['ko'] = a.kpi.map(KORD)
a['vo'] = a.variable.map({v: i for i, v in enumerate(VARS)})
a = a.sort_values(['zone', 'ko', 'vo', 'level']).drop(columns=['ko', 'vo'])
# NOTE: 'pct_change' is also a pandas method - bracket access only
a['curve_span'] = a.groupby(['zone', 'kpi', 'variable'])['pct_change'] \
                   .transform(lambda s: s.max() - s.min()).round(2)
A22 = a[['zone', 'kpi', 'variable', 'level', 'pct_change', 'curve_span']].rename(
    columns={'zone': 'Zone', 'kpi': 'KPI', 'variable': 'Variable', 'level': 'Level',
             'pct_change': 'Pct change vs median design', 'curve_span': 'Curve span (pp)'})
p22 = BASE + '/tables/table_A22_nomograph_atlas.csv'
A22.to_csv(p22, index=False)
print('=' * 76)
print(f'Table A22 - nomograph curve atlas ({len(A22)} rows) -> {p22}')
print('=' * 76)
print(A22.head(12).to_string(index=False))
print(f'... ({len(A22)} rows total = {A22.Zone.nunique()} zones x '
      f'{A22.groupby(["Zone", "KPI"]).size().iloc[0]} curve points per cell)')

# leverage ranking: mean curve span per variable, across cells where modeled
lev = (a.groupby(['variable', 'kpi']).curve_span.mean().unstack()
        .round(1).reindex(VARS))
print('\nmean curve span (pp) by variable and KPI - the leverage hierarchy:')
print(lev.to_string())

# ================= A23: dominant-element election ledger =================
r1 = sh[sh.shap_rank == 1]
rows = []
for z in ZONES:
    s = r1[r1.zone == z]
    cells = ', '.join(f'{k}:{v}' for k, v in
                      zip(s.kpi, s.input.map(FAMILY))) if len(s) else '--'
    if len(s) == 0:
        rows.append({'Zone': z, 'City': ZCITY[z], 'Rank-1 votes (KPI:family)': cells,
                     'Wall-family votes': 0, 'Roof-family votes': 0,
                     'Winner': '--', 'Tie-break': ''})
        continue
    vc = s.input.map(FAMILY).value_counts()
    top = vc[vc == vc.max()].index
    if len(top) > 1:                   # tie -> largest total SHAP share
        tot = s[s.input.map(FAMILY).isin(top)].groupby(
            s.input.map(FAMILY)).shap_share.sum()
        win, tb = tot.idxmax(), f'tie -> larger total SHAP share ({tot.max():.2f} vs ' \
                               f'{sorted(tot)[-2] if len(tot) > 1 else 0:.2f})'
    else:
        win, tb = top[0], ''
    rows.append({'Zone': z, 'City': ZCITY[z], 'Rank-1 votes (KPI:family)': cells,
                 'Votes: wall-side': int(sum('wall' in f for f in s.input.map(FAMILY))),
                 'Votes: roof-side': int(sum('roof' in f for f in s.input.map(FAMILY))),
                 'Winner': win, 'Tie-break': tb})
A23 = pd.DataFrame(rows)
p23 = BASE + '/tables/table_A23_dominant_election.csv'
A23.to_csv(p23, index=False)
print('\n' + '=' * 76)
print(f'Table A23 - dominant-element election ledger ({len(A23)} rows) -> {p23}')
print('=' * 76)
print(A23.to_string(index=False))

files.download(p22)
files.download(p23)

print('\nCaption A22:')
print('Table A22. Complete nomograph atlas: the marginal effect of each design '
      'variable on each modelled KPI, for all eight climate zones (Fig 14 '
      'shows four representative zones). Each row gives the percentage change '
      'of the KPI when one variable is set to the tabulated level and the '
      'remaining seven are held at their zone-median levels, evaluated on the '
      "zone's certified surrogate; 'Curve span' is the range of the curve "
      '(max - min, percentage points), the leverage of the variable in that '
      'cell. Cells of the three Section 3.1 exclusions are absent.')
print('\nCaption A23:')
print('Table A23. The election behind the dominant-element column of Table 6. '
      'For each zone, the SHAP rank-1 variable of every modelled cell (Table '
      'A17) casts one vote for its variable family (the eight families of '
      'the design space); the family with the most votes wins, and where the '
      'vote splits, the tie is broken by the larger total SHAP share of the '
      "tied families across the zone's cells. The wall-side and roof-side "
      'vote totals aggregate the four wall and four roof families for a '
      'compact view; the full per-cell votes are listed in the second '
      'column.')

print('\nDONE 17d - paste this output back.')
