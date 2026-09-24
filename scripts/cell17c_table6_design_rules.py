# ============================================================================
# scripts/cell17c_table6_design_rules.py
# Cell 17c (v2) — Table 6 (main text): zone-specific design rules
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 35
# (70 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 17c (v2) - TABLE 6 (main text): ZONE-SPECIFIC DESIGN RULES (after 17a)
# Format + caption + diagnostics for Section 3.8. Reads the frame written by
# Cell 17a (v2), which now states each knee twice - level codes and physical
# values decoded with the single-source-of-truth maps (level_decode_map.csv,
# the same file Cell 19a reads for the validation export).
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
t6 = pd.read_csv(BASE + '/tables/table6_design_rules.csv')
cf = pd.read_csv(BASE + '/tables/uq_conformal.csv')

# ---- diagnostics for the Section 3.8 text ----
print('=' * 76)
print('DIAGNOSTICS (for Section 3.8)')
print('=' * 76)
print('dominant element counts:')
print(t6.Dominant.value_counts().to_string())
cfc = cf.set_index(['zone', 'kpi'])
print('\nrisk (width_norm) detail:')
for z in t6.Zone:
    parts = []
    for k in ['EUI', 'IDD', 'TL']:
        if (z, k) in cfc.index:
            parts.append(f'{k} {100*cfc.loc[(z,k),"width_norm"]:.0f}%')
        else:
            parts.append(f'{k} --')
    print(f'  {z}: ' + ' | '.join(parts))
hi = [(z, k) for z in t6.Zone for k in ['EUI', 'IDD', 'TL']
      if (z, k) in cfc.index and cfc.loc[(z, k), 'width_norm'] > 0.30]
print(f'\ncells with risk > 30% of dynamic range: {hi}')

print('\n' + '=' * 76)
print('Table 6 - zone-specific design rules (codes + physical decode)')
print('=' * 76)
print(t6.to_string(index=False))

csv_path = BASE + '/tables/table6_design_rules.csv'
files.download(csv_path)

print('\nCaption Table 6:')
print('Table 6. Zone-specific design rules distilled from the multi-objective '
      'analysis: the recommended knee configuration in level codes (order '
      'type-thickness-melt point-position; Fig 12b, Table A20) and in physical '
      'units, decoded with the same value lists as the simulation model '
      '(PCM type 0/1 = BioPCM/InfiniteR; thickness 0-3 = 11.2/20.8/37.1/74.2 mm; '
      'melt point 0-4 = 18/21/23/25/29 \u00b0C for BioPCM and 21/23/25/27/29 \u00b0C '
      'for InfiniteR; position 0/1/2 = exterior/middle/interior; the decode map '
      'is provided in the repository as level_decode_map.csv and reused by the '
      'validation export of Section 3.10), the dominant envelope element '
      'identified by the convergence analysis of Section 3.4 (the SHAP rank-1 '
      'family shared by the zone\u2019s cells; the election is recorded in Table '
      'A25), and the prediction risk of each KPI - the 90% conformal interval '
      'width as a percentage of the zone\u2019s observed KPI range (Section 3.7). '
      'Em-dashes mark the Section 3.1 exclusions. Rules whose risk exceeds 30% '
      'of the dynamic range (0B) should be read as directional rather than '
      'quantitative.')

print('\nDONE 17c - paste this output back.')
