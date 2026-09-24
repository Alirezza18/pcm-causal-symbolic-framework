# ============================================================================
# scripts/cell11c_table_A17_shap.py
# Cell 11c — Table A17: SHAP per variable
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 18
# (109 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 11c - TABLE A17: SHAP attribution per variable (run AFTER Cell 11)
# A17: mean|SHAP|, normalized share, rank, top-3 lens flag, and perturbation
#      scheme for every variable in every modeled cell (21 x 8 = 168 rows).
# Also prints the text diagnostics for Section 3.4 (top-1 shares, rank-1
# agreement with Sobol argmax, per-KPI means).
# Read-only w.r.t. Cell 11 outputs; writes one formatted appendix CSV.
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
sh = pd.read_csv(BASE + '/tables/shap_attribution.csv')
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
dfc = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
CITY = dfc.groupby('climate_zone').city.first().to_dict()

VARS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
        'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
NAME = {'pcm_type_wall': 'PCM type (wall)', 'thickness_wall_level': 'PCM thickness (wall)',
        'melting_point_wall_level': 'PCM melt point (wall)', 'position_wall': 'PCM position (wall)',
        'pcm_type_roof': 'PCM type (roof)', 'thickness_roof_level': 'PCM thickness (roof)',
        'melting_point_roof_level': 'PCM melt point (roof)', 'position_roof': 'PCM position (roof)'}
ZONES = sorted(sh.zone.unique())
assert len(sh) == 168, f'expected 168 rows, got {len(sh)}'
assert sh.groupby(['zone', 'kpi']).size().eq(8).all(), 'cell without 8 variables'

# ---- text diagnostics ----
print('=' * 76)
print('DIAGNOSTICS (for Section 3.4)')
print('=' * 76)
sobv = sob[~sob.input.str.startswith('GROUP')]
t1 = sh.sort_values('shap_rank').groupby(['zone', 'kpi']).first()
agree = 0
for (z, k), r in t1.iterrows():
    srow = sobv[(sobv.zone == z) & (sobv.kpi == k)].sort_values('ST').iloc[-1]
    if srow.input == r.input:
        agree += 1
    else:
        print(f'  rank-1 differs in {z} {k}: SHAP={r.input} vs Sobol argmax={srow.input}')
print(f'rank-1 agreement (SHAP argmax == Sobol ST argmax): {agree}/21')
print(f'top-1 SHAP share: mean={t1.shap_share.mean():.3f} '
      f'min={t1.shap_share.min():.3f} ({t1.shap_share.idxmin()}) '
      f'max={t1.shap_share.max():.3f} ({t1.shap_share.idxmax()})')
print('top-1 identity by KPI:')
for k in ['EUI', 'IDD', 'TL']:
    sub = t1.xs(k, level='kpi')
    vc = sub.input.value_counts()
    print(f'  {k}: ' + ', '.join(f'{NAME[v]} in {n} of {len(sub)}' for v, n in vc.items()))
n_melt = int(sh[sh.input.str.contains('melt')].is_top3.sum())
n_rtype = int(sh[sh.input == 'pcm_type_roof'].is_top3.sum())
print(f'top-3 membership: melt-point vars {n_melt}/336 possible, roof PCM type {n_rtype}/42')
print('perturbation schemes used:')
print(sh.groupby(['adopted', 'perturbation']).size().to_string())

# ---- Table A17 ----
sh2 = sh.copy()
sh2['zo'] = sh2.zone.map({z: i for i, z in enumerate(ZONES)})
sh2['ko'] = sh2.kpi.map({'EUI': 0, 'IDD': 1, 'TL': 2})
sh2['vo'] = sh2.input.map({v: i for i, v in enumerate(VARS)})
sh2 = sh2.sort_values(['zo', 'ko', 'vo']).drop(columns=['zo', 'ko', 'vo'])
out = pd.DataFrame({
    'Zone': sh2.zone, 'City': sh2.zone.map(CITY), 'KPI': sh2.kpi,
    'Adopted model': sh2.adopted, 'Variable': sh2.input.map(NAME),
    'Mean |SHAP|': sh2.mean_abs_shap.round(4),
    'Share': sh2.shap_share.round(3),
    'Rank': sh2.shap_rank.astype(int),
    'Top-3': np.where(sh2.is_top3 == True, 'yes', ''),
    'Perturbation': sh2.get('perturbation', pd.Series('interventional', index=sh2.index)),
})

# ============================ SAVE & DOWNLOAD FIX ============================
tables_dir = BASE + '/tables'
os.makedirs(tables_dir, exist_ok=True)
csv_path = os.path.join(tables_dir, 'table_A17_shap_attribution.csv')

out.to_csv(csv_path, index=False)

print('\n' + '=' * 76)
print(f'Table A17 - SHAP attribution per variable (168 rows) '
      f'-> {csv_path}')
print('=' * 76)
print(out.head(16).to_string(index=False))
print('... (16 of 168 rows shown)')

# دانلود مستقیم و بدون خطا از مسیر گوگل درایو
files.download(csv_path)

print('\nCaption A17:')
print('Table A17. SHAP attribution on the adopted surrogate per zone-KPI cell: '
      'mean absolute SHAP value, its normalized share of the cell total, the '
      'within-cell rank, and membership of the top-3 SHAP lens (the shap_top3 '
      'set used in the cross-method convergence analysis). Values are exact '
      'TreeSHAP attributions on 2048 level-coded evaluation designs; the '
      'perturbation scheme (interventional with a 100-design background, or '
      'tree_path_dependent where the adopted XGBoost models require it) is '
      'recorded per row. The three combinations excluded in Section 3.1 are '
      'absent.')
print('\nDONE 11c - paste this output back.')
