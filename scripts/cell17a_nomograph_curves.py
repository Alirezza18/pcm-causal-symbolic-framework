# ============================================================================
# scripts/cell17a_nomograph_curves.py
# Cell 17a (v2) — nomograph marginal curves + Table 6 producer
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 33
# (162 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 17a (v2) - PHASE 10: NOMOGRAPH CURVES + TABLE 6 (producer, Fig 14+T6)
# For every zone and modeled KPI: marginal effect of each design variable -
# sweep the variable over its observed integer levels, hold the other seven
# at the zone's median level, predict with the adopted surrogate, and express
# the curve as % change from the all-median design prediction.
#
# Table 6 (v2) states each knee TWICE:
#   code      - level codes, order type-thickness-melt point-position (the
#               currency of Table 4 / A20 / Section 3.6)
#   physical  - decoded with the SINGLE SOURCE OF TRUTH below (the same maps
#               the Grasshopper/EnergyPlus model uses), so the rule is
#               readable by a practitioner and consumable by Cell 19a:
#               tables/level_decode_map.csv is written HERE and READ by the
#               validation export - one decode, used everywhere.
#   type 0/1 = BioPCM/InfiniteR | thickness 0-3 = 11.2/20.8/37.1/74.2 mm
#   melt 0-4 = 18/21/23/25/29 C (BioPCM), 21/23/25/27/29 C (InfiniteR)
#   position 0/1/2 = exterior/middle/interior
# Also: Dominant = mode of the SHAP rank-1 variable family across the zone's
#   cells (A17; ties broken by total SHAP share); Risk* = 90% conformal width
#   as % of the zone's observed KPI range (uq_conformal.width_norm).
# Saves: tables/nomograph_curves.csv + tables/level_decode_map.csv
#        + tables/table6_design_rules.csv
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import time, datetime, joblib
import numpy as np
import pandas as pd

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
sh = pd.read_csv(BASE + '/tables/shap_attribution.csv')
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')
cf = pd.read_csv(BASE + '/tables/uq_conformal.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
FAMILY = {'pcm_type_wall': 'wall PCM type', 'thickness_wall_level': 'wall thickness',
          'melting_point_wall_level': 'wall melt point', 'position_wall': 'wall position',
          'pcm_type_roof': 'roof PCM type', 'thickness_roof_level': 'roof thickness',
          'melting_point_roof_level': 'roof melt point', 'position_roof': 'roof position'}

# ---- PHYSICAL DECODE MAPS: single source of truth for Sections 3.8 -> 3.10 ----
TYPENAME = {0: 'BioPCM', 1: 'InfiniteR'}
THK      = {0: 11.2, 1: 20.8, 2: 37.1, 3: 74.2}
MELT_BP  = {0: 18, 1: 21, 2: 23, 3: 25, 4: 29}
MELT_IR  = {0: 21, 1: 23, 2: 25, 3: 27, 4: 29}
POS      = {0: 'exterior', 1: 'middle', 2: 'interior'}

def decode_side(t, thk, melt, pos):
    """level codes (t, thk, melt, pos) -> 'Product-XX.Xmm-YYC-position'."""
    m = MELT_BP if int(t) == 0 else MELT_IR
    return (f'{TYPENAME[int(t)]}-{THK[int(thk)]}mm-'
            f'{m[int(melt)]}C-{POS[int(pos)]}')

KNEE = kr[kr.type == 'knee'].set_index('zone')

def to_levels(X):
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

t0 = time.time()

# ---------------- nomograph curves ----------------
rows = []
for z in ZONES:
    g = df[df.climate_zone == z]
    Xl = to_levels(g[INPUTS]).reset_index(drop=True)
    med = {c: int(np.median(Xl[c])) for c in INPUTS}
    for t in TARGETS:
        k = SHORT[t]
        path = f'{BASE}/models/cat_{z}_{k}.joblib'
        if not os.path.exists(path):
            continue                                   # 3.1 exclusion
        m = joblib.load(path)
        base_X = pd.DataFrame([med], columns=INPUTS)
        y_base = float(m.predict(base_X)[0])
        for c in INPUTS:
            levs = sorted(Xl[c].unique())
            Xs = pd.DataFrame([med] * len(levs), columns=INPUTS)
            Xs[c] = levs
            yp = np.asarray(m.predict(Xs), float)
            for lv, y in zip(levs, yp):
                rows.append({'zone': z, 'kpi': k, 'variable': c, 'level': int(lv),
                             'pct_change': 100 * (y - y_base) / abs(y_base)
                             if y_base != 0 else np.nan})
    print(f'{z} curves done ({time.time()-t0:.0f}s)', flush=True)
cur = pd.DataFrame(rows)
cur.to_csv(BASE + '/tables/nomograph_curves.csv', index=False)
print(f'nomograph_curves.csv: {len(cur)} rows '
      f'({cur.zone.nunique()} zones x {cur.variable.nunique()} vars)')

# ---------------- decode map: written here, read by Cell 19a ----------------
dec = ([{'variable': 'pcm_type', 'level': k, 'value': v} for k, v in TYPENAME.items()]
       + [{'variable': 'thickness_mm', 'level': k, 'value': v} for k, v in THK.items()]
       + [{'variable': 'melt_bp_c', 'level': k, 'value': v} for k, v in MELT_BP.items()]
       + [{'variable': 'melt_ir_c', 'level': k, 'value': v} for k, v in MELT_IR.items()]
       + [{'variable': 'position', 'level': k, 'value': v} for k, v in POS.items()])
pd.DataFrame(dec).to_csv(BASE + '/tables/level_decode_map.csv', index=False)
print(f'level_decode_map.csv: {len(dec)} rows (5 variable groups)')

# ---------------- Table 6 frame ----------------
r1 = sh[sh.shap_rank == 1]
dom = {}
for z in ZONES:
    s = r1[r1.zone == z]
    if len(s) == 0:
        dom[z] = '--'; continue
    vc = s.input.value_counts()
    top = vc[vc == vc.max()].index
    if len(top) > 1:                       # tie -> largest total SHAP share
        tot = s[s.input.isin(top)].groupby('input').shap_share.sum()
        dom[z] = FAMILY[tot.idxmax()]
    else:
        dom[z] = FAMILY[top[0]]

cfc = cf.set_index(['zone', 'kpi'])
t6rows = []
for z in ZONES:
    kne = KNEE.loc[z]
    wall_c = '-'.join(str(int(kne[v])) for v in INPUTS[:4])
    roof_c = '-'.join(str(int(kne[v])) for v in INPUTS[4:])
    wall_p = decode_side(kne.pcm_type_wall, kne.thickness_wall_level,
                         kne.melting_point_wall_level, kne.position_wall)
    roof_p = decode_side(kne.pcm_type_roof, kne.thickness_roof_level,
                         kne.melting_point_roof_level, kne.position_roof)
    risk = {}
    for k in ['EUI', 'IDD', 'TL']:
        risk[k] = (f'{100 * cfc.loc[(z, k), "width_norm"]:.0f}%'
                   if (z, k) in cfc.index else '--')
    t6rows.append({'Zone': z, 'City': ZCITY[z], 'Dominant': dom[z],
                   'Knee wall code': wall_c, 'Knee wall': wall_p,
                   'Knee roof code': roof_c, 'Knee roof': roof_p,
                   'Risk EUI': risk['EUI'], 'Risk IDD': risk['IDD'],
                   'Risk TL': risk['TL']})
t6 = pd.DataFrame(t6rows)
t6.to_csv(BASE + '/tables/table6_design_rules.csv', index=False)
print('\nTable 6 frame (codes + physical):')
print(t6.to_string(index=False))

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()} | CELL 17a v2 | nomograph '
            f'curves ({len(cur)} rows) + decode map + table6 (decoded) saved\n')
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/nomograph_curves.csv '
      f'+ level_decode_map.csv + table6_design_rules.csv')
print('DONE 17a - paste this output back.')
