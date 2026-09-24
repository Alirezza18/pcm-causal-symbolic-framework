# ============================================================================
# scripts/cell19a_validation_design_set.py
# Cell 19a (v3) — validation set: 8 knees + 8 held-out designs
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 41
# (147 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 19a (v3) - PHASE 12: VALIDATION DESIGN SET - 8 KNEES + 8 HELD-OUT
# (producer for Fig 16 + Table 8; Section 3.10)
# knee    - the eight recommended configurations (Table 6). The optimizer
#           chose them inside the 14,400-point grid, so no engine has ever
#           simulated them: out-of-sample in BOTH model and design point.
# heldout - eight PRE-REGISTERED configurations spanning the design space
#           (fixed in this code before any simulation; rationales stated) -
#           they test surrogate accuracy away from the optima.
# Physical decode is loaded from tables/level_decode_map.csv - the single
# source of truth WRITTEN BY CELL 17a (run 17a first). Same maps as the
# Grasshopper/EnergyPlus model:
#   type 0/1 = BioPCM/InfiniteR | thickness 0-3 = 11.2/20.8/37.1/74.2 mm
#   melt 0-4 = 18/21/23/25/29 C (BioPCM), 21/23/25/27/29 C (InfiniteR)
#   position 0/1/2 = exterior/middle/interior
# Exclusions (3.1) propagate: 0B IDD, 2B TL, 4B TL -> no row (6 of 48 cells).
# Saves: tables/validation_designs_RAW.csv (16 designs, decoded)
#        tables/validation_results_TEMPLATE.csv (42 rows, sim_value empty)
# Then: run E+ once per design (16 runs; each yields all 3 KPIs), fill
#       sim_value, save as tables/validation_results.csv, run Cell 19b.
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import time, joblib
import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
t0 = time.time()
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level',
          'position_wall', 'pcm_type_roof', 'thickness_roof_level',
          'melting_point_roof_level', 'position_roof']
KPIS = ['EUI', 'IDD', 'TL']
EXCLUDED = {('0B', 'IDD'), ('2B', 'TL'), ('4B', 'TL')}          # Section 3.1

# ---- decode maps: single source of truth written by Cell 17a ----
DMAP = BASE + '/tables/level_decode_map.csv'
assert os.path.exists(DMAP), 'level_decode_map.csv not found - run Cell 17a first'
dec = pd.read_csv(DMAP)
def _grp(name):
    g = dec[dec.variable == name]
    return {int(r.level): r.value for r in g.itertuples()}
TYPENAME = {k: str(v) for k, v in _grp('pcm_type').items()}
THK      = {k: float(v) for k, v in _grp('thickness_mm').items()}
MELT_BP  = {k: float(v) for k, v in _grp('melt_bp_c').items()}
MELT_IR  = {k: float(v) for k, v in _grp('melt_ir_c').items()}
POS      = {k: str(v) for k, v in _grp('position').items()}

def decode_side(t, thk, melt, pos):
    """level codes (t, thk, melt, pos) -> 'Product-XX.Xmm-YC-position'."""
    m = MELT_BP if int(t) == 0 else MELT_IR
    return (f'{TYPENAME[int(t)]}-{THK[int(thk)]:g}mm-'
            f'{m[int(melt)]:g}C-{POS[int(pos)]}')

KNEE = kr[kr.type == 'knee'].set_index('zone')

# ---- held-out designs: pre-registered, fixed before any simulation ----
HELDOUT = [
    ('0B', 'BandarAbbas', 1, 0, 4, 2,  1, 0, 0, 2, 'Thin IR wall + low-melt roof'),
    ('1B', 'Bushehr',     0, 3, 3, 0,  0, 3, 1, 0, 'Thick BP exterior + BP roof'),
    ('2B', 'Kashan',      1, 0, 0, 1,  0, 0, 3, 1, 'Thin IR middle + thin BP middle'),
    ('3A', 'Rasht',       0, 3, 4, 0,  1, 0, 2, 0, 'Thick BP ext wall + thin IR ext roof'),
    ('3B', 'Tehran',      1, 3, 4, 2,  0, 0, 0, 1, 'Thick IR interior + thin BP middle roof'),
    ('4A', 'Hamedan',     0, 0, 1, 1,  1, 3, 3, 2, 'Thin BP middle + thick IR interior'),
    ('4B', 'Tabriz',      1, 0, 0, 0,  1, 3, 0, 1, 'Thin IR exterior + thick IR middle roof'),
    ('5C', 'Ardebil',     0, 3, 0, 1,  0, 0, 3, 0, 'Thick BP middle wall + thin BP ext roof'),
]

# ---- assemble the 16 designs ----
recs = []
for z in ZONES:
    k = KNEE.loc[z]
    recs.append({'case_id': f'{z}-KNEE', 'zone': z, 'city': ZCITY[z],
                 'design_type': 'knee', 'rationale': 'recommended knee (Table 6)',
                 'PCM_type_Wall': int(k.pcm_type_wall),
                 'Thickness_Wall': int(k.thickness_wall_level),
                 'Melting_Point_Wall': int(k.melting_point_wall_level),
                 'wall_PCM_Position': int(k.position_wall),
                 'PCM_type_Roof': int(k.pcm_type_roof),
                 'Thickness_Roof': int(k.thickness_roof_level),
                 'Melting_Point_Roof': int(k.melting_point_roof_level),
                 'Roof_PCM_Position': int(k.position_roof)})
for i, (z, c, wt, wthk, wm, wp, rt, rthk, rm, rp, rat) in enumerate(HELDOUT, 1):
    recs.append({'case_id': f'{z}-HO{i}', 'zone': z, 'city': c,
                 'design_type': 'heldout', 'rationale': rat,
                 'PCM_type_Wall': wt, 'Thickness_Wall': wthk,
                 'Melting_Point_Wall': wm, 'wall_PCM_Position': wp,
                 'PCM_type_Roof': rt, 'Thickness_Roof': rthk,
                 'Melting_Point_Roof': rm, 'Roof_PCM_Position': rp})
D = pd.DataFrame(recs)
D['wall_decoded'] = [decode_side(r.PCM_type_Wall, r.Thickness_Wall,
                                 r.Melting_Point_Wall, r.wall_PCM_Position)
                     for r in D.itertuples()]
D['roof_decoded'] = [decode_side(r.PCM_type_Roof, r.Thickness_Roof,
                                 r.Melting_Point_Roof, r.Roof_PCM_Position)
                     for r in D.itertuples()]

# ---- surrogate predictions for every modeled cell (42 rows) ----
models, rows, checks = {}, [], []
for r in D.itertuples():
    X = pd.DataFrame([[r.PCM_type_Wall, r.Thickness_Wall, r.Melting_Point_Wall,
                       r.wall_PCM_Position, r.PCM_type_Roof, r.Thickness_Roof,
                       r.Melting_Point_Roof, r.Roof_PCM_Position]], columns=INPUTS)
    for k in KPIS:
        if (r.zone, k) in EXCLUDED:
            continue
        key = (r.zone, k)
        if key not in models:
            models[key] = joblib.load(f'{BASE}/models/cat_{r.zone}_{k}.joblib')
        p = float(models[key].predict(X)[0])
        if r.design_type == 'knee':                 # contract check vs pareto_knee
            ref = float(KNEE.loc[r.zone, k])
            checks.append(abs(p - ref) / max(abs(ref), 1e-9))
        rows.append({'case_id': r.case_id, 'zone': r.zone, 'city': r.city,
                     'design_type': r.design_type, 'rationale': r.rationale,
                     'KPI': k, 'surrogate_pred': round(p, 3), 'sim_value': np.nan})
TPL = pd.DataFrame(rows)

print(f'designs: {len(D)} (8 knee + {len(D) - 8} held-out) | predicted cells: '
      f'{len(TPL)} of 48 (6 excluded by 3.1)')
print(f'knee re-prediction vs pareto_knee.csv: max rel. dev {max(checks):.2e}')
print('\nE+ INPUT SHEET (one simulation per row; each yields all 3 KPIs):')
for r in D.itertuples():
    print(f'  {r.case_id:>9} {r.city:>12}  wall: {r.wall_decoded:<26} '
          f'roof: {r.roof_decoded:<26}  [{r.rationale}]')

D.to_csv(BASE + '/tables/validation_designs_RAW.csv', index=False)
TPL.to_csv(BASE + '/tables/validation_results_TEMPLATE.csv', index=False)
print(f'\nsaved tables/validation_designs_RAW.csv ({len(D)} designs)')
print(f'saved tables/validation_results_TEMPLATE.csv ({len(TPL)} rows - fill '
      f'sim_value after each E+ run, save as tables/validation_results.csv)')
files.download(BASE + '/tables/validation_designs_RAW.csv')
files.download(BASE + '/tables/validation_results_TEMPLATE.csv')
print(f'\nTotal: {time.time() - t0:.0f}s | DONE 19a - paste this output back.')
