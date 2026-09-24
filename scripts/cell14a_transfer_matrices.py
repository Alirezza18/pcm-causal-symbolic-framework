# ============================================================================
# scripts/cell14a_transfer_matrices.py
# Cell 14a — cross-climate transfer matrices (producer)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 22
# (114 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 14a - PHASE 8: CROSS-CLIMATE TRANSFER MATRICES (producer for Fig 11+A19)
# For every KPI: the adopted surrogate of each DONOR zone is evaluated on the
# TARGET zone's level-coded designs -> shape-transfer R2 matrix (8x8).
#   * donor model  = models/cat_{zone}_{k}.joblib (same objects as Cells 7-11)
#   * target inputs = the TARGET zone's designs, level-coded with Cell 9's
#     to_levels (byte-identical logic - the representation all lenses share)
#   * shape recalibration = 2-parameter linear refit (offset+gain, least
#     squares on the target's true y) before R2 - removes the between-climate
#     response offset so R2 isolates the transferred SHAPE
#   * EXCLUDED cells (Section 3.1): 0B IDD, 2B TL, 4B TL - NaN as donor
#     (row) and as target (column)
# Saves: tables/transfer_shape_{EUI,IDD,TL}.csv (8x8, NaN = excluded)
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
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}

SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}
SKIP_CELLS = {('4B', 'TL'), ('0B', 'IDD'), ('2B', 'TL')}   # (zone, KPI)

def to_levels(X):
    """Level coding - byte-identical logic to Cell 9."""
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

def r2(y, yhat):
    y = np.asarray(y, float)
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - np.asarray(yhat, float)) ** 2).sum() / ss if ss > 0 else np.nan

t0 = time.time()
for t in TARGETS:
    k = SHORT[t]
    M = pd.DataFrame(np.nan, index=ZONES, columns=ZONES)

    # ---- designs + targets per zone (once per KPI) ----
    data = {}
    for z in ZONES:
        g = df[df.climate_zone == z]
        data[z] = {'X': to_levels(g[INPUTS]).reset_index(drop=True),
                   'y': g[t].reset_index(drop=True).astype(float)}

    # ---- donor surrogates: predict every zone's designs ----
    preds = {}
    for dz in ZONES:
        if (dz, k) in SKIP_CELLS:
            continue
        m = joblib.load(f'{BASE}/models/cat_{dz}_{k}.joblib')
        preds[dz] = {tz: np.asarray(m.predict(data[tz]['X']), float)
                     for tz in ZONES}
        ADOPTED = bm[(bm.zone == dz) & (bm.kpi == k)].iloc[0].best_model
        print(f'{k} donor {dz} loaded | adopted: {ADOPTED} '
              f'({time.time()-t0:.0f}s)', flush=True)

    # ---- fill matrix: shape R2 after 2-parameter recalibration ----
    for dz in ZONES:
        if (dz, k) in SKIP_CELLS:
            print(f'{dz} {k}: SKIPPED as donor - '
                  f'{SKIP.get((ZCITY[dz], k), "3.1 exclusion")}')
            continue
        for tz in ZONES:
            if (tz, k) in SKIP_CELLS:
                continue            # stays NaN in both directions
            y = data[tz]['y'].values
            ph = preds[dz][tz]
            # 2-parameter linear recalibration (offset + gain, least squares
            # on the target's true y) before R2
            Xc = np.column_stack([np.ones_like(ph), ph])
            coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
            rec = Xc @ coef
            M.loc[dz, tz] = round(r2(y, rec), 3)

    path = f'{BASE}/tables/transfer_shape_{k}.csv'
    M.to_csv(path)
    eye = np.eye(8, dtype=bool)
    n_all = int((~M.isna()).sum().sum())
    n_off = n_all - int(np.diag(M.notna().values).sum())
    print(f'\n=== {k} ===')
    print(M.round(3).to_string())
    print(f'filled {n_all}/64 cells ({n_off}/56 off-diagonal; excluded '
          f'rows+cols carry NaN), saved {path} ({time.time()-t0:.0f}s)')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()} | CELL 14a | transfer '
            f'matrices saved for EUI/IDD/TL\n')
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/transfer_shape_{{EUI,IDD,TL}}.csv')
print('DONE 14a - paste this output back.')
