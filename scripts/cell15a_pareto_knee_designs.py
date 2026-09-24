# ============================================================================
# scripts/cell15a_pareto_knee_designs.py
# Cell 15a — Pareto fronts + knee designs (producer)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 25
# (185 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 15a - PHASE 9: PARETO FRONTS + KNEE DESIGNS (producer for Fig 12 + A20)
# (run AFTER Cells 7-11; needs models + pcm_dataset_final.csv only)
# For every zone: exhaustive evaluation of the adopted surrogates over the
# zone's own level grid (the same design space the surrogate was trained on)
# -> KPI vector (EUI, IDD, TL) per design -> non-dominated (Pareto) front per
# zone, computed on min-max normalized KPIs (each KPI equally weighted in the
# dominance relation; raw values reported alongside).
#   * knee design  = front point minimizing Euclidean distance to the ideal
#     point (0,0,0) in normalized space (standard knee definition)
#   * per-KPI min designs = the single-objective optima, for reference
#   * front sizes / knee / per-KPI optima are diagnostics-printed and saved
#   * every design keeps its raw (EUI, IDD, TL) so Table A20 reports
#     engineering units, not normalized ones
# Saves: tables/pareto_front.csv      (one row per front point, all zones)
#        tables/pareto_knee.csv       (knee + 3 single-objective optima/zone)
#        tables/pareto_gridsummary.csv(n_grid, n_front, hypervolume ref)
# Runtime: a few minutes on Colab CPU.
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import time, datetime, itertools, joblib
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

def to_levels(X):
    """Level coding - byte-identical logic to Cell 9."""
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

from bisect import bisect_right

def pareto_front(Y):
    """Boolean mask of non-dominated points. Y: (n,m) - LOWER is better in
    every column (all KPIs are cost-like). Exact fast paths:
      m=2: lexsort by (o1,o2) + running minimum of o2 (ties on (o1,o2)
           impossible after dedup).
      m=3: lexsort by (o1,o2,o3) + staircase skyline over (o2,o3). The sort
           guarantees any dominator of a point precedes it; for m=3 the
           (o2,o3) test is then exact dominance (equal-o1 predecessors
           dominate iff they are <= in both remaining objectives).
    m>3 raises - not needed for this study (2 or 3 modeled KPIs per zone)."""
    m = Y.shape[1]
    U, inv = np.unique(Y, axis=0, return_inverse=True)   # dedup: identical
    order = np.lexsort(tuple(U[:, j] for j in range(m - 1, -1, -1)))
    nd_u = np.zeros(len(U), dtype=bool)
    if m == 2:
        best = np.inf
        for idx in order:
            if U[idx, 1] < best:
                nd_u[idx] = True
                best = U[idx, 1]
        return nd_u[inv]
    if m != 3:
        raise ValueError(f'pareto_front: exact paths implemented for m=2,3 '
                         f'(got m={m})')
    sk_o2, sk_o3 = [], []          # staircase over (o2,o3): o2 strictly
    for idx in order:              # increasing, o3 strictly decreasing
        o2, o3 = U[idx, 1], U[idx, 2]
        p = bisect_right(sk_o2, o2)
        if p > 0 and sk_o3[p - 1] <= o3:
            continue               # dominated by a processed point
        nd_u[idx] = True
        q = p
        while q < len(sk_o3) and sk_o3[q] >= o3:
            q += 1                 # these staircase points just got dominated
        del sk_o2[p:q], sk_o3[p:q]
        sk_o2.insert(p, float(o2)), sk_o3.insert(p, float(o3))
    return nd_u[inv]

t0 = time.time()
front_rows, knee_rows, sum_rows = [], [], []
for z in ZONES:
    g = df[df.climate_zone == z]
    # ---- zone's own level grid: every cell of the design space it trained on
    # (per-variable min..max range - the level-coded design space is dense by
    # construction, so the range IS the training support)
    grids = [range(int(g[c].min()), int(g[c].max()) + 1) for c in INPUTS]
    G = pd.DataFrame(list(itertools.product(*grids)), columns=INPUTS)
    n_grid = len(G)

    # ---- surrogate predictions of the three KPIs on the full grid ----
    EXCL = {('BandarAbbas', 'IDD'), ('Kashan', 'TL'), ('Tabriz', 'TL')}
    Y = {}
    for t in TARGETS:
        k = SHORT[t]
        if (ZCITY[z], k) in EXCL:
            Y[k] = None            # excluded cell (3.1)
            continue
        m = joblib.load(f'{BASE}/models/cat_{z}_{k}.joblib')
        Y[k] = np.asarray(m.predict(G), float)

    ok = [k for k in ('EUI', 'IDD', 'TL') if Y[k] is not None]
    if len(ok) < 2:
        print(f'{z}: fewer than 2 KPIs modeled - skipped')
        continue

    # ---- normalize each available KPI to [0,1] over the grid ----
    Ym = np.column_stack([Y[k] for k in ok])
    lo, hi = Ym.min(axis=0), Ym.max(axis=0)
    rng_ = np.where(hi > lo, hi - lo, 1.0)
    Yn = (Ym - lo) / rng_

    nd = pareto_front(Yn)
    F = Yn[nd]
    d_all = np.sqrt((F ** 2).sum(axis=1))      # dist to ideal (0,..,0)
    print(f'\n--- {z} {ZCITY[z]} | grid {n_grid} | front {int(nd.sum())} '
          f'({100*nd.mean():.1f}%) | knee dist {d_all.min():.3f} ---', flush=True)

    # ---- record the front in raw units ----
    idx_front = np.where(nd)[0]
    for i in idx_front:
        front_rows.append({
            'zone': z, 'city': ZCITY[z],
            **{c: int(G.loc[i, c]) for c in INPUTS},
            **{k: float(Y[k][i]) for k in ok},
            'n_kpi': len(ok)})
    sum_rows.append({'zone': z, 'city': ZCITY[z], 'n_grid': n_grid,
                     'n_front': int(nd.sum()), 'front_pct': float(nd.mean()),
                     'knee_dist': float(d_all.min()), 'n_kpi': len(ok),
                     'kpis': '|'.join(ok)})

    # ---- knee + per-KPI optima (raw units) ----
    ik = int(np.argmin(d_all))
    knee = idx_front[ik]
    knee_rows.append({'zone': z, 'city': ZCITY[z], 'type': 'knee',
                      **{c: int(G.loc[knee, c]) for c in INPUTS},
                      **{k: float(Y[k][knee]) for k in ok}})
    print(f'  knee: ' + ', '.join(f'{c}={int(G.loc[knee, c])}' for c in INPUTS
                                  if int(G.loc[knee, c]) != 0))
    for k in ok:
        j_opt = int(np.argmin(Y[k]))
        knee_rows.append({'zone': z, 'city': ZCITY[z], 'type': f'min_{k}',
                          **{c: int(G.loc[j_opt, c]) for c in INPUTS},
                          **{kk: (float(Y[kk][j_opt]) if Y[kk] is not None else np.nan)
                             for kk in ok}})
    m_eui = float(np.min(Y['EUI']))
    m_idd = f'{float(np.min(Y["IDD"])):.1f}' if Y['IDD'] is not None else '--'
    m_tl = f'{float(np.min(Y["TL"])):.1f}' if Y['TL'] is not None else '--'
    print(f'  min-EUI: {m_eui:.1f} | min-IDD: {m_idd} | min-TL: {m_tl}'
          f' ({time.time()-t0:.0f}s)', flush=True)

fr = pd.DataFrame(front_rows)
kr = pd.DataFrame(knee_rows)
sr = pd.DataFrame(sum_rows)
os.makedirs(BASE + '/tables', exist_ok=True)
fr.to_csv(BASE + '/tables/pareto_front.csv', index=False)
kr.to_csv(BASE + '/tables/pareto_knee.csv', index=False)
sr.to_csv(BASE + '/tables/pareto_gridsummary.csv', index=False)

print('\n' + '=' * 76)
print('DIAGNOSTICS (for Section 3.6)')
print('=' * 76)
print(sr.to_string(index=False))
print(f'\nfronts saved: {len(fr)} points across {len(sr)} zones')
print(f'knee + optima rows: {len(kr)}')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()} | CELL 15a | pareto '
            f'fronts saved for {len(sr)} zones\n')
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/pareto_front.csv '
      f'+ pareto_knee.csv + pareto_gridsummary.csv')
print('DONE 15a - paste this output back.')
