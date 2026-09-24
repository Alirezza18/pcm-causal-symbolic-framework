# ============================================================================
# scripts/cell11_shap_attribution.py
# Cell 11 — SHAP attribution on adopted surrogates, resume-safe
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 16
# (172 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 11 - PHASE 6: SHAP ATTRIBUTION on the adopted surrogates (resume-safe)
# Exact TreeSHAP values (CatBoost/XGBoost) per zone-KPI cell on the
# level-coded design matrix - the SAME input representation Sobol/Morris used,
# so SHAP shares and Sobol ST are directly comparable.
# Lens for Fig 10: shap_top3 = the 3 variables with largest mean|SHAP| share.
# Saves: tables/shap_attribution.csv (per variable: mean|SHAP|, share, rank,
#        is_top3, perturbation mode; background/eval counts) - incremental.
# NOTE:  runs on the ADOPTED model per cell (CatBoost everywhere except
#        Tabriz IDD and Ardebil TL = XGBoost). Skips the 3.1 exclusions:
#        Tabriz TL (no surrogate), BA IDD + Kashan TL (degenerate) -> 21 cells.
#   [6] FIX (crash at Tabriz IDD): recent shap raises the categorical-split
#        check for XGBoost models in BOTH perturbation modes -> layered
#        fallback ending in XGBoost-native pred_contribs TreeSHAP (exact,
#        tree_path_dependent algorithm), recorded in 'perturbation'.
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

try:
    import shap
except ImportError:
    # [Colab shell] pip install shap -q
    import shap

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}

SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}

BG_N, EV_N, RNG = 100, 2048, 42

def to_levels(X):
    """Level coding - byte-identical logic to Cell 9 so the lenses share units."""
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

def shap_values(m, Xbg, Xev):
    """[6] Exact SHAP with layered fallback:
       1) interventional TreeExplainer (CatBoost cells);
       2) tree_path_dependent TreeExplainer (standard shap advice);
       3) XGBoost-native pred_contribs TreeSHAP - recent shap versions raise
          the categorical-split check in BOTH modes for XGB models carrying
          categorical metadata, so we bypass shap's model parser entirely and
          take the exact TreeSHAP values from xgboost itself.
    Returns (values, mode)."""
    try:
        ex = shap.TreeExplainer(m, data=Xbg, feature_perturbation='interventional',
                                model_output='raw')
        return ex.shap_values(Xev, check_additivity=False), 'interventional'
    except NotImplementedError:
        pass
    try:
        ex = shap.TreeExplainer(m, feature_perturbation='tree_path_dependent',
                                model_output='raw')
        return ex.shap_values(Xev, check_additivity=False), 'tree_path_dependent'
    except NotImplementedError:
        pass
    import xgboost
    contribs = m.get_booster().predict(xgboost.DMatrix(Xev), pred_contribs=True)
    sv = np.asarray(contribs, dtype=float)[:, :-1]      # drop the bias column
    return sv, 'xgb_pred_contribs'

# ---- resume support: keep previously saved rows ----
res_path = BASE + '/tables/shap_attribution.csv'
rows = []
done = set()
if os.path.exists(res_path):
    prev = pd.read_csv(res_path)
    if len(prev):
        if 'perturbation' not in prev.columns:      # rows from pre-fix run
            prev['perturbation'] = 'interventional' # (CatBoost cells - correct)
        prev['n_background'] = 100                  # effective masker background
        done = set(zip(prev['zone'], prev['kpi']))
        rows = prev.to_dict('records')
        print(f'resuming - {len(done)} zone-KPI already computed, keeping them in file')
    else:
        print('existing shap_attribution.csv is empty - starting fresh')

t0 = time.time()
n_cells = 0
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    for t in TARGETS:
        k = SHORT[t]
        if (city, k) in SKIP:
            print(f'{zone} {city:11s} {k}: SKIPPED - {SKIP[(city, k)]}')
            continue
        if (zone, k) in done:
            print(f'{zone} {k} already done - skip')
            continue
        n_cells += 1
        X = g[INPUTS].reset_index(drop=True)
        y = g[t].reset_index(drop=True)
        ADOPTED = bm[(bm.zone == zone) & (bm.kpi == k)].iloc[0].best_model
        m = joblib.load(f'{BASE}/models/cat_{zone}_{k}.joblib')
        print(f'--- {zone} {city} {k} | adopted: {ADOPTED} ---', flush=True)

        # ---- background / evaluation designs (same distribution as Cell 9) ----
        rng = np.random.default_rng(RNG)
        idx = rng.choice(len(X), size=min(BG_N, len(X)), replace=False)
        Xbg = to_levels(X.iloc[idx])
        ev_idx = np.arange(EV_N) % len(X)          # deterministic eval design
        Xev = to_levels(X.iloc[ev_idx])

        # ---- exact SHAP values on the adopted tree model ----
        sv, mode = shap_values(m, Xbg, Xev)
        mean_abs = np.abs(sv).mean(axis=0)
        share = mean_abs / mean_abs.sum()
        order = np.argsort(-mean_abs)
        rank = {j: r + 1 for r, j in enumerate(order)}
        top3 = set(order[:3])

        for j, c in enumerate(INPUTS):
            rows.append({'zone': zone, 'city': city, 'kpi': k, 'input': c,
                         'adopted': ADOPTED,
                         'mean_abs_shap': float(mean_abs[j]),
                         'shap_share': float(share[j]),
                         'shap_rank': rank[j],
                         'is_top3': bool(j in top3),
                         'perturbation': mode,
                         'n_background': len(Xbg), 'n_eval': len(Xev)})
        pd.DataFrame(rows).to_csv(res_path, index=False)   # incremental save
        top3_names = [INPUTS[j] for j in order[:3]]
        print(f'{zone} {city:11s} {k}: top3={top3_names} '
              f'| share={share[order[0]]:.3f}/{share[order[1]]:.3f}/{share[order[2]]:.3f} '
              f'| shap={mode} ({time.time()-t0:.0f}s)', flush=True)

res = pd.DataFrame(rows)
res.to_csv(res_path, index=False)
print('\n' + '=' * 70)
print(f'PHASE 6 DONE - {n_cells} cells computed this run, {len(res)} rows saved')
print('=' * 70)
chk = res[res.input != 'GROUP_wall']
assert len(chk.groupby(['zone', 'kpi'])) == 21, 'expected 21 cells'
assert chk.groupby(['zone', 'kpi']).size().eq(8).all(), 'cell without 8 variables'
os.makedirs(BASE + '/logs', exist_ok=True)
print('perturbation modes used per adopted family:')
print(chk.groupby(['adopted', 'perturbation']).size().to_string())
w = chk.pivot_table(index='zone', columns='kpi', values='shap_share', aggfunc=lambda s: s.iloc[0])
print('SHAP top-variable share per zone (sanity glance):')
print(w.round(3).to_string())
print('\nshap_top3 lens per cell (for lens_convergence.csv):')
for (zone, k), grp in chk.groupby(['zone', 'kpi']):
    top = grp.sort_values('shap_rank').input.head(3).tolist()
    print(f'  {zone} {k:3s}: {"|".join(top)}')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 11 | shap done, {len(res)} rows saved\n")
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/shap_attribution.csv')
print('DONE - paste this output back.')
