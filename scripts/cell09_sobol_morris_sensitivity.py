# ============================================================================
# scripts/cell09_sobol_morris_sensitivity.py
# Cell 9 — global sensitivity: Morris -> Sobol -> D(Yk)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 8
# (205 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 9 — PHASE 4: GLOBAL SENSITIVITY (Morris -> Sobol -> D(Yk))
# FIXED - runs on the ADOPTED surrogate per (zone, KPI) from Cell 7;
# skips the degenerate cells excluded in 3.1. Expected: 21 cells.
# Saves: tables/sobol_indices.csv (incremental, resume-safe)
# Fixes: [1] *_level column names  [2] adopted model loaded per cell
#        [3] degenerate skips      [4] row-bootstrap (not seed-only)
#        [5] no model writing here
# Runtime ~1.5-3 h on Colab CPU. Resume-safe: disconnect costs nothing.
# ============================================================
import os, time, datetime, joblib
import numpy as np
import pandas as pd

try:
    import xgboost, lightgbm, catboost
except ImportError:
    # [Colab shell] pip install xgboost lightgbm catboost -q
    import xgboost, lightgbm, catboost
try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
except ImportError:
    # [Colab shell] pip install SALib -q
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

from joblib import Parallel, delayed

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}

LEVELS = {'pcm_type_wall': [0, 1], 'thickness_wall_level': [0, 1, 2, 3],
          'melting_point_wall_level': [0, 1, 2, 3, 4], 'position_wall': [0, 1, 2],
          'pcm_type_roof': [0, 1], 'thickness_roof_level': [0, 1, 2, 3],
          'melting_point_roof_level': [0, 1, 2, 3, 4], 'position_roof': [0, 1, 2]}

GROUPS = {'wall': ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall'],
          'roof': ['pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']}

SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}

N_SAMPLE = 2048          # Saltelli base sample (point estimate)
N_BOOT = 500             # row-bootstrap resamples (methodology number)
N_CHUNK = 50             # progress/save chunk for bootstrap
N_MORRIS_TRAJ = 50       # Morris trajectories
RNG = 42

problem = {'num_vars': len(INPUTS), 'names': INPUTS,
           'bounds': [[min(LEVELS[c]), max(LEVELS[c])] for c in INPUTS]}

def to_levels(X):
    Xl = np.round(X).astype(int)
    for j, c in enumerate(INPUTS):
        lv = LEVELS[c]
        Xl[:, j] = np.clip(Xl[:, j], lv[0], lv[-1])
    return pd.DataFrame(Xl, columns=INPUTS)

BUILDERS = {
    'xgb_default':      lambda: xgboost.XGBRegressor(random_state=RNG, n_jobs=-1),
    'xgb_regularized':  lambda: xgboost.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7,
        colsample_bytree=0.7, min_child_weight=5, reg_lambda=5.0, reg_alpha=1.0,
        gamma=0.5, random_state=RNG, n_jobs=-1),
    'lgbm_default':     lambda: lightgbm.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1),
    'catboost_default': lambda: catboost.CatBoostRegressor(iterations=500,
                                                           random_seed=RNG, verbose=False),
    'mlp':              lambda: None,   # MLP cells are skipped below; never reaches bootstrap
}

def boot_sti(b, Xv, yv, Xl_eval):
    """ROW-bootstrap: refit adopted family on resampled designs, recompute
    ST on the SAME fixed Saltelli design. Fixed model seed - variability
    comes from the data resample, not the seed."""
    rng_b = np.random.default_rng(RNG + b)          # reproducible resample
    take = rng_b.choice(len(Xv), size=len(Xv), replace=True)
    m = BUILDERS[ADOPTED]()
    m.fit(Xv.iloc[take], yv.iloc[take])
    return sobol.analyze(problem, m.predict(Xl_eval),
                         calc_second_order=True)['ST']

# ---- resume support: keep previously saved rows ----
res_path = BASE + '/tables/sobol_indices.csv'
rows = []
done = set()
if os.path.exists(res_path):
    prev = pd.read_csv(res_path)
    prev = prev[~prev.apply(lambda r: (r['city'], r['kpi']) in SKIP, axis=1)]  # drop stale skips
    done = set(zip(prev['zone'], prev['kpi']))
    rows = prev.to_dict('records')
    print(f'resuming - {len(done)} zone-KPI already computed, keeping them in file')

t0 = time.time()
n_cells = 0
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    for t in TARGETS:
        if (zone, SHORT[t]) in done:
            print(f'{zone} {SHORT[t]} already done - skip'); continue
        if (city, SHORT[t]) in SKIP:
            print(f'{zone} {city:11s} {SHORT[t]}: SKIPPED - {SKIP[(city, SHORT[t])]}')
            continue
        n_cells += 1
        X = g[INPUTS].reset_index(drop=True)
        y = g[t].reset_index(drop=True)

        # ---- 0. adopted surrogate: load the model Cell 7 saved ----
        ADOPTED = bm[(bm.zone == zone) & (bm.kpi == SHORT[t])].iloc[0].best_model
        m = joblib.load(f'{BASE}/models/cat_{zone}_{SHORT[t]}.joblib')
        print(f'--- {zone} {city} {SHORT[t]} | adopted: {ADOPTED} ---', flush=True)

        # ---- 1. Morris screening ----
        Xm = morris_sample.sample(problem, N_MORRIS_TRAJ, num_levels=4)
        Ym = m.predict(to_levels(Xm))
        Mm = morris_analyze.analyze(problem, Xm, Ym)
        mu_star = Mm['mu_star']

        # ---- 2. Sobol point estimate with CIs + second order ----
        Xs = sobol_sample.sample(problem, N_SAMPLE, calc_second_order=True)
        Xl = to_levels(Xs)
        Y = m.predict(Xl)
        Si = sobol.analyze(problem, Y, calc_second_order=True, num_resamples=1000)
        si1, sti, s2 = Si['S1'], Si['ST'], Si['S2']
        sti_conf = Si['ST_conf']
        print(f'    Sobol done - sum(ST)={sti.sum():.2f} (expect ~0.8-1.1 after level rounding)', flush=True)

        # ---- 3. grouped Sobol (wall vs roof) - Jansen estimator ----
        Nn = N_SAMPLE
        YA = Y[:Nn]
        A3 = Xl.values[:Nn].copy()
        B3 = Xl.values[Nn:2*Nn].copy()
        VarY = np.var(YA)
        def _group_ST(cols):
            ABg = A3.copy()
            ABg[:, cols] = B3[:, cols]
            YABg = m.predict(pd.DataFrame(ABg, columns=INPUTS))
            return float(np.sum((YA - YABg)**2) / (2 * Nn * VarY))
        wall_cols = [INPUTS.index(c) for c in GROUPS['wall']]
        roof_cols = [INPUTS.index(c) for c in GROUPS['roof']]
        stg = [_group_ST(wall_cols), _group_ST(roof_cols)]

        # ---- 4. row-bootstrap 500, chunked with progress ----
        Xl_eval = Xl
        sti_boot = np.empty((N_BOOT, len(INPUTS)))
        for start in range(0, N_BOOT, N_CHUNK):
            end = min(start + N_CHUNK, N_BOOT)
            chunk = np.array(Parallel(n_jobs=2)(
                delayed(boot_sti)(b, X, y, Xl_eval)
                for b in range(start, end)))
            sti_boot[start:end] = chunk
            print(f'      {zone} {SHORT[t]} bootstrap {end}/{N_BOOT} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        retain_frac = (sti_boot >= 0.05).mean(axis=0)
        drivers = [c for c, rf in zip(INPUTS, retain_frac) if rf >= 0.90]

        for j, c in enumerate(INPUTS):
            rows.append({'zone': zone, 'city': city, 'kpi': SHORT[t], 'input': c,
                         'adopted': ADOPTED,
                         'morris_mu_star': mu_star[j],
                         'S1': si1[j], 'ST': sti[j], 'ST_conf': sti_conf[j],
                         'S2_diag': s2[j, j],
                         'ST_boot_mean': sti_boot[:, j].mean(),
                         'retain_frac': retain_frac[j], 'in_driver_set': c in drivers})
        rows.append({'zone': zone, 'city': city, 'kpi': SHORT[t], 'input': 'GROUP_wall',
                     'adopted': ADOPTED,
                     'morris_mu_star': np.nan, 'S1': np.nan, 'ST': stg[0],
                     'ST_conf': np.nan, 'S2_diag': np.nan, 'ST_boot_mean': np.nan,
                     'retain_frac': np.nan, 'in_driver_set': np.nan})
        rows.append({'zone': zone, 'city': city, 'kpi': SHORT[t], 'input': 'GROUP_roof',
                     'adopted': ADOPTED,
                     'morris_mu_star': np.nan, 'S1': np.nan, 'ST': stg[1],
                     'ST_conf': np.nan, 'S2_diag': np.nan, 'ST_boot_mean': np.nan,
                     'retain_frac': np.nan, 'in_driver_set': np.nan})
        pd.DataFrame(rows).to_csv(res_path, index=False)   # incremental save
        print(f'{zone} {city:11s} {SHORT[t]}: drivers={drivers} '
              f'| wall_ST={stg[0]:.3f} roof_ST={stg[1]:.3f} ({time.time()-t0:.0f}s)', flush=True)

res = pd.DataFrame(rows)
res.to_csv(res_path, index=False)
print('\n' + '=' * 70)
print(f'PHASE 4 DONE - {n_cells} cells computed this run, {len(rows)} rows saved')
print('=' * 70)
print('DRIVER SETS D(Yk) - STi>=0.05 in >=90% of 500 row-bootstraps')
for (zone, kpi), grp in res[res.in_driver_set == True].groupby(['zone', 'kpi']):
    print(f'  {zone} {kpi:3s}: {", ".join(grp.input)}')
print('\nGROUPED SOBOL (wall vs roof ST):')
g = res[res.input.str.startswith("GROUP")].drop_duplicates(subset=['zone', 'city', 'input'])
print(g.pivot(index=['zone', 'city'], columns='input', values='ST').round(3).to_string())

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 9 | sobol done, {len(rows)} rows saved\n")
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/sobol_indices.csv')
print('DONE - paste this output back.')
