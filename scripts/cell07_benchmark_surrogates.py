# ============================================================================
# scripts/cell07_benchmark_surrogates.py
# Cell 7 — benchmark 5 model families -> winners + Table A11 + models/
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 5
# (153 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 7 — PHASE 3: BENCHMARK 5 FAMILIES -> BEST PER (ZONE, KPI)
#          + TABLE A11 (full CV statistics, all 24 zone-KPI combos)
# Saves: tables/model_benchmarks.csv (R2+RMSE, mean+-SD per family)
#        models/cat_{zone}_{KPI}.joblib
#        tables/table_A11_cv_statistics.csv (120 rows = 24 combos x 5 families)
# NOTE:  Tabriz TL is EXCLUDED from modeling (constant target -> R2 undefined;
#        predicting a constant gives artifact R2 = 1.0). Expected: 23 modeled
#        cells; the A11 table keeps the 24th row marked "excluded".
#        Winner claims come from THIS run only - no hardcoded numbers.
# ============================================================
import os, time, joblib, datetime
import numpy as np
import pandas as pd

try:
    import xgboost, lightgbm, catboost
except ImportError:
    # [Colab shell] pip install lightgbm catboost -q
    import xgboost, lightgbm, catboost

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = '/content/drive/MyDrive/PCM_Study'
os.makedirs(BASE + '/tables', exist_ok=True)
os.makedirs(BASE + '/models', exist_ok=True)
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}
FAMILIES = ['xgb_default', 'xgb_regularized', 'lgbm_default', 'catboost_default', 'mlp']
RNG = 42
OUTER = 5

def cv_stats(model_fn, X, y):
    """5-fold CV -> fold-level R2 and RMSE arrays (kept for Table A11)."""
    kf = KFold(OUTER, shuffle=True, random_state=RNG)
    r2s, rmses = [], []
    for tr, te in kf.split(X):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], p))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], p)))
    return np.array(r2s), np.array(rmses)

MODELS = {
    'xgb_default':      lambda: xgboost.XGBRegressor(random_state=RNG, n_jobs=-1),
    'xgb_regularized':  lambda: xgboost.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.7,
        colsample_bytree=0.7, min_child_weight=5, reg_lambda=5.0, reg_alpha=1.0,
        gamma=0.5, random_state=RNG, n_jobs=-1),
    'lgbm_default':     lambda: lightgbm.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1),
    'catboost_default': lambda: catboost.CatBoostRegressor(iterations=500,
                                                           random_seed=RNG, verbose=False),
    'mlp':              lambda: make_pipeline(StandardScaler(),
                                              MLPRegressor(hidden_layer_sizes=(32, 16),
                                                           max_iter=2000, random_state=RNG)),
}

# ════════════════════════ PART 1 - BENCHMARK ════════════════════════
rows = []
t0 = time.time()
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    X = g[INPUTS].reset_index(drop=True)
    for t in TARGETS:
        if city == 'Tabriz' and t == 'tl_h':          # constant target -> R2 meaningless
            print(f'{zone} {city:11s} TL : EXCLUDED (constant tl_h = -1)')
            continue
        y = g[t].reset_index(drop=True)

        r2m, r2s_, rmm, rms_ = {}, {}, {}, {}
        for name, fn in MODELS.items():
            r2s, rmses = cv_stats(fn, X, y)
            r2m[name], r2s_[name] = r2s.mean(), r2s.std()
            rmm[name], rms_[name] = rmses.mean(), rmses.std()

        best_name = max(r2m, key=r2m.get)
        if r2m[best_name] < 0:
            print(f'  !! {city} {SHORT[t]}: best R2 is NEGATIVE ({r2m[best_name]:.3f}) '
                  f'- target barely learnable, check outlier designs')

        winner = MODELS[best_name]()
        winner.fit(X, y)
        joblib.dump(winner, f'{BASE}/models/cat_{zone}_{SHORT[t]}.joblib')

        rows.append({'zone': zone, 'city': city, 'kpi': SHORT[t], 'best_model': best_name,
                     **{f'r2_{k}': round(v, 4) for k, v in r2m.items()},
                     **{f'r2std_{k}': round(v, 4) for k, v in r2s_.items()},
                     **{f'rmse_{k}': round(v, 4) for k, v in rmm.items()},
                     **{f'rmsestd_{k}': round(v, 4) for k, v in rms_.items()}})
        pd.DataFrame(rows).to_csv(f'{BASE}/tables/model_benchmarks.csv', index=False)
        print(f'{zone} {city:11s} {SHORT[t]}: best={best_name} R2={r2m[best_name]:.3f}'
              f'+-{r2s_[best_name]:.3f} RMSE={rmm[best_name]:.3f} ({time.time()-t0:.0f}s)', flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{BASE}/tables/model_benchmarks.csv', index=False)
print('\n' + '=' * 70)
print(f'PHASE 3 - FINAL RESULTS ({len(res)} modeled cells; Tabriz TL excluded)')
print('=' * 70)
model_cols = [c for c in res.columns if c.startswith('r2_') and not c.startswith('r2std')]
print('\nmean R2 per family (over all modeled cells):')
print(res[model_cols].mean().round(3).to_string())
rmse_cols = [c for c in res.columns if c.startswith('rmse_') and not c.startswith('rmsestd')]
print('\nmean RMSE per family (native units, not comparable across KPIs):')
print(res[rmse_cols].mean().round(3).to_string())
print('\nBEST MODEL PER (ZONE, KPI):')
print(res.pivot(index=['zone', 'city'], columns='kpi', values='best_model').to_string())
print('\nwinner counts:')
print(res.best_model.value_counts().to_string())

# ════════════════════════ PART 2 - TABLE A11 (appendix) ════════════════════════
rows_a11 = []
for _, r in res.iterrows():
    for m in FAMILIES:
        rows_a11.append({'zone': r.zone, 'city': r.city, 'kpi': r.kpi, 'model': m,
                         'r2_mean': r[f'r2_{m}'], 'r2_std': r[f'r2std_{m}'],
                         'rmse_mean': r[f'rmse_{m}'], 'rmse_std': r[f'rmsestd_{m}']})
a11 = pd.DataFrame(rows_a11)

# reinsert the excluded combination so the appendix shows all 24 rows honestly
if res[(res.city == 'Tabriz') & (res.kpi == 'TL')].empty:
    a11 = pd.concat([a11, pd.DataFrame(
        [{'zone': '4B', 'city': 'Tabriz', 'kpi': 'TL', 'model': m,
          'r2_mean': None, 'r2_std': None, 'rmse_mean': None, 'rmse_std': None}
         for m in FAMILIES])], ignore_index=True)
a11['note'] = np.where(a11.kpi.eq('TL') & a11.city.eq('Tabriz'),
                       'excluded: constant target (TL = -1 h)', '')
a11.to_csv(f'{BASE}/tables/table_A11_cv_statistics.csv', index=False)
print(f'\nTable A11 saved: {len(a11)} rows '
      f'({a11.zone.nunique()} zones x 3 KPI x {len(FAMILIES)} families)')
excluded = a11[a11.note != '']
if len(excluded):
    print('excluded rows kept for honesty:', len(excluded))

print('\nA11 preview - best mean R2 per (zone, KPI):')
print(a11.dropna(subset=['r2_mean']).pivot_table(index=['zone', 'city'], columns='kpi',
                                                 values='r2_mean', aggfunc='max').round(3).to_string())

# ════════════════════════ LOG + DONE ════════════════════════
with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 7 | benchmark + Table A11 done, "
            f"{len(res)} modeled cells, winner={res.best_model.mode()[0]}\n")
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/model_benchmarks.csv, '
      f'tables/table_A11_cv_statistics.csv, models/cat_*.joblib')
print('DONE - paste this output back.')
