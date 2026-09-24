# ============================================================================
# scripts/cell08_holdout_validation.py
# Cell 8 — hold-out (80/20) validation of adopted surrogates
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 7
# (100 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 8 — PHASE 3.5: HOLD-OUT VALIDATION OF ADOPTED SURROGATES
# Refits each ADOPTED model (from Cell 7) on 80%, scores once on an
# untouched 20% set. Independent check behind the §3.1 claim and the
# entry gate for §3.2 (Sobol/SHAP inherit these models).
# Saves: tables/holdout_validation.csv + tables/holdout_predictions.csv
# NOTE:  single 80/20 split on ~480 designs -> hold-out scores are noisier
#        than 5-fold CV; drops up to ~0.1 R2 are expected, not alarming.
#        Degenerate cells (BandarAbbas IDD, Kashan TL - see §3.1) are
#        scored for completeness but excluded from the pass judgment.
# ============================================================
import os, time, joblib, datetime
import numpy as np
import pandas as pd

try:
    import xgboost, lightgbm, catboost
except ImportError:
    # [Colab shell] pip install lightgbm catboost -q
    import xgboost, lightgbm, catboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = {'EUI': 'eui_kwh_m2', 'IDD': 'idd_deg_h', 'TL': 'tl_h'}
RNG = 42
DEGENERATE = {('BandarAbbas', 'IDD'), ('Kashan', 'TL')}   # §3.1: target-geometry limited

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

rows, preds = [], []
t0 = time.time()
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    X = g[INPUTS].reset_index(drop=True)
    for kpi, t in TARGETS.items():
        if city == 'Tabriz' and kpi == 'TL':
            continue
        b = bm[(bm.zone == zone) & (bm.kpi == kpi)].iloc[0]
        adopted, cv_r2 = b.best_model, float(b[f'r2_{b.best_model}'])
        y = g[t].reset_index(drop=True)

        Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=RNG)
        m = MODELS[adopted]()                       # refit adopted family on the 80%
        m.fit(Xtr, ytr)
        p = m.predict(Xho)
        ho_r2 = r2_score(yho, p)
        ho_rmse = float(np.sqrt(mean_squared_error(yho, p)))
        delta = ho_r2 - cv_r2

        degenerate = (city, kpi) in DEGENERATE
        flag = 'degenerate (see 3.1)' if degenerate else ('review: drop > 0.15' if delta < -0.15 else 'OK')
        rows.append({'zone': zone, 'city': city, 'kpi': kpi, 'adopted': adopted,
                     'cv_r2': round(cv_r2, 4), 'holdout_r2': round(ho_r2, 4),
                     'delta': round(delta, 4), 'holdout_rmse': round(ho_rmse, 4),
                     'n_test': len(yho), 'flag': flag})
        preds.append(pd.DataFrame({'zone': zone, 'city': city, 'kpi': kpi,
                                   'y_true': yho.values, 'y_pred': p}))
        print(f'{zone} {city:11s} {kpi:3s}: CV={cv_r2:.3f}  hold-out={ho_r2:.3f}  '
              f'delta={delta:+.3f}  [{flag}] ({time.time()-t0:.0f}s)', flush=True)

h = pd.DataFrame(rows)
h.to_csv(BASE + '/tables/holdout_validation.csv', index=False)
pd.concat(preds, ignore_index=True).to_csv(BASE + '/tables/holdout_predictions.csv', index=False)

ok = h[(h.flag == 'OK')]
print('\n' + '=' * 70)
print(f'HOLD-OUT SUMMARY ({len(ok)} healthy cells judged; degenerate cells scored but not judged)')
print('=' * 70)
print(f'mean delta (hold-out - CV): {ok.delta.mean():+.3f}   worst drop: {ok.delta.min():+.3f} '
      f'({ok.loc[ok.delta.idxmin(), "city"]} {ok.loc[ok.delta.idxmin(), "kpi"]})')
print(f'healthy cells with hold-out R2 >= CV - 0.15: {(ok.delta >= -0.15).sum()}/{len(ok)}')
print(h.to_string(index=False))

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 8 | hold-out validation done, "
            f"mean delta={ok.delta.mean():+.3f}, worst={ok.delta.min():+.3f}\n")
print('\nsaved: tables/holdout_validation.csv + tables/holdout_predictions.csv')
print('DONE - paste this output back.')
