# ============================================================================
# scripts/cell16a_uncertainty_quantification.py
# Cell 16a — uncertainty quantification (producer)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 29
# (187 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 16a - PHASE 9: UNCERTAINTY QUANTIFICATION (producer for Fig 13+T5+A21)
# Two independent instruments on the SAME adopted surrogates as Cells 9-15:
#
# (1) SPLIT-CONFORMAL PREDICTION INTERVALS (distribution-free, finite-sample;
#     Vovk et al. 2005). Per cell, N_SPLITS repeated random splits of the
#     zone's designs into train 60% / calibration 20% / test 20%:
#     fit the ADOPTED family from scratch on train; qhat = finite-sample
#     corrected (1-alpha)-quantile of |residuals| on calibration; PI =
#     yhat +/- qhat; nominal coverage 90% (alpha=0.10) measured on test.
#     Saves mean coverage + mean width (engineering AND normalized = width /
#     range(y)) with across-split SEs.
# (2) VARIANCE DECOMPOSITION of the point prediction:
#     * EPISTEMIC (model) variance: B=30 pairs-bootstrap refits (resample
#       rows, refit adopted family from scratch) -> Var of predictions
#       across refits, averaged over the zone's designs
#     * ALEATORIC (noise) variance: sigma_resid^2 from the full-data fit
#     * fractions f_model, f_noise reported per cell
#     LENS WARNING: this quantifies SURROGATE uncertainty, not scenario or
#     weather-file uncertainty; stated in the captions.
# Design points scored: each zone's own level-coded designs (same set as 14a).
# EXCLUDED cells (Section 3.1): 0B IDD, 2B TL, 4B TL -> absent everywhere.
# Resume-safe: incremental saves of tables/uq_conformal.csv + uq_vardec.csv.
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import time, datetime
import numpy as np
import pandas as pd

try:
    import catboost
except ImportError:
    # [Colab shell] pip install catboost -q
    import catboost
try:
    import xgboost
except ImportError:
    # [Colab shell] pip install xgboost -q
    import xgboost

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']

SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}

N_SPLITS, ALPHA = 10, 0.10       # repeated split-conformal, nominal 90%
N_BOOT           = 30            # epistemic bootstrap refits
SEED             = 42

def to_levels(X):
    """Level coding - byte-identical logic to Cell 9."""
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

def make_model(family, seed):
    """Adopted-family factory - Cell 7 default hyperparameters."""
    if family.startswith('catboost'):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6,
                                 random_seed=seed, verbose=False)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                        random_state=seed, verbosity=0)

cf_path, vd_path = BASE + '/tables/uq_conformal.csv', BASE + '/tables/uq_vardec.csv'
rows_cf, rows_vd = [], []
for p, rows in ((cf_path, rows_cf), (vd_path, rows_vd)):
    if os.path.exists(p):
        prev = pd.read_csv(p)
        if len(prev):
            rows.extend(prev.to_dict('records'))
done = ({(r['zone'], r['kpi']) for r in rows_cf} &
        {(r['zone'], r['kpi']) for r in rows_vd})
if done:
    print(f'resuming - {len(done)} cells already computed, keeping them')

t0 = time.time()
n_new = 0
for z in ZONES:
    g = df[df.climate_zone == z]
    city = g['city'].iloc[0]
    Xall = to_levels(g[INPUTS]).reset_index(drop=True)
    for t in TARGETS:
        k = SHORT[t]
        if (z, k) in done:
            print(f'{z} {k} already done - skip'); continue
        if (city, k) in SKIP:
            print(f'{z} {city:11s} {k}: SKIPPED - {SKIP[(city, k)]}'); continue
        n_new += 1
        y = g[t].reset_index(drop=True).astype(float).values
        yrange = float(y.max() - y.min())
        ADOPTED = bm[(bm.zone == z) & (bm.kpi == k)].iloc[0].best_model
        print(f'--- {z} {city} {k} | adopted: {ADOPTED} ---', flush=True)
        n = len(y)
        rng = np.random.default_rng(SEED + abs(hash((z, k))) % 10000)

        # ---------- (1) repeated split-conformal (60/20/20) ----------
        ntr, nca = int(0.6 * n), int(0.2 * n)
        covs, wids = [], []
        for s in range(N_SPLITS):
            idx = rng.permutation(n)
            itr, ica, ite = idx[:ntr], idx[ntr:ntr + nca], idx[ntr + nca:]
            m = make_model(ADOPTED, SEED + s)
            m.fit(Xall.iloc[itr], y[itr])
            res = np.abs(y[ica] - m.predict(Xall.iloc[ica]))
            qh = np.quantile(res, min(1.0, np.ceil((len(ica) + 1) * (1 - ALPHA)) / len(ica)))
            pe = m.predict(Xall.iloc[ite])
            covs.append(float(np.mean(np.abs(y[ite] - pe) <= qh)))
            wids.append(2 * qh)
        cov, wid = float(np.mean(covs)), float(np.mean(wids))
        rows_cf.append({'zone': z, 'city': city, 'kpi': k, 'adopted': ADOPTED,
                        'coverage': cov, 'width': wid,
                        'width_norm': wid / yrange,
                        'cov_se': float(np.std(covs, ddof=1) / np.sqrt(N_SPLITS)),
                        'width_se': float(np.std(wids, ddof=1) / np.sqrt(N_SPLITS)),
                        'alpha': ALPHA, 'n_splits': N_SPLITS,
                        'n_train': ntr, 'n_cal': nca, 'n_test': n - ntr - nca})
        print(f'    conformal : coverage={cov:.3f} (SE {rows_cf[-1]["cov_se"]:.3f})  '
              f'width={wid:.2f} (norm {wid/yrange:.3f})  ({time.time()-t0:.0f}s)', flush=True)

        # ---------- (2) epistemic bootstrap + aleatoric residual ----------
        m_full = make_model(ADOPTED, SEED)
        m_full.fit(Xall, y)
        sig2 = float((y - m_full.predict(Xall)).var(ddof=1))   # aleatoric
        preds = np.empty((N_BOOT, n))
        for b in range(N_BOOT):
            rb = np.random.default_rng(SEED + 1000 + b)
            idxb = rb.choice(n, size=n, replace=True)
            mb = make_model(ADOPTED, SEED + 100 + b)
            mb.fit(Xall.iloc[idxb], y[idxb])
            preds[b] = mb.predict(Xall)
        var_mod = float(preds.var(axis=0, ddof=1).mean())      # epistemic
        tot = var_mod + sig2
        rows_vd.append({'zone': z, 'city': city, 'kpi': k, 'adopted': ADOPTED,
                        'sigma_resid': float(np.sqrt(sig2)),
                        'sd_model': float(np.sqrt(var_mod)),
                        'sd_total': float(np.sqrt(tot)),
                        'sd_norm': float(np.sqrt(tot) / yrange),
                        'f_model': var_mod / tot, 'f_noise': sig2 / tot,
                        'n_boot': N_BOOT})
        print(f'    vardec    : sd_model={np.sqrt(var_mod):.3f}  '
              f'sigma_resid={np.sqrt(sig2):.3f}  f_model={var_mod/tot:.2f}  '
              f'({time.time()-t0:.0f}s)', flush=True)

        pd.DataFrame(rows_cf).to_csv(cf_path, index=False)   # incremental save
        pd.DataFrame(rows_vd).to_csv(vd_path, index=False)

res_cf, res_vd = pd.DataFrame(rows_cf), pd.DataFrame(rows_vd)
print('\n' + '=' * 76)
print(f'PHASE 9 DONE - {n_new} cells computed, '
      f'{len(res_cf)} conformal + {len(res_vd)} vardec rows saved')
print('=' * 76)
if n_new == 0:
    print('nothing new to compute - tables complete')
print('\nconformal summary (mean across cells per KPI):')
print(res_cf.groupby('kpi')[['coverage', 'width_norm']].mean().round(3).to_string())
print('\ncells below 90% nominal coverage:',
      res_cf[res_cf.coverage < 0.90][['zone', 'kpi', 'coverage']].to_string(index=False)
      if (res_cf.coverage < 0.90).any() else 'none')
print('\nvardec summary (mean across cells per KPI):')
print(res_vd.groupby('kpi')[['f_model', 'sd_norm']].mean().round(3).to_string())

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()} | CELL 16a | UQ done, '
            f'{len(res_cf)}+{len(res_vd)} rows\n')
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/uq_conformal.csv + uq_vardec.csv')
print('DONE 16a - paste this output back.')
