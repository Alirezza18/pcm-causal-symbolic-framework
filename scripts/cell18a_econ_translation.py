# ============================================================================
# scripts/cell18a_econ_translation.py
# Cell 18a — economic & environmental translation (Fig 15 + Table 7)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 37
# (143 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 18a - PHASE 11: ECONOMIC & ENVIRONMENTAL TRANSLATION (Fig 15 + Table 7)
# Translates the knee designs (Table 6) into numbers a practitioner can use:
#
#   (1) EUI SAVING vs BASELINE - two references computed from the zone's own
#       surrogate, both on the level grid:
#         typical  = prediction at the modal level of every variable
#         worst    = prediction at the WORST level of every variable (each
#                    variable taken at the level that maximizes EUI marginally)
#       saving_pct = 100 * (baseline_EUI - knee_EUI) / baseline_EUI
#   (2) ANNUAL COST + CO2 - EUI saving converted at REFERENCE TARIFF
#       (USD 0.08/kWh, electricity; stated in caption) and at an emission
#       factor of EF = 0.55 kgCO2e/kWh (regional grid default; CONFIG below).
#       Scaled to a 100 m2 classroom. For KPI units we assume EUI already is
#       annual kWh/m2; IDD/TL are NOT monetized (no defensible price).
#   (3) COMFORT CO-BENEFIT - knee IDD vs the zone-MEAN IDD (mean of simulated
#       IDD over the zone's designs, the 'status-quo' expectation).
#
# Baseline definitions are CONFIG constants - they are scenario choices, not
# results, and are stated in the Table 7 caption verbatim.
# EXCLUDED cells (Section 3.1) propagate: no IDD row for 0B, no TL for 2B/4B.
# Saves: tables/econ_translation.csv (one row per zone)
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
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')

# ---------------------- scenario CONFIG (stated in caption) --------------
TARIFF_USD_KWH = 0.08        # reference electricity price, USD/kWh
EF_KGCO2_KWH   = 0.55        # grid emission factor, kgCO2e/kWh
FLOOR_M2       = 100.0       # classroom floor area for scaling

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}

KNEE = kr[kr.type == 'knee'].set_index('zone')

def to_levels(X):
    Xl = np.round(X).astype(int)
    for c in INPUTS:
        lo, hi = int(Xl[c].min()), int(Xl[c].max())
        Xl[c] = Xl[c].clip(lo, hi)
    return Xl

t0 = time.time()
rows = []
for z in ZONES:
    g = df[df.climate_zone == z]
    Xl = to_levels(g[INPUTS]).reset_index(drop=True)
    knee = KNEE.loc[z]

    # ---- EUI surrogate + baseline designs ----
    m = joblib.load(f'{BASE}/models/cat_{z}_EUI.joblib')
    knee_X = pd.DataFrame([{c: int(knee[c]) for c in INPUTS}], columns=INPUTS)
    eui_knee = float(m.predict(knee_X)[0])

    typ_X = pd.DataFrame([{c: int(Xl[c].mode()[0]) for c in INPUTS}], columns=INPUTS)
    eui_typ = float(m.predict(typ_X)[0])

    # worst-case: the level maximizing marginal EUI per variable (greedy,
    # one-variable-at-a-time from the modal design - sufficient for a ref.)
    worst = {c: int(Xl[c].mode()[0]) for c in INPUTS}
    for c in INPUTS:
        levs = sorted(Xl[c].unique())
        Xs = pd.DataFrame([{cc: worst[cc] for cc in INPUTS}] * len(levs),
                          columns=INPUTS)
        Xs[c] = levs
        worst[c] = int(levs[int(np.argmax(np.asarray(m.predict(Xs), float)))])
    eui_worst = float(m.predict(pd.DataFrame([worst], columns=INPUTS))[0])

    sav_typ = 100 * (eui_typ - eui_knee) / eui_typ if eui_typ else np.nan
    sav_worst = 100 * (eui_worst - eui_knee) / eui_worst if eui_worst else np.nan
    kwh_saved = eui_typ - eui_knee                    # per m2 per year vs typical

    # ---- IDD co-benefit: knee vs zone-mean simulated IDD ----
    if os.path.exists(f'{BASE}/models/cat_{z}_IDD.joblib'):
        mi = joblib.load(f'{BASE}/models/cat_{z}_IDD.joblib')
        idd_knee = float(mi.predict(knee_X)[0])
        idd_mean = float(g['idd_deg_h'].mean())
        idd_red = idd_mean - idd_knee
    else:
        idd_knee = idd_mean = idd_red = np.nan

    # ---- TL context (not monetized) ----
    if os.path.exists(f'{BASE}/models/cat_{z}_TL.joblib'):
        mt = joblib.load(f'{BASE}/models/cat_{z}_TL.joblib')
        tl_knee = float(mt.predict(knee_X)[0])
    else:
        tl_knee = np.nan

    rows.append({
        'zone': z, 'city': ZCITY[z],
        'eui_typ': eui_typ, 'eui_worst': eui_worst, 'eui_knee': eui_knee,
        'sav_typ_pct': sav_typ, 'sav_worst_pct': sav_worst,
        'kwh_saved_m2': kwh_saved,
        'usd_per_m2': kwh_saved * TARIFF_USD_KWH,
        'usd_100m2': kwh_saved * TARIFF_USD_KWH * FLOOR_M2,
        'co2_kg_m2': kwh_saved * EF_KGCO2_KWH,
        'co2_ton_100m2': kwh_saved * EF_KGCO2_KWH * FLOOR_M2 / 1000.0,
        'idd_knee': idd_knee, 'idd_mean': idd_mean, 'idd_red': idd_red,
        'tl_knee': tl_knee,
    })
    print(f'{z} {ZCITY[z]:11s} EUI {eui_typ:6.1f}->{eui_knee:6.1f} '
          f'(typ {sav_typ:.1f}% | worst-case {sav_worst:.1f}%) | '
          f'${kwh_saved*TARIFF_USD_KWH*FLOOR_M2:5.1f}/yr | '
          f'{kwh_saved*EF_KGCO2_KWH*FLOOR_M2/1000:.3f} tCO2/yr | '
          f'IDD {idd_red:+.2f} ({time.time()-t0:.0f}s)', flush=True)

res = pd.DataFrame(rows)
res.to_csv(BASE + '/tables/econ_translation.csv', index=False)

print('\n' + '=' * 76)
print(f'econ_translation.csv: {len(res)} rows | tariff ${TARIFF_USD_KWH}/kWh, '
      f'EF {EF_KGCO2_KWH} kgCO2/kWh, floor {FLOOR_M2:.0f} m2')
print('=' * 76)
print(res.round(2).to_string(index=False))
print('\nTotals across zones (informational): median saving '
      f'{res.sav_typ_pct.median():.1f}% | median ${res.usd_100m2.median():.1f}/yr '
      f'per 100 m2 | median {res.co2_ton_100m2.median():.3f} tCO2/yr')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f'{datetime.datetime.now().isoformat()} | CELL 18a | econ '
            f'translation saved ({len(res)} zones)\n')
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/econ_translation.csv')
print('DONE 18a - paste this output back.')
