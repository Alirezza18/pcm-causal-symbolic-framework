# ============================================================================
# scripts/cell18d_tables_A24_A25.py
# Cell 18d — Tables A24 + A25 for Section 3.9
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 40
# (123 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 18d - TABLES A24 + A25: APPENDIX FOR SECTION 3.9 (run AFTER 18a/18c)
# Pure arithmetic on already-saved certified tables; no model calls.
#
#   A24 - RESOLVABILITY LEDGER (one row per zone):
#     the section's censoring test, shown zone by zone: energy saving vs the
#     typical design (kWh/m2/yr), the 90% conformal width (kWh/m2, A21),
#     their ratio (# of error bars spanned), the verdict
#     (resolvable if ratio > 1), and the same ratio vs the WORST-case saving
#     (the best case for PCM, for context).
#   A25 - SCENARIO GRID: the min/median/max saving zones re-priced across a
#     tariff range (0.05-0.15 USD/kWh) and emission factors (0.35-0.75
#     kgCO2e/kWh). States the caption's "savings scale linearly with both"
#     claim as a lookup table reviewers can read directly.
#
# Reads: tables/econ_translation.csv (18a) + tables/uq_conformal.csv (16a)
# Saves: tables/table_A24_resolvability.csv + tables/table_A25_scenario_grid.csv
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
res = pd.read_csv(BASE + '/tables/econ_translation.csv')
uq = pd.read_csv(BASE + '/tables/uq_conformal.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
res = res.set_index('zone').reindex(ZONES).reset_index()

# conformal EUI widths in engineering units (lowercase kpi per Cell 16a)
weui = {r.zone: r.width for _, r in uq[uq.kpi == 'EUI'].iterrows()}
risk = {r.zone: r.width_norm for _, r in uq[uq.kpi == 'EUI'].iterrows()}

# ================= A24: resolvability ledger =================
rows = []
for _, r in res.iterrows():
    w = weui[r.zone]
    ratio_typ = r.kwh_saved_m2 / w if w else np.nan
    kwh_worst = r.eui_worst - r.eui_knee          # worst-case saving, abs
    ratio_worst = kwh_worst / w if w else np.nan
    rows.append({
        'Zone': r.zone, 'City': r.city,
        'Saving vs typical (kWh/m2/yr)': round(r.kwh_saved_m2, 2),
        '90% conformal width (kWh/m2)': round(w, 2),
        'Ratio (typ)': round(ratio_typ, 2),
        'Saving vs worst case (kWh/m2/yr)': round(kwh_worst, 2),
        'Ratio (worst)': round(ratio_worst, 2),
        'Conformal risk (share of range)': f"{risk[r.zone]*100:.0f}%",
        'Verdict (typ)': 'resolvable' if ratio_typ > 1 else 'NOT resolvable',
    })
A24 = pd.DataFrame(rows)
p24 = BASE + '/tables/table_A24_resolvability.csv'
A24.to_csv(p24, index=False)
print('=' * 76)
print(f'Table A24 - resolvability ledger ({len(A24)} rows) -> {p24}')
print('=' * 76)
print(A24.to_string(index=False))
n_res = (A24['Ratio (typ)'] > 1).sum()
print(f'\nresolvable at 90% confidence: {n_res} of {len(A24)} zones '
      f'({", ".join(A24.loc[A24["Ratio (typ)"] > 1, "Zone"])})')
print('NOT resolvable: '
      + ', '.join(A24.loc[A24['Ratio (typ)'] <= 1, 'Zone']))

# ================= A25: tariff x EF scenario grid =================
TARIFFS = [0.05, 0.08, 0.10, 0.15]
EFS = [0.35, 0.55, 0.75]
q1 = res.sort_values('usd_100m2').iloc[len(res) // 2]           # median zone
hi = res.loc[res.usd_100m2.idxmax()]
lo = res.loc[res.usd_100m2.idxmin()]
grid_rows = []
for lbl, r in [('min zone', lo), ('median zone', q1), ('max zone (0B*)', hi)]:
    row = {'Case': lbl}
    for t in TARIFFS:
        row[f'${t:.2f}/kWh'] = round(r.kwh_saved_m2 * t * 100, 0)
    for ef in EFS:
        row[f'EF {ef}'] = round(r.kwh_saved_m2 * ef * 100 / 1000, 3)
    grid_rows.append(row)
A25 = pd.DataFrame(grid_rows)
p25 = BASE + '/tables/table_A25_scenario_grid.csv'
A25.to_csv(p25, index=False)
print('\n' + '=' * 76)
print(f'Table A25 - tariff & emission-factor scenario grid -> {p25}')
print('=' * 76)
print('USD/yr per 100 m2 (left) and tCO2/yr per 100 m2 (right):')
print(A25.to_string(index=False))
print('(* the 0B saving is directional-only: conformal risk 61%, Section 3.7)')

files.download(p24)
files.download(p25)

print('\nCaption A24:')
print('Table A24. Statistical resolvability of the energy saving, zone by '
      'zone. The saving of the knee design against the typical design '
      '(kWh/m2 per year) is divided by the 90% conformal prediction-interval '
      'width of the same cell (Table A21, Section 3.7): the resulting ratio '
      'counts how many error bars the saving spans, and a ratio above one '
      'means the surrogate distinguishes the saving from zero at 90% '
      'confidence. The worst-case columns repeat the computation against the '
      'EUI-maximising design - the most favourable baseline - for context. '
      'The conformal risk column restates the width as a share of the '
      "zone's observed EUI range (Section 3.7 threshold: 30%).")
print('\nCaption A25:')
print('Table A25. Sensitivity of the monetary and carbon translation to the '
      'two scenario assumptions. Each cell re-prices one reference zone '
      '(the minimum, median, and maximum saving zone of Table 7) across '
      'electricity tariffs of 0.05-0.15 USD/kWh (cost columns) and grid '
      'emission factors of 0.35-0.75 kgCO2e/kWh (carbon columns). Because '
      'the translation is linear in both parameters, the grid brackets any '
      'intermediate tariff or factor without further computation. The '
      'Bandar Abbas row is directional-only (conformal risk 61%).')

print('\nDONE 18d - paste this output back.')
