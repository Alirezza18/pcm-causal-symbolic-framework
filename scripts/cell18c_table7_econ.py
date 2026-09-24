# ============================================================================
# scripts/cell18c_table7_econ.py
# Cell 18c — Table 7: economic & environmental translation
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 39
# (110 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 18c - TABLE 7: ECONOMIC & ENVIRONMENTAL TRANSLATION (run AFTER 18a)
# Formats tables/econ_translation.csv into the main-text Table 7:
#   Zone | City | EUI saving vs typical (%) | EUI saving vs worst-case (%) |
#   kWh saved (kWh/m2/yr) | USD/yr per 100 m2 | tCO2/yr per 100 m2 |
#   IDD reduction knee vs zone mean (deg-h) | TL at knee (h)
# Also prints Section 3.9 text diagnostics: savings spread, best/worst
# monetary cases, CO2 totals, the IDD co-benefit ranking, and the risk-flagged
# cells (>30% conformal risk) whose numbers must be called directional.
# Saves one formatted appendix-free CSV + auto-download.
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
res = res.set_index('zone').reindex(ZONES).reset_index()

# risk flags from Section 3.7 (>30% of dynamic range => directional only)
# NOTE: uq_conformal.csv columns are lowercase (Cell 16a): kpi, width_norm
risk = {(r.zone, r.kpi): r.width_norm for _, r in uq.iterrows()}
flagged = sorted([(z, k) for (z, k), v in risk.items()
                  if pd.notna(v) and v > 0.30])

# ---- diagnostics ----
print('=' * 76)
print('DIAGNOSTICS (for Section 3.9)')
print('=' * 76)
print(f'EUI saving vs typical: mean {res.sav_typ_pct.mean():.1f}%, '
      f'median {res.sav_typ_pct.median():.1f}%, '
      f'range {res.sav_typ_pct.min():.1f}-{res.sav_typ_pct.max():.1f}%')
print(f'EUI saving vs worst-case: mean {res.sav_worst_pct.mean():.1f}%, '
      f'range {res.sav_worst_pct.min():.1f}-{res.sav_worst_pct.max():.1f}%')
best = res.loc[res.usd_100m2.idxmax()]
worst = res.loc[res.usd_100m2.idxmin()]
print(f'monetary: best {best.zone} ${best.usd_100m2:.0f}/yr per 100 m2 | '
      f'worst {worst.zone} ${worst.usd_100m2:.0f}/yr | '
      f'median ${res.usd_100m2.median():.0f}/yr')
print(f'CO2: median {res.co2_ton_100m2.median():.3f} t/yr per 100 m2 | '
      f'fleet of 1,000 classrooms ~ '
      f'{res.co2_ton_100m2.median() * 1000:.0f} t/yr at the median knee')
idd = res.dropna(subset=['idd_red']).sort_values('idd_red', ascending=False)
print('IDD co-benefit ranking (knee vs zone mean, deg-h):')
for _, r in idd.iterrows():
    pct = 100 * r.idd_red / r.idd_mean if r.idd_mean else np.nan
    print(f"  {r.zone} {r.city:11s} {r.idd_red:+7.2f} ({pct:+.1f}%)")
q = res.dropna(subset=['sav_typ_pct'])
f_by = q.set_index('zone')
corr_z = q.sort_values('sav_typ_pct').zone.tolist()
print(f'savings order: {" < ".join(f"{z} {f_by.loc[z, 'sav_typ_pct']:.1f}%" for z in corr_z)}')
print(f'risk-flagged (directional-only) cells: {flagged}')

# ---- Table 7 ---- ('*' suffix = directional-only, conformal risk > 30%)
FLAG_SET = set(flagged)
def star(z, kpi, txt):
    return f'{txt}*' if (z, kpi) in FLAG_SET else txt
t7 = pd.DataFrame({
    'Zone': res.zone, 'City': res.city,
    'EUI saving vs typical (%)': [star(z, 'EUI', f'{v:.1f}')
                                  for z, v in zip(res.zone, res.sav_typ_pct)],
    'EUI saving vs worst-case (%)': res.sav_worst_pct.round(1),
    'kWh saved (kWh/m2/yr)': res.kwh_saved_m2.round(1),
    'USD/yr per 100 m2': res.usd_100m2.round(0).astype(int),
    'tCO2/yr per 100 m2': res.co2_ton_100m2.round(3),
    'IDD reduction (deg-h)': [star(z, 'IDD', f'{v:+.2f}') if pd.notna(v) else ''
                              for z, v in zip(res.zone, res.idd_red)],
    'TL at knee (h)': [star(z, 'TL', f'{v:.1f}') if pd.notna(v) else ''
                       for z, v in zip(res.zone, res.tl_knee)],
})
csv_path = BASE + '/tables/table7_econ_translation.csv'
t7.to_csv(csv_path, index=False)
print('\n' + '=' * 76)
print(f'Table 7 - economic & environmental translation ({len(t7)} rows) -> {csv_path}')
print('=' * 76)
print(t7.to_string(index=False))
files.download(csv_path)

print('\nCaption Table 7:')
print('Table 7. Economic and environmental translation of the recommended '
      'PCM designs per climate zone. EUI savings are computed on each '
      "zone's certified surrogate (Section 3.7) against two baselines: the "
      'typical design (all variables at their modal level) and the worst '
      'case (each variable at its EUI-maximising level). Monetary values '
      'apply the reference tariff of USD 0.08/kWh and scale the annual '
      'saving to a 100 m2 classroom; CO2 uses an emission factor of 0.55 '
      'kgCO2e/kWh. IDD reduction compares the knee design with the zone '
      'mean of the simulated designs (status-quo expectation); TL at the '
      'knee is reported in hours for context and is not monetised. An '
      'asterisk marks KPI cells whose Section 3.7 conformal risk exceeds '
      '30% of the dynamic range (the flagged list is printed in the '
      'diagnostics above); such values are directional rather than '
      'quantitative. Savings against the worst case use a greedy '
      'per-variable EUI-maximising design. The tariff and emission factor '
      'are scenario assumptions, not findings; savings scale linearly '
      'with both.')

print('\nDONE 18c - paste this output back.')
