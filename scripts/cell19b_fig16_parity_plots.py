# ============================================================================
# scripts/cell19b_fig16_parity_plots.py
# Cell 19b (v5) — Fig 16: parity plots with conformal PIs
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 42
# (176 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 19b (v5) - PARITY PLOT WITH REGRESSION RELATIONSHIP FORMULAS
# Reads tables/validation_results.csv (the filled template from 16 E+ runs).
# Produces:
#   - Fig 16  : Publication-quality Parity Plots (y=x) with dynamic
#               regression formulas (y = m*x + c, R^2) and 90% Conformal PIs
#   - Table 8 : Validation results table saved to tables/table8_validation.csv
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
res = pd.read_csv(BASE + '/tables/validation_results.csv')
uq  = pd.read_csv(BASE + '/tables/uq_conformal.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
KPIS = ['EUI', 'IDD', 'TL']
UNITS = {'EUI': 'kWh/m²', 'IDD': '°C·h', 'TL': 'h'}
KORD = {'EUI': 0, 'IDD': 1, 'TL': 2}
ZI = {z: i for i, z in enumerate(ZONES)}

res.columns = [c.strip() for c in res.columns]
res['zone'] = res['zone'].astype(str)
res['KPI'] = res['KPI'].astype(str)
n_filled = int(res.sim_value.notna().sum())
if n_filled < len(res):
    print(f'WARNING: {len(res) - n_filled} of {len(res)} rows have no sim_value - skipped')

# ---- verdict per cell ----
rows, misses = [], []
for _, r in res.iterrows():
    if pd.isna(r['sim_value']):
        continue
    z, k = r['zone'], r['KPI']
    pred, sim = float(r['surrogate_pred']), float(r['sim_value'])
    u = uq[(uq.zone == z) & (uq.kpi == k)]
    if not len(u):
        print(f'WARNING: no conformal width for ({z}, {k}) - skipped'); continue
    w = float(u.iloc[0]['width'])
    lo, hi = pred - w / 2, pred + w / 2
    inside = bool(lo <= sim <= hi)
    rows.append({'Case': r['case_id'], 'Design': r['design_type'], 'Zone': z,
                 'KPI': k, 'Surrogate': round(pred, 2),
                 'PI lo': round(lo, 2), 'PI hi': round(hi, 2),
                 'Width (90%)': round(w, 2), 'Simulated (E+)': round(sim, 2),
                 'Error (units)': round(sim - pred, 2),
                 'Error (%)': round(100 * (sim - pred) / pred, 1) if pred else np.nan,
                 'In PI?': 'yes' if inside else 'NO'})
    if not inside:
        misses.append((r['case_id'], k, round(sim - pred, 2)))
T8 = pd.DataFrame(rows)
T8['_zo'] = T8['Zone'].map(ZI)
T8['_ko'] = T8['KPI'].map(KORD)
T8['_do'] = (T8['Design'] != 'knee').astype(int)
T8 = T8.sort_values(['_ko', '_zo', '_do']).drop(columns=['_zo', '_ko', '_do'])

def cov(d):
    return float((d['In PI?'] == 'yes').mean()) if len(d) else np.nan
kn, ho_ = T8[T8.Design == 'knee'], T8[T8.Design == 'heldout']
p, n = cov(T8), len(T8)

# ---------- DIAGNOSTICS ----------
print('=' * 76)
print('DIAGNOSTICS (for Section 3.10)')
print('=' * 76)
print(f'validated cells: {len(T8)} of 42 modeled (16 designs x 3 KPIs - 6 excluded by 3.1)')
print(f'realized coverage: knee {cov(kn):.2f} (n={len(kn)}) | '
      f'held-out {cov(ho_):.2f} (n={len(ho_)}) | overall {cov(T8):.2f} | nominal 0.90')

# ---------- FIG 16 (PARITY PLOT WITH REGRESSION FORMULAS) ----------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#1F2937'
plt.rcParams['axes.linewidth'] = 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=300)

for idx, (ax, k) in enumerate(zip(axes, KPIS)):
    s = T8[T8.KPI == k].copy()
    if not len(s):
        ax.set_visible(False)
        continue

    min_val = min(s['Surrogate'].min(), s['Simulated (E+)'].min(), s['PI lo'].min())
    max_val = max(s['Surrogate'].max(), s['Simulated (E+)'].max(), s['PI hi'].max())
    padding = (max_val - min_val) * 0.12 if max_val != min_val else 1.0
    lims = [min_val - padding, max_val + padding]

    # 1. رسم خط قطری ایده‌آل y = x
    ax.plot(lims, lims, color='#6B7280', ls='--', lw=1.2, zorder=1, label='Identity (y = x)')

    # 2. محاسبه رابطه برازش خطی (Linear Regression Fit)
    x_val = s['Surrogate'].values
    y_val = s['Simulated (E+)'].values
    slope, intercept = np.polyfit(x_val, y_val, 1)
    r2 = np.corrcoef(x_val, y_val)[0, 1]**2

    # خط برازش روی داده‌ها
    x_fit = np.linspace(lims[0], lims[1], 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='#DC2626', ls=':', lw=1.5, zorder=2, label='Linear Fit')

    # 3. رسم نقاط و بازه‌های اطمینان Conformal PI
    for _, row in s.iterrows():
        is_inside = row['In PI?'] == 'yes'
        pred, sim = row['Surrogate'], row['Simulated (E+)']

        ax.errorbar(pred, sim, xerr=[[pred - row['PI lo']], [row['PI hi'] - pred]],
                    fmt='none', ecolor='#93C5FD', elinewidth=1.5, capsize=3, capthick=1.2, zorder=3)

        if not is_inside:
            ax.scatter(pred, sim, marker='X', color='#DC2626', s=70, zorder=5)
        else:
            m = 'o' if row['Design'] == 'knee' else '^'
            c = '#059669' if row['Design'] == 'knee' else '#2563EB'
            ax.scatter(pred, sim, marker=m, color=c, s=55, alpha=0.9, zorder=4)

    # 4. درج فرمول رابطه برازش روی شکل
    sign = '+' if intercept >= 0 else '-'
    formula_str = f"$y = {slope:.2f}x {sign} {abs(intercept):.2f}$\n$R^2 = {r2:.3f}$"
    ax.text(0.04, 0.85, formula_str, transform=ax.transAxes, fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9FAFB', edgecolor='#D1D5DB', alpha=0.95),
            zorder=6)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(f'Surrogate Predicted {k} [{UNITS[k]}]', fontsize=9.5, fontweight='semibold')
    ax.set_ylabel(f'EnergyPlus Simulated {k} [{UNITS[k]}]', fontsize=9.5, fontweight='semibold')
    ax.set_title(f'({chr(97 + idx)}) {k} Validation', fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, ls=':', alpha=0.5, color='#D1D5DB')

legend_elements = [
    plt.Line2D([0], [0], color='#6B7280', ls='--', lw=1.2, label='Identity (y = x)'),
    plt.Line2D([0], [0], color='#DC2626', ls=':', lw=1.5, label='Linear Fit'),
    plt.Line2D([0], [0], color='#93C5FD', lw=2, label='90% Conformal PI'),
    plt.Line2D([0], [0], marker='o', color='#059669', ms=7, ls='', label='Knee Design'),
    plt.Line2D([0], [0], marker='^', color='#2563EB', ms=7, ls='', label='Held-out Design'),
    plt.Line2D([0], [0], marker='X', color='#DC2626', ms=8, ls='', label='Outside PI')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=6, frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=8.5)

plt.tight_layout()

os.makedirs(BASE + '/figures/published', exist_ok=True)
for ext in ['png', 'pdf', 'svg']:
    fig.savefig(BASE + f'/figures/published/fig16_validation.{ext}', dpi=300, bbox_inches='tight')

print('\nsaved figures/published/fig16_validation.png')
files.download(BASE + '/figures/published/fig16_validation.png')

# ---------- TABLE 8 ----------
csv_path = BASE + '/tables/table8_validation.csv'
T8.to_csv(csv_path, index=False)
print(f'\nTable 8 - E+ validation ({len(T8)} rows) -> {csv_path}')
print(T8.to_string(index=False))
files.download(csv_path)

print('\nDONE 19b - paste this output back.')
