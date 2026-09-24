# ============================================================================
# scripts/cell18b_fig15_economic_bars.py
# Fig 15 — economic results bars (unlabeled cell)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 38
# (145 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
# HIGH-IMPACT PUBLICATION SETUP (Elsevier / Nature Standards)
# ============================================================================
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

BASE = '/content/drive/MyDrive/PCM_Study'
res = pd.read_csv(BASE + '/tables/econ_translation.csv')

SHORT = {'BandarAbbas': 'B.Abbas', 'Bushehr': 'Bushehr', 'Kashan': 'Kashan',
         'Rasht': 'Rasht', 'Tehran': 'Tehran', 'Hamedan': 'Hamedan',
         'Tabriz': 'Tabriz', 'Ardebil': 'Ardebil'}

LBL = [f'{z}\n{SHORT[c]}' for z, c in zip(res.zone, res.city)]

# Create Wide Figure for Maximum Readability
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))
plt.subplots_adjust(wspace=0.35)

# Academic Color Scheme
C1 = '#2b5c8f'  # Dark Steel Blue
C2 = '#7ca1cc'  # Light Blue
C3 = '#2d8259'  # Forest Green

# ----------------------------------------------------------------------------
# Panel (a) EUI Savings vs Two Baselines
# ----------------------------------------------------------------------------
ax_a = axes[0]
x = np.arange(len(res))
w = 0.35

ax_a.grid(True, axis='y', linestyle=':', alpha=0.5, color='#aaaaaa', zorder=0)

b1 = ax_a.bar(x - w/2, res.sav_typ_pct, w, color=C1, edgecolor='black', lw=0.6, label='vs. Typical Design', zorder=3)
b2 = ax_a.bar(x + w/2, res.sav_worst_pct, w, color=C2, edgecolor='black', lw=0.6, hatch='//', label='vs. Worst-Case Design', zorder=3)

# Data Value Annotations
for xi, v1, v2 in zip(x, res.sav_typ_pct, res.sav_worst_pct):
    ax_a.text(xi - w/2, v1 + 0.15, f'{v1:.1f}', ha='center', va='bottom', fontsize=7.5, color='black')
    ax_a.text(xi + w/2, v2 + 0.15, f'{v2:.1f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

ax_a.axhline(0, color='black', lw=0.8, zorder=2)
ax_a.set_xticks(x)
ax_a.set_xticklabels(LBL, fontsize=8)
ax_a.set_ylabel('EUI Saving of Knee Design (%)', fontsize=9.5, fontweight='bold')
ax_a.set_title('(a) EUI Saving vs. Baseline Designs', fontsize=10.5, fontweight='bold', loc='left', pad=10)
ax_a.legend(fontsize=8, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc')
ax_a.set_ylim(0, max(res.sav_worst_pct) * 1.18)
ax_a.tick_params(labelsize=8)

# ----------------------------------------------------------------------------
# Panel (b) Annual Cost & CO2 Savings (Dual Y-Axis)
# ----------------------------------------------------------------------------
ax_b = axes[1]
order = res.sort_values('usd_100m2', ascending=False).reset_index(drop=True)
x_b = np.arange(len(order))

ax_b.grid(True, axis='y', linestyle=':', alpha=0.5, color='#aaaaaa', zorder=0)

bars_usd = ax_b.bar(x_b - w/2, order.usd_100m2, w, color=C1, edgecolor='black', lw=0.6, zorder=3)
ax_b.set_ylabel('Annual Cost Saving (USD / 100 m²)', fontsize=9.5, fontweight='bold', color=C1)

ax_b2 = ax_b.twinx()
co2_kg = order.co2_ton_100m2 * 1000
bars_co2 = ax_b2.bar(x_b + w/2, co2_kg, w, color=C3, alpha=0.85, edgecolor='black', lw=0.6, hatch='\\\\', zorder=3)
ax_b2.set_ylabel('Annual CO₂ Saving (kg / 100 m²)', fontsize=9.5, fontweight='bold', color=C3)

# Clean, Non-Overlapping Value Labels
for xi, (u, c_) in enumerate(zip(order.usd_100m2, co2_kg)):
    if u > 5:
        ax_b.text(xi - w/2, u + max(order.usd_100m2)*0.02, f'${u:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color=C1)
        ax_b2.text(xi + w/2, c_ + max(co2_kg)*0.02, f'{c_:.0f}', ha='center', va='bottom', fontsize=7, color=C3)

ax_b.set_ylim(0, max(order.usd_100m2) * 1.22)
ax_b2.set_ylim(0, max(co2_kg) * 1.22)
ax_b.set_xticks(x_b)
ax_b.set_xticklabels([f"{z}\n{SHORT[c]}" for z, c in zip(order.zone, order.city)], fontsize=8)
ax_b.set_title('(b) Annual Savings per 100 m² ($0.08/kWh)', fontsize=10.5, fontweight='bold', loc='left', pad=10)

ax_b.tick_params(axis='y', colors=C1, labelsize=8)
ax_b2.tick_params(axis='y', colors=C3, labelsize=8)

handles_b = [Patch(facecolor=C1, edgecolor='black', label='Cost Saving ($)'),
             Patch(facecolor=C3, alpha=0.85, edgecolor='black', hatch='\\\\', label='CO₂ Saving (kg)')]
ax_b.legend(handles=handles_b, loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff')

# ----------------------------------------------------------------------------
# Panel (c) Comfort Co-benefit Dumbbells (Clear & Spaced Out)
# ----------------------------------------------------------------------------
ax_c = axes[2]
dd = res.dropna(subset=['idd_knee']).sort_values('idd_mean', ascending=True).reset_index(drop=True)
y = np.arange(len(dd))

ax_c.grid(True, axis='x', linestyle=':', alpha=0.5, color='#aaaaaa', zorder=0)

# Lines
for i, r in dd.iterrows():
    ax_c.plot([r.idd_mean, r.idd_knee], [i, i], color='#888888', lw=1.8, zorder=1)

# Status Quo Points (Circles)
ax_c.scatter(dd.idd_mean, y, facecolors='white', edgecolors='black', s=50, lw=1.0, zorder=3, label='Status Quo (Zone Mean)')

# Knee Design Points (Stars)
ax_c.scatter(dd.idd_knee, y, marker='*', s=110, c=C3, edgecolors='black', lw=0.5, zorder=4, label='Knee Design (Optimal)')

# Difference Labels next to points
for i, r in dd.iterrows():
    diff = r.idd_knee - r.idd_mean
    max_x = max(r.idd_mean, r.idd_knee)
    ax_c.text(max_x + 0.25, i, f'{diff:+.1f} °C·h', ha='left', va='center', fontsize=7.5, color='#333333')

ax_c.set_yticks(y)
ax_c.set_yticklabels([f"{z} ({SHORT[c]})" for z, c in zip(dd.zone, dd.city)], fontsize=8.5)
ax_c.set_xlim(dd[['idd_mean', 'idd_knee']].min().min() - 0.4, dd[['idd_mean', 'idd_knee']].max().max() + 2.2)
ax_c.set_ylim(-0.8, len(dd) - 0.2)
ax_c.set_xlabel('Indoor Discomfort Degree (IDD, °C·h)', fontsize=9.5, fontweight='bold')
ax_c.set_title('(c) Thermal Comfort Improvement', fontsize=10.5, fontweight='bold', loc='left', pad=10)
ax_c.legend(fontsize=7.5, loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc')
ax_c.tick_params(labelsize=8)

# Overall Title
fig.suptitle('Figure 15. Economic and Environmental Impact of Recommended PCM Configurations',
             fontsize=12, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# Save Outputs
os.makedirs(BASE + '/figures/published', exist_ok=True)
out = BASE + '/figures/published/fig15_clean_published'
for ext in ['png', 'pdf']:
    fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')

print(f'Successfully saved publication-quality plot to: {out}.png and .pdf')
