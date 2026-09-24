# ============================================================================
# scripts/cell15b_fig12_pareto_masterplot.py
# Cell 15b — Fig 12: integrated Pareto masterplot
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 26
# (117 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 15b - FIG 12: INTEGRATED HIGH-IMPACT PUBLICATION MASTERPLOT
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
from google.colab import files

# تنظیمات استایل حرفه‌ای مقالات
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#2b2b2b'
plt.rcParams['axes.linewidth'] = 0.8

BASE = '/content/drive/MyDrive/PCM_Study'
fr = pd.read_csv(BASE + '/tables/pareto_front.csv')
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')
KNEE = kr[kr.type == 'knee'].set_index('zone')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITYD = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level',
          'position_wall', 'pcm_type_roof', 'thickness_roof_level',
          'melting_point_roof_level', 'position_roof']
VARLBL = ['Wall PCM Type', 'Wall Thickness', 'Wall Melt Point', 'Wall Position',
          'Roof PCM Type', 'Roof Thickness', 'Roof Melt Point', 'Roof Position']

PALETTE = ['#2b5c8f', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#333333']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'P']
ZCOL = dict(zip(ZONES, PALETTE))
ZMARK = dict(zip(ZONES, MARKERS))

fig = plt.figure(figsize=(15, 7.5), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.20)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ---------------- Panel (a): Trade-off Space (EUI vs IDD) ----------------
# نرمال‌سازی اهداف برای نمایش یکپارچه در یک Scatter plot جامع
for z in ZONES:
    fz = fr[fr.zone == z].copy()
    if 'EUI' in fz.columns and 'IDD' in fz.columns:
        # رسم کلیه نقاط پارتو
        eui_norm = (fz['EUI'] - fz['EUI'].min()) / (fz['EUI'].max() - fz['EUI'].min() + 1e-9)
        idd_norm = (fz['IDD'] - fz['IDD'].min()) / (fz['IDD'].max() - fz['IDD'].min() + 1e-9)

        ax1.scatter(eui_norm, idd_norm, c=ZCOL[z], alpha=0.25, s=20, edgecolors='none')

        # رسم نقطه Knee
        rk = KNEE.loc[z]
        rk_eui = (rk['EUI'] - fz['EUI'].min()) / (fz['EUI'].max() - fz['EUI'].min() + 1e-9)
        rk_idd = (rk['IDD'] - fz['IDD'].min()) / (fz['IDD'].max() - fz['IDD'].min() + 1e-9)

        ax1.scatter(rk_eui, rk_idd, c=ZCOL[z], marker=ZMARK[z], s=120, edgecolors='k',
                    linewidths=1.0, zorder=5, label=f'{z} ({CITYD[z]})')

ax1.set_xlabel('Normalized EUI* (0 = Climate Best)', fontsize=9, fontweight='bold')
ax1.set_ylabel('Normalized IDD* (0 = Climate Best)', fontsize=9, fontweight='bold')
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_title('(a) Multi-Objective Pareto Space & Selected Knee Points', fontsize=11, fontweight='bold', loc='left', pad=12)
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='#ccc', ncol=2)

# ---------------- Panel (b): Integrated Design + Performance Matrix ----------------
M_design = np.zeros((len(INPUTS), len(ZONES)))
for j, z in enumerate(ZONES):
    r = KNEE.loc[z]
    for i, v in enumerate(INPUTS):
        M_design[i, j] = int(r[v])

cmap = plt.cm.get_cmap('YlGnBu', 5)
im = ax2.imshow(M_design, cmap=cmap, vmin=-0.5, vmax=4.5, aspect='auto')

# گرید سفید بین سلول‌ها
ax2.set_xticks(np.arange(len(ZONES)) - 0.5, minor=True)
ax2.set_yticks(np.arange(len(INPUTS)) - 0.5, minor=True)
ax2.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax2.tick_params(which="minor", bottom=False, left=False)

for i in range(len(INPUTS)):
    for j in range(len(ZONES)):
        val = int(M_design[i, j])
        ax2.text(j, i, val, ha='center', va='center', fontsize=8.5, fontweight='bold',
                 color='white' if val >= 3 else '#222222')

ax2.set_xticks(range(len(ZONES)))
ax2.set_xticklabels(ZONES, fontsize=9, fontweight='bold', color='#111111')
ax2.set_yticks(range(len(INPUTS)))
ax2.set_yticklabels(VARLBL, fontsize=8.5, color='#222222')
ax2.tick_params(length=0)
ax2.set_title('(b) Optimal Knee Configuration Profile (Level Codes)', fontsize=11, fontweight='bold', loc='left', pad=12)

cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, ticks=range(5))
cbar.set_label('Discrete Level Code (0 to 4)', fontsize=8, rotation=270, labelpad=12)
cbar.ax.tick_params(labelsize=8)

# Save
out = BASE + '/figures/published'
os.makedirs(out, exist_ok=True)
fig.savefig(out + '/fig12_pareto_knee_master.png', dpi=300, bbox_inches='tight')
fig.savefig(out + '/fig12_pareto_knee_master.pdf', bbox_inches='tight')
print('Saved Masterplot to:', out + '/fig12_pareto_knee_master.png')
files.download(out + '/fig12_pareto_knee_master.png')
