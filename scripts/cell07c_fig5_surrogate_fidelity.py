# ============================================================================
# scripts/cell07c_fig5_surrogate_fidelity.py
# Cell 7c — Fig 5: surrogate fidelity heatmaps
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 6
# (98 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 7c — FIG 5 rebuilt from tables/model_benchmarks.csv (no re-training)
# (a) CV R2 of adopted surrogate per cell  (b) RMSE (log, native units)
# Red border = weakest modeled cell (computed, not chosen).
# Saves: figures/published/fig5_surrogate_fidelity.{png,pdf}
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Rectangle

BASE = '/content/drive/MyDrive/PCM_Study'
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITY  = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
KPIS  = ['EUI', 'IDD', 'TL']
UNITS = {'EUI': 'kWh/m2.yr', 'IDD': 'degC.h', 'TL': 'h'}

r2_mat, rmse_mat = np.full((3, 8), np.nan), np.full((3, 8), np.nan)
for _, r in bm.iterrows():
    i, j = KPIS.index(r.kpi), ZONES.index(r.zone)
    r2_mat[i, j] = r[f"r2_{r.best_model}"]
    rmse_mat[i, j] = r[f"rmse_{r.best_model}"]

wi, wj = np.unravel_index(np.nanargmin(r2_mat), r2_mat.shape)
print(f"weakest modeled cell: {CITY[ZONES[wj]]} {KPIS[wi]} R2={r2_mat[wi, wj]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'wspace': 0.35})

# color scale clipped at -1 so the degenerate cell (-4.65) does not squash
# the contrast of the 0.7-0.97 band; printed numbers are unaffected.
im1 = ax1.imshow(r2_mat, cmap=plt.cm.RdYlGn,
                 norm=Normalize(vmin=-1.0, vmax=1.0), aspect='auto')
for i in range(3):
    for j in range(8):
        if np.isnan(r2_mat[i, j]):
            ax1.text(j, i, 'excluded', ha='center', va='center', fontsize=8,
                     color='gray', style='italic')
            continue
        v = r2_mat[i, j]
        ax1.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9,
                 color='white' if v < 0.50 or v > 0.93 else 'black',
                 fontweight='bold' if (i, j) == (wi, wj) else 'normal')
ax1.add_patch(Rectangle((wj - .5, wi - .5), 1, 1, linewidth=2.5, edgecolor='red',
                        facecolor='none', zorder=10))
ax1.set_xticks(range(8))
ax1.set_xticklabels([f'{z}\n{CITY[z]}' for z in ZONES], fontsize=8,
                    rotation=20, ha='right', rotation_mode='anchor')
ax1.set_yticks(range(3))
ax1.set_yticklabels([f'{k}  ({UNITS[k]})' for k in KPIS], fontsize=10, fontweight='bold')
ax1.set_xlabel('Climate zone', fontsize=10)
ax1.set_title('A · Cross-validated $R^2$\n(adopted surrogate per cell)', fontsize=11, pad=8)
cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label('CV $R^2$', fontsize=9)
cb1.ax.tick_params(labelsize=8)

im2 = ax2.imshow(rmse_mat, cmap=plt.cm.YlOrRd,
                 norm=LogNorm(vmin=max(1e-3, float(np.nanmin(rmse_mat))),
                              vmax=max(100, float(np.nanmax(rmse_mat)))), aspect='auto')
for i in range(3):
    for j in range(8):
        if np.isnan(rmse_mat[i, j]):
            ax2.text(j, i, 'excluded', ha='center', va='center', fontsize=8,
                     color='gray', style='italic')
            continue
        v = rmse_mat[i, j]
        ax2.text(j, i, f'{v:.2f}' if v >= 0.01 else f'{v:.3f}', ha='center', va='center',
                 fontsize=8.5, color='white' if v > 20 else 'black')
ax2.set_xticks(range(8))
ax2.set_xticklabels([f'{z}\n{CITY[z]}' for z in ZONES], fontsize=8,
                    rotation=20, ha='right', rotation_mode='anchor')
ax2.set_yticks(range(3))
ax2.set_yticklabels([f'{k}  ({UNITS[k]})' for k in KPIS], fontsize=10, fontweight='bold')
ax2.set_xlabel('Climate zone', fontsize=10)
ax2.set_title('B · RMSE (log scale, native units)', fontsize=11, pad=8)
cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label('RMSE (native units, log)', fontsize=9)

fig.suptitle('Cross-validated surrogate fidelity by climate zone and performance indicator',
             fontsize=13, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs(BASE + '/figures/published', exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(f'{BASE}/figures/published/fig5_surrogate_fidelity.{ext}', dpi=300,
                bbox_inches='tight')
plt.close()

import datetime
with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 7c | fig5 rebuilt from model_benchmarks.csv\n")
print('saved:', BASE + '/figures/published/fig5_surrogate_fidelity.png + .pdf')
print('DONE - paste this output back.')
