# ============================================================================
# scripts/cell09d_fig6_fig7_st_plots.py
# Cell 9d — Fig 6 (grouped ST bars) + Fig 7 (ST heatmap)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 12
# (119 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 9d — FIG 6 (grouped wall/roof ST bars) + FIG 7 (per-variable ST heatmap)
# Runs after Cell 9 (uses tables/sobol_indices.csv). Saves:
#   figures/published/fig6_grouped_st.png + .pdf
#   figures/published/fig7_st_heatmap.png + .pdf
# Design notes:
#   Fig 6: one panel per KPI, wall vs roof bars per zone -> the
#          "where does thermal inertia act" localization story.
#   Fig 7: ST heatmap 21 cells x 8 variables, drivers framed in red.
# ============================================================================
# ---- mount Drive if this is a fresh session (no-op if already mounted) ----
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
from matplotlib.patches import Rectangle

BASE = '/content/drive/MyDrive/PCM_Study'
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITY  = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
KPIS  = ['EUI', 'IDD', 'TL']
VARS  = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
         'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
VLBL_FULL = ['PCM type (wall)', 'PCM thickness (wall)', 'Melt point (wall)', 'Position (wall)',
             'PCM type (roof)', 'PCM thickness (roof)', 'Melt point (roof)', 'Position (roof)']

vars_only = sob[~sob.input.str.startswith('GROUP')].copy()
grp_only  = sob[sob.input.str.startswith('GROUP')].copy()
os.makedirs(BASE + '/figures/published', exist_ok=True)

# ---------------- FIG 6: grouped wall vs roof ST ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
x = np.arange(len(ZONES)); w = 0.38
for ax, kpi in zip(axes, KPIS):
    gw = grp_only[(grp_only.kpi == kpi) & (grp_only.input == 'GROUP_wall')].set_index('zone').reindex(ZONES)
    gr = grp_only[(grp_only.kpi == kpi) & (grp_only.input == 'GROUP_roof')].set_index('zone').reindex(ZONES)
    wall_v = gw.ST.values; roof_v = gr.ST.values
    ax.bar(x - w/2, wall_v, w, label='Wall assembly', color='#c0392b', edgecolor='white')
    ax.bar(x + w/2, roof_v, w, label='Roof assembly', color='#2471a3', edgecolor='white')
    ax.set_title(kpi, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{z}\n{CITY[z].split()[0]}' for z in ZONES], fontsize=8)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    for xi, v in zip(x - w/2, wall_v):
        if not np.isnan(v):
            ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=7)
    for xi, v in zip(x + w/2, roof_v):
        if not np.isnan(v):
            ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0, 1.25)
axes[0].set_ylabel('Grouped total-order $S_{T}$', fontsize=10)
axes[0].legend(fontsize=9, loc='upper right')
fig.suptitle('Wall vs. roof contribution to response variance by climate zone and KPI',
             fontsize=13, y=1.03)
fig.tight_layout(rect=[0, 0, 1, 0.98])
for ext in ['png', 'pdf']:
    fig.savefig(f'{BASE}/figures/published/fig6_grouped_st.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig6 saved')

# ---------------- FIG 7: per-variable ST heatmap (21 cells x 8 vars) ----------------
cell_rows = []
for kpi in KPIS:
    for z in ZONES:
        sub = vars_only[(vars_only.kpi == kpi) & (vars_only.zone == z)]
        if len(sub) == 0:
            continue
        cell_rows.append((kpi, z, sub.set_index('input').reindex(VARS)))
heat = np.vstack([s.ST.values for _, _, s in cell_rows])
drv   = np.vstack([s.in_driver_set.fillna(False).astype(bool).values for _, _, s in cell_rows])
labels = [f'{kpi} - {CITY[z]}' for kpi, z, _ in cell_rows]

fig, ax = plt.subplots(figsize=(7.5, 8.5))
im = ax.imshow(heat, cmap='YlOrRd', vmin=0, vmax=max(0.35, np.nanmax(heat)), aspect='auto')
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                color='white' if v > 0.55 * np.nanmax(heat) else 'black')
    for j in range(heat.shape[1]):
        if drv[i, j]:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor='#c0392b', lw=1.6))
ax.set_xticks(range(len(VARS))); ax.set_xticklabels(VLBL_FULL, rotation=45, ha='right', fontsize=8.5)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
for bi, (kpi, z, s) in enumerate(cell_rows):
    if bi and kpi != cell_rows[bi-1][0]:
        ax.axhline(bi - 0.5, color='black', lw=1.2)
ax.set_title('Total-order Sobol $S_{T}$ per variable\n(red frame = driver set $D(Y_k)$, '
             '$S_{T}\\geq 0.05$ in $\\geq$90% of 500 bootstraps)', fontsize=10.5, pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03); cb.set_label('$S_{T}$', fontsize=9)
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(f'{BASE}/figures/published/fig7_st_heatmap.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig7 saved')

print('\nwall vs roof leaders per KPI (mean grouped ST):')
for kpi in KPIS:
    gw = grp_only[(grp_only.kpi == kpi) & (grp_only.input == 'GROUP_wall')].ST.mean()
    gr = grp_only[(grp_only.kpi == kpi) & (grp_only.input == 'GROUP_roof')].ST.mean()
    lead = 'wall' if gw > gr else 'roof'
    print(f'  {kpi:3s}: mean ST_wall={gw:.3f}  ST_roof={gr:.3f}  -> {lead} leads')

print('\nDONE 9d - download the two figures and attach them here.')
