# ============================================================================
# scripts/cell13b_fig10_lens_consensus.py
# Cell 13b — Fig 10: four-lens consensus figure
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 21
# (151 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 13b - FIG 10: FOUR-LENS CONSENSUS FIGURE   (run AFTER Cell 13)
# Fixed version of the user's draft:
#   [1] canonical input names (*_level), not thickness_wall_mm/melting_point_c
#   [2] 21 modeled cells only (3.1 exclusions) -> no KeyError on missing rows
#   [3] renamed fig8_consensus -> fig10_lens_consensus (Fig 8 = causal, Fig 9
#       = SHAP - the published numbering chain stays intact)
#   [4] xlabel says 21 pairs
#   [5] set parsing unified on '|' (Cell 13's writer) - no ';'-vs-'|' bug
#   [6] KPI-block separators computed from cumulative block sizes (the 3.1
#       exclusions make some zones 2 cells wide - fixed z*3 lines drift)
#   [7] panel-b highlight marks the strongest-agreeing pair dynamically
# Reads tables/lens_convergence.csv. Saves figures/published/fig10_lens_consensus.{png,pdf}
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
KPIS = ['EUI', 'IDD', 'TL']
CITY = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
        '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
LENSES = ['sobol_drivers', 'causal_parents', 'shap_top3', 'symbolic_vars']
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
IN_LABEL = ['W\u00b7type', 'W\u00b7thk', 'W\u00b7melt', 'W\u00b7pos',
            'R\u00b7type', 'R\u00b7thk', 'R\u00b7melt', 'R\u00b7pos']
JPAIRS = [('J_sobol_causal', 'Sobol\u2013Causal'), ('J_sobol_shap', 'Sobol\u2013SHAP'),
          ('J_sobol_symbolic', 'Sobol\u2013Symbolic'), ('J_causal_shap', 'Causal\u2013SHAP'),
          ('J_causal_symbolic', 'Causal\u2013Symbolic'), ('J_shap_symbolic', 'SHAP\u2013Symbolic')]

df = pd.read_csv(BASE + '/tables/lens_convergence.csv').drop_duplicates(subset=['zone', 'kpi'])
df['pair'] = df['zone'] + '_' + df['kpi']
CELLS = [(z, k) for z in ZONES for k in KPIS
         if (z, k) not in [('0B', 'IDD'), ('2B', 'TL'), ('4B', 'TL')]]
assert len(CELLS) == 21, f'expected 21 modeled cells, got {len(CELLS)}'
pairs = [f'{z}_{k}' for z, k in CELLS]
missing = [p for p in pairs if p not in set(df.pair)]
assert not missing, f'lens_convergence.csv missing rows (run Cell 13 first): {missing}'

def parse(s):
    if pd.isna(s) or str(s).strip().lower() in ('', 'nan'):
        return set()
    return set(str(s).split('|'))

M = np.zeros((len(INPUTS), len(pairs)), dtype=int)
for j, pr in enumerate(pairs):
    sets = [parse(df[df.pair == pr].iloc[0][L]) for L in LENSES]
    for i, v in enumerate(INPUTS):
        M[i, j] = sum(v in s for s in sets)

cmap = ListedColormap(['#f0f0f0', '#d5e4f2', '#8bc1e4', '#327fb9', '#042c67'])
norm = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

fig = plt.figure(figsize=(13.0, 5.2))
gs = fig.add_gridspec(1, 4, width_ratios=[2.4, 0.05, 0.35, 1.0], wspace=0.08)

ax = fig.add_subplot(gs[0, 0])    # heatmap
cax = fig.add_subplot(gs[0, 1])   # colorbar (dedicated column, no label)
ax2 = fig.add_subplot(gs[0, 3])   # bars

# ---------- panel (a): consensus heatmap ----------
im = ax.imshow(M, aspect='auto', cmap=cmap, norm=norm)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if v >= 1:
            ax.text(j, i, str(v), ha='center', va='center', fontsize=6.5,
                    color='white' if v >= 3 else '#404040')

blocks, pos = [], 0
for z in ZONES:
    n = sum(1 for k in KPIS if (z, k) not in
            [('0B', 'IDD'), ('2B', 'TL'), ('4B', 'TL')])
    blocks.append((pos, n))
    pos += n
for pos0, n in blocks:
    ax.axvline(pos0 - 0.5, color='k', lw=1.4)
    for kk in range(1, n):
        ax.axvline(pos0 + kk - 0.5, color='k', lw=0.35, alpha=0.7)
ax.axvline(len(pairs) - 0.5, color='k', lw=1.4)

ax.set_xticks([pos0 + (n - 1) / 2 for pos0, n in blocks])
ax.set_xticklabels([f'{z}\n{CITY[z]}' for z in ZONES], fontsize=7.5)
ax.set_yticks(range(len(INPUTS)))
ax.set_yticklabels(IN_LABEL, fontsize=8.5)
ax.tick_params(length=0)
ax.set_title('(a)  Lenses agreeing on each input per zone\u2013KPI', fontsize=10, loc='left')

cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
cbar.set_ticklabels(['0', '1', '2', '3', '4'])
cbar.ax.tick_params(labelsize=8, length=0)

# ---------- panel (b): pairwise Jaccard ----------
means = {lab: df[c].mean() for c, lab in JPAIRS}
order = sorted(JPAIRS, key=lambda x: means[x[1]])
y = np.arange(len(order))
vals = [means[lab] for _, lab in order]
best_lab = max(order, key=lambda x: means[x[1]])[1]
cols = ['#042c67' if lab == best_lab else '#8bc1e4' for _, lab in order]
ax2.barh(y, vals, color=cols, edgecolor='k', lw=0.4, height=0.62)
for yi, v in zip(y, vals):
    ax2.text(v + 0.02, yi, f'{v:.2f}', va='center', fontsize=9)
ax2.set_yticks(y)
ax2.set_yticklabels([lab for _, lab in order], fontsize=9)
ax2.tick_params(axis='y', pad=10)   # keep labels clear of the colorbar ticks
ax2.set_xlim(0, 0.92)
ax2.set_xlabel('Mean pairwise Jaccard (21 modeled cells)', fontsize=9)
ax2.set_title('(b)  Lens-pair agreement', fontsize=10, loc='left')
ax2.grid(axis='x', color='0.85', lw=0.5)
ax2.set_axisbelow(True)

# ============================ SAVE & DOWNLOAD FIX ============================
out = BASE + '/figures/published'
os.makedirs(out, exist_ok=True)

png_path = os.path.join(out, 'fig10_lens_consensus.png')
pdf_path = os.path.join(out, 'fig10_lens_consensus.pdf')

fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print('saved:', png_path, '+', pdf_path)

# دانلود امن مستقیماً از مسیر گوگل درایو
files.download(png_path)
files.download(pdf_path)

print('\nCaption Fig 10:')
print('Figure 10. Convergence of four independent attribution lenses across '
      'the 21 modeled zone-KPI cells. (a) Number of lenses (stability-filtered '
      'Sobol driver set, consensus causal parents, top-3 SHAP variables, '
      'symbolic-regression consensus) that include each input in each cell. '
      '(b) Mean pairwise Jaccard overlap of every lens pair over the 21 '
      'modeled cells. The three combinations excluded in Section 3.1 are '
      'omitted.')
print('\nDONE 13b - download the figure and paste this output back.')
