# ============================================================================
# scripts/cell11b_fig9_shap_agreement.py
# Cell 11b — Fig 9: SHAP attribution + SHAP-Sobol agreement
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 17
# (152 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 11b - FIG 9: SHAP attribution + SHAP-Sobol agreement (run AFTER Cell 11)
# (a) SHAP mean|value| share heatmap: 21 modeled cells x 8 inputs
#     (red frame = top-3 SHAP variables = the shap_top3 lens for Fig 10)
# (b) SHAP share vs Sobol S_T (bootstrap mean): 168 points, per-KPI Spearman
# Reads: tables/shap_attribution.csv + tables/sobol_indices.csv
# Saves: figures/published/fig9_shap_attribution.{png,pdf}
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
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
sh = pd.read_csv(BASE + '/tables/shap_attribution.csv')
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
IN_LABEL = ['W-type', 'W-thickness', 'W-melt', 'W-position',
            'R-type', 'R-thickness', 'R-melt', 'R-position']
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
KPIS = ['EUI', 'IDD', 'TL']
EXCLUDED = [('0B', 'IDD'), ('2B', 'TL'), ('4B', 'TL')]
CELLS = [(z, k) for z in ZONES for k in KPIS if (z, k) not in EXCLUDED]
assert len(CELLS) == 21, f'expected 21 cells, got {len(CELLS)}'
assert len(sh) == 168, f'expected 168 shap rows, got {len(sh)}'

sobv = sob[~sob.input.str.startswith('GROUP')][['zone', 'kpi', 'input', 'ST', 'ST_boot_mean']]

# ---- panel (a) matrix ----
sh_idx = sh.set_index(['zone', 'kpi', 'input'])
M = np.full((len(INPUTS), len(CELLS)), np.nan)
TOP = np.zeros_like(M, dtype=bool)
for j, (z, k) in enumerate(CELLS):
    for i, v in enumerate(INPUTS):
        M[i, j] = sh_idx.loc[(z, k, v), 'shap_share']
        TOP[i, j] = bool(sh_idx.loc[(z, k, v), 'is_top3'])

# ---- panel (b) merged frame + agreement diagnostics ----
mg = sh.merge(sobv, on=['zone', 'kpi', 'input'], how='inner')
assert len(mg) == 168, f'shap-sobol merge incomplete: {len(mg)} (check sobol_indices.csv)'
diag = {}
for k in KPIS:
    s = mg[mg.kpi == k]
    diag[k] = spearmanr(s.shap_share, s.ST_boot_mean)
rho_all, _ = spearmanr(mg.shap_share, mg.ST_boot_mean)
agree_cells = []
for z, k in CELLS:
    s = mg[(mg.zone == z) & (mg.kpi == k)]
    if s.loc[s.shap_share.idxmax(), 'input'] == s.loc[s.ST.idxmax(), 'input']:
        agree_cells.append((z, k))

# ============================ figure ============================
KPI_COL = {'EUI': '#0072B2', 'IDD': '#D55E00', 'TL': '#009E73'}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                              gridspec_kw={'width_ratios': [2.5, 1.15]},
                              constrained_layout=True)

cmap = plt.cm.Blues.copy()
cmap.set_bad('#f0f0f0')
im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=np.nanmax(M))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j] >= 0.10:
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if M[i, j] > 0.6 * np.nanmax(M) else '#303030')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if TOP[i, j]:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='red', lw=1.1))
# separators between KPI blocks (EUI|IDD|TL), accounting for excluded cells
block_edges = [CELLS.index(('0B', 'TL')) - 0.5, CELLS.index(('1B', 'TL')) - 0.5,
               CELLS.index(('4A', 'TL')) - 0.5]
for x in block_edges:
    ax.axvline(x, color='k', lw=1.3)
for lab, c0, c1 in [('EUI', 0, CELLS.index(('0B', 'TL'))),
                    ('IDD', CELLS.index(('0B', 'TL')), CELLS.index(('4A', 'TL'))),
                    ('TL', CELLS.index(('4A', 'TL')), len(CELLS))]:
    ax.text((c0 + c1) / 2 - 0.5, -1.15, lab, ha='center', va='center',
            fontsize=9, fontweight='bold')
ax.set_xticks(range(len(CELLS)))
ax.set_xticklabels([z for z, _ in CELLS], fontsize=7.5)
ax.set_yticks(range(len(INPUTS)))
ax.set_yticklabels(IN_LABEL, fontsize=8.5)
ax.tick_params(length=0)
ax.set_title('(a)  SHAP share per input and zone–KPI cell', fontsize=10, loc='left')
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)

for k in KPIS:
    s = mg[mg.kpi == k]
    ax2.scatter(s.ST_boot_mean, s.shap_share, s=16, alpha=0.65,
                color=KPI_COL[k], edgecolors='none',
                label=f'{k}  (\u03c1={diag[k][0]:.2f})')
lim = max(mg.ST_boot_mean.max(), mg.shap_share.max()) * 1.06
ax2.plot([0, lim], [0, lim], ls='--', c='0.6', lw=0.9)
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel('Sobol $S_T$ (bootstrap mean)', fontsize=9)
ax2.set_ylabel('SHAP share (mean |SHAP|, normalized)', fontsize=9)
ax2.set_title(f'(b)  SHAP vs Sobol  (pooled \u03c1={rho_all:.2f})', fontsize=10, loc='left')
ax2.legend(fontsize=8.5, frameon=False, loc='upper left')
ax2.grid(color='0.88', lw=0.5)
ax2.set_axisbelow(True)

# ============================ SAVE & DOWNLOAD FIX ============================
out_dir = BASE + '/figures/published'
os.makedirs(out_dir, exist_ok=True)

png_path = os.path.join(out_dir, 'fig9_shap_attribution.png')
pdf_path = os.path.join(out_dir, 'fig9_shap_attribution.pdf')

fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')

# دانلود امن و مستقیم از مسیر گوگل درایو
files.download(png_path)

# ============================ diagnostics ============================
print('=' * 72)
print('SHAP-Sobol agreement diagnostics (for Section 3.4 text)')
print('=' * 72)
for k in KPIS:
    print(f'  {k}: Spearman rho(shap_share, ST_boot_mean) = {diag[k][0]:.3f} (p={diag[k][1]:.2g})')
print(f'  pooled rho = {rho_all:.3f}')
print(f'  rank-1 agreement (SHAP argmax == Sobol ST argmax): {len(agree_cells)}/21')
if len(agree_cells) < 21:
    dis = [(z, k) for (z, k) in CELLS if (z, k) not in agree_cells]
    print(f'  disagreeing cells: {dis}')
t1 = sh.sort_values('shap_rank').groupby(['zone', 'kpi']).first()
print(f'  mean top-1 SHAP share = {t1.shap_share.mean():.3f} '
      f'(min {t1.shap_share.min():.3f}, max {t1.shap_share.max():.3f})')

print('\nCaption Fig 9:')
print('Fig. 9. SHAP attribution on the adopted surrogates. (a) Mean |SHAP| share '
      'per input across the 21 modeled zone-KPI cells (red frames: top-3 SHAP '
      'variables per cell). (b) SHAP share versus Sobol S_T (bootstrap mean) for '
      'all 168 variable-cell pairs; per-KPI Spearman rank correlations in the '
      'legend. Excluded combinations (Section 3.1) are omitted.')
print('\nDONE 11b - download the figure and paste this output back.')
