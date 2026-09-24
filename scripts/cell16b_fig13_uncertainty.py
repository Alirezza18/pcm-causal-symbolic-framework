# ============================================================================
# scripts/cell16b_fig13_uncertainty.py
# Cell 16b — Fig 13: surrogate prediction uncertainty
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 30
# (118 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 16b - FIG 13: SURROGATE PREDICTION UNCERTAINTY (run AFTER Cell 16a)
#   (a) Per-cell bubble grid: bubble size = NORMALIZED conformal width
#       (width / range(y)); colour = achieved coverage vs 90% nominal
#       (blue >= 90%, red < 90%); grey hatch = Section 3.1 exclusions.
#   (b) Mean uncertainty decomposition per KPI: epistemic (30 bootstrap
#       refits) vs aleatoric (residual noise) variance fractions.
# Reads: tables/uq_conformal.csv + tables/uq_vardec.csv (Cell 16a)
# Saves: figures/published/fig13_uncertainty.{png,pdf}
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
from matplotlib.lines import Line2D
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
cf = pd.read_csv(BASE + '/tables/uq_conformal.csv')
vd = pd.read_csv(BASE + '/tables/uq_vardec.csv')

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITYD = {'0B': 'Bandar\nAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
KPIS  = ['EUI', 'IDD', 'TL']

assert len(cf) == 21 and len(vd) == 21, \
    f'expected 21 modeled cells, got {len(cf)} / {len(vd)} - run Cell 16a first'
assert set(cf.kpi) <= set(KPIS) and set(cf.zone) <= set(ZONES)

fig = plt.figure(figsize=(13.0, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

# ------------------------------- panel (a) -------------------------------
ax = fig.add_subplot(gs[0, 0])
cfc = cf.set_index(['zone', 'kpi'])
SMAX = cf.width_norm.max()
for xi, z in enumerate(ZONES):
    for yi, k in enumerate(KPIS):
        if (z, k) in cfc.index:
            r = cfc.loc[(z, k)]
            size = 40 + 900 * (r.width_norm / SMAX) ** 2
            col = '#2166ac' if r.coverage >= 0.90 else '#b2182b'
            ax.scatter(xi, yi, s=size, c=col, alpha=0.85, zorder=3,
                       edgecolors='black', linewidths=0.6)
            ax.text(xi, yi - 0.30, f'{r.coverage:.2f}', ha='center', va='top',
                    fontsize=6.3, color=col, zorder=4)
        else:
            ax.add_patch(Rectangle((xi - 0.42, yi - 0.28), 0.84, 0.56,
                                   facecolor='0.92', edgecolor='0.62',
                                   hatch='///', lw=0.5, zorder=2))
ax.set_xticks(range(8)); ax.set_xticklabels([CITYD[z] for z in ZONES], fontsize=7.5)
ax.set_yticks(range(3)); ax.set_yticklabels(KPIS, fontsize=9)
ax.set_xlim(-0.6, 7.6); ax.set_ylim(2.7, -0.75)   # EUI top row
ax.set_xlabel('Climate zone', fontsize=9)
ax.set_title('(a) Conformal 90% prediction intervals per cell', fontsize=10, pad=10)
ax.text(3.5, -0.62, 'bubble area ∝ normalized width', ha='center',
        fontsize=7.5, style='italic', color='0.35')
leg = [Line2D([], [], marker='o', ls='', mfc='#2166ac', mec='k', ms=9, label='coverage ≥ 90%'),
       Line2D([], [], marker='o', ls='', mfc='#b2182b', mec='k', ms=9, label='coverage < 90%'),
       Line2D([], [], marker='s', ls='', mfc='0.92', mec='0.62', ms=10, label='excluded (Sec 3.1)')]
ax.legend(handles=leg, loc='lower left', fontsize=7.2, framealpha=0.9)

# ------------------------------- panel (b) -------------------------------
ax2 = fig.add_subplot(gs[0, 1])
gm = vd.groupby('kpi')[['f_model', 'f_noise']].mean().reindex(KPIS)
x = np.arange(3)
ax2.bar(x, gm.f_model, width=0.55, color='#4393c3', label='epistemic (model refit)')
ax2.bar(x, gm.f_noise, width=0.55, bottom=gm.f_model, color='#f4a582',
        label='aleatoric (residual noise)')
for xi, k in zip(x, KPIS):
    if gm.f_model[k] >= 0.08:
        ax2.text(xi, gm.f_model[k] / 2, f'{gm.f_model[k]*100:.0f}%',
                 ha='center', va='center', fontsize=8.5, color='white',
                 fontweight='bold')
    else:
        ax2.annotate(f'{gm.f_model[k]*100:.0f}%', (xi, gm.f_model[k]),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', fontsize=7.5, color='#2166ac')
    ax2.text(xi, gm.f_model[k] + gm.f_noise[k] / 2, f'{gm.f_noise[k]*100:.0f}%',
             ha='center', va='center', fontsize=8.5, color='0.15')
ax2.set_xticks(x); ax2.set_xticklabels(KPIS, fontsize=9)
ax2.set_ylim(0, 1.0); ax2.set_ylabel('fraction of prediction variance', fontsize=9)
ax2.set_title('(b) Uncertainty decomposition (mean across zones)', fontsize=10, pad=10)
ax2.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

os.makedirs(BASE + '/figures/published', exist_ok=True)
out = BASE + '/figures/published/fig13_uncertainty'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=300, bbox_inches='tight')
print(f'saved: {out}.png')

print('\nCaption Fig 13:')
print('Figure 13. Surrogate prediction uncertainty across climate zones. '
      '(a) Split-conformal 90% prediction intervals for every modeled cell: '
      'bubble area is proportional to the normalized interval width (2q-hat '
      'divided by the range of the observed KPI in that zone); colour marks '
      'achieved coverage against the 90% nominal (blue >= 90%, red < 90%), '
      'with the per-cell value printed; hatched grey cells are the Section '
      '3.1 exclusions. (b) Mean decomposition of the prediction variance '
      'into an epistemic component (30 paired bootstrap refits of the '
      'adopted surrogate family) and an aleatoric component (residual '
      'variance of the full-data fit), averaged across the eight zones per '
      'KPI. Per-cell values are given in Table A21.')

# files.download(f'{out}.png')
print('\nDONE 16b - paste this output back (and send fig13_uncertainty.png).')
