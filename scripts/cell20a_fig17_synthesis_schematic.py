# ============================================================================
# scripts/cell20a_fig17_synthesis_schematic.py
# Cell 20a — Fig 17: synthesis schematic (the certified chain)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 44
# (150 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 20a - FIG 17: SYNTHESIS SCHEMATIC (the certified chain; Section 4.1)
# The Discussion's visual anchor: the methodological pipeline as a chain of
# certificates, with the engine audit of Section 3.10 closing the loop -
# decisions preserved (solid), intervals repriced (dashed).
# Pure diagram: no inputs, no models, deterministic.
# Saves: figures/published/fig17_synthesis.{png(600dpi),pdf}
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'

BLUE, LBLUE, GREY, DGREY = '#2166ac', '#d6e4f0', '#f2f2f2', '#333333'
RED, LRED = '#cb181d', '#fde0dd'
GREEN = '#1b7837'

fig, ax = plt.subplots(figsize=(14.6, 8.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def box(x0, y0, x1, y1, title, lines, fc=LBLUE, ec=BLUE):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=0.6,rounding_size=1.2',
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text((x0 + x1) / 2, y1 - 2.6, title, ha='center', va='top',
            fontsize=9.3, fontweight='bold', color=DGREY, zorder=3)
    ax.text((x0 + x1) / 2, y1 - 7.0, '\n'.join(lines), ha='center', va='top',
            fontsize=7.8, color=DGREY, zorder=3, linespacing=1.45)

def arrow(x0, y0, x1, y1, color=DGREY, ls='-', lw=1.6, label=None,
          lx=None, ly=None, lfs=7.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=15, color=color, ls=ls, lw=lw,
                                 zorder=4))
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2,
                ly if ly is not None else (y0 + y1) / 2 + 1.2, label,
                ha='center', va='bottom', fontsize=lfs, color=color,
                style='italic', zorder=5)

# ---- Tier A: physics and design space (y 78-96) ----
box(2, 78, 30, 96, 'Design space & physics',
    ['8 level-coded variables · 14,400 grid',
     '~500 LHS designs per zone',
     '8 climates (0B-5C), one weather file each'])
box(36, 78, 64, 96, 'EnergyPlus campaign',
    ['24 zone-KPI cells -> 21 modeled',
     '(3.1: 0B IDD, 2B TL, 4B TL excluded)',
     'annual EUI, IDD, TL per design'])
box(70, 78, 98, 96, 'Zone surrogates',
    ['CatBoost adopted in 19/23 cells',
     'holdout delta: mean +0.019',
     '21/21 within 0.15 of CV'])
arrow(30, 87, 36, 87); arrow(64, 87, 70, 87)

# ---- Tier B: uncertainty and attribution (y 54-72) ----
box(2, 54, 30, 72, 'Certified error budget',
    ['90% split-conformal intervals',
     'internal coverage 88-93%',
     'epistemic fraction 74-100%'])
box(36, 54, 64, 72, 'Five attribution lenses',
    ['Sobol · causal · SHAP · symbolic',
     '126 lens pairs: mean J = 0.674',
     'convergence itself the finding'])
box(70, 54, 98, 72, 'Convergent hierarchy',
    ['wall/roof thickness + position lead',
     'melt point: null in 0/21 cells',
     '(five independent instruments)'])
arrow(84, 78, 84, 72, label='predict', ly=74.6)
arrow(64, 66, 70, 66)
arrow(30, 66, 36, 66, label='calibrate', ly=67.6)

# ---- Tier C: decisions (y 29-47) ----
box(2, 29, 30, 47, 'Multi-objective design',
    ['exact fronts: 1-178 points',
     '8 knees; loss <= 0.6% EUI',
     'NSGA-II/MOTPE HV >= 0.998 in 7/8'])
box(36, 29, 64, 47, 'Transfer law',
    ['chains 1B-2B-{3A,3B,4A}, 4B-5C',
     '0B an island (mean shape R2: EUI 0.286)',
     'IDD 0.426 · TL 0.125 - the inversion'])
box(70, 29, 98, 47, 'Rules & economics',
    ['2 families + position invariant',
     'risk-labeled (30% threshold)',
     'median saving $5/yr per 100 m2'])
arrow(84, 54, 84, 47, label='inform', ly=49.6)
arrow(64, 40, 70, 40); arrow(30, 40, 36, 40)

# ---- Tier D: engine audit (y 6-22) ----
box(14, 6, 48, 22, 'Engine audit (16 EnergyPlus runs)',
    ['8 knees + 8 pre-registered held-outs',
     'knee ranks preserved exactly (rho = 1.00)',
     'coverage 0.31 vs nominal 0.90'], fc=LRED, ec=RED)
box(58, 6, 92, 22, 'Demonstrated floors',
    ['energy 3.1% · discomfort 4.1%',
     'lag keeps its certified width',
     'applicability domain, on the record'],
    fc=GREY, ec='0.4')
arrow(70, 29, 44, 22, label='predict', lx=57, ly=24.2)
arrow(48, 14, 58, 14, color=RED, label='measure', ly=15.6)

# ---- feedback (dashed red): floors reprice the risk labels & economics ----
arrow(84, 22, 84, 29, color=RED, ls='--', lw=1.4)

# ---- green: decisions preserved (audit -> decision layer) ----
arrow(31, 22, 31, 29, color=GREEN, lw=1.4)

ax.text(50, 99.5, 'The certified chain: every layer checked by the next',
        ha='center', va='top', fontsize=11.5, fontweight='bold')
ax.text(50, -1.5, 'solid = production flow · red = the engine audit (measure) '
        'and its feedback (dashed: demonstrated floors reprice the risk labels, '
        'the economics, and the working intervals) · green = decisions '
        'preserved through the audit',
        ha='center', va='top', fontsize=7.8, color='0.25')

os.makedirs(BASE + '/figures/published', exist_ok=True)
out = BASE + '/figures/published/fig17_synthesis'
fig.savefig(f'{out}.png', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}.pdf', bbox_inches='tight')
print(f'saved: {out}.png (600 dpi) + {out}.pdf')
files.download(f'{out}.png')

print('\nCaption Fig 17 (locked):')
print('Figure 17. Synthesis of the certified chain. Each layer of the study '
      'hands its output to the next with an attached certificate: the '
      'level-coded design space and EnergyPlus campaign produce 21 modeled '
      'zone-KPI cells; zone surrogates (holdout-verified) carry a conformal '
      'error budget; five independent attribution lenses converge on a '
      'driver hierarchy with a melt-point null; exact Pareto fronts yield '
      'eight knee designs whose prescriptions export along climate-transfer '
      'chains and condense into two design families, a position invariant, '
      'risk-labeled rules, and tariff-scaled economics. The engine audit '
      '(sixteen out-of-sample EnergyPlus runs) closes the loop: it preserves '
      'every decision (rank agreement, green) while repricing the certified '
      'intervals at the demonstrated error floors (dashed red), which flow '
      'back into the risk labels and the economics.')

print('\nDONE 20a - paste this output back (and send fig17_synthesis.png).')
