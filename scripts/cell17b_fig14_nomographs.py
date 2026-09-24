# ============================================================================
# scripts/cell17b_fig14_nomographs.py
# Cell 17b (v3) — Fig 14: design nomographs
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 34
# (187 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 17b (v3) - FIG 14: DESIGN NOMOGRAPHS  (run AFTER Cell 17a)
# Journal-grade small multiples: rows = four representative zones across the
# climate gradient (0B hot desert, 2B hot semi-arid, 3B temperate, 5C cold),
# columns = the 8 design variables. Each panel: marginal % change of the KPI
# vs variable level, all other inputs held at the zone median.
#   * x-ticks are the OBSERVED levels of each column read from the curve CSV
#     (v2 bug fixed: the melt-point columns now show all five levels incl.
#     29 C - the old hard-coded 0..3 axis silently clipped level 4)
#   * ticks annotated with the physical value where unique (thickness in mm,
#     position ext/mid/int); PCM type and melt point stay as level codes
#     (decode in Table 6)
#   * dual axis per panel: EUI and IDD on the LEFT, thermal lag on the RIGHT
#     (small lag baseline inflates relative changes - stated in the caption,
#     absolute spans in Table A24)
#   * solid thick = Sobol driver of that cell; dashed = secondary;
#     grey panels = melt-point variables (the null result of 3.2-3.4)
#   * no figure number in the artwork - numbering lives in the caption
# Reads: tables/nomograph_curves.csv + level_decode_map.csv + sobol_indices.csv
# Saves: figures/published/fig14_nomographs.{png(600dpi),pdf}
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

BASE = '/content/drive/MyDrive/PCM_Study'
cur = pd.read_csv(BASE + '/tables/nomograph_curves.csv')
dec = pd.read_csv(BASE + '/tables/level_decode_map.csv')
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')

ZROW = ['0B', '2B', '3B', '5C']
ZLAB = {'0B': '0B  Bandar Abbas\n(hot desert)',
        '2B': '2B  Kashan\n(hot semi-arid)',
        '3B': '3B  Tehran\n(temperate)',
        '5C': '5C  Ardebil\n(cold)'}
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level',
          'position_wall', 'pcm_type_roof', 'thickness_roof_level',
          'melting_point_roof_level', 'position_roof']
VLAB = ['Wall\nPCM type', 'Wall\nthickness', 'Wall\nmelt point', 'Wall\nposition',
        'Roof\nPCM type', 'Roof\nthickness', 'Roof\nmelt point', 'Roof\nposition']
KPIS = ['EUI', 'IDD', 'TL']
KCOL = {'EUI': '#2166ac', 'IDD': '#1b7837', 'TL': '#e08214'}
KMARK = {'EUI': 'o', 'IDD': 's', 'TL': '^'}
MELTVARS = {'melting_point_wall_level', 'melting_point_roof_level'}

# ---- decode lookup: (group, level) -> physical value ----
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v
D = {(r.variable, int(r.level)): _num(r.value) for r in dec.itertuples()}

def tick_labels(c, levs):
    """single-line compact labels: physical value where unique, level code
    where the decode is product-dependent (melt point -> Table 6)."""
    lab = []
    for lv in levs:
        if c in ('pcm_type_wall', 'pcm_type_roof'):
            lab.append(f'{lv}')                     # decode: Table 6
        elif c in ('thickness_wall_level', 'thickness_roof_level'):
            v = D.get(('thickness_mm', int(lv)))
            lab.append(f'{v:g}' if v is not None else f'{lv}')   # mm
        elif c in ('position_wall', 'position_roof'):
            p = str(D.get(('position', int(lv)), ''))
            lab.append({'exterior': 'ext', 'middle': 'mid',
                        'interior': 'int'}.get(p, p) or f'{lv}')
        else:                                       # melt point: level codes
            lab.append(f'{lv}')
    return lab

# driver sets per (zone, kpi) from the stability-filtered Sobol table
drv = sob[sob.in_driver_set == True].groupby(['zone', 'kpi']).input.agg(set).to_dict()
modelled = cur.groupby('zone').kpi.agg(set).to_dict()

fig, axes = plt.subplots(len(ZROW), 8, figsize=(15.0, 9.8))
for ri, z in enumerate(ZROW):
    for ci, c in enumerate(INPUTS):
        ax = axes[ri, ci]
        levs = sorted(cur[(cur.zone == z) & (cur.variable == c)].level.unique())
        ax.set_xticks(levs)
        ax.set_xticklabels(tick_labels(c, levs), fontsize=5.9)
        ax.set_xlim(min(levs) - 0.4, max(levs) + 0.4)
        ax.axhline(0, color='0.6', lw=0.7, zorder=1)
        if c in MELTVARS:
            ax.set_facecolor('#f2f2f2')
        ax2 = ax.twinx() if 'TL' in modelled.get(z, set()) else None

        for k in KPIS:
            s = cur[(cur.zone == z) & (cur.kpi == k) & (cur.variable == c)]
            if not len(s):
                continue
            s = s.sort_values('level')
            target = ax2 if (k == 'TL' and ax2 is not None) else ax
            grey = c in MELTVARS
            solid = (not grey) and (c in drv.get((z, k), set()))
            target.plot(s.level, s['pct_change'],
                        color='0.62' if grey else KCOL[k],
                        lw=1.4 if grey else (2.2 if solid else 0.9),
                        ls='--' if (not grey and not solid) else '-',
                        alpha=0.95 if (grey or solid) else 0.8,
                        marker=KMARK[k], ms=3.4, zorder=3 if solid else 2)

        if ri == 0:
            ax.set_title(VLAB[ci], fontsize=9)
        if ci == 0:
            ax.set_ylabel(ZLAB[z], fontsize=8.5)
        if ax2 is not None:
            ax2.tick_params(labelsize=6.5, colors='0.35')
            if ri == len(ZROW) - 1 and ci == len(INPUTS) - 1:
                ax2.set_ylabel('TL (right)', fontsize=7, color='0.35')
        ax.tick_params(labelsize=6.5)

for ax in axes[-1]:
    ax.set_xlabel('level', fontsize=7.5)

from matplotlib.lines import Line2D
leg = [Line2D([], [], color=KCOL['EUI'], lw=2, marker='o', ms=4,
              label='EUI - driver (left)'),
       Line2D([], [], color=KCOL['IDD'], lw=2, marker='s', ms=4,
              label='IDD - driver (left)'),
       Line2D([], [], color=KCOL['TL'], lw=2, marker='^', ms=4,
              label='TL - driver (right)'),
       Line2D([], [], color='0.4', lw=0.9, ls='--', label='secondary (dashed)'),
       Line2D([], [], color='0.62', lw=1.4, marker='o', ms=4,
              label='melt point (null)')]
fig.legend(handles=leg, loc='lower center', ncol=5, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.text(0.5, -0.036, 'axis labels: thickness in mm; ext / mid / int = '
         'exterior / middle / interior; PCM type and melt point are level '
         'codes (physical decode in Table 6)',
         ha='center', fontsize=7.8, color='0.25')
fig.suptitle('Marginal design effects of the eight envelope variables '
             '(%, vs the zone-median design)', fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

# ---- self-checks: the level-clipping bug can never return silently ----
for c in sorted(MELTVARS):
    mx = int(cur[cur.variable == c].level.max())
    print(f'self-check {c}: max observed level = {mx} '
          f'({"OK - all 5 levels incl. 29 C" if mx == 4 else "CHECK: < 4"})')
for z in ZROW:
    miss = [k for k in KPIS if k not in modelled.get(z, set())]
    if miss:
        print(f'self-check {z}: excluded KPI(s) {miss} - curves absent (3.1)')

os.makedirs(BASE + '/figures/published', exist_ok=True)
out = BASE + '/figures/published/fig14_nomographs'
fig.savefig(f'{out}.png', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}.pdf', bbox_inches='tight')
print(f'saved: {out}.png (600 dpi) + {out}.pdf')
files.download(f'{out}.png')

print('\nCaption Fig 14 (locked):')
print('Figure 14. Design nomographs for PCM-integrated envelopes in four '
      'representative climates. Each panel shows the marginal effect of one '
      'design variable on a KPI (percent change from the all-median design '
      'of the zone), with all other inputs held at the zone median level; '
      'rows are climate regimes (hot desert, hot semi-arid, temperate, '
      'cold), columns the eight design variables; axes carry the physical '
      'value where it is unique (thickness in mm, position as '
      'exterior/middle/interior) and the level code where the decode is '
      'product-dependent (PCM type, melt point - Table 6). EUI and IDD are '
      'read on the left axis of each panel and thermal lag on the right; '
      'the lag baseline is small, which inflates its relative changes, and '
      'absolute spans are tabulated in Table A24. Solid lines mark variables '
      'in the stability-filtered Sobol driver set of that cell (Section '
      '3.2); dashed lines are modelled secondary variables; grey panels '
      'give the melt-point variables, whose flat responses reproduce the '
      'null result of Sections 3.2-3.4. Absent curves mark the Section 3.1 '
      'exclusions. Four representative zones are shown; the complete '
      'eight-zone atlas, with each curve\u2019s leverage span, is tabulated '
      'in Table A24.')

print('\nDONE 17b - paste this output back (and send fig14_nomographs.png).')
