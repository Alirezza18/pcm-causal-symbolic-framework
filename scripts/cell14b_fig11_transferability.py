# ============================================================================
# scripts/cell14b_fig11_transferability.py
# Cell 14b — Fig 11: cross-climate shape transferability
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 23
# (166 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 14b - FIG 11: CROSS-CLIMATE SHAPE TRANSFERABILITY
# (run AFTER Cell 14a, which writes tables/transfer_shape_{EUI,IDD,TL}.csv)
#   [1] Fig 11 numbering (9 = SHAP, 10 = four-lens consensus)
#   [2] regime separators after 2B and after 3B (hot | temperate | cold)
#   [3] means computed FROM the matrices - no hardcoded values
#   [4] 3.1 exclusion NaN pattern VERIFIED (EUI: full 8x8; IDD: row+col 0B
#       NaN; TL: rows+cols 2B and 4B NaN) - anything else fails loudly
#   [5] NaN-safe diagnostics: own-zone reference, within- vs cross-regime,
#       3x3 regime-pair means, best/worst donor->target pairs
#   [6] excluded matrix cells render light gray, no 'nan' labels
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
KPIS = ['EUI', 'IDD', 'TL']
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
CITY = {'0B': 'Bandar Abbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
        '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}
REG = [[0, 1, 2], [3, 4], [5, 6, 7]]          # hot | temperate | cold
REG_LAB = ['hot (0B-2B)', 'temperate (3A-3B)', 'cold (4A-5C)']
SEP = [3, 5]                                   # rules AFTER 2B and AFTER 3B
reg_of = np.zeros(8, dtype=int)
for r, idx in enumerate(REG):
    reg_of[idx] = r
# excluded (zone, KPI) cells of Section 3.1 -> NaN rows+cols in the matrices
EXCL_ZONE = {'EUI': [], 'IDD': ['0B'], 'TL': ['2B', '4B']}

M = {}
for k in KPIS:
    df = pd.read_csv(f'{BASE}/tables/transfer_shape_{k}.csv', index_col=0)
    dm = df.reindex(index=ZONES, columns=ZONES)
    for z in EXCL_ZONE[k]:
        assert dm.loc[z].isna().all(), f'{k}: row {z} must be all-NaN (3.1)'
        assert dm[z].isna().all(), f'{k}: col {z} must be all-NaN (3.1)'
    for z in ZONES:
        if z in EXCL_ZONE[k]:
            continue
        n_exp = 8 - len(EXCL_ZONE[k])
        assert dm.loc[z].notna().sum() == n_exp, \
            f'{k}: row {z} must have exactly {n_exp} values'
    M[k] = dm.values.astype(float)
print('3.1 exclusion NaN pattern verified for all three matrices.')

# ---------------- diagnostics for the Section 3.5 text ----------------
print('=' * 76)
print('DIAGNOSTICS (for Section 3.5)')
print('=' * 76)
eye = np.eye(8, dtype=bool)
same_reg = (reg_of[:, None] == reg_of[None, :]) & ~eye
for k in KPIS:
    mat = M[k]
    diag = np.diag(mat)
    od = mat[~eye]
    sr = same_reg & ~np.isnan(mat)
    print(f'\n{k}:')
    print(f'  own-zone (donor=target) R2: mean={np.nanmean(diag):.3f} '
          f'min={np.nanmin(diag):.3f} max={np.nanmax(diag):.3f}')
    print(f'  transferred R2: mean={np.nanmean(od):.3f} '
          f'| within-regime={np.nanmean(mat[sr]):.3f} '
          f'| cross-regime={np.nanmean(mat[~eye & ~sr & ~np.isnan(mat)]):.3f} '
          f'(ratio {np.nanmean(mat[sr]) / np.nanmean(mat[~eye & ~sr & ~np.isnan(mat)]):.2f}x)')
    print('  regime-pair means (donor rows x target cols):')
    for r, ri in enumerate(REG):
        cells = []
        for c, ci in enumerate(REG):
            blk = mat[np.ix_(ri, ci)]
            if r == c:
                blk = blk[~np.eye(len(ri), dtype=bool)]
            cells.append(f'{np.nanmean(blk):.3f}' if np.isfinite(blk).any()
                         else '  n/a')
        print(f'    {REG_LAB[r]:18s} -> ' + '  '.join(cells))
    off = mat.copy()
    np.fill_diagonal(off, np.nan)
    i, j = np.unravel_index(np.nanargmax(off), off.shape)
    off2 = off.copy()
    off2[i, j] = np.nan
    i2, j2 = np.unravel_index(np.nanargmin(off2), off2.shape)
    print(f'  best transfer: {ZONES[i]}->{ZONES[j]} {off[i, j]:.3f}   '
          f'worst: {ZONES[i2]}->{ZONES[j2]} {off2[i2, j2]:.3f}')

# ---------------- figure ----------------
cmap = plt.cm.Blues.copy()
cmap.set_bad('#e8e8e8')                     # excluded cells -> light gray

fig = plt.figure(figsize=(16.0, 4.9))
gs = fig.add_gridspec(1, 4, width_ratios=[2.2, 2.2, 2.2, 0.14], wspace=0.16)

axs, ims = [], []
for p, k in enumerate(KPIS):
    ax = fig.add_subplot(gs[0, p])
    axs.append(ax)
    mat = np.ma.masked_invalid(M[k])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ims.append(im)
    mt = np.nanmean(mat.filled(np.nan)[~eye])
    for i in range(8):
        for j in range(8):
            v = M[k][i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.5, color='white' if v >= 0.60 else 'black')
    for i in range(8):          # within-zone reference (donor = target)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='black', lw=1.6))
    for e in SEP:               # regime separators after 2B and after 3B
        ax.axvline(e - 0.5, color='black', lw=1.4)
        ax.axhline(e - 0.5, color='black', lw=1.4)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{z} {CITY[z]}' for z in ZONES], fontsize=7,
                       rotation=30, ha='right')
    ax.set_yticks(range(8))
    ax.set_yticklabels(ZONES, fontsize=8)
    ax.set_title(f'({chr(97 + p)}) {k}   \u00b7   mean transferred '
                 f'R\u00b2 = {mt:.3f}', fontsize=10, pad=12)

axs[0].set_ylabel('Source zone (donor)', fontsize=9)

cax = fig.add_subplot(gs[0, 3])
cb = fig.colorbar(ims[2], cax=cax)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_label('Shape-transfer R\u00b2 (level-calibrated)', fontsize=8.5,
             rotation=270, labelpad=14)
cb.ax.tick_params(labelsize=8)

# ============================ SAVE & DOWNLOAD FIX ============================
out = BASE + '/figures/published'
os.makedirs(out, exist_ok=True)

png_path = os.path.join(out, 'fig11_transfer.png')
pdf_path = os.path.join(out, 'fig11_transfer.pdf')

fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'\nsaved: {png_path} + {pdf_path}')

# دانلود امن مستقیماً از مسیر گوگل درایو
files.download(png_path)
files.download(pdf_path)

print('\nCaption Fig 11 (provisional - finalized after verification):')
print('Figure 11. Cross-climate shape transferability of the adopted '
      'surrogates. Rows give the donor (training) climate zone, columns the '
      'target zone; cell values are the R2 of the donor surrogate on the '
      'target designs after level recalibration, which removes the '
      'between-climate offset in mean response so that values isolate the '
      'transferred response shape. Black frames mark within-zone performance '
      '(the hold-out reference); heavy rules group the hot (0B-2B), temperate '
      '(3A-3B) and cold (4A-5C) regimes; gray cells are the combinations '
      'excluded in Section 3.1. Per-pair values are tabulated in Table A19.')
print('\nDONE 14b - download the figure and paste this output back.')
