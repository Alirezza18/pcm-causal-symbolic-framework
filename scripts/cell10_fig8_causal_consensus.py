# ============================================================================
# scripts/cell10_fig8_causal_consensus.py
# Fig 8 — causal consensus dot/bubble matrix (unlabeled cell)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 14
# (81 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# REVISED FIG 8: Dot Matrix / Bubble Plot for Causal Consensus
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = '/content/drive/MyDrive/PCM_Study'
res_path = BASE + '/tables/causal_edges.csv'

if not os.path.exists(res_path):
    raise FileNotFoundError(f"فایل {res_path} یافت نشد. ابتدا Cell 10 را اجرا کنید.")

res = pd.read_csv(res_path)

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
LABELS = ['PCM type\n(W)', 'Thick.\n(W)', 'Melt pt.\n(W)', 'Position\n(W)',
          'PCM type\n(R)', 'Thick.\n(R)', 'Melt pt.\n(R)', 'Position\n(R)']

fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

for ax, kpi in zip(axs, ['EUI', 'IDD', 'TL']):
    ax.set_title(f'Target KPI: {kpi}', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(range(len(INPUTS)))
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_yticks(range(len(ZONES)))
    ax.set_yticklabels(ZONES, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    for i, z in enumerate(ZONES):
        sub = res[(res.zone == z) & (res.kpi == kpi)].set_index('input')
        for j, c in enumerate(INPUTS):
            if c in sub.index:
                freq = sub.loc[c, 'mean_freq']
                is_parent = sub.loc[c, 'causal_parent']

                # مقیاس‌دهی اندازه حباب متناسب با فرکانس
                size = max(freq * 350, 15)

                # تفکیک رنگ بر اساس Causal Parent بودن
                color = '#d73027' if is_parent else '#4575b4'
                alpha = 0.9 if is_parent else 0.35

                ax.scatter(j, i, s=size, color=color, alpha=alpha,
                           edgecolors='black', linewidth=0.8, zorder=2)

                # نمایش اعداد فرکانس روی حباب‌ها
                if freq > 0.05:
                    ax.text(j, i, f'{freq:.2f}', ha='center', va='center', fontsize=6,
                            fontweight='bold' if is_parent else 'normal',
                            color='white' if is_parent else 'black', zorder=3)
            else:
                ax.text(j, i, '-', ha='center', va='center', color='gray', fontsize=8, zorder=3)

    ax.invert_yaxis()  # نمایش اقلیم‌ها از بالا به پایین (0B تا 5C)

# راهنمای نمادها (Legend)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Consensus Causal Parent (freq >= 0.90 in >=2 algos)',
           markerfacecolor='#d73027', markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='Sub-threshold Edge Frequency',
           markerfacecolor='#4575b4', alpha=0.5, markersize=8, markeredgecolor='k')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=2, frameon=True, facecolor='white', edgecolor='none', fontsize=9.5)

plt.tight_layout()

# ذخیره خروجی جدید
out_dir = BASE + '/figures/published'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f'{out_dir}/fig8_causal_consensus.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig8_causal_consensus.pdf', bbox_inches='tight')
plt.show()

print("نمودار Dot Matrix با موفقیت جایگزین شد و در پوشه figures/published ذخیره گردید.")
