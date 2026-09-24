# ============================================================================
# scripts/cell10_causal_discovery.py
# Cell 10 — causal discovery: PC + GES + NOTEARS, resume-safe
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 13
# (277 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 10 — PHASE 5: CAUSAL DISCOVERY (quality-first, resume-safe) [FIXED]
# PC + GES + NOTEARS x 500 row-bootstrap -> dual-consensus causal parents
# per (zone, KPI) -> cross-lens agreement vs Sobol drivers (Jaccard +
# Spearman) -> SHD sanity -> Fig 8.
# Saves: tables/causal_edges.csv, tables/causal_shd.csv,
#        tables/causal_parents.csv, tables/causal_summary.csv
#        figures/published/fig8_causal_consensus.png + .pdf
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

import time, datetime
import numpy as np
import pandas as pd

# ---- Fixed package installation & imports ----
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges
except ImportError:
    # [Colab shell] pip install git+https://github.com/cmu-phil/causal-learn.git -q
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges

try:
    from notears.linear import notears_linear
except ImportError:
    # [Colab shell] pip install notears -q
    from notears.linear import notears_linear

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')

# [1] canonical level column names
INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}

# [2] exclusions consistent with 3.1/3.2
SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}

N_BOOT = 500          # bootstrap resamples (matches Phase 4)
N_CHUNK = 50          # progress/save chunk
ALPHA = 0.05          # PC conditional-independence alpha
RNG = 42
NODE_T = 8            # target column index (9 nodes: 8 inputs + target)
P = len(INPUTS) + 1   # total nodes

def skeleton(M):
    A = np.abs(np.asarray(M)) > 0
    return (A | A.T).astype(int)

def run_algos(Xn):
    """Standardized (n,9) array -> {algo: skeleton} + {algo: directed matrix}."""
    sk, dirs = {}, {}
    try:
        try:
            cg = pc(Xn, ALPHA, 'fisherz', show_progress=False)
        except TypeError:
            cg = pc(Xn, ALPHA, 'fisherz')
        M = np.asarray(cg.G.graph)
        sk['pc'], dirs['pc'] = skeleton(M), M
    except Exception:
        sk['pc'], dirs['pc'] = np.zeros((P, P), int), np.zeros((P, P))
    try:
        rec, _ = ges(Xn, score_func='local_score_BIC', maxP=4)
        G = rec.G if hasattr(rec, 'G') else rec['G']
        M = np.asarray(G.graph)
        sk['ges'], dirs['ges'] = skeleton(M), M
    except Exception:
        sk['ges'], dirs['ges'] = np.zeros((P, P), int), np.zeros((P, P))
    try:
        W = notears_linear(Xn, lambda1=0.1, loss_type='l2',
                           max_iter=200, w_threshold=0.0)
        W = np.nan_to_num(W)
        thr = max(0.1 * np.abs(W).max(), 1e-4)
        M = W * (np.abs(W) > thr)
        sk['notears'], dirs['notears'] = skeleton(M), M
    except Exception:
        sk['notears'], dirs['notears'] = np.zeros((P, P), int), np.zeros((P, P))
    return sk, dirs

def boot_causal(b, Xv, yv):
    rng = np.random.RandomState(RNG + b)
    idx = rng.randint(0, len(Xv), size=len(Xv))
    D = np.column_stack([Xv.values[idx], yv.values[idx]]).astype(float)
    D = StandardScaler().fit_transform(D)
    sk, _ = run_algos(D)
    return {a: sk[a][:NODE_T, NODE_T] for a in ('pc', 'ges', 'notears')}

# ---- resume support (tolerates an empty file from an interrupted run) ----
res_path = BASE + '/tables/causal_edges.csv'
shd_path = BASE + '/tables/causal_shd.csv'
done = set()
if os.path.exists(res_path) and os.path.getsize(res_path) > 0:
    try:
        prev = pd.read_csv(res_path)
        done = set(zip(prev['zone'], prev['kpi']))
        print(f'resuming - {len(done)} zone-KPI already computed, keeping them in file')
    except pd.errors.EmptyDataError:
        print('causal_edges.csv is empty - starting fresh')

rows, shd_rows, par_rows = [], [], []
t0 = time.time()
n_cells = 0
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    X = g[INPUTS].reset_index(drop=True)
    for t in TARGETS:
        k = SHORT[t]
        if (zone, k) in done:
            print(f'{zone} {k} already done - skip'); continue
        if (city, k) in SKIP:
            print(f'{zone} {city:11s} {k}: SKIPPED - {SKIP[(city, k)]}'); continue
        n_cells += 1
        y = g[t].reset_index(drop=True)

        # ---- full-data graph: SHD between algos + sanity checks ----
        Df = np.column_stack([X.values, y.values]).astype(float)
        Df = StandardScaler().fit_transform(Df)
        skf, dirf = run_algos(Df)
        for a, b in (('pc', 'ges'), ('pc', 'notears'), ('ges', 'notears')):
            shd_rows.append({'zone': zone, 'kpi': k, 'algo_a': a, 'algo_b': b,
                             'shd': int(np.triu(np.abs(skf[a] - skf[b]), k=1).sum())})
        ii = {a: int(np.triu(skf[a][:8, :8], k=1).sum()) for a in ('pc', 'ges', 'notears')}
        rev = {a: int((np.abs(dirf[a]) > 0)[NODE_T, :8].sum()) for a in ('pc', 'ges', 'notears')}
        print(f'  {zone} {k} input-input edges {ii} (LHS->~0) | reverse edges {rev} (feed-forward->~0)', flush=True)

        # ---- 500-bootstrap edge frequencies, chunked ----
        freq = {a: np.zeros(len(INPUTS)) for a in ('pc', 'ges', 'notears')}
        for start in range(0, N_BOOT, N_CHUNK):
            end = min(start + N_CHUNK, N_BOOT)
            chunk = Parallel(n_jobs=2)(delayed(boot_causal)(b, X, y) for b in range(start, end))
            for r in chunk:
                for a in freq:
                    freq[a] += r[a]
            print(f'      {zone} {k} causal {end}/{N_BOOT} ({time.time()-t0:.0f}s)', flush=True)
        for a in freq:
            freq[a] /= N_BOOT

        agree = (freq['pc'] >= 0.90).astype(int) + (freq['ges'] >= 0.90).astype(int) \
                + (freq['notears'] >= 0.90).astype(int)
        causal = [c for j, c in enumerate(INPUTS) if agree[j] >= 2]

        for j, c in enumerate(INPUTS):
            rows.append({'zone': zone, 'city': city, 'kpi': k, 'input': c,
                         'pc_freq': round(freq['pc'][j], 4), 'ges_freq': round(freq['ges'][j], 4),
                         'notears_freq': round(freq['notears'][j], 4),
                         'mean_freq': round(float(np.mean([freq['pc'][j], freq['ges'][j],
                                                           freq['notears'][j]])), 4),
                         'n_algos_agree': int(agree[j]),
                         'causal_parent': bool(agree[j] >= 2)})
        par_rows.append({'zone': zone, 'city': city, 'kpi': k, 'causal_parents': '; '.join(causal)})
        pd.DataFrame(rows).to_csv(res_path, index=False)
        pd.DataFrame(shd_rows).to_csv(shd_path, index=False)
        pd.DataFrame(par_rows).to_csv(BASE + '/tables/causal_parents.csv', index=False)
        print(f'{zone} {city:11s} {k}: causal parents = {causal} ({time.time()-t0:.0f}s)', flush=True)

res = pd.DataFrame(rows)
if len(rows):
    res.to_csv(res_path, index=False)
if shd_rows:
    pd.DataFrame(shd_rows).to_csv(shd_path, index=False)
if par_rows:
    pd.DataFrame(par_rows).to_csv(BASE + '/tables/causal_parents.csv', index=False)
res = pd.read_csv(res_path)          # reload: includes previously-done pairs on resume
if os.path.exists(shd_path) and os.path.getsize(shd_path) > 0:
    shd = pd.read_csv(shd_path)
else:
    shd = pd.DataFrame(columns=['zone', 'kpi', 'algo_a', 'algo_b', 'shd'])
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
sob = sob[~sob.input.str.startswith('GROUP')]

# ---- consensus + cross-lens summary (21 modeled pairs) ----
summary = []
pairs = sorted(set(zip(res.zone, res.kpi)))
print('\n' + '=' * 72)
print(f'PHASE 5 DONE - {n_cells} cells computed this run, {len(res)} rows saved')
print('CONSENSUS CAUSAL PARENTS (>=2 of 3 algos at >=90% of 500 bootstraps)')
print('=' * 72)
for (zone, k) in pairs:
    grp = res[(res.zone == zone) & (res.kpi == k)]
    causal = set(grp[grp.causal_parent == True].input)
    sobdrv = set(sob[(sob.zone == zone) & (sob.kpi == k) & (sob.in_driver_set == True)].input)
    inter = causal & sobdrv
    jac = len(inter) / len(causal | sobdrv) if (causal | sobdrv) else float('nan')
    sub = grp.set_index('input')
    sb = sob[(sob.zone == zone) & (sob.kpi == k)].set_index('input')
    vals = [(sb.loc[c, 'ST_boot_mean'], sub.loc[c, 'mean_freq']) for c in INPUTS
            if c in sb.index and c in sub.index and not pd.isna(sb.loc[c, 'ST_boot_mean'])]
    rho = p = np.nan
    if len(vals) >= 3:
        rho, p = spearmanr([v[0] for v in vals], [v[1] for v in vals])
    shd_mean = shd[(shd.zone == zone) & (shd.kpi == k)].shd.mean() if len(shd) else float('nan')
    summary.append({'zone': zone, 'kpi': k,
                    'causal_parents': '; '.join(sorted(causal)),
                    'n_causal': len(causal), 'n_sobol_drivers': len(sobdrv),
                    'jaccard': jac, 'spearman_rho': rho, 'spearman_p': p,
                    'shd_mean': shd_mean})
    print(f'  {zone} {k:3s}: causal={sorted(causal)} | J={jac:.2f} rho={rho:.2f}')

pd.DataFrame(summary).to_csv(BASE + '/tables/causal_summary.csv', index=False)

print('\nSHD between algorithm skeletons (mean over pairs):')
if len(shd):
    print(shd.pivot_table(index='algo_a', columns='algo_b', values='shd', aggfunc='mean').round(2).to_string())
else:
    print('  (resumed run - SHD rows unchanged in causal_shd.csv)')

print('\nRank agreement with Sobol - Spearman rho over the 21 pairs:')
rr = [s['spearman_rho'] for s in summary if not pd.isna(s['spearman_rho'])]
print(f'  mean={np.mean(rr):.3f}  min={np.min(rr):.3f}  max={np.max(rr):.3f}')

# ---- FIG 8: one 3-panel figure, causal edge frequency per KPI ----
os.makedirs(BASE + '/figures/published', exist_ok=True)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for a, kpi in zip(axs, ['EUI', 'IDD', 'TL']):
        F = np.full((8, len(INPUTS)), np.nan)
        C = np.zeros((8, len(INPUTS)), bool)
        for i, z in enumerate(ZONES):
            sub = res[(res.zone == z) & (res.kpi == kpi)].set_index('input')
            for j, c in enumerate(INPUTS):
                if c in sub.index:
                    F[i, j] = sub.loc[c, 'mean_freq']
                    C[i, j] = bool(sub.loc[c, 'causal_parent'])
        im = a.imshow(F, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        a.set_xticks(range(len(INPUTS)))
        a.set_xticklabels(['PCM type\n(W)', 'Thick.\n(W)', 'Melt pt.\n(W)', 'Position\n(W)',
                           'PCM type\n(R)', 'Thick.\n(R)', 'Melt pt.\n(R)', 'Position\n(R)'],
                          fontsize=7)
        a.set_yticks(range(8)); a.set_yticklabels(ZONES, fontsize=9)
        a.set_title(kpi, fontsize=12, fontweight='bold')
        for i in range(8):
            for j in range(len(INPUTS)):
                if np.isnan(F[i, j]):
                    a.text(j, i, '-', ha='center', va='center', color='gray', fontsize=9)
                else:
                    a.text(j, i, f'{F[i, j]:.1f}', ha='center', va='center', fontsize=6.5,
                           color='white' if F[i, j] > 0.6 else 'black')
                if C[i, j]:
                    a.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                              edgecolor='darkred', lw=1.8))
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02); cb.set_label('edge frequency', fontsize=9)
    fig.suptitle('Bootstrap edge frequency, PC + GES + NOTEARS consensus\n'
                 '(red frame = causal parent: frequency >= 0.90 in >= 2 of 3 algorithms)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    for ext in ['png', 'pdf']:
        fig.savefig(f'{BASE}/figures/published/fig8_causal_consensus.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figure saved: figures/published/fig8_causal_consensus.png + .pdf')
except Exception as e:
    print('figure skipped (non-fatal):', e)

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 10 | causal done, {len(res)} rows saved\n")
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/causal_edges.csv, causal_shd.csv, causal_parents.csv, causal_summary.csv')
print('DONE - paste this output back.')
