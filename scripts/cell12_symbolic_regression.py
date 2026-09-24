# ============================================================================
# scripts/cell12_symbolic_regression.py
# Cell 12 — symbolic regression, 3-seed GP consensus
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 19
# (157 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 12 - PHASE 7: SYMBOLIC REGRESSION - 3-seed GP consensus (resume-safe)
# Genetic-programming symbolic regression (gplearn) per (zone, KPI) on the
# level-coded simulation data (NOT the surrogate - this lens is independent
# of the tree models by construction).
# Lens for Fig 10: symbolic_vars = variables appearing in the best expression
# of >=2 of 3 independent GP runs (seeds 42/43/44) - a 2-of-3 consensus rule
# paralleling the causal-discovery criterion.
# Saves: tables/symbolic_expressions.csv (21 cells x 3 seeds)
#        tables/symbolic_variables.csv  (21 x 8, in_symbolic_lens flag)
# Features are z-scored inside each cell (identity preserved); y is z-scored
# (R2 is scale-invariant). Function set: +, -, *, protected division.
# Runtime ~1-2 h total on Colab CPU; resume-safe per (cell, seed).
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import re, time, datetime
import numpy as np
import pandas as pd

try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    # [Colab shell] pip install gplearn -q
    from gplearn.genetic import SymbolicRegressor

BASE = '/content/drive/MyDrive/PCM_Study'
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')
bm = pd.read_csv(BASE + '/tables/model_benchmarks.csv')

INPUTS = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall',
          'pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
TARGETS = ['eui_kwh_m2', 'idd_deg_h', 'tl_h']
SHORT = {'eui_kwh_m2': 'EUI', 'idd_deg_h': 'IDD', 'tl_h': 'TL'}

SKIP = {('Tabriz', 'TL'): 'no surrogate (constant target)',
        ('BandarAbbas', 'IDD'): 'degenerate target (3.1)',
        ('Kashan', 'TL'): 'degenerate target (3.1)'}

SEEDS = [42, 43, 44]
# سازگاری کامل با نسخه‌های جدید gplearn
GP = dict(population_size=1500, generations=30, tournament_size=20,
          function_set=['add', 'sub', 'mul', 'div'],
          p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
          p_point_mutation=0.1, parsimony_coefficient=0.0005, metric='mse', n_jobs=2)
CURRENT = {'kpi': None}   # provenance hook (used by test harnesses)

def zscore(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)

def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - yhat) ** 2).sum() / ss if ss > 0 else np.nan

def free_vars(expr):
    return sorted({INPUTS[int(i)] for i in re.findall(r'X(\d+)', expr)})

# ---- resume support: per (cell, seed) ----
expr_path = BASE + '/tables/symbolic_expressions.csv'
var_path = BASE + '/tables/symbolic_variables.csv'
expr_rows = []
if os.path.exists(expr_path):
    prev = pd.read_csv(expr_path)
    if len(prev):
        expr_rows = prev.to_dict('records')
        print(f'resuming - {len(prev)} expressions already computed '
              f'({prev.groupby(["zone", "kpi"]).size().max()} seeds/cell max)')

t0 = time.time()
n_runs = 0
for zone, g in df.groupby('climate_zone'):
    city = g['city'].iloc[0]
    for t in TARGETS:
        k = SHORT[t]
        if (city, k) in SKIP:
            print(f'{zone} {city:11s} {k}: SKIPPED - {SKIP[(city, k)]}')
            continue
        X = g[INPUTS].reset_index(drop=True)
        y = pd.Series(zscore(g[t].reset_index(drop=True)))
        Xz = pd.DataFrame({c: zscore(X[c]) for c in INPUTS})
        ADOPTED = bm[(bm.zone == zone) & (bm.kpi == k)].iloc[0].best_model
        for seed in SEEDS:
            if any(r['zone'] == zone and r['kpi'] == k and r['seed'] == seed
                   for r in expr_rows):
                continue
            n_runs += 1
            CURRENT['kpi'] = k
            est = SymbolicRegressor(random_state=seed, **GP)
            est.fit(Xz.values, y.values)
            prog = est._program
            expr = str(prog)
            run_r2 = r2(y.values, est.predict(Xz.values))

            # اصلاح خطای TypeError: استفاده از prog.program برای شمارش گره‌ها
            n_nodes = len(prog.program) if hasattr(prog, 'program') else len(expr.split())

            expr_rows.append({'zone': zone, 'city': city, 'kpi': k, 'seed': seed,
                              'adopted': ADOPTED, 'expression': expr,
                              'r2': round(float(run_r2), 4), 'n_nodes': int(n_nodes),
                              'vars_used': '|'.join(free_vars(expr))})
            pd.DataFrame(expr_rows).to_csv(expr_path, index=False)  # incremental
            print(f'{zone} {city:11s} {k} s{seed}: R2={run_r2:.3f} '
                  f'nodes={n_nodes} vars={free_vars(expr)} '
                  f'({time.time()-t0:.0f}s)', flush=True)

exprs = pd.DataFrame(expr_rows)
exprs.to_csv(expr_path, index=False)

# ---- 2-of-3 consensus lens per cell ----
var_rows = []
print('\n' + '=' * 70)
print(f'PHASE 7 DONE - {n_runs} GP runs this time, {len(exprs)} expressions saved')
print('=' * 70)
for (zone, k), grp in exprs.groupby(['zone', 'kpi']):
    counts = {v: 0 for v in INPUTS}
    for _, r in grp.iterrows():
        for v in str(r.vars_used).split('|'):
            if v in counts:
                counts[v] += 1
    lens = [v for v in INPUTS if counts[v] >= 2]
    for v in INPUTS:
        var_rows.append({'zone': zone, 'kpi': k, 'input': v,
                         'seeds_present': int(counts[v]),
                         'in_symbolic_lens': bool(counts[v] >= 2)})
    best = grp.loc[grp.r2.idxmax()]
    print(f'{zone} {k:3s}: lens={lens}')
    for _, r in grp.sort_values('seed').iterrows():
        print(f'      s{r.seed} R2={r.r2:.3f} nodes={r.n_nodes}: {r.expression[:90]}')

vars_df = pd.DataFrame(var_rows)
assert len(vars_df.groupby(['zone', 'kpi'])) == 21, 'expected 21 cells'
assert vars_df.groupby(['zone', 'kpi']).size().eq(8).all(), 'cell without 8 variables'
os.makedirs(BASE + '/logs', exist_ok=True)
vars_df.to_csv(var_path, index=False)
print('\nsymbolic lens summary per variable (of 21 cells):')
for v in INPUTS:
    n = int(vars_df[(vars_df.input == v) & (vars_df.in_symbolic_lens)].shape[0])
    print(f'  {v:28s} {n:2d}')
print(f'\nmedian expression R2 = {exprs.r2.median():.3f} '
      f'(min {exprs.r2.min():.3f}, max {exprs.r2.max():.3f})')

with open(BASE + '/logs/pipeline_log.txt', 'a') as f:
    f.write(f"{datetime.datetime.now().isoformat()} | CELL 12 | symbolic done, "
            f"{len(exprs)} expressions saved\n")
print(f'\nTotal: {time.time()-t0:.0f}s | saved tables/symbolic_expressions.csv '
      f'+ tables/symbolic_variables.csv')
print('DONE - paste this output back.')
