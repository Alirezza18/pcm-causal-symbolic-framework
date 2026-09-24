# ============================================================================
# scripts/cell15d_moo_validation_table4.py
# Cell 15d — MOO algorithm validation + Table 4
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 28
# (254 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 15d - PHASE 9c: MOO ALGORITHM VALIDATION + TABLE 4 (FINAL FORMATTED)
# ============================================================================
import os, time, datetime, itertools, joblib
from bisect import bisect_right
import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
SEED = 42
N_EVAL = 1240                    # ~8.6% of the 14,400-design grid

fr = pd.read_csv(BASE + '/tables/pareto_front.csv')
kr = pd.read_csv(BASE + '/tables/pareto_knee.csv')
gs = pd.read_csv(BASE + '/tables/pareto_gridsummary.csv').drop_duplicates('zone')
sob = pd.read_csv(BASE + '/tables/sobol_indices.csv')
KNEE = kr[kr.type == 'knee'].set_index('zone')
OPTS = kr[kr.type.str.startswith('min_')]

ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']
ZCITY = {'0B': 'BandarAbbas', '1B': 'Bushehr', '2B': 'Kashan', '3A': 'Rasht',
         '3B': 'Tehran', '4A': 'Hamedan', '4B': 'Tabriz', '5C': 'Ardebil'}

INPUTS_WALL = ['pcm_type_wall', 'thickness_wall_level', 'melting_point_wall_level', 'position_wall']
INPUTS_ROOF = ['pcm_type_roof', 'thickness_roof_level', 'melting_point_roof_level', 'position_roof']
INPUTS = INPUTS_WALL + INPUTS_ROOF

ALLK = ['EUI', 'IDD', 'TL']
df = pd.read_csv(BASE + '/data/processed/pcm_dataset_final.csv')

# ---------------- exact Pareto + hypervolume ----------------
def pareto_front(Y):
    m = Y.shape[1]
    U, inv = np.unique(Y, axis=0, return_inverse=True)
    order = np.lexsort(tuple(U[:, j] for j in range(m - 1, -1, -1)))
    nd_u = np.zeros(len(U), dtype=bool)
    if m == 2:
        best = np.inf
        for idx in order:
            if U[idx, 1] < best:
                nd_u[idx] = True
                best = U[idx, 1]
        return nd_u[inv]
    assert m == 3, 'm=2 or 3 only'
    sk2, sk3 = [], []
    for idx in order:
        o2, o3 = U[idx, 1], U[idx, 2]
        p = bisect_right(sk2, o2)
        if p > 0 and sk3[p - 1] <= o3:
            continue
        nd_u[idx] = True
        q = p
        while q < len(sk3) and sk3[q] >= o3:
            q += 1
        del sk2[p:q], sk3[p:q]
        sk2.insert(p, float(o2)), sk3.insert(p, float(o3))
    return nd_u[inv]

def _stair_insert(sk_a, sk_b, a, b):
    p = bisect_right(sk_a, a)
    if p > 0 and sk_b[p - 1] <= b:
        return False
    q = p
    while q < len(sk_b) and sk_b[q] >= b:
        q += 1
    del sk_a[p:q], sk_b[p:q]
    sk_a.insert(p, float(a)), sk_b.insert(p, float(b))
    return True

def _hv2_of_stair(sk_a, sk_b, ra, rb):
    hv, prev = 0.0, ra
    for a, b in zip(sk_a, sk_b):
        hv += (prev - a) * (rb - b)
        prev = a
    return hv

def hypervolume(P, ref):
    m = P.shape[1]
    if m == 2:
        sk_a, sk_b = [], []
        for p in P:
            _stair_insert(sk_a, sk_b, p[0], p[1])
        return _hv2_of_stair(sk_a, sk_b, ref[0], ref[1])
    order = np.lexsort((P[:, 2], P[:, 1], P[:, 0]))
    sk_a, sk_b = [], []
    hv, prev1, i, n = 0.0, None, 0, len(P)
    while i < n:
        v1 = P[order[i], 0]
        if prev1 is not None:
            hv += (v1 - prev1) * _hv2_of_stair(sk_a, sk_b, ref[1], ref[2])
        prev1 = v1
        while i < n and P[order[i], 0] == v1:
            _stair_insert(sk_a, sk_b, P[order[i], 1], P[order[i], 2])
            i += 1
    hv += (ref[0] - prev1) * _hv2_of_stair(sk_a, sk_b, ref[1], ref[2])
    return hv

# ==== ALGO-RUNNER BLOCK ====
try:
    import pymoo
except ImportError:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, '-m', 'pip', 'install', '-q', 'pymoo'], check=False)
    import pymoo
try:
    import optuna
except ImportError:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, '-m', 'pip', 'install', '-q', 'optuna'], check=False)
    import optuna

def nsga2_archive(pb, xl, xu, m, seed, n_eval, grid_X, grid_O, ref_vecs):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as _minimize

    class _Prob(Problem):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=m, n_constr=0,
                             xl=np.asarray(xl, float), xu=np.asarray(xu, float))
            self.hist = []
        def _evaluate(self, X, out, *args, **kwargs):
            Xr = np.clip(np.round(np.atleast_2d(X)), xl, xu)
            self.hist.append(Xr.copy())
            out['F'] = pb(Xr)

    pop = 40
    algo = NSGA2(pop_size=pop, n_offsprings=pop, seed=seed)
    n_gen = max(1, int(np.ceil((n_eval - pop) / pop)))
    prob = _Prob()
    _minimize(prob, algo, ('n_gen', n_gen), seed=seed, verbose=False)
    return np.vstack(prob.hist)

def motpe_archive(pb, xl, xu, m, seed, n_eval, grid_X, grid_O, ref_vecs):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(directions=['minimize'] * m, sampler=sampler)

    def objective(trial):
        x = np.array([trial.suggest_int(v, int(lo), int(hi))
                      for v, lo, hi in zip(INPUTS, xl, xu)], float)
        return tuple(np.asarray(pb(x[None, :])[0], float))

    study.optimize(objective, n_trials=n_eval, show_progress_bar=False)
    return np.array([[t.params[v] for v in INPUTS] for t in study.trials], dtype=float)

# ---------------- per-zone benchmark ----------------
t0 = time.time()
rows = []
print('=' * 78)
print('MOO VALIDATION vs EXHAUSTIVE REFERENCE FRONT (budget %d evals/zone)' % N_EVAL)
print('=' * 78)

for z in ZONES:
    g = df[df.climate_zone == z]
    kpis = gs[gs.zone == z].iloc[0]['kpis'].split('|')
    m = len(kpis)

    lv = [range(int(g[c].min()), int(g[c].max()) + 1) for c in INPUTS]
    G = np.array(list(itertools.product(*lv)), float)
    models = {k: joblib.load(f'{BASE}/models/cat_{z}_{k}.joblib') for k in kpis}
    Y = np.column_stack([np.asarray(models[k].predict(
        pd.DataFrame(G, columns=INPUTS)), float) for k in kpis])
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    rng_ = np.where(hi > lo, hi - lo, 1.0)

    def pb(X):
        Xd = pd.DataFrame(np.clip(np.round(np.atleast_2d(X)),
                                  np.array([r.start for r in lv]),
                                  np.array([r.stop - 1 for r in lv])),
                          columns=INPUTS)
        P = np.column_stack([np.asarray(models[k].predict(Xd), float)
                             for k in kpis])
        return (P - lo) / rng_

    grid_O = (Y - lo) / rng_

    fz = fr[fr.zone == z]
    ref_raw = np.column_stack([fz[k].values for k in kpis])
    ref_norm = np.unique(np.round((ref_raw - lo) / rng_, 6), axis=0)
    ref_set = {tuple(v) for v in ref_norm}
    n_ref = len(ref_set)

    xl = np.array([r.start for r in lv], float)
    xu = np.array([r.stop - 1 for r in lv], float)
    bench = {}
    for name, runner in [('NSGA-II', nsga2_archive), ('MOTPE', motpe_archive)]:
        ax_ = runner(pb, xl, xu, m, SEED, N_EVAL, G, grid_O, ref_norm)
        A = np.round(pb(ax_), 6)
        nd = pareto_front(A)
        hv_a = hypervolume(A[nd], np.full(m, 1.05))
        hv_r = hypervolume(ref_norm, np.full(m, 1.05))

        hv_ratio = min(1.00, float(hv_a / hv_r)) if hv_r > 0 else 1.00
        cov = len(ref_set & {tuple(v) for v in A}) / n_ref if n_ref > 0 else 1.0
        bench[name] = (hv_ratio, cov, int(nd.sum()))

    kne = KNEE.loc[z]

    # 1. جداسازی Knee Wall و Knee Roof
    knee_wall = '-'.join(str(int(kne[v])) for v in INPUTS_WALL)
    knee_roof = '-'.join(str(int(kne[v])) for v in INPUTS_ROOF)

    # 2. محاسبه ΔEUI, ΔIDD (% positive-baseline) و ΔTL (ساعت/hours)
    dl = {}
    for k in ALLK:
        o = OPTS[(OPTS.zone == z) & (OPTS.type == f'min_{k}')]
        if len(o) and k in kne and not pd.isna(kne[k]):
            ov, kv = float(o.iloc[0][k]), float(kne[k])
            if k == 'TL':
                dl[k] = ('h', kv - ov)          # hours, positive = knee pays
            else:
                dl[k] = ('%', 100 * (kv - ov) / ov if ov != 0 else np.nan)
        else:
            dl[k] = (None, None)

    # 3. محاسبه شاخص Jaccard (J)
    active = {v for v in INPUTS if fz[v].nunique() > 1}
    drv = set()
    for k in kpis:
        drv |= set(sob[(sob.zone == z) & (sob.kpi == k) &
                       (sob.in_driver_set == True)].input)
    uni = active | drv
    J = (len(active & drv) / len(uni)) if uni else np.nan

    rows.append({
        'Zone': z,
        'City': ZCITY[z],
        'n': n_ref,
        'HV (NSGA-II)': round(bench['NSGA-II'][0], 3),
        'Cov (NSGA-II)': round(bench['NSGA-II'][1], 2),
        'HV (MOTPE)': round(bench['MOTPE'][0], 3),
        'Cov (MOTPE)': round(bench['MOTPE'][1], 2),
        'Knee wall': knee_wall,
        'Knee roof': knee_roof,
        'ΔEUI (%)': '--' if dl['EUI'][0] is None else f"{dl['EUI'][1]:.1f}%",
        'ΔIDD (%)': '--' if dl['IDD'][0] is None else f"{dl['IDD'][1]:.1f}%",
        'ΔTL (h)':  '--' if dl['TL'][0]  is None else f"{dl['TL'][1]:+.1f}",
        'J': '--' if np.isnan(J) else f"{J:.2f}"
    })

    print(f"{z} ({ZCITY[z]}) | n={n_ref:3d} | HV NSGA: {bench['NSGA-II'][0]:.3f} | HV MOTPE: {bench['MOTPE'][0]:.3f} | J={J:.2f}", flush=True)

T4 = pd.DataFrame(rows)

csv_path = BASE + '/tables/table4_moo_benchmark.csv'
T4.to_csv(csv_path, index=False)
print('\n' + '=' * 78)
print(f'Table 4 - Formatted MOO Validation per Zone -> {csv_path}')
print('=' * 78)
print(T4.to_string(index=False))

files.download(csv_path)
