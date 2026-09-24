# ============================================================================
# scripts/cell06_eda_summary.py
# Cell 6 — EDA: ranges, TL regimes, Spearman, eta-squared
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 4
# (387 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 6 — EDA SUMMARY
# Checks:
# 1. Rows per city
# 2. EUI variation and TL regimes
# 3. Input-input Spearman correlation
# 4. eta² drivers for EUI, IDD, and TL
# ============================================================

import os
import datetime
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE = '/content/drive/MyDrive/PCM_Study'
DATA_FILE = BASE + '/data/processed/pcm_dataset_final.csv'

# ------------------------------------------------------------
# CHECK DATASET
# ------------------------------------------------------------

if not os.path.exists(DATA_FILE):
    raise SystemExit(
        f'STOP: canonical dataset not found:\n{DATA_FILE}\n'
        'Run Cell 5 first.'
    )

df = pd.read_csv(DATA_FILE)

print('=' * 75)
print('CELL 6 — EDA SUMMARY')
print('=' * 75)

print(
    f'\nLoaded canonical dataset: '
    f'{df.shape[0]} rows x {df.shape[1]} columns'
)

# ------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------

INPUTS = [
    'pcm_type_wall',
    'thickness_wall_level',
    'melting_point_wall_level',
    'position_wall',
    'pcm_type_roof',
    'thickness_roof_level',
    'melting_point_roof_level',
    'position_roof'
]

TARGETS = [
    'eui_kwh_m2',
    'idd_deg_h',
    'tl_h'
]

# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

required = (
    ['case_id', 'city', 'climate_zone']
    + INPUTS
    + TARGETS
)

missing_columns = [
    c for c in required
    if c not in df.columns
]

if missing_columns:
    raise SystemExit(
        f'STOP: missing required columns: {missing_columns}'
    )

missing_values = int(df[required].isna().sum().sum())

if missing_values != 0:
    raise SystemExit(
        f'STOP: dataset contains {missing_values} missing values.'
    )

# ------------------------------------------------------------
# 1. ROWS PER CITY
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('1. ROWS PER CITY')
print('-' * 75)

city_counts = df.groupby('city').size()

print(city_counts.to_string())

# ------------------------------------------------------------
# 2. TARGET VARIATION BY CITY
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('2. TARGET RANGES BY CITY')
print('-' * 75)

for city, g in df.groupby('city'):

    print(
        f'  {city:12s} | '
        f'EUI {g["eui_kwh_m2"].min():8.2f} - '
        f'{g["eui_kwh_m2"].max():8.2f} | '
        f'IDD {g["idd_deg_h"].min():6.2f} - '
        f'{g["idd_deg_h"].max():6.2f} | '
        f'TL {g["tl_h"].min():6.2f} - '
        f'{g["tl_h"].max():6.2f}'
    )

# ------------------------------------------------------------
# 3. TL REGIMES
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('3. TL REGIMES BY CITY')
print('-' * 75)

for city, g in df.groupby('city'):

    unique_tl = sorted(
        g['tl_h']
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f'  {city:12s} | '
        f'n_unique={len(unique_tl):3d} | '
        f'TL values={unique_tl}'
    )

# ------------------------------------------------------------
# 4. INPUT-INPUT SPEARMAN CORRELATION
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('4. INPUT-INPUT SPEARMAN CORRELATION')
print('-' * 75)

print(
    'Screening rule: |r| > 0.30 is flagged for inspection.'
)

corr = df[INPUTS].corr(
    method='spearman'
).abs()

upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

strong_corr = (
    upper
    .stack()
    .sort_values(ascending=False)
)

strong_corr = strong_corr[
    strong_corr > 0.30
]

if len(strong_corr) == 0:

    print(
        '\n  No input pairs have |Spearman r| > 0.30.'
    )

else:

    print(
        '\n  Input pairs with |Spearman r| > 0.30:'
    )

    for (v1, v2), r in strong_corr.items():

        print(
            f'  {v1:30s} <-> {v2:30s} | '
            f'|r| = {r:.3f}'
        )

# ------------------------------------------------------------
# 5. ETA-SQUARED FUNCTION
# ------------------------------------------------------------

def eta2(x, y):

    x = pd.Categorical(x)
    y = np.asarray(y, dtype=float)

    grand_mean = y.mean()

    ss_between = 0.0

    for level in x.categories:

        mask = (x == level)

        if mask.sum() == 0:
            continue

        group_mean = y[mask].mean()

        ss_between += (
            mask.sum()
            * (group_mean - grand_mean) ** 2
        )

    ss_total = (
        (y - grand_mean) ** 2
    ).sum()

    if ss_total == 0:
        return np.nan

    return ss_between / ss_total

# ------------------------------------------------------------
# 6. ETA² DRIVERS
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('5. ETA² — INPUT EFFECT SIZE BY TARGET')
print('-' * 75)

print(
    'Values are calculated separately by city, '
    'then averaged across cities.'
)

eta_results = {}

for target in TARGETS:

    city_results = []

    for city, g in df.groupby('city'):

        row = {}

        for variable in INPUTS:

            row[variable] = eta2(
                g[variable],
                g[target]
            )

        city_results.append(row)

    eta_df = pd.DataFrame(city_results)

    mean_eta = (
        eta_df
        .mean(skipna=True)
        .sort_values(ascending=False)
    )

    eta_results[target] = mean_eta

    print(f'\n{target}:')

    for variable, value in mean_eta.items():

        if pd.isna(value):
            print(
                f'  {variable:30s} = NaN'
            )
        else:
            print(
                f'  {variable:30s} = {value:.4f}'
            )

# ------------------------------------------------------------
# 7. CONSTANT TARGET CHECK
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('6. CONSTANT TARGET CHECK')
print('-' * 75)

for target in TARGETS:

    constant_cities = []

    for city, g in df.groupby('city'):

        if g[target].nunique() <= 1:
            constant_cities.append(city)

    if constant_cities:

        print(
            f'  {target}: constant in '
            f'{constant_cities}'
        )

    else:

        print(
            f'  {target}: no city has a constant value'
        )

# ------------------------------------------------------------
# 8. INPUT LEVEL COUNTS
# ------------------------------------------------------------

print('\n' + '-' * 75)
print('7. INPUT LEVEL COUNTS')
print('-' * 75)

for variable in INPUTS:

    n_levels = df[variable].nunique()

    print(
        f'\n{variable} '
        f'({n_levels} levels):'
    )

    print(
        df[variable]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

# ------------------------------------------------------------
# 9. FINAL CHECK
# ------------------------------------------------------------

print('\n' + '=' * 75)
print('CELL 6 COMPLETE')
print('=' * 75)

print(
    f'Rows checked : {len(df)}'
)

print(
    f'Cities       : {df["city"].nunique()}'
)

print(
    f'Inputs       : {len(INPUTS)}'
)

print(
    f'Targets      : {len(TARGETS)}'
)

# ------------------------------------------------------------
# LOG
# ------------------------------------------------------------

os.makedirs(
    BASE + '/logs',
    exist_ok=True
)

with open(
    BASE + '/logs/pipeline_log.txt',
    'a'
) as f:

    f.write(
        f'{datetime.datetime.now().isoformat()} | '
        f'CELL 6 | EDA summary complete '
        f'({len(df)} rows)\n'
    )

print('\nDONE — paste the complete output back here.')
