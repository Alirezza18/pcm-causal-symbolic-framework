# ============================================================================
# scripts/cell04_clean_audit.py
# Cell 4 — clean + audit: 11,988 raw rows -> 3,861 clean rows
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 2
# (380 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# CELL 4 — CLEAN + AUDIT

import os
import glob
import pandas as pd
import datetime

BASE = '/content/drive/MyDrive/PCM_Study'
RAW = BASE + '/data/raw'
CLEAN = BASE + '/data/clean'

os.makedirs(CLEAN, exist_ok=True)

INPUTS = [
    'PCM_type_Wall',
    'Thickness_Wall',
    'Melting_Point_Wall',
    'wall_PCM_Position',
    'PCM_type_Roof',
    'Thickness_Roof',
    'Melting_Point_Roof',
    'Roof_PCM_Position'
]

TARGETS = ['EUI', 'IDD', 'TL']

# ------------------------------------------------------------
# FIND THE 8 CITY FILES
# ------------------------------------------------------------

files = sorted(
    glob.glob(os.path.join(RAW, '*.csv'))
)

print('=' * 75)
print('CELL 4 — CLEANING DATA')
print('=' * 75)

print(f'\nFound {len(files)} CSV files.')

if len(files) != 8:
    raise SystemExit(
        f'\nSTOP: Expected 8 CSV files, but found {len(files)}.\n'
        f'Please check: {RAW}'
    )

for f in files:
    print('  ', os.path.basename(f))

# ------------------------------------------------------------
# READ FILES
# ------------------------------------------------------------

all_dfs = []

for f in files:

    filename = os.path.basename(f)

    try:
        df = pd.read_csv(f)
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='utf-8-sig')

    # Clean column names
    df.columns = [
        c.strip().split('.')[-1].strip()
        for c in df.columns
    ]

    # Check columns
    missing = [
        c for c in INPUTS + TARGETS
        if c not in df.columns
    ]

    if missing:
        raise SystemExit(
            f'\nSTOP: {filename} is missing columns:\n{missing}'
        )

    # --------------------------------------------------------
    # CITY NAME FROM FILENAME
    # --------------------------------------------------------

    city = filename

    if city.startswith('LHS_'):
        city = city[4:]

    if city.lower().endswith('.csv'):
        city = city[:-4]

    city = city.replace('(1)', '').strip()

    # Normalize city names
    city_lower = city.lower()

    city_names = {
        'ardebial': 'Ardebial',
        'bandarabbas': 'Bandarabbas',
        'bushhr': 'Bushhr',
        'hamedan': 'Hamedan',
        'kashan': 'Kashan',
        'rasht': 'Rasht',
        'tabriz': 'Tabriz',
        'tehran': 'Tehran'
    }

    city = city_names.get(
        city_lower,
        city.capitalize()
    )

    df['city'] = city

    all_dfs.append(df)

    print(
        f'Read {city:15s}: {len(df):4d} rows'
    )

# ------------------------------------------------------------
# COMBINE ALL CITIES
# ------------------------------------------------------------

full = pd.concat(
    all_dfs,
    ignore_index=True
)

before = len(full)

print('\n' + '=' * 75)
print(f'TOTAL ROWS BEFORE CLEANING: {before}')
print('=' * 75)

# ------------------------------------------------------------
# 1. REMOVE DUPLICATE LOG ROWS
# ------------------------------------------------------------

d1 = full.duplicated(
    subset=INPUTS + TARGETS,
    keep='first'
)

n_dup = int(d1.sum())

print(
    f'\n[1] Duplicate log rows removed: {n_dup}'
)

# ------------------------------------------------------------
# 2. REMOVE IDD == EUI WIRING GLITCH
# ------------------------------------------------------------

eui = pd.to_numeric(
    full['EUI'],
    errors='coerce'
)

idd = pd.to_numeric(
    full['IDD'],
    errors='coerce'
)

tl = pd.to_numeric(
    full['TL'],
    errors='coerce'
)

d2 = (
    (idd - eui).abs() < 1e-6
)

n_wire = int(d2.sum())

print(
    f'[2] IDD == EUI wiring-glitch rows removed: {n_wire}'
)

if n_wire > 0:

    cols = [
        'city',
        'EUI',
        'IDD',
        'TL'
    ]

    if 'time' in full.columns:
        cols.insert(1, 'time')

    print('\nWiring-glitch rows:')
    print(
        full.loc[d2, cols].to_string(index=False)
    )

# ------------------------------------------------------------
# 3. REMOVE IMPOSSIBLE PHYSICS
# ------------------------------------------------------------

d3 = (
    (tl.abs() > 24) |
    (idd < 0.5)
)

n_imp = int(d3.sum())

print(
    f'\n[3] Impossible rows removed '
    f'(|TL| > 24 OR IDD < 0.5): {n_imp}'
)

if n_imp > 0:

    cols = [
        'city',
        'EUI',
        'IDD',
        'TL'
    ]

    if 'time' in full.columns:
        cols.insert(1, 'time')

    print('\nImpossible rows:')
    print(
        full.loc[d3, cols].to_string(index=False)
    )

# ------------------------------------------------------------
# FINAL CLEANING
# ------------------------------------------------------------

remove = d1 | d2 | d3

clean = full.loc[
    ~remove
].copy()

# ------------------------------------------------------------
# PER-CITY COUNTS
# ------------------------------------------------------------

print('\n' + '=' * 75)
print('PER-CITY COUNTS AFTER CLEANING')
print('=' * 75)

for city in sorted(clean['city'].unique()):

    n = int(
        (clean['city'] == city).sum()
    )

    print(
        f'{city:15s}: {n:4d}'
    )

# ------------------------------------------------------------
# TOTAL
# ------------------------------------------------------------

removed = before - len(clean)

print('\n' + '=' * 75)
print('CLEANING SUMMARY')
print('=' * 75)

print(
    f'Rows before cleaning : {before}'
)

print(
    f'Duplicate rows       : {n_dup}'
)

print(
    f'Wiring-glitch rows   : {n_wire}'
)

print(
    f'Impossible rows      : {n_imp}'
)

print(
    f'Total rows removed   : {removed}'
)

print(
    f'Rows after cleaning  : {len(clean)}'
)

# ------------------------------------------------------------
# POST-CLEAN CHECK
# ------------------------------------------------------------

print('\n' + '=' * 75)
print('POST-CLEAN TARGET CHECK')
print('=' * 75)

for city in sorted(clean['city'].unique()):

    g = clean[
        clean['city'] == city
    ]

    print(f'\n{city}  (n={len(g)})')

    print(
        f'  EUI: {g["EUI"].min():.3f} '
        f'-> {g["EUI"].max():.3f}'
    )

    print(
        f'  IDD: {g["IDD"].min():.3f} '
        f'-> {g["IDD"].max():.3f}'
    )

    print(
        f'  TL : {g["TL"].min():.3f} '
        f'-> {g["TL"].max():.3f}'
    )

# ------------------------------------------------------------
# SAVE CLEAN DATASET
# ------------------------------------------------------------

output_file = os.path.join(
    CLEAN,
    'all_cities_CLEANED.csv'
)

clean.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# SAVE LOG
# ------------------------------------------------------------

os.makedirs(
    BASE + '/logs',
    exist_ok=True
)

log_file = BASE + '/logs/pipeline_log.txt'

with open(log_file, 'a') as f:

    f.write(
        f'{datetime.datetime.now().isoformat()} | '
        f'CELL 4 | '
        f'before={before} | '
        f'duplicates={n_dup} | '
        f'wiring={n_wire} | '
        f'impossible={n_imp} | '
        f'after={len(clean)}\n'
    )

# ------------------------------------------------------------
# FINISHED
# ------------------------------------------------------------

print('\n' + '=' * 75)
print('DONE')
print('=' * 75)

print(
    '\nCleaned file saved to:'
)

print(
    output_file
)

print(
    f'\nFinal dataset: {len(clean)} rows x {len(clean.columns)} columns'
)
