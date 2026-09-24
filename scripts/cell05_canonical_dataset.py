# ============================================================================
# scripts/cell05_canonical_dataset.py
# Cell 5 — canonical dataset pcm_dataset_final.csv (3,861 x 14)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 3
# (327 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 5 — CANONICAL DATASET
# Creates:
# data/processed/pcm_dataset_final.csv
# Expected: 3,861 rows x 14 columns
# ============================================================

import os
import datetime
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE = '/content/drive/MyDrive/PCM_Study'

CLEAN_FILE = BASE + '/data/clean/all_cities_CLEANED.csv'
OUT_DIR = BASE + '/data/processed'
OUT = OUT_DIR + '/pcm_dataset_final.csv'

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# CITY METADATA
# ------------------------------------------------------------

META = {
    'ardebial':    ('08', '5C', 'Ardebil'),
    'bandarabbas': ('01', '0B', 'BandarAbbas'),
    'bushhr':      ('02', '1B', 'Bushehr'),
    'hamedan':     ('07', '4A', 'Hamedan'),
    'kashan':      ('03', '2B', 'Kashan'),
    'rasht':       ('06', '3A', 'Rasht'),
    'tabriz':      ('05', '4B', 'Tabriz'),
    'tehran':      ('04', '3B', 'Tehran'),
}

EXPECTED = {
    'Ardebil': 479,
    'BandarAbbas': 482,
    'Bushehr': 486,
    'Hamedan': 484,
    'Kashan': 489,
    'Rasht': 475,
    'Tabriz': 482,
    'Tehran': 484
}

# ------------------------------------------------------------
# COLUMN RENAMING
# ------------------------------------------------------------

RENAME = {
    'PCM_type_Wall': 'pcm_type_wall',
    'Thickness_Wall': 'thickness_wall_level',
    'Melting_Point_Wall': 'melting_point_wall_level',
    'wall_PCM_Position': 'position_wall',

    'PCM_type_Roof': 'pcm_type_roof',
    'Thickness_Roof': 'thickness_roof_level',
    'Melting_Point_Roof': 'melting_point_roof_level',
    'Roof_PCM_Position': 'position_roof',

    'EUI': 'eui_kwh_m2',
    'IDD': 'idd_deg_h',
    'TL': 'tl_h'
}

COLS = [
    'case_id',
    'city',
    'climate_zone',

    'pcm_type_wall',
    'thickness_wall_level',
    'melting_point_wall_level',
    'position_wall',

    'pcm_type_roof',
    'thickness_roof_level',
    'melting_point_roof_level',
    'position_roof',

    'eui_kwh_m2',
    'idd_deg_h',
    'tl_h'
]

PARAMS8 = [
    'pcm_type_wall',
    'thickness_wall_level',
    'melting_point_wall_level',
    'position_wall',
    'pcm_type_roof',
    'thickness_roof_level',
    'melting_point_roof_level',
    'position_roof'
]

# ------------------------------------------------------------
# CHECK CLEAN FILE
# ------------------------------------------------------------

if not os.path.exists(CLEAN_FILE):
    raise SystemExit(
        f'STOP: Clean file not found:\n{CLEAN_FILE}\n'
        'Run Cell 4 first.'
    )

# ------------------------------------------------------------
# LOAD CLEAN DATA
# ------------------------------------------------------------

clean = pd.read_csv(CLEAN_FILE)

print('=' * 75)
print('CELL 5 — CANONICAL DATASET')
print('=' * 75)

print(
    f'\nLoaded cleaned dataset: '
    f'{clean.shape[0]} rows x {clean.shape[1]} columns'
)

# ------------------------------------------------------------
# CLEAN COLUMN NAMES
# ------------------------------------------------------------

clean.columns = clean.columns.str.strip()

# ------------------------------------------------------------
# RENAME COLUMNS
# ------------------------------------------------------------

clean = clean.rename(columns=RENAME)

# ------------------------------------------------------------
# CREATE NORMALIZED CITY KEY
# ------------------------------------------------------------

clean['city_key'] = (
    clean['city']
    .astype(str)
    .str.replace(r'\.csv$', '', regex=True)
    .str.replace(r'^lhs[_ ]*', '', case=False, regex=True)
    .str.replace(r'\(1\)', '', regex=True)
    .str.strip()
    .str.lower()
)

# ------------------------------------------------------------
# CHECK CITY NAMES
# ------------------------------------------------------------

unknown = set(clean['city_key'].unique()) - set(META.keys())

if unknown:
    raise SystemExit(
        f'Unrecognized city key(s): {unknown}\n'
        f'Expected keys: {sorted(META.keys())}'
    )

# ------------------------------------------------------------
# BUILD CANONICAL DATASET
# ------------------------------------------------------------

frames = []

for key, (code, zone, display) in META.items():

    g = clean[
        clean['city_key'] == key
    ].copy()

    if len(g) == 0:
        raise SystemExit(
            f'STOP: No rows found for city: {key}'
        )

    # Create stable hash from the 8 PCM design parameters
    g['_hash'] = pd.util.hash_pandas_object(
        g[PARAMS8],
        index=False
    )

    # Sort consistently
    g = g.sort_values(
        ['eui_kwh_m2', 'idd_deg_h']
    ).reset_index(drop=True)

    # Create case IDs
    g['case_id'] = [
        f'{code}-{r:04d}'
        for r in g['_hash']
        .rank(method='first')
        .astype(int)
    ]

    g['city'] = display
    g['climate_zone'] = zone

    frames.append(
        g.drop(columns='_hash')
    )

# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

df = pd.concat(
    frames,
    ignore_index=True
)

# Keep exactly the canonical columns
df = df[COLS]

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print('\nRows per city:')
print(
    df.groupby('city')
      .size()
      .to_string()
)

got = df.groupby('city').size().to_dict()

if got != EXPECTED:
    raise SystemExit(
        f'\nSTOP — COUNT MISMATCH:\n'
        f'Got:      {got}\n'
        f'Expected: {EXPECTED}'
    )

# Check missing values
missing = int(
    df.isna().sum().sum()
)

if missing != 0:
    raise SystemExit(
        f'STOP — dataset contains {missing} missing values.'
    )

# Check dimensions
if df.shape != (3861, 14):
    raise SystemExit(
        f'STOP — unexpected dataset shape: {df.shape}. '
        f'Expected (3861, 14).'
    )

# Check duplicate case IDs
duplicate_ids = int(
    df['case_id'].duplicated().sum()
)

if duplicate_ids != 0:
    raise SystemExit(
        f'STOP — found {duplicate_ids} duplicate case IDs.'
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUT,
    index=False
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
        f'CELL 5 | '
        f'pcm_dataset_final.csv '
        f'created/verified '
        f'({len(df)} rows x {len(df.columns)} cols)\n'
    )

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print('\n' + '=' * 75)
print('CANONICAL DATASET READY')
print('=' * 75)

print(
    f'Rows    : {df.shape[0]}'
)

print(
    f'Columns : {df.shape[1]}'
)

print(
    f'Missing : {missing}'
)

print(
    f'Dup IDs : {duplicate_ids}'
)

print(
    '\nSaved to:'
)

print(OUT)

print('\nDONE — paste the output back here.')
