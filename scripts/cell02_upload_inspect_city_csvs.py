# ============================================================================
# scripts/cell02_upload_inspect_city_csvs.py
# Cells 2+3 — upload + inspect the 8 city LHS CSVs
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 1
# (164 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================
# CELL 2+3 — UPLOAD + SAVE + INSPECT THE 8 CITY CSVs
# ============================================================

import os
import glob
import pandas as pd
from google.colab import files

# ------------------------------------------------------------
# 1. Set raw-data folder
# ------------------------------------------------------------
RAW = BASE + '/data/raw'
os.makedirs(RAW, exist_ok=True)

# ------------------------------------------------------------
# 2. Upload CSV files
# ------------------------------------------------------------
print('Select your 8 CSV files:')
uploaded = files.upload()

saved_files = []

for name, content in uploaded.items():

    if not name.lower().endswith('.csv'):
        print(f'SKIPPED (not CSV): {name}')
        continue

    out_path = os.path.join(RAW, name)

    with open(out_path, 'wb') as fh:
        fh.write(content)

    saved_files.append(out_path)
    print(f'Saved: {name} -> {out_path}')

print(f'\nSaved {len(saved_files)} CSV file(s).')

# ------------------------------------------------------------
# 3. Expected columns
# ------------------------------------------------------------
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
# 4. Find CSV files
# ------------------------------------------------------------
files_found = sorted(
    glob.glob(os.path.join(RAW, '**', '*.csv'), recursive=True)
)

print('\n' + '=' * 78)
print(f'Found {len(files_found)} CSV file(s)')
print('=' * 78)

# ------------------------------------------------------------
# 5. Inspect CSV files
# ------------------------------------------------------------
for f in files_found:

    print('\n' + '=' * 78)
    print('FILE:', os.path.basename(f))
    print('=' * 78)

    try:
        df = pd.read_csv(f)
    except UnicodeDecodeError:
        df = pd.read_csv(f, encoding='utf-8-sig')

    # Clean column names
    df.columns = [
        c.strip().split('.')[-1].strip()
        for c in df.columns
    ]

    # Check required columns
    missing = [
        c for c in INPUTS + TARGETS
        if c not in df.columns
    ]

    if missing:
        print('!! MISSING COLUMNS:', missing)
        continue

    print(f'shape: {df.shape[0]} rows x {df.shape[1]} cols')

    print('\nunique values per input column:')

    for c in INPUTS:
        values = sorted(
            df[c].dropna().unique().tolist(),
            key=lambda x: str(x)
        )

        print(
            f'  {c:22s} '
            f'n={df[c].nunique():3d} -> {values}'
        )

    print('\ntarget stats (min / mean / max / std):')

    for c in TARGETS:
        s = pd.to_numeric(df[c], errors='coerce')

        print(
            f'  {c:8s} '
            f'min={s.min():10.3f}  '
            f'mean={s.mean():10.3f}  '
            f'max={s.max():10.3f}  '
            f'std={s.std():10.3f}'
        )

    missing_total = int(df.isna().sum().sum())

    dup = int(
        df.duplicated(
            subset=INPUTS + TARGETS
        ).sum()
    )

    eui = pd.to_numeric(df['EUI'], errors='coerce')
    idd = pd.to_numeric(df['IDD'], errors='coerce')
    tl = pd.to_numeric(df['TL'], errors='coerce')

    wire = int(
        ((idd - eui).abs() < 1e-6).sum()
    )

    imp = int(
        ((tl > 24) | (idd < 0.5)).sum()
    )

    print('\nmissing values:', missing_total)

    print(
        f'health: '
        f'duplicate log rows={dup} | '
        f'IDD==EUI rows={wire} | '
        f'impossible rows={imp}'
    )

# ------------------------------------------------------------
# 6. Finish
# ------------------------------------------------------------
plog(
    'CELL 2+3',
    f'uploaded and inspected {len(files_found)} CSV files'
)

print('\n' + '=' * 78)
print('DONE — upload + inspection completed.')
print('=' * 78)
