# ============================================================================
# scripts/cell01_setup_environment.py
# Cell 1 — setup: Drive mount, folder scaffolding, plog logger
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 0
# (52 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

from google.colab import drive
import os
import datetime

MOUNTPOINT = '/content/drive'

drive.mount(MOUNTPOINT, force_remount=True)

# Fail early if Google Drive did not actually mount
DRIVE_ROOT = os.path.join(MOUNTPOINT, 'MyDrive')
if not os.path.isdir(DRIVE_ROOT):
    raise RuntimeError(
        f"Google Drive is not mounted correctly at {DRIVE_ROOT}"
    )

BASE = os.path.join(DRIVE_ROOT, 'PCM_Study')

for sub in [
    'data/raw',
    'data/clean',
    'data/processed',
    'tables',
    'figures',
    'models',
    'predictions',
    'logs',
    'backup'
]:
    os.makedirs(os.path.join(BASE, sub), exist_ok=True)

RNG = 42

def plog(cell, note):
    os.makedirs(os.path.join(BASE, 'logs'), exist_ok=True)
    with open(
        os.path.join(BASE, 'logs', 'pipeline_log.txt'),
        'a',
        encoding='utf-8'
    ) as f:
        f.write(
            f"{datetime.datetime.now().isoformat()} | "
            f"{cell} | {note}\n"
        )

plog('CELL 1', 'setup complete')

print('Setup OK — BASE:', BASE)
print(
    'Folders ready:',
    'data/raw, data/clean, data/processed, tables, models, '
    'predictions, figures, logs, backup'
)
