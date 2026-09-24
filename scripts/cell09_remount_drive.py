# ============================================================================
# scripts/cell09_remount_drive.py
# Utility — Drive remount + tables folder listing (unlabeled cell)
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 9
# (13 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

from google.colab import drive
import os

# اتصال مجدد و اجباری گوگل درایو
drive.mount('/content/drive', force_remount=True)

# بررسی محتویات پوشه
path = '/content/drive/MyDrive/PCM_Study/tables'
if os.path.exists(path):
    print("فایل‌های موجود در پوشه tables:")
    print(os.listdir(path))
else:
    print("مسیر PCM_Study/tables پیدا نشد.")
