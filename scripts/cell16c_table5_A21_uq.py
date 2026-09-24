# ============================================================================
# scripts/cell16c_table5_A21_uq.py
# Cell 16c — Table 5 (main text) + Table A21
# Source: pcm_causal_symbolic_framework.ipynb, notebook cell index 31
# (119 lines of code, unchanged from the executed Colab notebook
#  except that Colab '!' shell lines are commented out).
# Environment: written for Google Colab (Drive mount, google.colab imports).
# To run locally, install requirements.txt and point BASE at a local folder
# with the same structure (data/, tables/, figures/, models/, logs/).
# Run order + full paper mapping: see README.md.
# ============================================================================

# ============================================================================
# CELL 16c - TABLE 5 (main text) + TABLE A21 (appendix)  (run AFTER Cell 16a)
# Table 5  : one row per KPI, averaged across the modeled zones - conformal
#            coverage (mean +/- SE), normalized interval width, epistemic
#            variance fraction, half-width in engineering units.
# Table A21: per-cell detail (21 rows).
# Read-only w.r.t. Cell 16a; prints diagnostics for Section 3.7.
# ============================================================================
import os
try:
    list('/content/drive/MyDrive')
    print('Drive already mounted.')
except OSError:
    from google.colab import drive
    drive.mount('/content/drive')
    print('Drive mounted.')

import numpy as np
import pandas as pd
from google.colab import files

BASE = '/content/drive/MyDrive/PCM_Study'
cf = pd.read_csv(BASE + '/tables/uq_conformal.csv')
vd = pd.read_csv(BASE + '/tables/uq_vardec.csv')
KPIS = ['EUI', 'IDD', 'TL']
ZONES = ['0B', '1B', '2B', '3A', '3B', '4A', '4B', '5C']

assert len(cf) == 21 and len(vd) == 21, 'run Cell 16a first'
m = cf.merge(vd[['zone', 'kpi', 'sigma_resid', 'sd_model', 'f_model', 'sd_norm']],
             on=['zone', 'kpi'], suffixes=('', '_vd'))
assert len(m) == 21

# ---------------- diagnostics for the Section 3.7 text ----------------
print('=' * 76)
print('DIAGNOSTICS (for Section 3.7)')
print('=' * 76)
below = m[m.coverage < 0.90]
print(f'cells below nominal 90% coverage: {len(below)}/21')
if len(below):
    print(below[['zone', 'kpi', 'coverage', 'width_norm']].round(3).to_string(index=False))
print(f'coverage range: {m.coverage.min():.3f} ({m.loc[m.coverage.idxmin(),"zone"]} '
      f'{m.loc[m.coverage.idxmin(),"kpi"]}) .. {m.coverage.max():.3f}')
print(f'normalized width range: {m.width_norm.min():.4f} .. {m.width_norm.max():.4f}')
wid = m.pivot_table(index='zone', columns='kpi', values='width_norm')
print('\nnormalized width by zone x KPI:')
print(wid.reindex(ZONES).round(3).to_string())
print('\nmean f_model by KPI:')
print(m.groupby('kpi').f_model.mean().round(3).to_string())
corr = np.corrcoef(m.f_model, m.width_norm)[0, 1]
print(f'\ncorr(f_model, width_norm) = {corr:.2f} (informational: width tracks '
      f'residual noise via the calibration quantile; f_model is a variance '
      f'ratio - the two need not move together)')
sdn = m.pivot_table(index='zone', columns='kpi', values='sd_norm')
print('\nnormalized total SD by zone x KPI:')
print(sdn.reindex(ZONES).round(4).to_string())

# ---------------- Table 5 (main text) ----------------
t5 = pd.DataFrame({
    'KPI': KPIS,
    'Cells': [int((m.kpi == k).sum()) for k in KPIS],
    'Coverage (%)': [f'{m[m.kpi==k].coverage.mean()*100:.1f} ± '
                     f'{m[m.kpi==k].cov_se.mean()*100:.1f}' for k in KPIS],
    'Width (eng.)': [f'{m[m.kpi==k].width.mean():.2f}' for k in KPIS],
    'Width (norm.)': [f'{m[m.kpi==k].width_norm.mean():.3f}' for k in KPIS],
    'Half-width (eng.)': [f'{m[m.kpi==k].width.mean()/2:.2f}' for k in KPIS],
    'f_model (%)': [f'{m[m.kpi==k].f_model.mean()*100:.0f}' for k in KPIS],
    'sigma_resid': [f'{m[m.kpi==k].sigma_resid.mean():.2f}' for k in KPIS],
})
p5 = BASE + '/tables/table5_uncertainty.csv'
t5.to_csv(p5, index=False)
print('\n' + '=' * 76)
print(f'Table 5 - conformal + decomposition by KPI -> {p5}')
print('=' * 76)
print(t5.to_string(index=False))

# ---------------- Table A21 (appendix) ----------------
ZA = {z: i for i, z in enumerate(ZONES)}
KA = {'EUI': 0, 'IDD': 1, 'TL': 2}
m['zo'], m['ko'] = m.zone.map(ZA), m.kpi.map(KA)
mA = m.sort_values(['zo', 'ko']).drop(columns=['zo', 'ko'])
a21 = pd.DataFrame({
    'Zone': mA.zone, 'City': mA.city, 'KPI': mA.kpi, 'Adopted': mA.adopted,
    'Coverage': mA.coverage.round(3), 'Width': mA.width.round(2),
    'Width (norm.)': mA.width_norm.round(4),
    'sigma_resid': mA.sigma_resid.round(3), 'sd_model': mA.sd_model.round(3),
    'f_model': mA.f_model.round(3), 'sd_total (norm.)': mA.sd_norm.round(4),
})
p21 = BASE + '/tables/table_A21_uncertainty.csv'
a21.to_csv(p21, index=False)
print('\n' + '=' * 76)
print(f'Table A21 - per-cell uncertainty (21 rows) -> {p21}')
print('=' * 76)
print(a21.head(9).to_string(index=False))
print('... (9 of 21 rows shown)')

print('\nCaption Table 5:')
print('Table 5. Conformal prediction intervals and uncertainty decomposition '
      'by performance indicator, averaged across the modeled climate zones. '
      'Coverage: empirical coverage of the nominal 90% split-conformal '
      'interval (10 repeated 60/20/20 splits; +/- the across-split standard '
      'error). Width: mean interval width in engineering units and '
      'normalized by the observed range of the KPI in each zone; the '
      'half-width is the marginal error budget at the 90% level. f_model: '
      'epistemic fraction of the prediction variance (30 paired bootstrap '
      'refits) with the complement attributable to residual noise; '
      'sigma_resid is the residual standard deviation of the full-data fit. '
      'The three Section 3.1 exclusions are absent; per-cell values in '
      'Table A21.')

print('\nCaption Table A21:')
print('Table A21. Per-cell uncertainty quantities behind Table 5: achieved '
      'conformal coverage and interval width (engineering and normalized), '
      'the residual and epistemic standard deviations, and the epistemic '
      'variance fraction for every modeled zone-KPI cell, with the adopted '
      'surrogate family recorded per row.')

files.download(p5)
files.download(p21)
print('\nDONE 16c - paste this output back.')
