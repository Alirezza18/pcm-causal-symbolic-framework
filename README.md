# PCM Causal–Symbolic Framework

Code companion to the paper *"Certified, transfer-limited design rules for
PCM-integrated envelopes across eight Iranian climates"* (under review).

The pipeline turns 3,861 EnergyPlus simulations of PCM-integrated envelope
designs (8 Iranian climate zones, Latin-Hypercube-sampled design space) into
gradient-boosting surrogate models, then interrogates the surrogates with
global sensitivity analysis (Morris → Sobol), causal discovery (PC, GES,
NOTEARS), SHAP attribution, and symbolic regression — converging on design
rules, Pareto-optimal knee configurations, uncertainty bands, and an
EnergyPlus audit of the recommended designs.

## Repository layout

```
├── data/
│   ├── raw/                     # 8 raw per-city LHS CSVs, 11,988 rows total (EnergyPlus campaign)
│   │   ├── LHS_Ardebial .csv
│   │   ├── LHS_Bandarabbas.csv
│   │   ├── LHS_Bushhr .csv
│   │   ├── LHS_Hamedan.csv
│   │   ├── LHS_Kashan .csv
│   │   ├── LHS_Tehran.csv
│   │   ├── LHS_rasht.csv
│   │   └── LHS_tabriz.csv
│   └── pcm_results_clean.csv    # cleaned, merged 3,861-run harvest (inputs + EUI/IDD/TL + city)
├── scripts/                     # 44 numbered scripts, one per notebook cell
│   ├── cell01_setup_environment.py
│   ├── cell02_upload_inspect_city_csvs.py
│   ├── ...
│   └── cell20a_fig17_synthesis_schematic.py
├── pcm_causal_symbolic_framework.ipynb   # original Colab notebook (reference)
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline at a glance

| Stage | Scripts | Paper section | Key outputs |
|---|---|---|---|
| Setup + ingestion | 01–02 | Methods | Drive folder scaffolding, per-file audit |
| Cleaning + audit | 04 | Methods | 11,988 → 3,861 rows (duplicates, IDD==EUI wiring, impossible physics) |
| Canonical dataset | 05 | Methods | `pcm_dataset_final.csv`, 3,861 × 14, stable case IDs |
| EDA | 06 | §2/§3 | ranges, TL regimes, Spearman, η² drivers |
| Surrogate benchmark | 07, 07c | §3.1, Table 3/Fig 5 | 5 families × 23 modeled cells; CatBoost wins 19 |
| Hold-out validation | 08 | §3.1 | untouched-20% scores |
| Sensitivity | 09, 09b–9d | §3.2, Table 2, Figs 6–7 | Morris, Sobol ST, D(Yk) |
| Causal discovery | 10, Fig8, 10b | §3.3, Figs 8, Tables A15–A16 | PC + GES + NOTEARS consensus |
| SHAP | 11, 11b, 11c | §3.4, Fig 9, Table A17 | per-variable attribution, SHAP–Sobol agreement |
| Symbolic regression | 12 | §3.5 | 3-seed GP consensus formulas |
| Four-lens convergence | 13, 13b | §3.6, Table A18 | lens agreement matrix |
| Transferability | 14a–c | §3.7, Fig 11, Table A19 | cross-climate transfer matrices |
| Pareto / MOO | 15a–d | §3.8, Table 4, Figs 12 | knee designs, NSGA-II validation |
| Uncertainty | 16a–c | §3.9, Table 5, Fig 13 | conformal-style prediction uncertainty |
| Nomographs | 17a–d | §3.10, Table 6, Fig 14 | marginal effect curves, design rules |
| Economics | 18a–d | §3.11, Table 7, Fig 15 | cost/CO₂ translation |
| EnergyPlus audit | 19a–c | §3.12, Table 8, Fig 16 | 16 E+ runs vs surrogate parity |
| Synthesis | 20a | §4 | Fig 17 schematic |

## Getting started

The scripts were written for Google Colab (Drive-mounted `BASE = /content/drive/MyDrive/PCM_Study`).
To run them:

1. **Colab (as-is):** open the included notebook, or run the `scripts/` in order in Colab after
   mounting Drive with the `PCM_Study` folder structure (`data/raw`, `data/clean`,
   `data/processed`, `tables`, `figures`, `models`, `predictions`, `logs`, `backup`).
2. **Locally:** install the requirements, create the same folder structure, and replace `BASE`
   with a local path in the scripts. Cell 09/10/… helpers that call `google.colab` need a
   no-op shim or removal.

```bash
pip install -r requirements.txt
python scripts/cell04_clean_audit.py        # ...then each script in numeric order
```

Raw per-city LHS CSVs (8 files, `LHS_<city>.csv`, 11,988 rows before cleaning) are included in
`data/raw/`. Cell 02 uploads/inspects them and Cell 04 cleans them (removing duplicate log rows,
IDD==EUI wiring glitches, and physically impossible readings) into the 3,861-row dataset that
matches `data/pcm_results_clean.csv`, so the full pipeline reproduces from the raw campaign
output to the final tables and figures.

| Raw rows | Duplicates | Wiring glitches | Impossible physics | Clean rows |
|---|---|---|---|---|
| 11,988 | 8,116 | 18 | 6 | **3,861** |

## Design variables (inputs, coded levels)

`pcm_type_{wall,roof}` (2) · `thickness_{wall,roof}_level` (4) ·
`melting_point_{wall,roof}_level` (5) · `position_{wall,roof}` (3, interior/middle/exterior)

## Performance indicators (targets)

- `eui_kwh_m2` — annual energy use intensity
- `idd_deg_h` — integrated degree of discomfort
- `tl_h` — time lag (constant at −1 h in Tabriz; excluded from modeling there)

## Citation

If you use this code, please cite the paper (citation to be added on acceptance) and this
repository.
