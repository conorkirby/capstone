# Capstone: Statistical Analysis of Bubble Dynamics in 2D Wet Foams

Physics capstone project analysing the trajectories of individual bubbles in simulated
two-dimensional wet foams. The bubble motion is characterised with tools from statistical
physics and econophysics (displacement distributions, log-returns, mean-squared
displacement, autocorrelation, volatility clustering and coordination-number statistics),
and compared against synthetic Gaussian/Brownian random walks generated in this repo as a
null model.

Core questions the notebooks address:

- Do bubble displacements follow Gaussian statistics, or heavy-tailed (Lévy-like) statistics?
- Do the displacement PDFs at different lag times Δt collapse under a scaling exponent β?
- Is the mean-squared displacement diffusive (∝ t) or anomalous (∝ t^α)?
- Are there long-range correlations / volatility clustering, as seen in financial time series?
- How does foam topology (coordination number Z) evolve as the foam coarsens?

---

## Requirements

- **Python 3.9+** (the committed virtual environment was built with the macOS system Python 3.9.6;
  anything ≥3.9 works)
- **Jupyter** (JupyterLab, classic Notebook, or the VS Code notebook editor)
- A LaTeX distribution *only* if you want to rebuild `Figures/Latex_w_Figures/Capstone_figures.pdf`

### Python packages

| Package | Version used |
| --- | --- |
| numpy | 2.0.2 |
| pandas | 2.3.2 |
| matplotlib | 3.9.4 |
| scipy | 1.13.1 |
| seaborn | 0.13.2 |
| jupyterlab | 4.4.7 |
| notebook | 7.4.5 |
| ipykernel | 6.30.1 |

These are pinned in [requirements.txt](requirements.txt).

> **Note:** [ProjectNotebooks/10_diffcoeff.py](ProjectNotebooks/10_diffcoeff.py) sets
> `matplotlib.use('Qt5Agg')` for interactive plotting. That script additionally needs `PyQt5`
> (`pip install PyQt5`). None of the notebooks need it; they use the inline backend.

---

## Setup

```bash
git clone https://github.com/conorkirby/capstone.git
cd capstone

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# register the venv as a Jupyter kernel
python -m ipykernel install --user --name capstone --display-name "Python (capstone)"

jupyter lab
```

All data files needed to run every notebook are committed under [DataFiles/](DataFiles/),
so there is nothing external to download.

---

## ⚠️ Before you run anything: hard-coded paths

The notebooks were written against one machine and contain **absolute paths** that will not
resolve on yours. Two places need fixing:

1. **The data file.** [ProjectNotebooks/data_loader.py](ProjectNotebooks/data_loader.py) sets
   `DATA_FILE` to an absolute path beginning
   `/Users/conorkirby/Library/Mobile Documents/...`.
2. **Figure output.** Nearly every `plt.savefig(...)` call in the notebooks writes to an
   absolute path under `.../capstone/Figures/...` or `.../capstone/Figures_Sample2/...`.

The cheapest fix for `data_loader.py` is to make the path relative to the file itself:

```python
DATA_FILE = Path(__file__).resolve().parent.parent / "DataFiles" / "wetfoam2_bub_RH2_0.020000_0.507713.txt"
```

For figures, a project-wide find-and-replace of the absolute prefix with a relative
`../Figures/` works, e.g. from the repo root:

```bash
grep -rl "/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/" \
  ProjectNotebooks/*.ipynb \
  | xargs sed -i '' 's#/Users/conorkirby/Library/Mobile Documents/com~apple~CloudDocs/Coding/projects/python/capstone/#../#g'
```

(On Linux drop the `''` after `-i`. Commit first; this rewrites the notebooks in place.)

---

## The data

### Format

Each file in [DataFiles/](DataFiles/) is a whitespace-separated text dump of a wet-foam
simulation, one block per timestep. Comment lines start with `#`; a line beginning `#id`
marks the start of a new timestep, which is how `data_loader.py` reconstructs the time axis,
since **there is no explicit time column**.

```
#400 bubbles
#width=20.203051, height=20.203051, time =1
###vcentrex= 10.039337, ...
#id,,x,y,area,pressure,Z
1  16.830467  10.024328  1.505199  1.051348  16
...
```

Columns as read by the loader: `id`, `x`, `y`, `area`, `pressure`, `Z`.

**Important:** the `area` column actually holds a **radius**. `data_loader.py` derives the true
area as `actual_area = π · area²`.

### Datasets

Filenames follow `wetfoam<N>_bub_RH<N>_<liquid_fraction>_<something>.txt`. The number in the
middle is the liquid fraction φ.

| File | φ | Box width | Timesteps | Columns |
| --- | --- | --- | --- | --- |
| `wetfoam2_bub_RH2_0.020000_0.507713.txt` | 0.02 | 20.203051 | 726 | 6 |
| `wetfoam2_bub_RH2_0.200000_0.501910.txt` | 0.20 | 22.360680 | 3427 | 6 |
| `wetfoam_bub_RH2_0.080000_0.507713.txt` | 0.08 | 20.851441 | 1203 | 5 (no Z) |
| `wetfoam_bub_RH2_0.140000_0.502800.txt` | 0.14 | 20.000000 | 2523 | 5 (no Z) |
| `wetfoam3_bub_RH3_0.020000_0.500686.txt` | 0.02 | 32.665584 † | 293 | 7 |
| `wetfoam3_bub_RH3_0.080000_0.501221.txt` | 0.08 | 39.562828 | 198 | 7 |
| `wetfoam3_bub_RH3_0.140000_0.501910.txt` | 0.14 | 41.343710 | 308 | 7 |

† non-square box (`height=31.347984`); the loader assumes a square box, so this one needs care.

The `bubblesample*.txt`, `bubbletroubleshoot.txt`, `bubbledisappear.txt` and
`TroubleshootDataRemoval.txt` files are small hand-made extracts used by the troubleshooting
notebooks.

#### Known trap: column-count mismatch

`data_loader.py` hard-codes six column names. This is **only correct for the `wetfoam2_*`
files**:

- **5-column files** (`wetfoam_bub_RH2_*`) → `Z` comes back as all-`NaN`, silently breaking
  [07_CoordinationNumber.ipynb](ProjectNotebooks/07_CoordinationNumber.ipynb).
- **7-column files** (`wetfoam3_*`) → pandas consumes the first column as the DataFrame index
  and **every column is shifted by one** (`x` holds y, `area` holds pressure, …). Results will
  be wrong, not error out.

If you switch datasets, fix the `names=[...]` list in `read_csv` to match the actual column
count before trusting any output.

### Switching dataset

`data_loader.py` keeps the alternatives as commented-out lines. To switch you must change
**both** `DATA_FILE` and the matching `BOX_SIZE`. They are separate constants and nothing
checks that they agree. `BOX_SIZE` is used for the periodic-boundary unwrapping, so a
mismatch corrupts every trajectory.

---

## What `data_loader.py` does

It is a **script, not a library of functions**: notebooks run `from data_loader import *`,
which executes the whole file and pulls the resulting globals into the notebook namespace.
Steps:

1. Reads the raw file into a DataFrame.
2. Reconstructs `timestep` by counting `#id` header lines.
3. Flags bubbles whose last appearance precedes the final timestep (`disappearing_ids`),
   the bubbles that pop or coarsen away.
4. **Unwraps periodic boundaries**: if a bubble jumps more than half a box in one step, the
   rest of its track is shifted by ±`BOX_SIZE`, turning wrapped coordinates back into
   continuous trajectories. This is the correction motivated in
   [OriginalTroubleshootBubbles.ipynb](OriginalTroubleshootBubbles.ipynb).
5. Computes derived quantities.

Names it exports into the notebook namespace:

| Name | Meaning |
| --- | --- |
| `df` | raw DataFrame (wrapped coordinates) |
| `df_corrected` | boundary-corrected DataFrame, plus `actual_area` |
| `disappearing_ids` | set of bubble IDs that vanish before the end |
| `bubbles_per_timestep` | bubble count vs time |
| `avg_area_per_timestep` | mean bubble area vs time |
| `approx_avg_area` | area estimated from liquid fraction: `L²(1−φ)/N` |
| `A_0` | initial mean bubble area (used to non-dimensionalise lengths) |
| `max_timestep`, `final_step` | last timestep index |
| `box_area` | `BOX_SIZE²` |
| `BOX_SIZE`, `PERIODIC_THRESHOLD`, `LIQUID_FRACTION` | constants |

Because it is import-executed, notebooks re-run it with
`importlib.reload(data_loader)` after editing constants:

```python
import data_loader, importlib
importlib.reload(data_loader)
from data_loader import *
```

---

## File structure

```
capstone/
├── README.md
├── requirements.txt
│
├── DataFiles/                       # all simulation output + small test extracts
│
├── ProjectNotebooks/                # the main analysis pipeline (run in numeric order)
│   ├── data_loader.py               # shared loader: edit paths/constants HERE
│   ├── 01_BubblePaths.ipynb
│   ├── 02_SingleBubbleAnalysis.ipynb
│   ├── 03_DisplacementDistributions.ipynb
│   ├── 03.1_DispDist_Extra.ipynb
│   ├── 03.2_ScalingInvestigation.ipynb
│   ├── 03.5_Gauss_DisplacementDistributions.ipynb
│   ├── 04_LogReturns.ipynb
│   ├── 05_AreaAnalysis.ipynb
│   ├── 06_MSD_Analysis.ipynb
│   ├── 07_CoordinationNumber.ipynb
│   ├── 08_MagnitudeDisplacements.ipynb
│   ├── 09_AutoCorrelation.ipynb
│   ├── 09.5_AutoCorrelation_LogReturns.ipynb
│   ├── 10_Volatility.ipynb
│   ├── 10.5_Volatility_Gaussian.ipynb
│   ├── 10_diffcoeff.py              # notebook 10 exported as a Qt5-backend script
│   └── 11_DiffusionCoefficient.ipynb
│
├── Figures/                         # output for the primary dataset (φ = 0.02)
│   ├── 00_Thesis_Figures/           # curated final figures used in the write-up
│   ├── 01_Figures/ … 10_Figures/    # per-notebook output, numbering matches notebooks
│   └── Latex_w_Figures/             # LaTeX doc collecting the figures + built PDF
│
├── Figures_Sample2/                 # same figures for the second dataset (files suffixed _2)
│   └── 01_Figures/ … 10_Figures/
│
├── 1DBrownianMotion.ipynb           # 1D random-walk reference model
├── 2DGaussianWalk.ipynb             # 2D Gaussian random walk, null model for comparison
├── BubbleDataSelect.ipynb           # earlier exploratory notebook (varying Δt, correlations)
├── OriginalTroubleshootBubbles.ipynb   # why periodic boundaries must be unwrapped, not dropped
└── TroubleshootDataRemoval.ipynb       # experiments with removing "bad" rows (superseded)
```

### Notebook guide

Each is self-contained after the `data_loader` import.

| Notebook | What it produces |
| --- | --- |
| `01_BubblePaths` | trajectory plots, zoomed walks, mean position vs time, path heatmap, bubble lifetime histogram and survival curve |
| `02_SingleBubbleAnalysis` | position changes and log-returns for individual tracked bubbles |
| `03_DisplacementDistributions` | displacement PDFs across Δt, CCDFs of x/y components, PDF collapse under scaling exponent β, Lévy fit, x–y symmetry check |
| `03.1_DispDist_Extra` | alternative ways to extract β: two-regime power-law peak model vs smooth spline on width-vs-Δt |
| `03.2_ScalingInvestigation` | sensitivity of the CCDF cut-scaling / collapse to the fitted constants |
| `03.5_Gauss_DisplacementDistributions` | the same displacement analysis applied to Brownian data, the control |
| `04_LogReturns` | log-returns of bubble position (with the sign/zero-crossing correction discussed in-notebook), CCDFs, collapse, ACF |
| `05_AreaAnalysis` | mean bubble area vs time; measured area vs the liquid-fraction approximation `L²(1−φ)/N`, with absolute and percent error |
| `06_MSD_Analysis` | mean-squared displacement, power-law fit for the anomalous exponent, MSD next to the Gaussian walk, statistics-vs-time diagnostics |
| `07_CoordinationNumber` | distribution of Z over time, ⟨Z⟩ evolution, second moment μ₂ / variance of Z, area-vs-Z relation, topological rearrangement activity |
| `08_MagnitudeDisplacements` | per-step displacement magnitude vs time, moving average, normalised step-size histogram |
| `09_AutoCorrelation` | displacement autocorrelation via the FFT (Wiener–Khinchin) method |
| `09.5_AutoCorrelation_LogReturns` | ACF of log-returns at several Δt, on linear and log–log axes |
| `10_Volatility` | volatility series and the ACF of |log-return|, the volatility-clustering test |
| `10.5_Volatility_Gaussian` | volatility of the Gaussian walk, for contrast |
| `11_DiffusionCoefficient` | diffusion coefficient from the scaling exponent (α = 1/β) |

---

## Rebuilding the LaTeX figure document

```bash
cd Figures/Latex_w_Figures
latexmk -pdf Capstone_figures.tex
```

Build artefacts (`.aux`, `.fls`, `.fdb_latexmk`, `.log`, `.synctex.gz`) are currently committed
alongside the source; they are safe to delete.

---

## Notes for continuing the work

Things a future contributor should know about the state of the repo:

- **No `.gitignore`.** `.DS_Store` files are tracked, and there is nothing stopping `.venv/`
  from being committed (it currently is not; the venv carries its own catch-all ignore file).
  Adding a `.gitignore` covering `.DS_Store`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
  and the LaTeX build artefacts is the first cleanup worth doing.
- **Notebook outputs are committed**, which is why several `.ipynb` files run to megabytes
  (`BubbleDataSelect.ipynb` is ~2.7 MB). Consider `nbstripout` if diffs become unmanageable.
- **`data_loader.py` is a script, not a module.** Refactoring it into a
  `load(path, box_size)` function returning a DataFrame would remove the
  `importlib.reload` dance, make the box-size/data-file pairing impossible to get wrong, and
  let the column names be inferred from the actual column count (fixing the 5/7-column trap
  above).
- **Two figure trees, one code path.** `Figures/` and `Figures_Sample2/` hold the same analyses
  for two datasets, distinguished only by a `_2` filename suffix that is typed by hand in each
  `savefig` call. Driving the output directory from the dataset name would remove the manual
  step.
- **Duplicate figure names across trees** (e.g. `displacement_overlay_2.png` appears in both
  `Figures/03_Figures/` and `Figures_Sample2/03_Figures/`), so check which tree a notebook
  is writing to before assuming a figure is stale.
- **`10_diffcoeff.py` is a stale export** of the volatility/diffusion notebook, kept for
  interactive Qt5 plotting. It will drift from the notebook; treat the notebook as the source
  of truth.

---

## Repository

<https://github.com/conorkirby/capstone>
