# KD Comparison Tool

A tool for analyzing and comparing dissociation constant (KD) values from BLI (Bio-Layer Interferometry) and SPR (Surface Plasmon Resonance) kinetic result tables. Available as both a **Streamlit web app** and a **Jupyter notebook**.

## Features

- **Automatic fit type detection** from filenames (`fit_1_1` or `fit_2_1`)
- **Data cleaning pipeline**:
  - Removes buffer reference rows
  - Filters out rows with missing concentrations
  - Excludes rows below a configurable R² threshold (default: 0.95)
  - Parses below-detection-limit values (e.g., `<1.0E-12`) and flags them
  - Skips rows with missing or unparseable KD values
- **Geometric mean KD** calculated per Antigen x Antibody pair across all concentration points
- **Fold-change analysis** vs. a WT (wild-type) reference antibody (auto-detected or user-selected)
- **Interactive visualizations** (Plotly):
  - Log2 fold-change heatmap
  - Grouped bar chart (log scale)
  - Scatter plot of individual KD values per concentration, faceted by antigen
- **Export** cleaned comparison table as CSV

## Input Format

The tool expects a **Kinetic Result Table CSV** exported from BLI/SPR software with the following required columns:

| Column | Description |
|---|---|
| `Sample ID` | Antigen name |
| `Loading Sample ID` | Antibody name |
| `Conc. (nM)` | Analyte concentration in nM |
| `Full R^2` | Goodness of fit |
| `KD2` or `KD (M)` | Dissociation constant (depends on fit model) |

- **`fit_2_1`** files use the `KD2` column
- **`fit_1_1`** files use the `KD (M)` column

The fit type is auto-detected from the filename (looks for `fit_2_1` or `fit_1_1`) but can be overridden manually.

## Installation

```bash
git clone https://github.com/mahdishafiei/KD_compare.git
cd KD_compare
pip install -r requirements.txt
```

## Usage

### Option 1: Jupyter Notebook (recommended for quick analysis)

```bash
jupyter notebook kd_analysis.ipynb
```

Edit the **Settings** cell to point to your CSV file:

```python
CSV_PATH = "your_kinetic_result_table.csv"
R2_THRESHOLD = 0.95
FIT_TYPE_OVERRIDE = None  # Set to "KD2" or "KD (M)" to override auto-detection
```

Then **Run All** cells. Results are displayed inline and exported to `kd_comparison_output.csv`.

### Option 2: Streamlit Web App

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501` where you can:
- Upload a CSV file via the sidebar
- Adjust R² threshold with a slider
- Override fit type detection
- Select the WT reference antibody
- Browse results across four tabs: **KD Summary**, **Fold Changes**, **Plots**, **Raw Data**
- Download the comparison table as CSV

## Output

### KD Summary Table
Geometric mean KD per Antigen x Antibody pair with automatic unit formatting (pM / nM / uM). Below-detection-limit values are flagged with `*`.

### Fold Change Table
KD(variant) / KD(WT) for each non-WT antibody. Values > 1 indicate weaker binding; values < 1 indicate tighter binding vs. the reference.

### Plots
- **Heatmap**: Log2 fold-change vs. WT reference (RdBu color scale, midpoint at 0)
- **Bar chart**: Geometric mean KD in nM grouped by antigen (log scale)
- **Scatter**: Individual KD values per concentration point, faceted by antigen

## Dependencies

- Python >= 3.8
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.18.0
- openpyxl >= 3.1.0
- streamlit >= 1.32.0 (for the web app)
- jupyter (for the notebook)

## License

MIT
