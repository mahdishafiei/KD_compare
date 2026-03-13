# KD Comparison Tool

A pipeline for analyzing and comparing dissociation constant (KD) values from BLI/SPR kinetic result tables. Processes all CSV files in batch and generates publication-quality tables and comparison plots.

## Project Structure

```
KD_compare/
├── csvs/                  # Input: drop CSV files here
├── output/                # Output: generated per-run (gitignored)
│   ├── <csv_name>/        #   Per-antigen KD table PNGs
│   └── comparison_plots/  #   Cross-round comparison plots
├── run_pipeline.py        # Main pipeline (batch processing)
├── kd_analysis.ipynb      # Jupyter notebook (single-file analysis)
├── app.py                 # Streamlit web app (interactive)
├── generate_tables.py     # Standalone table generator
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/mahdishafiei/KD_compare.git
cd KD_compare
pip install -r requirements.txt
```

## Usage

### Option 1: Pipeline (recommended)

Drop your CSV files into `csvs/`, then run:

```bash
python3 run_pipeline.py
```

This will:
1. Process all CSVs in `csvs/`
2. Generate per-antigen KD tables (PNG) in `output/<csv_name>/`
3. Generate comparison plots (bar charts + fold-change heatmaps) in `output/comparison_plots/`

Options:
```bash
python3 run_pipeline.py --r2 0.90          # custom R² threshold
python3 run_pipeline.py --input my_csvs/   # custom input directory
```

### Option 2: Jupyter Notebook

```bash
jupyter notebook kd_analysis.ipynb
```

Edit the Settings cell to point to your CSV, then Run All. Good for exploring a single file interactively.

### Option 3: Streamlit Web App

```bash
streamlit run app.py
```

Upload a CSV via the browser UI for interactive analysis with adjustable settings.

## Input Format

Kinetic Result Table CSV exported from BLI/SPR software with these required columns:

| Column | Description |
|---|---|
| `Sample ID` | Antigen name |
| `Loading Sample ID` | Antibody name |
| `Conc. (nM)` | Analyte concentration in nM |
| `Full R^2` | Goodness of fit |
| `KD2` or `KD (M)` | Dissociation constant (depends on fit model) |

Fit type is auto-detected from the filename (`fit_2_1` → `KD2`, `fit_1_1` → `KD (M)`), with fallback to `KD2`.

## Data Cleaning

- Buffer reference rows removed
- Rows with missing concentrations excluded
- R² threshold filter (default: 0.95)
- Below-detection-limit values (e.g., `<1.0E-12`) parsed and flagged with `*`
- Missing/unparseable KD values skipped

## Output

- **Per-antigen tables**: KD and R² at each dilution, WT and variants side by side, with geometric mean
- **Bar charts**: Grouped KD comparison across antigens (log scale)
- **Fold-change heatmaps**: Log2 fold change vs. WT (blue = tighter binding)

## Dependencies

- Python >= 3.8
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- plotly >= 5.18.0
- openpyxl >= 3.1.0
- streamlit >= 1.32.0 (web app only)

## License

MIT
