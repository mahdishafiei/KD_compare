"""
Batch process all CSV files in csvs/ folder and generate per-antigen PNG tables in output/.
Each CSV gets its own subfolder under output/ named after the CSV file.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path("csvs")
OUTPUT_DIR = Path("output")
R2_THRESHOLD = 0.95

FIT_TYPE_MAP = {"fit_2_1": "KD2", "fit_1_1": "KD (M)"}
UNIT_THRESHOLDS = [
    (1e-9, "pM", 1e12),
    (1e-6, "nM", 1e9),
    (1.0,  "\u00b5M", 1e6),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_kd(value_m):
    if value_m is None or np.isnan(value_m):
        return "N/A"
    for threshold, unit, scale in UNIT_THRESHOLDS:
        if value_m < threshold:
            return f"{value_m * scale:.3g} {unit}"
    return f"{value_m * 1e6:.3g} \u00b5M"


def detect_fit_type(filename):
    name_lower = filename.lower()
    for pattern, col in FIT_TYPE_MAP.items():
        if pattern in name_lower:
            return col
    return None


def parse_kd_value(raw):
    if raw is None:
        return None, False
    s = str(raw).strip()
    if s == "" or s.lower() in ("nan", "n/a", "na", "#num!", "error"):
        return None, False
    below_limit = s.startswith("<")
    s_clean = s.lstrip("<").strip()
    try:
        return float(s_clean), below_limit
    except ValueError:
        return None, False


def clean_csv(filepath, kd_column, r2_threshold):
    df = pd.read_csv(filepath, dtype=str)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    for col in ("Sample ID", "Loading Sample ID", "Conc. (nM)", "Full R^2", kd_column):
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}', skipping file.")
            return None
        df[col] = df[col].astype(str).str.strip()

    # Filter buffer
    df = df[~(df["Sample ID"].str.lower() == "buffer") & ~(df["Loading Sample ID"].str.lower() == "buffer")].copy()
    # Filter N/A concentrations
    df = df[~df["Conc. (nM)"].str.lower().isin(["n/a", "na", "nan", ""])].copy()
    # Filter by R^2
    df["_r2"] = pd.to_numeric(df["Full R^2"], errors="coerce")
    df = df[df["_r2"] >= r2_threshold].copy()
    # Parse KD
    parsed = df[kd_column].apply(parse_kd_value)
    df["kd_value"] = [v for v, _ in parsed]
    df["below_limit"] = [b for _, b in parsed]
    df = df[df["kd_value"].notna()].copy()

    return df


def make_safe_name(name):
    return re.sub(r'[^\w\-]', '_', name).strip('_')


def render_table_png(df, antigen, antibody_order, output_path):
    ag_df = df[df["Sample ID"] == antigen].copy()
    ag_df["Conc_nM"] = pd.to_numeric(ag_df["Conc. (nM)"], errors="coerce")
    dilutions = sorted(ag_df["Conc_nM"].dropna().unique(), reverse=True)

    # Column headers
    col_headers = ["Conc. (nM)"]
    for ab in antibody_order:
        col_headers += [f"{ab}\nKD", f"{ab}\nR\u00b2"]

    # Data rows
    cell_text = []
    for conc in dilutions:
        row = [f"{conc:g}"]
        for ab in antibody_order:
            ab_conc = ag_df[(ag_df["Loading Sample ID"] == ab) & (ag_df["Conc_nM"] == conc)]
            if ab_conc.empty:
                row += ["\u2014", "\u2014"]
            else:
                r = ab_conc.iloc[0]
                kd_str = format_kd(r["kd_value"])
                if r["below_limit"]:
                    kd_str = "<" + kd_str
                row += [kd_str, f"{r['_r2']:.4f}"]
        cell_text.append(row)

    # Geo mean row
    mean_row = ["Geo. Mean"]
    for ab in antibody_order:
        ab_vals = ag_df[ag_df["Loading Sample ID"] == ab]["kd_value"].dropna().values
        if len(ab_vals) > 0:
            gm = float(np.exp(np.mean(np.log(ab_vals))))
            mean_row += [format_kd(gm), ""]
        else:
            mean_row += ["\u2014", ""]
    cell_text.append(mean_row)

    # Render
    n_rows = len(cell_text)
    n_cols = len(col_headers)
    fig_width = max(8, n_cols * 1.4)
    fig_height = max(2, (n_rows + 1) * 0.45 + 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(antigen, fontsize=14, fontweight="bold", pad=12, loc="left")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#cccccc")

    # Style data rows
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor("#cccccc")
            if j == 0:
                cell.set_facecolor("#ecf0f1")
                cell.set_text_props(fontweight="bold")
            if i == n_rows:
                cell.set_facecolor("#f7f9f9")
                cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#2c3e50")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}/")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {INPUT_DIR}/\n")

    for csv_path in csv_files:
        print(f"Processing: {csv_path.name}")

        # Detect fit type
        kd_column = detect_fit_type(csv_path.name)
        if kd_column is None:
            kd_column = "KD2"
            print(f"  Could not auto-detect fit type, defaulting to KD2")
        else:
            print(f"  KD column: {kd_column}")

        # Clean data
        df = clean_csv(csv_path, kd_column, R2_THRESHOLD)
        if df is None or df.empty:
            print(f"  No valid data after filtering, skipping.\n")
            continue

        # Create output subfolder named after the CSV
        folder_name = make_safe_name(csv_path.stem)
        out_dir = OUTPUT_DIR / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sort antibodies: WT first
        antibody_list = sorted(df["Loading Sample ID"].unique())
        antibody_order = sorted(antibody_list, key=lambda x: (0 if "wt" in x.lower() else 1, x.lower()))
        antigen_list = sorted(df["Sample ID"].unique())

        print(f"  Antigens: {antigen_list}")
        print(f"  Antibodies: {antibody_order}")
        print(f"  Output: {out_dir}/")

        for antigen in antigen_list:
            safe_ag = make_safe_name(antigen)
            png_path = out_dir / f"{safe_ag}.png"
            render_table_png(df, antigen, antibody_order, png_path)
            print(f"    Saved: {png_path}")

        print()

    print("Done!")


if __name__ == "__main__":
    main()
