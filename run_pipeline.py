#!/usr/bin/env python3
"""
KD Comparison Pipeline
======================
Single entry point that processes all CSV files in csvs/ and generates:
  1. Per-antigen KD tables (PNG) in output/<csv_name>/
  2. Comparison plots (PNG) in output/comparison_plots/

Usage:
    python3 run_pipeline.py
    python3 run_pipeline.py --r2 0.90          # custom R² threshold
    python3 run_pipeline.py --input my_csvs/   # custom input directory
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIT_TYPE_MAP = {"fit_2_1": "KD2", "fit_1_1": "KD (M)"}
UNIT_THRESHOLDS = [
    (1e-9, "pM", 1e12),
    (1e-6, "nM", 1e9),
    (1.0,  "\u00b5M", 1e6),
]
COLORS = {
    "WT": "#2c3e50",
    "Fab22": "#e74c3c",
    "Fab6": "#2980b9",
    "Fab10": "#27ae60",
    "Fab11": "#f39c12",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_kd(value_m: float) -> str:
    if value_m is None or np.isnan(value_m):
        return "N/A"
    for threshold, unit, scale in UNIT_THRESHOLDS:
        if value_m < threshold:
            return f"{value_m * scale:.3g} {unit}"
    return f"{value_m * 1e6:.3g} \u00b5M"


def detect_fit_type(filename: str) -> str | None:
    name_lower = filename.lower()
    for pattern, col in FIT_TYPE_MAP.items():
        if pattern in name_lower:
            return col
    return None


def parse_kd_value(raw) -> tuple[float | None, bool]:
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


def safe_name(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', name).strip('_')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_clean(filepath: Path, kd_column: str, r2_threshold: float) -> pd.DataFrame | None:
    df = pd.read_csv(filepath, dtype=str)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    required = ("Sample ID", "Loading Sample ID", "Conc. (nM)", "Full R^2", kd_column)
    for col in required:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}', skipping.")
            return None
        df[col] = df[col].astype(str).str.strip()

    # Filter buffer rows
    df = df[
        ~(df["Sample ID"].str.lower() == "buffer")
        & ~(df["Loading Sample ID"].str.lower() == "buffer")
    ].copy()

    # Filter N/A concentrations
    df = df[~df["Conc. (nM)"].str.lower().isin(["n/a", "na", "nan", ""])].copy()

    # Filter by R²
    df["_r2"] = pd.to_numeric(df["Full R^2"], errors="coerce")
    df = df[df["_r2"] >= r2_threshold].copy()

    # Parse KD values
    parsed = df[kd_column].apply(parse_kd_value)
    df["kd_value"] = [v for v, _ in parsed]
    df["below_limit"] = [b for _, b in parsed]
    df = df[df["kd_value"].notna()].copy()

    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (sample, loading), grp in df.groupby(["Sample ID", "Loading Sample ID"], sort=True):
        vals = grp["kd_value"].dropna().values
        if len(vals) == 0:
            continue
        geomean = float(np.exp(np.mean(np.log(vals))))
        records.append({
            "antigen": sample,
            "antibody": loading,
            "geomean_kd_M": geomean,
            "geomean_kd_nM": geomean * 1e9,
            "n_points": len(vals),
            "any_below_limit": bool(grp["below_limit"].any()),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def render_table_png(df: pd.DataFrame, antigen: str, antibody_order: list[str], output_path: Path):
    ag_df = df[df["Sample ID"] == antigen].copy()
    ag_df["Conc_nM"] = pd.to_numeric(ag_df["Conc. (nM)"], errors="coerce")
    dilutions = sorted(ag_df["Conc_nM"].dropna().unique(), reverse=True)

    col_headers = ["Conc. (nM)"]
    for ab in antibody_order:
        col_headers += [f"{ab}\nKD", f"{ab}\nR\u00b2"]

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

    # Geometric mean row
    mean_row = ["Geo. Mean"]
    for ab in antibody_order:
        ab_vals = ag_df[ag_df["Loading Sample ID"] == ab]["kd_value"].dropna().values
        if len(ab_vals) > 0:
            gm = float(np.exp(np.mean(np.log(ab_vals))))
            mean_row += [format_kd(gm), ""]
        else:
            mean_row += ["\u2014", ""]
    cell_text.append(mean_row)

    n_rows = len(cell_text)
    n_cols = len(col_headers)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.4), max(2, (n_rows + 1) * 0.45 + 0.8)))
    ax.axis("off")
    ax.set_title(antigen, fontsize=14, fontweight="bold", pad=12, loc="left")

    table = ax.table(cellText=cell_text, colLabels=col_headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("#cccccc")

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
# Comparison plots
# ---------------------------------------------------------------------------
def generate_comparison_plots(all_summaries: dict[str, pd.DataFrame], output_dir: Path):
    """Generate comparison bar charts and heatmaps from summaries grouped by round."""
    plot_dir = output_dir / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for label, summary in all_summaries.items():
        if summary.empty:
            continue

        tag = safe_name(label)
        antigens = sorted(summary["antigen"].unique())
        antibodies_all = sorted(summary["antibody"].unique())

        # Order: WT first, then alphabetical
        antibodies = sorted(antibodies_all, key=lambda x: (0 if "wt" in x.lower() else 1, x.lower()))

        # --- Bar chart ---
        fig, ax = plt.subplots(figsize=(max(10, len(antigens) * 2.2), 6))
        n_abs = len(antibodies)
        x = np.arange(len(antigens))
        bar_width = 0.8 / n_abs

        for i, ab in enumerate(antibodies):
            ab_data = summary[summary["antibody"] == ab]
            vals = []
            for ag in antigens:
                row = ab_data[ab_data["antigen"] == ag]
                vals.append(row["geomean_kd_nM"].values[0] if not row.empty else 0)
            color = COLORS.get(ab, "#95a5a6")
            bars = ax.bar(x + i * bar_width, vals, bar_width, label=ab,
                          color=color, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                            fontweight="bold", color=color)

        ax.set_yscale("log")
        ax.set_xlabel("Antigen", fontsize=13, fontweight="bold")
        ax.set_ylabel("KD (nM)", fontsize=13, fontweight="bold")
        ax.set_title(f"{label} \u2014 KD Comparison", fontsize=16, fontweight="bold", pad=15)
        ax.set_xticks(x + bar_width * (n_abs - 1) / 2)
        ax.set_xticklabels(antigens, fontsize=12, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        plt.tight_layout()
        path = plot_dir / f"{tag}_bar.png"
        fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    Saved: {path}")

        # --- Fold-change heatmap vs WT ---
        wt_ab = next((ab for ab in antibodies if "wt" in ab.lower()), None)
        if wt_ab and len(antibodies) >= 2:
            pivot = summary.pivot(index="antigen", columns="antibody", values="geomean_kd_nM")
            pivot = pivot.reindex(index=antigens, columns=antibodies)
            wt_vals = pivot[wt_ab]
            non_wt = [ab for ab in antibodies if ab != wt_ab]
            fc_display = pd.DataFrame({ab: pivot[ab] / wt_vals for ab in non_wt}, index=antigens)
            log2fc = np.log2(fc_display)

            fig, ax = plt.subplots(figsize=(max(6, len(non_wt) * 2), max(4, len(antigens) * 0.9)))
            vmax = max(abs(np.nanmin(log2fc.values)), abs(np.nanmax(log2fc.values)))
            im = ax.imshow(log2fc.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

            ax.set_xticks(range(len(non_wt)))
            ax.set_xticklabels(non_wt, fontsize=12, fontweight="bold")
            ax.set_yticks(range(len(antigens)))
            ax.set_yticklabels(antigens, fontsize=12, fontweight="bold")

            for i in range(len(antigens)):
                for j in range(len(non_wt)):
                    fc_val = fc_display.values[i, j]
                    l2_val = log2fc.values[i, j]
                    if not np.isnan(fc_val):
                        text_color = "white" if abs(l2_val) > 1.0 else "black"
                        ax.text(j, i, f"{fc_val:.2f}x\n({l2_val:+.1f})", ha="center",
                                va="center", fontsize=10, fontweight="bold", color=text_color)

            ax.set_title(f"{label} \u2014 Fold Change vs {wt_ab}\n(< 1 = tighter binding)",
                         fontsize=14, fontweight="bold", pad=15)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Log\u2082(FC vs WT)", fontsize=11)
            plt.tight_layout()
            path = plot_dir / f"{tag}_foldchange_heatmap.png"
            fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KD Comparison Pipeline")
    parser.add_argument("--input", default="csvs", help="Input directory with CSV files (default: csvs/)")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    parser.add_argument("--r2", type=float, default=0.95, help="R\u00b2 threshold (default: 0.95)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    r2_threshold = args.r2

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}/")
        sys.exit(1)

    print(f"KD Comparison Pipeline")
    print(f"{'=' * 50}")
    print(f"Input:  {input_dir}/  ({len(csv_files)} CSV files)")
    print(f"Output: {output_dir}/")
    print(f"R\u00b2 threshold: {r2_threshold}")
    print()

    # Group summaries by round for comparison plots
    round_summaries: dict[str, list[pd.DataFrame]] = {}

    for csv_path in csv_files:
        print(f"[1/2] Processing: {csv_path.name}")

        kd_column = detect_fit_type(csv_path.name)
        if kd_column is None:
            kd_column = "KD2"
            print(f"  Fit type: defaulting to KD2")
        else:
            print(f"  Fit type: {kd_column}")

        df = load_and_clean(csv_path, kd_column, r2_threshold)
        if df is None or df.empty:
            print(f"  No valid data after filtering, skipping.\n")
            continue

        # Determine round label from filename
        name_lower = csv_path.name.lower()
        if "round_2" in name_lower or "round2" in name_lower:
            round_label = "Round 2"
        elif "round_3" in name_lower or "round3" in name_lower:
            round_label = "Round 3"
        else:
            round_label = "Round 1"

        # Create output subfolder
        folder_name = safe_name(csv_path.stem)
        out_dir = output_dir / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        antibody_list = sorted(df["Loading Sample ID"].unique())
        antibody_order = sorted(antibody_list, key=lambda x: (0 if "wt" in x.lower() else 1, x.lower()))
        antigen_list = sorted(df["Sample ID"].unique())

        print(f"  Round:      {round_label}")
        print(f"  Antigens:   {antigen_list}")
        print(f"  Antibodies: {antibody_order}")

        for antigen in antigen_list:
            png_path = out_dir / f"{safe_name(antigen)}.png"
            render_table_png(df, antigen, antibody_order, png_path)
            print(f"  \u2713 {png_path}")

        # Collect summary for comparison plots
        summary = compute_summary(df)
        # Standardize antigen names
        summary["antigen"] = summary["antigen"].replace({"h2": "H2", "Vtn": "Vietnam", "Tx24": "Texas"})
        round_summaries.setdefault(round_label, []).append(summary)
        print()

    # Merge round summaries and generate comparison plots
    print(f"[2/2] Generating comparison plots...")
    merged = {}
    for label, dfs in round_summaries.items():
        merged[label] = pd.concat(dfs, ignore_index=True)

    generate_comparison_plots(merged, output_dir)

    print(f"\n{'=' * 50}")
    print(f"Pipeline complete.")
    print(f"Tables and plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
