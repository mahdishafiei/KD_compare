"""
KD Comparison Tool
==================
Streamlit application for comparing KD values from BLI/SPR kinetic result tables.
Supports fit_1_1 (KD column) and fit_2_1 (KD2 column) formats.
"""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KD Comparison Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIT_TYPE_MAP = {
    "fit_2_1": "KD2",
    "fit_1_1": "KD (M)",
}

UNIT_THRESHOLDS = [
    (1e-9, "pM", 1e12),   # < 1 nM  → display in pM
    (1e-6, "nM", 1e9),    # < 1 µM  → display in nM
    (1.0,  "µM", 1e6),    # < 1 M   → display in µM
]


# ---------------------------------------------------------------------------
# Helper: unit formatting
# ---------------------------------------------------------------------------
def format_kd(value_m: float) -> str:
    """Return a human-readable KD string with appropriate SI prefix."""
    if value_m is None or np.isnan(value_m):
        return "N/A"
    for threshold, unit, scale in UNIT_THRESHOLDS:
        if value_m < threshold:
            return f"{value_m * scale:.3g} {unit}"
    return f"{value_m * 1e6:.3g} µM"


# ---------------------------------------------------------------------------
# Helper: detect fit type from filename
# ---------------------------------------------------------------------------
def detect_fit_type(filename: str) -> str | None:
    """
    Return 'KD2' for fit_2_1, 'KD (M)' for fit_1_1, or None if undetermined.
    """
    name_lower = filename.lower()
    for pattern, col in FIT_TYPE_MAP.items():
        if pattern in name_lower:
            return col
    return None


# ---------------------------------------------------------------------------
# Helper: parse a KD string value
# ---------------------------------------------------------------------------
def parse_kd_value(raw) -> tuple[float | None, bool]:
    """
    Parse a KD cell value (possibly '<1.0E-12' style).

    Returns
    -------
    (value, below_limit)
        value       : float in Molar, or None if unparseable
        below_limit : True when the raw value started with '<'
    """
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


# ---------------------------------------------------------------------------
# Helper: load and clean a CSV file
# ---------------------------------------------------------------------------
def load_and_clean(
    file_obj,
    filename: str,
    kd_column: str,
    r2_threshold: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Read a kinetic result table CSV and apply all cleaning rules.

    Returns
    -------
    (df_clean, stats)
        df_clean : cleaned DataFrame with extra columns:
                   - kd_value  (float, Molar)
                   - below_limit (bool)
        stats    : dict with counts for the summary panel
    """
    # ---- read ---------------------------------------------------------------
    try:
        df = pd.read_csv(file_obj, dtype=str)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    # Drop fully-unnamed columns (trailing comma artefact)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    stats = {"rows_raw": len(df)}

    # ---- require the KD column to exist ------------------------------------
    if kd_column not in df.columns:
        st.error(
            f"Column **{kd_column!r}** not found in the file. "
            f"Available columns: {list(df.columns)}"
        )
        st.stop()

    # ---- required columns --------------------------------------------------
    for col in ("Sample ID", "Loading Sample ID", "Conc. (nM)", "Full R^2"):
        if col not in df.columns:
            st.error(f"Required column **{col!r}** not found in the file.")
            st.stop()

    # ---- strip whitespace from key string columns --------------------------
    for col in ("Sample ID", "Loading Sample ID", "Conc. (nM)", "Full R^2", kd_column):
        df[col] = df[col].astype(str).str.strip()

    # ---- filter: skip Buffer rows ------------------------------------------
    mask_buffer_sample = df["Sample ID"].str.lower() == "buffer"
    mask_buffer_loading = df["Loading Sample ID"].str.lower() == "buffer"
    df = df[~mask_buffer_sample & ~mask_buffer_loading].copy()
    stats["rows_after_buffer"] = len(df)

    # ---- filter: skip Conc. == N/A -----------------------------------------
    mask_conc_na = df["Conc. (nM)"].str.lower().isin(["n/a", "na", "nan", ""])
    df = df[~mask_conc_na].copy()
    stats["rows_after_conc"] = len(df)

    # ---- parse Full R^2 as numeric -----------------------------------------
    df["_r2"] = pd.to_numeric(df["Full R^2"], errors="coerce")

    # ---- filter: R² threshold ----------------------------------------------
    mask_r2 = df["_r2"] >= r2_threshold
    stats["rows_excluded_r2"] = int((~mask_r2).sum())
    df = df[mask_r2].copy()
    stats["rows_after_r2"] = len(df)

    # ---- parse KD values ---------------------------------------------------
    parsed = df[kd_column].apply(parse_kd_value)
    df["kd_value"] = [v for v, _ in parsed]
    df["below_limit"] = [b for _, b in parsed]

    # ---- filter: skip rows where KD is missing after parsing ---------------
    mask_kd_missing = df["kd_value"].isna()
    stats["rows_excluded_kd_missing"] = int(mask_kd_missing.sum())
    df = df[~mask_kd_missing].copy()
    stats["rows_used"] = len(df)
    stats["rows_excluded_total"] = stats["rows_raw"] - stats["rows_used"]

    return df, stats


# ---------------------------------------------------------------------------
# Helper: compute geometric mean KD per (Sample ID × Loading Sample ID)
# ---------------------------------------------------------------------------
def compute_geomean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Sample ID, Loading Sample ID) group compute:
      - geomean_kd   : geometric mean of kd_value
      - kd_min       : min kd_value
      - kd_max       : max kd_value
      - n_points     : count of valid rows used
      - any_below    : True if at least one value was below-detection-limit
    """
    records = []
    for (sample, loading), grp in df.groupby(
        ["Sample ID", "Loading Sample ID"], sort=True
    ):
        vals = grp["kd_value"].dropna().values
        if len(vals) == 0:
            continue
        log_vals = np.log(vals)
        geomean = float(np.exp(np.mean(log_vals)))
        records.append(
            {
                "antigen": sample,
                "antibody": loading,
                "geomean_kd_M": geomean,
                "kd_min_M": float(vals.min()),
                "kd_max_M": float(vals.max()),
                "n_points": len(vals),
                "any_below_limit": bool(grp["below_limit"].any()),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper: detect WT reference
# ---------------------------------------------------------------------------
def detect_wt(antibodies: list[str]) -> str:
    """
    Return the WT antibody name: first one containing 'wt' (case-insensitive),
    else the first alphabetically.
    """
    for ab in sorted(antibodies, key=str.lower):
        if "wt" in ab.lower():
            return ab
    return sorted(antibodies)[0]


# ---------------------------------------------------------------------------
# Helper: build pivot tables
# ---------------------------------------------------------------------------
def build_pivot_tables(
    summary: pd.DataFrame, wt_ref: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    kd_pivot      : rows=antigen, cols=antibody, values=formatted KD string
    fc_pivot      : rows=antigen, cols=antibody (non-WT), values=fold-change string
    """
    antigens = sorted(summary["antigen"].unique())
    antibodies = sorted(summary["antibody"].unique())

    # Numeric pivot for fold-change computation
    numeric_pivot = summary.pivot(
        index="antigen", columns="antibody", values="geomean_kd_M"
    ).reindex(index=antigens, columns=antibodies)

    # KD pivot with formatting (use map for pandas >= 2.1 compatibility)
    kd_pivot = numeric_pivot.map(format_kd)

    # Mark below-limit cells
    below_pivot = summary.pivot(
        index="antigen", columns="antibody", values="any_below_limit"
    ).reindex(index=antigens, columns=antibodies)
    for col in kd_pivot.columns:
        if col in below_pivot.columns:
            mask = below_pivot[col].fillna(False).astype(bool)
            kd_pivot.loc[mask, col] = kd_pivot.loc[mask, col] + " *"

    # Fold-change pivot
    if wt_ref in numeric_pivot.columns:
        wt_vals = numeric_pivot[wt_ref]
        non_wt = [ab for ab in antibodies if ab != wt_ref]
        fc_rows = {}
        for ab in non_wt:
            fc = numeric_pivot[ab] / wt_vals
            fc_rows[ab] = fc.apply(
                lambda v: f"{v:.2f}x" if not np.isnan(v) else "N/A"
            )
        fc_pivot = pd.DataFrame(fc_rows, index=antigens)
    else:
        fc_pivot = pd.DataFrame()

    return kd_pivot, fc_pivot


# ---------------------------------------------------------------------------
# Helper: build numeric pivot for heatmap/plot
# ---------------------------------------------------------------------------
def build_numeric_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    antigens = sorted(summary["antigen"].unique())
    antibodies = sorted(summary["antibody"].unique())
    return summary.pivot(
        index="antigen", columns="antibody", values="geomean_kd_M"
    ).reindex(index=antigens, columns=antibodies)


# ---------------------------------------------------------------------------
# Helper: build download CSV
# ---------------------------------------------------------------------------
def build_download_df(summary: pd.DataFrame, wt_ref: str | None) -> pd.DataFrame:
    rows = []
    antigens = sorted(summary["antigen"].unique())
    antibodies = sorted(summary["antibody"].unique())

    for antigen in antigens:
        row: dict = {"Antigen": antigen}
        ag_data = summary[summary["antigen"] == antigen]
        wt_kd = None
        if wt_ref:
            wt_row = ag_data[ag_data["antibody"] == wt_ref]
            if not wt_row.empty:
                wt_kd = wt_row.iloc[0]["geomean_kd_M"]
        for ab in antibodies:
            ab_row = ag_data[ag_data["antibody"] == ab]
            if ab_row.empty:
                row[f"KD_{ab}_(M)"] = np.nan
                row[f"KD_{ab}_(nM)"] = np.nan
                row[f"n_points_{ab}"] = 0
                if wt_ref and ab != wt_ref:
                    row[f"FoldChange_{ab}_vs_{wt_ref}"] = np.nan
            else:
                kd_m = ab_row.iloc[0]["geomean_kd_M"]
                row[f"KD_{ab}_(M)"] = kd_m
                row[f"KD_{ab}_(nM)"] = kd_m * 1e9
                row[f"n_points_{ab}"] = int(ab_row.iloc[0]["n_points"])
                if wt_ref and ab != wt_ref and wt_kd is not None:
                    row[f"FoldChange_{ab}_vs_{wt_ref}"] = kd_m / wt_kd
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_heatmap(summary: pd.DataFrame, wt_ref: str | None) -> go.Figure:
    """Log2 fold-change heatmap (RdBu_r, midpoint at 0)."""
    antigens = sorted(summary["antigen"].unique())
    antibodies = sorted(summary["antibody"].unique())
    num_pivot = build_numeric_pivot(summary)

    if wt_ref and wt_ref in antibodies:
        wt_vals = num_pivot[wt_ref]
        log2fc_pivot = num_pivot.copy()
        for ab in antibodies:
            log2fc_pivot[ab] = np.log2(num_pivot[ab] / wt_vals)
        title = f"Log₂ Fold Change vs. {wt_ref}"
        colorbar_title = f"Log₂(FC vs {wt_ref})"
    else:
        log2fc_pivot = np.log2(num_pivot * 1e9)  # log2(nM)
        title = "Log₂ KD (nM)"
        colorbar_title = "Log₂ KD (nM)"

    z = log2fc_pivot.values.tolist()
    text_vals = [
        [f"{v:.2f}" if not np.isnan(v) else "N/A" for v in row] for row in log2fc_pivot.values
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(log2fc_pivot.columns),
            y=list(log2fc_pivot.index),
            text=text_vals,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title=colorbar_title),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Antibody",
        yaxis_title="Antigen",
        height=max(300, len(antigens) * 45 + 150),
    )
    return fig


def plot_bar_chart(summary: pd.DataFrame) -> go.Figure:
    """Bar chart: KD in nM, grouped by antigen, log Y, color by antibody."""
    plot_df = summary.copy()
    plot_df["KD (nM)"] = plot_df["geomean_kd_M"] * 1e9

    fig = px.bar(
        plot_df,
        x="antigen",
        y="KD (nM)",
        color="antibody",
        barmode="group",
        log_y=True,
        title="Geometric Mean KD by Antigen (log scale)",
        labels={"antigen": "Antigen", "KD (nM)": "KD (nM)", "antibody": "Antibody"},
        text_auto=False,
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        legend_title="Antibody",
        height=450,
    )
    return fig


def plot_scatter(df_clean: pd.DataFrame) -> go.Figure:
    """
    Scatter: individual KD values per concentration point,
    faceted by antigen, color by antibody.
    """
    plot_df = df_clean.copy()
    plot_df["KD (nM)"] = plot_df["kd_value"] * 1e9
    plot_df["Conc_nM"] = pd.to_numeric(plot_df["Conc. (nM)"], errors="coerce")

    antigens = sorted(plot_df["Sample ID"].unique())
    n_antigens = len(antigens)
    cols = min(3, n_antigens)
    rows = int(np.ceil(n_antigens / cols))

    fig = px.scatter(
        plot_df,
        x="Conc_nM",
        y="KD (nM)",
        color="Loading Sample ID",
        facet_col="Sample ID",
        facet_col_wrap=cols,
        log_y=True,
        log_x=True,
        title="KD Values per Concentration Point",
        labels={
            "Conc_nM": "Conc. (nM)",
            "KD (nM)": "KD (nM)",
            "Loading Sample ID": "Antibody",
            "Sample ID": "Antigen",
        },
    )
    fig.update_layout(height=max(350, rows * 280), legend_title="Antibody")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    st.title("KD Comparison Tool")
    st.caption("BLI / SPR Kinetic Result Table Analyzer")

    # ---- Sidebar -----------------------------------------------------------
    st.sidebar.header("Settings")

    uploaded = st.sidebar.file_uploader(
        "Upload kinetic result CSV",
        type=["csv"],
        help="Upload a Kinetics Result Table exported from the BLI/SPR software.",
    )

    r2_threshold = st.sidebar.slider(
        "Minimum Full R² threshold",
        min_value=0.80,
        max_value=1.00,
        value=0.95,
        step=0.01,
        help="Rows with Full R² below this value will be excluded.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fit Type Override")
    fit_override = st.sidebar.radio(
        "KD column to use",
        options=["Auto-detect from filename", "KD2 (fit_2_1)", "KD (M) (fit_1_1)"],
        index=0,
        help=(
            "Auto-detect reads the filename for 'fit_2_1' or 'fit_1_1'. "
            "Override if the filename doesn't match the expected pattern."
        ),
    )

    # ---- No file yet -------------------------------------------------------
    if uploaded is None:
        st.info(
            "Upload a **Kinetic Result Table CSV** in the sidebar to begin analysis.",
            icon="📂",
        )
        st.markdown(
            """
**Supported formats**
- `fit_2_1` files → `KD2` column is used automatically
- `fit_1_1` files → `KD (M)` column is used automatically
- Override available in the sidebar if needed

**Data cleaning applied**
- Rows with `Sample ID` or `Loading Sample ID` = "Buffer" are skipped
- Rows with `Conc. (nM)` = N/A are skipped
- Rows with `Full R² < threshold` are excluded
- KD values like `<1.0E-12` are parsed and flagged with `*`
- Rows with missing/unparseable KD are skipped

**Output tabs**
`KD Summary` · `Fold Changes` · `Plots` · `Raw Data`
"""
        )
        return

    # ---- Determine KD column -----------------------------------------------
    filename = uploaded.name
    auto_kd_col = detect_fit_type(filename)

    if fit_override == "KD2 (fit_2_1)":
        kd_column = "KD2"
        fit_source = "user override"
    elif fit_override == "KD (M) (fit_1_1)":
        kd_column = "KD (M)"
        fit_source = "user override"
    elif auto_kd_col is not None:
        kd_column = auto_kd_col
        fit_source = f"auto-detected from filename (`{'fit_2_1' if kd_column == 'KD2' else 'fit_1_1'}`)"
    else:
        st.sidebar.warning(
            "Could not detect fit type from filename. Please select manually."
        )
        manual_col = st.sidebar.selectbox(
            "Select KD column manually",
            options=["KD2", "KD (M)"],
            index=0,
        )
        kd_column = manual_col
        fit_source = "manual selection (filename unrecognised)"

    # ---- Load data ---------------------------------------------------------
    df_clean, stats = load_and_clean(
        uploaded, filename, kd_column, r2_threshold
    )

    # ---- Check we have data ------------------------------------------------
    if df_clean.empty:
        st.error(
            "**No valid rows remain after filtering.** "
            f"Try lowering the R² threshold (currently {r2_threshold:.2f}) in the sidebar."
        )
        st.stop()

    # ---- Compute summary ---------------------------------------------------
    summary = compute_geomean_table(df_clean)

    if summary.empty:
        st.error("Could not compute any KD summaries from the filtered data.")
        st.stop()

    antigens = sorted(summary["antigen"].unique().tolist())
    antibodies = sorted(summary["antibody"].unique().tolist())

    # ---- WT reference selector ---------------------------------------------
    auto_wt = detect_wt(antibodies)
    st.sidebar.markdown("---")
    st.sidebar.subheader("WT Reference")
    wt_ref = st.sidebar.selectbox(
        "Select WT / reference antibody",
        options=antibodies,
        index=antibodies.index(auto_wt),
        help="Used as denominator for fold-change calculations.",
    )

    # ---- Top-level metrics -------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Antigens", len(antigens))
    col2.metric("Antibodies", len(antibodies))
    col3.metric("Rows used", stats["rows_used"])
    col4.metric("Rows excluded (R²)", stats.get("rows_excluded_r2", 0))
    col5.metric("Rows excluded (KD missing)", stats.get("rows_excluded_kd_missing", 0))

    st.caption(
        f"File: **{filename}** | KD column: **{kd_column}** ({fit_source}) | "
        f"R² ≥ **{r2_threshold}**"
    )

    # ---- Below-limit notice ------------------------------------------------
    n_below = int(df_clean["below_limit"].sum())
    if n_below > 0:
        st.info(
            f"**{n_below} row(s)** had KD values reported as below the detection limit "
            f"(e.g., `<1.0E-12`). The numeric value after `<` was used for calculations "
            f"and these entries are flagged with **`*`** in the KD table.",
            icon="ℹ️",
        )

    # ---- Build pivot tables ------------------------------------------------
    kd_pivot, fc_pivot = build_pivot_tables(summary, wt_ref)
    numeric_pivot = build_numeric_pivot(summary)

    # ---- Tabs --------------------------------------------------------------
    tab_kd, tab_fc, tab_plots, tab_raw = st.tabs(
        ["KD Summary", "Fold Changes", "Plots", "Raw Data"]
    )

    # ------------------------------------------------------------------ Tab 1
    with tab_kd:
        st.subheader("Geometric Mean KD per Antigen × Antibody")
        st.markdown(
            "Values shown as geometric mean across all concentration points. "
            "**`*`** = at least one value was below the instrument detection limit."
        )
        st.dataframe(kd_pivot, use_container_width=True)

        # Point count sub-table
        with st.expander("Show number of valid points used"):
            n_pivot = summary.pivot(
                index="antigen", columns="antibody", values="n_points"
            ).reindex(
                index=sorted(summary["antigen"].unique()),
                columns=sorted(summary["antibody"].unique()),
            )
            st.dataframe(n_pivot, use_container_width=True)

        # Download button
        dl_df = build_download_df(summary, wt_ref)
        csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download comparison table (CSV)",
            data=csv_bytes,
            file_name="kd_comparison.csv",
            mime="text/csv",
        )

    # ------------------------------------------------------------------ Tab 2
    with tab_fc:
        st.subheader(f"Fold Change vs. {wt_ref}")
        if len(antibodies) < 2:
            st.warning(
                "Only one antibody found in the data. "
                "Fold-change comparison requires at least two antibodies."
            )
        elif fc_pivot.empty:
            st.warning(
                f"Reference antibody **{wt_ref}** not found in data for fold-change calculation."
            )
        else:
            st.markdown(
                f"Fold change = KD (variant) / KD ({wt_ref}). "
                "Values **> 1** indicate weaker binding vs. reference. "
                "Values **< 1** indicate tighter binding."
            )
            st.dataframe(fc_pivot, use_container_width=True)

            # Numeric fold-change table
            with st.expander("Show numeric fold-change values"):
                non_wt_abs = [ab for ab in antibodies if ab != wt_ref]
                wt_num = numeric_pivot[wt_ref]
                fc_num = numeric_pivot[non_wt_abs].copy()
                for ab in non_wt_abs:
                    fc_num[ab] = fc_num[ab] / wt_num
                st.dataframe(fc_num.style.format("{:.3f}"), use_container_width=True)

    # ------------------------------------------------------------------ Tab 3
    with tab_plots:
        st.subheader("Visualizations")

        plot_col1, plot_col2 = st.columns([1, 1])

        with plot_col1:
            if len(antibodies) >= 2:
                fig_heatmap = plot_heatmap(summary, wt_ref)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info(
                    "Heatmap requires at least two antibodies for fold-change display. "
                    "Showing log₂ KD (nM) instead."
                )
                fig_heatmap = plot_heatmap(summary, wt_ref=None)
                st.plotly_chart(fig_heatmap, use_container_width=True)

        with plot_col2:
            fig_bar = plot_bar_chart(summary)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        fig_scatter = plot_scatter(df_clean)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------------------------------------------ Tab 4
    with tab_raw:
        st.subheader("Filtered Raw Data")
        st.caption(
            f"Showing {len(df_clean)} rows after all filters. "
            "KD values are parsed numerics (Molar)."
        )

        display_cols = [
            "Sample ID",
            "Loading Sample ID",
            "Conc. (nM)",
            kd_column,
            "kd_value",
            "below_limit",
            "Full R^2",
            "_r2",
        ]
        display_cols = [c for c in display_cols if c in df_clean.columns]

        col_rename = {kd_column: f"{kd_column} (raw)", "kd_value": "KD parsed (M)", "_r2": "R²"}
        st.dataframe(
            df_clean[display_cols].rename(columns=col_rename),
            use_container_width=True,
        )

        # Full raw data
        with st.expander("Show all columns"):
            st.dataframe(df_clean.drop(columns=["kd_value", "below_limit", "_r2"], errors="ignore"), use_container_width=True)


if __name__ == "__main__":
    main()
