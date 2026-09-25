import os
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    print(f"[dashboard] Missing required package: {exc.name}")
    print(f"{sys.executable} -m pip install pandas numpy")
    raise SystemExit(1)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import ALL, Dash, Input, Output, State, dash_table, dcc, html
except ImportError as exc:
    print(f"[dashboard] Missing required package: {exc.name}")
    print(f"{sys.executable} -m pip install dash plotly")
    raise SystemExit(1)

# =========================
# PLOTLY VERSION COMPAT
# =========================
import plotly as _plotly
_PLOTLY_V6 = int(_plotly.__version__.split(".")[0]) >= 6

def _scatter_map(**kwargs):
    """Return Scattermap (v6+) or Scattermapbox (v5)."""
    if _PLOTLY_V6:
        return go.Scattermap(**kwargs)
    return _scatter_map(**kwargs)

def _px_scatter_map(df, **kwargs):
    """Return px.scatter_map (v6+) or px.scatter_mapbox (v5)."""
    if _PLOTLY_V6:
        return px.scatter_map(df, **kwargs)
    return _px_scatter_map(df, **kwargs)

def _map_layout(**kwargs):
    """Return layout kwargs for map config, version-safe."""
    if _PLOTLY_V6:
        return {
            "map_style": kwargs.get("style", "carto-positron"),
            "map": dict(
                center=kwargs.get("center", {"lat": 36.7, "lon": -119.5}),
                zoom=kwargs.get("zoom", 5.2),
            ),
        }
    return {
        "mapbox_style": kwargs.get("style", "carto-positron"),
        "mapbox": dict(
            center=kwargs.get("center", {"lat": 36.7, "lon": -119.5}),
            zoom=kwargs.get("zoom", 5.2),
        ),
    }


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

REQUIRED_TRACK2_FILES = [
    "dashboard_cell_risk.csv",
    "dashboard_hourly_summary.csv",
    "dashboard_topk_snapshot.csv",
]
OPTIONAL_FILES = {
    "erc_locations": ["erc_locations.csv", "dashboard_erc_locations.csv"],
    "track1_demand_cells": ["dashboard_demand_cells.csv"],
    "track1_erc_metrics": ["dashboard_erc_metrics.csv"],
    "track1_demand_assignment": ["dashboard_demand_assignment.csv"],
}


def resolve_existing_file(candidates):
    """Check OUTPUT_DIR first, then BASE_DIR (so files placed in either location work)."""
    for search_dir in [OUTPUT_DIR, BASE_DIR]:
        for candidate in candidates:
            for name in (candidate, candidate + ".gz"):  # large outputs ship gzipped
                path = search_dir / name
                if path.exists():
                    return path
    return None


def load_csv(label, candidates, required):
    path = resolve_existing_file(candidates)
    if path is None:
        if required:
            return None, None
        print(f"[dashboard] MISSING optional file for {label}: searched in {OUTPUT_DIR} and {BASE_DIR}")
        return None, pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[dashboard] FOUND {label}: {path} ({len(df)} rows)")
    return path, df


def fail_for_missing_track2(missing_names):
    print(f"[dashboard] Base directory: {BASE_DIR}")
    print(f"[dashboard] Output directory: {OUTPUT_DIR}")
    print("[dashboard] Missing required Track 2 CSVs:")
    for name in missing_names:
        print(f"  - {OUTPUT_DIR / name}")
    raise SystemExit(1)


print(f"[dashboard] Base directory: {BASE_DIR}")
print(f"[dashboard] Output directory: {OUTPUT_DIR}")

missing_track2 = [name for name in REQUIRED_TRACK2_FILES if resolve_existing_file([name]) is None]
if missing_track2:
    fail_for_missing_track2(missing_track2)

cell_risk_path, cell_risk = load_csv("dashboard_cell_risk", ["dashboard_cell_risk.csv"], required=True)
hourly_summary_path, hourly_summary = load_csv("dashboard_hourly_summary", ["dashboard_hourly_summary.csv"], required=True)
topk_snapshot_path, topk_snapshot = load_csv("dashboard_topk_snapshot", ["dashboard_topk_snapshot.csv"], required=True)
erc_locations_path, erc_locations = load_csv("erc_locations", OPTIONAL_FILES["erc_locations"], required=False)
_, demand_cells = load_csv("track1_demand_cells", OPTIONAL_FILES["track1_demand_cells"], required=False)
_, erc_metrics = load_csv("track1_erc_metrics", OPTIONAL_FILES["track1_erc_metrics"], required=False)
_, demand_assignment = load_csv("track1_demand_assignment", OPTIONAL_FILES["track1_demand_assignment"], required=False)


# =========================
# CLEAN / SHAPE  (Track 2)
# =========================
REQUIRED_CELL_RISK_COLS = [
    "cell_id", "cell_lat", "cell_lng", "hour_of_week",
    "day_name", "hour", "risk_score", "pred_rank",
]
missing_cols = [c for c in REQUIRED_CELL_RISK_COLS if c not in cell_risk.columns]
if missing_cols:
    print(f"[dashboard] dashboard_cell_risk.csv is missing columns: {missing_cols}")
    raise SystemExit(1)

REQUIRED_HOURLY_COLS = ["hour_of_week", "day_name", "hour", "avg_risk_score"]
missing_hourly_cols = [c for c in REQUIRED_HOURLY_COLS if c not in hourly_summary.columns]
if missing_hourly_cols:
    print(f"[dashboard] dashboard_hourly_summary.csv is missing columns: {missing_hourly_cols}")
    raise SystemExit(1)

cell_risk["cell_id"] = cell_risk["cell_id"].astype(str)
cell_risk["cell_lat"] = pd.to_numeric(cell_risk["cell_lat"], errors="coerce")
cell_risk["cell_lng"] = pd.to_numeric(cell_risk["cell_lng"], errors="coerce")
cell_risk["hour"] = pd.to_numeric(cell_risk["hour"], errors="coerce").fillna(0).astype(int)
cell_risk["pred_rank"] = pd.to_numeric(cell_risk["pred_rank"], errors="coerce")
cell_risk["risk_score"] = pd.to_numeric(cell_risk["risk_score"], errors="coerce")
cell_risk["day_name"] = cell_risk["day_name"].astype(str).str.strip()

for opt_col in ["cell_rate", "cell_how_rate", "neighbor_rate", "is_top_10", "is_top_50", "is_top_100"]:
    if opt_col in cell_risk.columns:
        cell_risk[opt_col] = pd.to_numeric(cell_risk[opt_col], errors="coerce")

hourly_summary["day_name"] = hourly_summary["day_name"].astype(str).str.strip()
hourly_summary["hour"] = pd.to_numeric(hourly_summary["hour"], errors="coerce").fillna(0).astype(int)
hourly_summary["avg_risk_score"] = pd.to_numeric(hourly_summary["avg_risk_score"], errors="coerce").fillna(0.0)

if not topk_snapshot.empty:
    topk_snapshot["cell_id"] = topk_snapshot["cell_id"].astype(str)
    topk_snapshot["day_name"] = topk_snapshot["day_name"].astype(str).str.strip()

DAY_OPTIONS = sorted(
    cell_risk["day_name"].dropna().unique().tolist(),
    key=lambda d: (
        hourly_summary["day_name"].drop_duplicates().tolist().index(d)
        if d in hourly_summary["day_name"].drop_duplicates().tolist()
        else 999,
        d,
    ),
)
DEFAULT_DAY = "Friday" if "Friday" in DAY_OPTIONS else DAY_OPTIONS[0]
GLOBAL_HOURLY_MAX = float(hourly_summary["avg_risk_score"].max()) if not hourly_summary.empty else 1.0
GLOBAL_HOURLY_YMAX = GLOBAL_HOURLY_MAX * 1.15 if GLOBAL_HOURLY_MAX > 0 else 1.0


def infer_topk_values():
    topk_values = []
    for v in [10, 50, 100]:
        col = f"is_top_{v}"
        if col in cell_risk.columns:
            topk_values.append(v)
    if topk_values:
        return topk_values
    if "k_label" in topk_snapshot.columns:
        parsed = []
        for v in topk_snapshot["k_label"].dropna().astype(str).unique():
            digits = "".join(ch for ch in v if ch.isdigit())
            if digits:
                parsed.append(int(digits))
        if parsed:
            return sorted(set(parsed))
    return [10, 50, 100]


TOPK_VALUES = infer_topk_values()
for _extra_k in [10, 25, 50, 100]:
    if _extra_k not in TOPK_VALUES:
        TOPK_VALUES.append(_extra_k)
TOPK_VALUES = sorted(set(TOPK_VALUES))
DEFAULT_TOPK = 50 if 50 in TOPK_VALUES else TOPK_VALUES[0]


# =========================
# CLEAN / SHAPE  (Track 1)
# =========================
def normalize_method(v):
    if pd.isna(v):
        return v
    txt = str(v).strip().lower().replace("_", "-")
    if txt in ["kmeans", "k-means"]:
        return "kmeans"
    if txt in ["pmedian", "p-median", "p median"]:
        return "pmedian"
    return txt


for df in [erc_locations, erc_metrics, demand_assignment]:
    if not df.empty and "method" in df.columns:
        df["method"] = df["method"].apply(normalize_method)
    if not df.empty and "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce")

if not demand_cells.empty:
    for col in ["lat", "lng", "count", "weight"]:
        if col in demand_cells.columns:
            demand_cells[col] = pd.to_numeric(demand_cells[col], errors="coerce")

if not demand_assignment.empty:
    for col in ["lat", "lng", "nearest_erc_lat", "nearest_erc_lng", "geo_dist_km"]:
        if col in demand_assignment.columns:
            demand_assignment[col] = pd.to_numeric(demand_assignment[col], errors="coerce")

TRACK1_AVAILABLE = not demand_cells.empty and not erc_locations.empty
METHOD_LABELS = {"kmeans": "K-Means", "pmedian": "P-Median"}


def method_values():
    values = []
    if not erc_locations.empty and "method" in erc_locations.columns:
        values.extend(erc_locations["method"].dropna().unique().tolist())
    if not values and not erc_metrics.empty and "method" in erc_metrics.columns:
        values.extend(erc_metrics["method"].dropna().unique().tolist())
    return sorted(set(values))



method_options = [
    {"label": METHOD_LABELS.get(m, str(m).title()), "value": m}
    for m in method_values()
]

def _k_slider_marks(method):
    """Return slider marks dict for available k values of a method."""
    opts = get_k_options_for_method(method)
    if not opts:
        return {10: "10", 100: "100", 1000: "1000"}
    return {int(o["value"]): str(int(o["value"])) for o in opts}

def _k_slider_range(method):
    opts = get_k_options_for_method(method)
    vals = [int(o["value"]) for o in opts] if opts else [10, 100, 1000]
    return min(vals), max(vals), vals


def get_k_options_for_method(method):
    if method is None:
        return []
    values = []
    if not erc_locations.empty and {"method", "k"}.issubset(erc_locations.columns):
        values.extend(
            erc_locations.loc[erc_locations["method"] == method, "k"]
            .dropna().astype(int).sort_values().unique().tolist()
        )
    if not values and not erc_metrics.empty and {"method", "k"}.issubset(erc_metrics.columns):
        values.extend(
            erc_metrics.loc[erc_metrics["method"] == method, "k"]
            .dropna().astype(int).sort_values().unique().tolist()
        )
    return [{"label": str(v), "value": int(v)} for v in sorted(set(values))]


def best_erc_config():
    required_cols = ["method", "k", "coverage_10_min", "coverage_15_min", "wtd_mean_min"]
    if erc_metrics.empty or not set(required_cols).issubset(erc_metrics.columns):
        return None
    metrics = erc_metrics.dropna(subset=["method", "k"]).copy()
    if metrics.empty:
        return None
    metrics = metrics.sort_values(
        by=["coverage_10_min", "coverage_15_min", "wtd_mean_min"],
        ascending=[False, False, True],
    )
    return metrics.iloc[0].to_dict()


# Precompute distinct service-area colors per ERC for k<=100
_ERC_PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
]


def _erc_color_map(erc_ids):
    """Return dict mapping erc_id -> hex color."""
    unique_ids = sorted(set(erc_ids))
    cmap = {}
    for i, eid in enumerate(unique_ids):
        cmap[eid] = _ERC_PALETTE[i % len(_ERC_PALETTE)]
    return cmap


# =========================
# CALIFORNIA POLYGON (for Track 2 clipping)
# =========================
CALIFORNIA_POLYGON = [
    (-124.48, 42.00), (-124.35, 41.20), (-124.30, 40.50), (-124.20, 39.70),
    (-123.95, 38.95), (-123.75, 38.20), (-123.30, 37.50), (-122.90, 37.10),
    (-122.60, 36.70), (-122.40, 36.20), (-121.90, 35.70), (-121.40, 35.20),
    (-120.80, 34.90), (-120.30, 34.60), (-119.70, 34.45), (-119.10, 34.25),
    (-118.50, 34.05), (-117.95, 33.80), (-117.55, 33.45), (-117.15, 32.55),
    (-114.63, 32.72), (-114.52, 33.80), (-114.45, 35.00), (-114.55, 36.20),
    (-114.65, 37.20), (-114.72, 38.20), (-114.75, 39.00), (-120.00, 39.00),
    (-120.00, 42.00), (-124.48, 42.00),
]


def point_in_poly(lon, lat, poly):
    inside = False
    for i in range(len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        intersects = ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / ((y2 - y1) + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


# =========================
# STYLE CONSTANTS
# =========================
COLORS = {
    "bg": "#f0f2f5",
    "card": "#ffffff",
    "text": "#1a2535",
    "muted": "#6b7a90",
    "accent": "#1c3d5a",
    "accent2": "#2563a8",
    "border": "#d4dae4",
    "warn_bg": "#fff8e6",
    "warn_border": "#f0c36d",
    "warn_text": "#7a5000",
    "success": "#166534",
    "success_bg": "#f0fdf4",
    "tab_selected": "#1c3d5a",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 1px 6px rgba(0,0,0,0.06)",
}


def blank_figure(message):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(
            text=message, x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False,
            font=dict(size=15, color=COLORS["muted"]),
        )],
    )
    return fig


def make_kpi(title, value, subtitle=None, highlight=False):
    border_color = "#2563a8" if highlight else COLORS["border"]
    bg_color = "#eff6ff" if highlight else COLORS["card"]
    return html.Div([
        html.Div(title, style={
            "fontSize": "12px", "color": COLORS["muted"],
            "marginBottom": "6px", "textTransform": "uppercase",
            "letterSpacing": "0.05em", "fontWeight": "600",
        }),
        html.Div(value, style={
            "fontSize": "22px", "fontWeight": "700",
            "color": COLORS["accent2"] if highlight else COLORS["accent"],
        }),
        *(
            [html.Div(subtitle, style={"fontSize": "11px", "color": COLORS["muted"], "marginTop": "4px"})]
            if subtitle else []
        ),
    ], style={
        **CARD_STYLE,
        "textAlign": "center",
        "padding": "16px 10px",
        "backgroundColor": bg_color,
        "borderColor": border_color,
        "borderWidth": "1.5px" if highlight else "1px",
    })


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{100 * float(value):.1f}%"


def format_minutes(value):
    if pd.isna(value):
        return "N/A"
    v = float(value)
    if v < 0.5:
        return f"{v:.2f} min"
    return f"{v:.1f} min"


# =========================
# TRACK 1 MAP BUILDER
# =========================
def build_permanent_map(method, k):
    if not TRACK1_AVAILABLE or demand_cells.empty or not {"lat", "lng"}.issubset(demand_cells.columns):
        return blank_figure("Track 1 data not available.")
    if method is None or k is None:
        return blank_figure("Select a method and ERC count to see the plan.")

    k_int = int(k)

    # Filter ERC locations and assignment for selected method/k
    erc_filtered = erc_locations.copy()
    if {"method", "k"}.issubset(erc_locations.columns):
        erc_filtered = erc_filtered[
            (erc_filtered["method"] == method) & (erc_filtered["k"] == k_int)
        ].copy()

    asgn_filtered = demand_assignment.copy() if not demand_assignment.empty else pd.DataFrame()
    if not asgn_filtered.empty and {"method", "k"}.issubset(asgn_filtered.columns):
        asgn_filtered = asgn_filtered[
            (asgn_filtered["method"] == method) & (asgn_filtered["k"] == k_int)
        ].copy()

    # ── Demand cells: color by nearest_erc_id when k is small enough to distinguish ──
    show_assignment_colors = k_int <= 100 and not asgn_filtered.empty

    if show_assignment_colors and "nearest_erc_id" in asgn_filtered.columns:
        # Merge assignment data back into demand_cells for coloring
        merged = demand_cells.merge(
            asgn_filtered[["cell_id", "nearest_erc_id", "geo_dist_km"]],
            on="cell_id", how="left",
        )
        color_map = _erc_color_map(merged["nearest_erc_id"].dropna().astype(int).tolist())
        merged["marker_color"] = merged["nearest_erc_id"].apply(
            lambda eid: color_map.get(int(eid), "#aab4c4") if pd.notna(eid) else "#aab4c4"
        )
        merged["hover_erc"] = merged["nearest_erc_id"].apply(
            lambda eid: f"ERC {int(eid)}" if pd.notna(eid) else "Unassigned"
        )
        merged["hover_dist"] = merged["geo_dist_km"].apply(
            lambda d: f"{d:.1f} km" if pd.notna(d) else "N/A"
        )

        fig = go.Figure()
        # Plot demand cells grouped by ERC for clean legend
        for erc_id, group in merged.groupby("nearest_erc_id"):
            color = color_map.get(int(erc_id), "#aab4c4")
            erc_label = f"ERC {int(erc_id)}"
            fig.add_trace(_scatter_map(
                lat=group["lat"], lon=group["lng"],
                mode="markers",
                marker=dict(
                    size=(group["weight"] / merged["weight"].max() * 18 + 5).clip(5, 23).tolist()
                    if "weight" in group.columns else 9,
                    color=color, opacity=0.72,
                ),
                name=erc_label,
                customdata=np.column_stack([
                    group["cell_id"].values,
                    group.get("count", pd.Series(["N/A"] * len(group))).values,
                    group.get("weight", pd.Series(["N/A"] * len(group))).values,
                    group["hover_dist"].values,
                    group["hover_erc"].values,
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b><br>"
                    "Incidents: %{customdata[1]}<br>"
                    "Weight: %{customdata[2]}<br>"
                    "Nearest ERC: %{customdata[4]}<br>"
                    "Distance: %{customdata[3]}<extra></extra>"
                ),
                showlegend=(k_int <= 20),
            ))
    else:
        # For k=1000 or missing assignment: simple uniform demand cell display
        fig = go.Figure()
        size_vals = None
        if "weight" in demand_cells.columns:
            w = demand_cells["weight"].clip(lower=0)
            max_w = w.max()
            size_vals = ((w / max_w) * 14 + 4).clip(4, 18).tolist() if max_w > 0 else 8

        hover_cols = {}
        if "cell_id" in demand_cells.columns:
            hover_cols["cell_id"] = demand_cells["cell_id"].values
        if "count" in demand_cells.columns:
            hover_cols["count"] = demand_cells["count"].values

        fig.add_trace(_scatter_map(
            lat=demand_cells["lat"], lon=demand_cells["lng"],
            mode="markers",
            marker=dict(
                size=size_vals if size_vals is not None else 8,
                color="#5b8db8", opacity=0.55,
            ),
            name="Demand Cells",
            hovertemplate="<b>Cell %{customdata[0]}</b><br>Incidents: %{customdata[1]}<extra></extra>"
            if len(hover_cols) == 2 else "Demand cell<extra></extra>",
            customdata=np.column_stack(list(hover_cols.values())) if hover_cols else None,
            showlegend=False,
        ))

    # ── Assignment lines: single efficient trace using None separators ──
    if not asgn_filtered.empty and {
        "lat", "lng", "nearest_erc_lat", "nearest_erc_lng"
    }.issubset(asgn_filtered.columns):
        # For k=1000 skip lines (too many, hurts readability)
        if k_int <= 100:
            lats, lons = [], []
            for _, row in asgn_filtered.iterrows():
                lats += [row["lat"], row["nearest_erc_lat"], None]
                lons += [row["lng"], row["nearest_erc_lng"], None]
            line_opacity = 0.18 if k_int >= 100 else 0.28
            fig.add_trace(_scatter_map(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(width=1, color=f"rgba(28,61,90,{line_opacity})"),
                hoverinfo="skip",
                showlegend=False,
                name="Assignment lines",
            ))

    # ── Underserved cells layer (geo_dist_km > 30 km) ──
    UNDERSERVED_KM = 30
    if not asgn_filtered.empty and {"lat", "lng", "geo_dist_km"}.issubset(asgn_filtered.columns):
        underserved = asgn_filtered[asgn_filtered["geo_dist_km"] > UNDERSERVED_KM].copy()
        if not underserved.empty:
            # Merge with demand_cells to get weight for sizing
            uc = underserved.merge(
                demand_cells[["cell_id", "weight"]].rename(columns={"weight": "dc_weight"}),
                on="cell_id", how="left"
            )
            uc["dc_weight"] = uc["dc_weight"].fillna(uc["weight"])
            max_w = float(uc["dc_weight"].max()) if uc["dc_weight"].max() > 0 else 1
            uc_sizes = ((uc["dc_weight"] / max_w) * 12 + 6).clip(6, 18).tolist()
            fig.add_trace(_scatter_map(
                lat=uc["lat"], lon=uc["lng"],
                mode="markers",
                marker=dict(
                    size=uc_sizes,
                    color="#f97316",
                    opacity=0.75,
                    symbol="circle",
                ),
                name=f"Underserved (>{UNDERSERVED_KM} km)",
                customdata=np.column_stack([
                    uc["cell_id"].values,
                    uc["geo_dist_km"].round(1).values,
                    uc["dc_weight"].round(0).values,
                ]),
                hovertemplate=(
                    "<b>Cell %{customdata[0]}</b> — Underserved<br>"
                    "Distance to nearest ERC: %{customdata[1]} km<br>"
                    "Demand weight: %{customdata[2]}<extra></extra>"
                ),
                showlegend=True,
            ))

    # ── ERC markers ──
    if not erc_filtered.empty and {"lat", "lng"}.issubset(erc_filtered.columns):
        erc_size = 18 if k_int <= 10 else (12 if k_int <= 100 else 7)
        show_text = k_int <= 100
        labels = [f"ERC {int(v)}" if pd.notna(v) else "ERC"
                  for v in erc_filtered.get("ERC_ID", pd.Series([None] * len(erc_filtered)))]

        fig.add_trace(_scatter_map(
            lat=erc_filtered["lat"],
            lon=erc_filtered["lng"],
            mode="markers+text" if show_text else "markers",
            marker=dict(
                size=erc_size,
                color="#dc2626",
                symbol="circle",
                opacity=0.95,
            ),
            text=labels if show_text else None,
            textposition="top right",
            textfont=dict(size=10, color="#dc2626"),
            name="ERC Locations",
            hovertemplate=(
                "<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lng: %{lon:.4f}<extra></extra>"
                if show_text else
                "<b>ERC</b><br>Lat: %{lat:.4f}<br>Lng: %{lon:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_map_layout(style="carto-positron", center={"lat": 37.26, "lon": -119.3}, zoom=5.5),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        uirevision=f"track1-map-{method}-{k}",
    )
    return fig



# =========================
# TRACK 2 PRECOMPUTED DATA
# =========================
_DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Day × Hour avg risk heatmap (all cells)
_heatmap_pivot = (
    cell_risk.groupby(["day_name", "hour"])["risk_score"]
    .mean()
    .unstack(fill_value=0.0)
    .reindex(_DAY_ORDER)
)

# Persistent hotspots: cells that appear in Top-10 most across all 168 hour_of_week slots
_top10_cells = cell_risk[cell_risk["is_top_10"] == 1] if "is_top_10" in cell_risk.columns else cell_risk[cell_risk["pred_rank"] <= 10]
_hotspot_freq = (
    _top10_cells.groupby("cell_id")
    .agg(
        times_in_top10=("hour_of_week", "count"),
        avg_risk=("risk_score", "mean"),
        cell_lat=("cell_lat", "first"),
        cell_lng=("cell_lng", "first"),
    )
    .sort_values("times_in_top10", ascending=False)
    .head(10)
    .reset_index()
)
_MAX_HOW = int(cell_risk["hour_of_week"].nunique())  # typically 168

def _hour_label(h):
    """Convert 0-23 to '12:00 AM', '5:00 PM' etc."""
    h = int(h)
    if h == 0:   return "12:00 AM"
    if h < 12:   return f"{h}:00 AM"
    if h == 12:  return "12:00 PM"
    return f"{h-12}:00 PM"


# =========================
# TRACK 2 HELPERS
# =========================
def get_track2_slot(day_name, hour, top_k, log_debug=True):
    frame = cell_risk[
        (cell_risk["day_name"] == day_name) & (cell_risk["hour"] == int(hour))
    ].copy()
    if frame.empty:
        return frame

    flag_col = f"is_top_{int(top_k)}"
    if flag_col in frame.columns:
        frame = frame[frame[flag_col].fillna(0).astype(int) == 1]
    else:
        frame = frame[frame["pred_rank"] <= int(top_k)]
    frame = frame.sort_values("pred_rank", ascending=True).head(int(top_k)).copy()

    before_clip = len(frame)
    clipped = frame[
        frame.apply(
            lambda row: point_in_poly(float(row["cell_lng"]), float(row["cell_lat"]), CALIFORNIA_POLYGON),
            axis=1,
        )
    ].copy()

    if clipped.empty and not frame.empty:
        plot_frame = frame
    else:
        plot_frame = clipped

    return plot_frame



def build_coverage_curve(selected_method=None, selected_k=None):
    """Coverage vs k curve for both methods, both thresholds."""
    if erc_metrics.empty:
        return blank_figure("Metrics not available.")

    fig = go.Figure()
    method_colors = {"pmedian": {"10": "#1c3d5a", "15": "#5b8db8"},
                     "kmeans":  {"10": "#b45309", "15": "#f59e0b"}}
    method_names  = {"pmedian": "P-Median", "kmeans": "K-Means"}
    dash_styles   = {"10": "solid", "15": "dot"}

    for method in ["pmedian", "kmeans"]:
        sub = erc_metrics[erc_metrics["method"] == method].sort_values("k")
        if sub.empty:
            continue
        for thresh, col_key in [("coverage_10_min", "10"), ("coverage_15_min", "15")]:
            label = f"{method_names[method]} ≤{col_key} min"
            fig.add_trace(go.Scatter(
                x=sub["k"].astype(int),
                y=sub[thresh] * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=method_colors[method][col_key],
                          dash=dash_styles[col_key], width=2.5),
                marker=dict(size=8),
                hovertemplate=f"k=%{{x}}<br>{label}: %{{y:.1f}}%<extra></extra>",
            ))

    fig.add_hline(y=90, line_width=1.5, line_dash="dash", line_color="#9ca3af",
                  annotation_text="90% target", annotation_position="right")

    # Vertical marker + dots for selected config
    k_vals = sorted(erc_metrics["k"].astype(int).unique().tolist())
    if selected_k is not None:
        sk = int(selected_k)
        fig.add_vline(
            x=sk, line_width=1.5, line_dash="dot", line_color="#64748b",
        )
        # Dots on each curve at selected k
        mn = METHOD_LABELS.get(selected_method, selected_method) if selected_method else None
        for method in (["pmedian", "kmeans"] if selected_method is None else [selected_method]):
            sub = erc_metrics[(erc_metrics["method"] == method) & (erc_metrics["k"] == sk)]
            if sub.empty:
                continue
            mrow = sub.iloc[0]
            mc = method_colors.get(method, {})
            for thresh, col_key, label_suffix in [
                ("coverage_10_min", "10", "≤10 min"),
                ("coverage_15_min", "15", "≤15 min"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[sk], y=[mrow[thresh] * 100],
                    mode="markers",
                    marker=dict(size=12, color=mc.get(col_key, "#333"),
                                line=dict(color="white", width=2)),
                    showlegend=False,
                    hovertemplate=f"{METHOD_LABELS.get(method, method)} {label_suffix}<br>k={sk}: %{{y:.1f}}%<extra></extra>",
                ))

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=80, t=10, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            title="Number of ERCs (k)",
            tickvals=k_vals, ticktext=[str(v) for v in k_vals],
            type="log" if max(k_vals) / min(k_vals) > 20 else "linear",
            gridcolor="#f0f0f0", fixedrange=True,
        ),
        yaxis=dict(title="% Demand Covered", range=[0, 105],
                   gridcolor="#f0f0f0", fixedrange=True),
        legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=COLORS["border"], borderwidth=1,
                    x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        uirevision=f"coverage-curve-{selected_method}-{selected_k}",
    )
    return fig


def build_distance_distribution(method, k):
    """Box plot of geo_dist_km per config for selected method, all k values."""
    if demand_assignment.empty:
        return blank_figure("Assignment data not available.")

    method_sub = demand_assignment[demand_assignment["method"] == method].copy()
    if method_sub.empty:
        return blank_figure("No assignment data for selected method.")

    k_vals = sorted(method_sub["k"].astype(int).unique().tolist())
    colors = ["#5b8db8", "#2563a8", "#1c3d5a"]

    fig = go.Figure()
    for i, kv in enumerate(k_vals):
        sub = method_sub[method_sub["k"] == kv]["geo_dist_km"].dropna()
        color = colors[i % len(colors)]
        highlight = (kv == int(k)) if k is not None else False
        fig.add_trace(go.Box(
            y=sub,
            name=f"k = {kv}",
            marker_color=color,
            line=dict(color=color, width=2.5 if highlight else 1.5),
            fillcolor=color if highlight else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)",
            boxmean=True,
            hovertemplate="k=%{x}<br>Distance: %{y:.1f} km<extra></extra>",
        ))

    method_label = METHOD_LABELS.get(method, str(method).title())
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(
            text=f"Method: {method_label}  ·  highlighted = selected k",
            font=dict(size=11, color=COLORS["muted"]),
            x=0, xanchor="left", pad=dict(l=4),
        ),
        xaxis=dict(title="", gridcolor="#f0f0f0", fixedrange=True),
        yaxis=dict(title="Travel Distance (km)", gridcolor="#f0f0f0", fixedrange=True),
        showlegend=False,
        uirevision=f"dist-dist-{method}-{k}",
    )
    return fig


def build_erc_load_chart(method, k):
    """Bar chart of demand weight served per ERC for selected config."""
    if demand_assignment.empty:
        return blank_figure("Assignment data not available.")

    k_int = int(k) if k is not None else 10
    if k_int > 100:
        return blank_figure("Load balance chart available for k ≤ 100.")

    sub = demand_assignment[
        (demand_assignment["method"] == method) &
        (demand_assignment["k"] == k_int)
    ].copy()
    if sub.empty:
        return blank_figure("No data for this configuration.")

    load = (sub.groupby("nearest_erc_id")
              .agg(total_weight=("weight", "sum"),
                   n_cells=("cell_id", "count"),
                   mean_dist=("geo_dist_km", "mean"))
              .reset_index()
              .sort_values("total_weight", ascending=False))

    mean_load = float(load["total_weight"].mean())
    load["pct_of_mean"] = load["total_weight"] / mean_load
    load["bar_color"] = load["pct_of_mean"].apply(
        lambda r: "#dc2626" if r > 1.5 else ("#f59e0b" if r > 1.1 else "#2563a8")
    )
    # Use rank labels (1st, 2nd...) — ERC IDs are meaningless and unreadable at k>20
    load = load.reset_index(drop=True)
    load["rank_label"] = [f"#{i+1}" for i in range(len(load))]
    load["erc_label"] = load["nearest_erc_id"].apply(lambda x: f"ERC {int(x)}")

    # For small k show ERC IDs, for large k show rank only
    use_rank = k_int > 15
    load["x_label"] = load["rank_label"] if use_rank else load["erc_label"]

    fig = go.Figure(go.Bar(
        x=load["x_label"], y=load["total_weight"],
        marker_color=load["bar_color"],
        customdata=np.column_stack([
            load["erc_label"],
            load["n_cells"],
            load["mean_dist"].round(1),
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (rank %{x})<br>"
            "Demand weight: %{y:,.0f}<br>"
            "Cells assigned: %{customdata[1]}<br>"
            "Avg distance: %{customdata[2]} km<extra></extra>"
        ) if use_rank else (
            "<b>%{x}</b><br>"
            "Demand weight: %{y:,.0f}<br>"
            "Cells assigned: %{customdata[1]}<br>"
            "Avg distance: %{customdata[2]} km<extra></extra>"
        ),
    ))
    fig.add_hline(y=mean_load, line_dash="dash", line_color="#9ca3af",
                  line_width=1.5,
                  annotation_text="Mean", annotation_position="right")

    chart_height = 400 if k_int >= 100 else (320 if k_int > 20 else 280)
    tick_font_size = 9 if k_int >= 100 else (10 if k_int > 20 else 12)
    fig.update_layout(
        height=chart_height,
        margin=dict(l=20, r=60, t=10, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(
            title="ERCs ranked by demand weight served (highest to lowest)" if use_rank else "",
            tickfont=dict(size=tick_font_size),
            tickangle=-60 if k_int >= 50 else (-30 if k_int > 15 else 0),
            gridcolor="#f0f0f0", fixedrange=True,
        ),
        yaxis=dict(title="Demand Weight", gridcolor="#f0f0f0", fixedrange=True),
        showlegend=False,
        uirevision=f"erc-load-{method}-{k}",
    )
    return fig



def build_day_hour_heatmap_data(selected_day=None, selected_hour=None):
    """Return the heatmap as a list-of-dicts for HTML rendering."""
    days  = _DAY_ORDER
    hours = list(range(24))
    rows  = []
    # Compute global min/max for color normalization
    all_vals = [float(_heatmap_pivot.loc[d, h])
                for d in days for h in hours
                if h in _heatmap_pivot.columns]
    v_min = min(all_vals) if all_vals else 0.0
    v_max = max(all_vals) if all_vals else 1.0
    v_range = v_max - v_min if v_max > v_min else 1.0

    for day in days:
        for h in hours:
            val = float(_heatmap_pivot.loc[day, h]) if h in _heatmap_pivot.columns else 0.0
            norm = (val - v_min) / v_range  # 0..1
            is_sel = (day == selected_day and h == int(selected_hour) if selected_hour is not None else False)
            rows.append({
                "day": day, "hour": h,
                "val": round(val, 4),
                "norm": round(norm, 4),
                "selected": is_sel,
                "hour_label": _hour_label(h),
            })
    return rows, v_min, v_max

def build_hourly_chart_with_history(day_name, selected_hour, top_k):
    """Hourly bar chart overlaid with historical accident count."""
    hourly_rows = []
    for h in range(24):
        slot = get_track2_slot(day_name, h, top_k, log_debug=False)
        avg_risk = float(slot["risk_score"].mean()) if not slot.empty else 0.0
        hourly_rows.append({"hour": h, "avg_risk_score": avg_risk})

    frame = pd.DataFrame(hourly_rows)
    sel_h = int(selected_hour)
    frame["bar_color"] = frame["hour"].apply(
        lambda h: "#dc2626" if h == sel_h else "#93b8e0"
    )
    day_ymax = float(frame["avg_risk_score"].max()) * 1.2 if frame["avg_risk_score"].max() > 0 else 1.0

    # Historical accident data for this day
    hist = hourly_summary[hourly_summary["day_name"] == day_name].sort_values("hour")

    fig = go.Figure()

    # Bars: Top-K avg risk (primary y)
    fig.add_trace(go.Bar(
        x=frame["hour"].astype(str),
        y=frame["avg_risk_score"],
        marker_color=frame["bar_color"],
        width=0.6,
        name=f"Avg Top-{int(top_k)} Risk",
        yaxis="y",
        hovertemplate="Hour: %{x}:00<br>Avg Risk: %{y:.4f}<extra></extra>",
    ))

    # Historical accident line (secondary y)
    if not hist.empty and "historical_accident_count" in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist["hour"].astype(str),
            y=hist["historical_accident_count"],
            mode="lines+markers",
            name="Historical Accidents",
            line=dict(color="#7c3aed", width=2, dash="dot"),
            marker=dict(size=5),
            yaxis="y2",
            hovertemplate="Hour: %{x}:00<br>Accidents: %{y:,.0f}<extra></extra>",
        ))

    # Mean line
    fig.add_hline(
        y=float(frame["avg_risk_score"].mean()),
        line_width=1.5, line_dash="dash", line_color="#9ca3af",
        yref="y",
    )

    y2_max = float(hist["historical_accident_count"].max()) * 1.2 if not hist.empty else 1.0

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=60, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        bargap=0.25,
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
        xaxis=dict(
            title="Hour of Day", type="category",
            categoryorder="array",
            categoryarray=[str(v) for v in range(24)],
            fixedrange=True,
        ),
        yaxis=dict(
            title=f"Avg Top-{int(top_k)} Risk",
            range=[0, day_ymax], fixedrange=True,
            gridcolor="#f0f0f0",
        ),
        yaxis2=dict(
            title=dict(text="Historical Accidents", font=dict(color="#7c3aed")),
            overlaying="y", side="right",
            range=[0, y2_max], fixedrange=True,
            showgrid=False,
            tickfont=dict(color="#7c3aed"),
        ),
        uirevision=f"hourly-{day_name}-{top_k}",
    )
    return fig


def build_hotspot_map():
    """Persistent hotspot cells (always in Top-10). Static across all time."""
    if _hotspot_freq.empty:
        return blank_figure("No persistent hotspot data available.")

    max_t = float(_hotspot_freq["times_in_top10"].max())

    fig = go.Figure()
    fig.add_trace(_scatter_map(
        lat=_hotspot_freq["cell_lat"],
        lon=_hotspot_freq["cell_lng"],
        mode="markers+text",
        marker=dict(
            size=(_hotspot_freq["times_in_top10"] / max_t * 20 + 8).tolist(),
            color=_hotspot_freq["avg_risk"].tolist(),
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Avg Risk", thickness=12),
            opacity=0.9,
        ),
        text=[f"#{i+1}" for i in range(len(_hotspot_freq))],
        textposition="top right",
        textfont=dict(size=10, color="#1c3d5a"),
        customdata=np.column_stack([
            _hotspot_freq["cell_id"].values,
            _hotspot_freq["times_in_top10"].values,
            (_hotspot_freq["times_in_top10"] / _MAX_HOW * 100).round(1).values,
            _hotspot_freq["avg_risk"].round(4).values,
        ]),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "In Top-10: %{customdata[1]} / %{_MAX_HOW} slots (%{customdata[2]}%%)<br>"
            "Avg Risk Score: %{customdata[3]}<extra></extra>"
        ).replace("%{_MAX_HOW}", str(_MAX_HOW)),
        showlegend=False,
    ))

    # Compute bounding box of hotspot cells
    if not _hotspot_freq.empty:
        _lat_c = float(_hotspot_freq["cell_lat"].mean())
        _lon_c = float(_hotspot_freq["cell_lng"].mean())
        _lat_range = float(_hotspot_freq["cell_lat"].max() - _hotspot_freq["cell_lat"].min())
        _zoom = 9.5 if _lat_range < 0.5 else (8.0 if _lat_range < 1.5 else 6.0)
    else:
        _lat_c, _lon_c, _zoom = 36.7, -119.5, 5.5

    fig.update_layout(
        **_map_layout(style="carto-positron", center={"lat": _lat_c, "lon": _lon_c}, zoom=_zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        uirevision="hotspot-map",
    )
    return fig


def build_dynamic_map(day_name, hour, top_k, selected_cell_id=None):
    frame = get_track2_slot(day_name, hour, top_k)
    if frame.empty:
        return blank_figure("No Track 2 rows found for this day/hour/top-K selection.")

    fig = _px_scatter_map(
        frame, lat="cell_lat", lon="cell_lng", color="risk_score",
        hover_data={"cell_id": True, "risk_score": ":.4f", "pred_rank": True},
        color_continuous_scale="YlOrRd",
        center={"lat": 36.7, "lon": -119.5}, zoom=5.4, height=620,
    )
    fig.update_traces(marker=dict(size=15, opacity=0.84))

    top3 = frame[frame["pred_rank"] <= 3].copy()
    if not top3.empty:
        fig.add_trace(_scatter_map(
            lat=top3["cell_lat"], lon=top3["cell_lng"], mode="text",
            text=[f"Rank {int(r)}" for r in top3["pred_rank"]],
            textposition="top right",
            textfont=dict(size=11, color="#23374d"),
            showlegend=False, hoverinfo="skip",
        ))

    # Highlight selected cell from table click
    if selected_cell_id is not None:
        sel = frame[frame["cell_id"] == str(selected_cell_id)]
        if not sel.empty:
            # Outer white halo (largest, behind everything)
            fig.add_trace(_scatter_map(
                lat=sel["cell_lat"], lon=sel["cell_lng"],
                mode="markers",
                marker=dict(size=38, color="rgba(255,255,255,0.95)", symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))
            # Bright cyan ring
            fig.add_trace(_scatter_map(
                lat=sel["cell_lat"], lon=sel["cell_lng"],
                mode="markers",
                marker=dict(size=30, color="#00d4ff", opacity=0.9, symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))
            # Inner white core
            fig.add_trace(_scatter_map(
                lat=sel["cell_lat"], lon=sel["cell_lng"],
                mode="markers",
                marker=dict(size=20, color="#ffffff", opacity=1.0, symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))
            # Dark center dot with label
            fig.add_trace(_scatter_map(
                lat=sel["cell_lat"], lon=sel["cell_lng"],
                mode="markers+text",
                marker=dict(size=12, color="#1c3d5a", opacity=1.0, symbol="circle"),
                text=["▶ Selected"],
                textposition="top right",
                textfont=dict(size=12, color="#1c3d5a"),
                name=f"Selected: {selected_cell_id}",
                hovertemplate=f"<b>Selected: {selected_cell_id}</b><br>Click map to deselect<extra></extra>",
            ))

    fig.update_layout(
        **_map_layout(style="carto-positron", center={"lat": 37.26, "lon": -121.5}, zoom=5.6),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Risk score"),
        uirevision=f"track2-map-{day_name}-{hour}-{top_k}-{selected_cell_id}",
    )
    return fig


def build_hourly_chart(day_name, selected_hour, top_k):
    hourly_rows = []
    for h in range(24):
        slot = get_track2_slot(day_name, h, top_k, log_debug=False)
        avg_risk = float(slot["risk_score"].mean()) if not slot.empty else 0.0
        hourly_rows.append({"hour": h, "avg_risk_score": avg_risk})

    frame = pd.DataFrame(hourly_rows)
    frame["bar_color"] = np.where(frame["hour"] == int(selected_hour), "#dc2626", "#93b8e0")
    day_ymax = float(frame["avg_risk_score"].max()) * 1.15 if frame["avg_risk_score"].max() > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=frame["hour"].astype(str), y=frame["avg_risk_score"],
        marker_color=frame["bar_color"], width=0.65,
        hovertemplate="Hour: %{x}:00<br>Avg Risk: %{y:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=float(frame["avg_risk_score"].mean()),
        line_width=2, line_dash="dash", line_color="#7f8c8d",
    )
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=20, b=40),
        plot_bgcolor="white", paper_bgcolor="white", bargap=0.25,
        xaxis=dict(
            title="Hour of Day", type="category",
            categoryorder="array",
            categoryarray=[str(v) for v in range(24)], fixedrange=True,
        ),
        yaxis=dict(
            title=f"Avg Top-{int(top_k)} Risk Score",
            range=[0, day_ymax], fixedrange=True,
        ),
        showlegend=False,
        uirevision=f"hourly-{day_name}-{top_k}",
    )
    return fig


# =========================
# APP
# =========================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Emergency Response Allocation Dashboard"
server = app.server

default_method = "pmedian" if "pmedian" in method_values() else (method_values()[0] if method_values() else None)
default_k_options = get_k_options_for_method(default_method) if default_method else []
default_k = default_k_options[0]["value"] if default_k_options else None

_best = best_erc_config()
best_banner = None


# =========================
# LAYOUT
# =========================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1(
            "Emergency Response Allocation · California",
            style={"margin": "0", "color": COLORS["accent"], "fontSize": "28px", "fontWeight": "700"},
        ),
        html.Div(
            "Track 1: Permanent ERC placement optimization  ·  Track 2: Dynamic risk-based deployment",
            style={"color": COLORS["muted"], "marginTop": "5px", "fontSize": "14px"},
        ),
    ], style={"marginBottom": "20px"}),

    dcc.Tabs(
        id="main-tabs", value="tab-track1",
        children=[
            dcc.Tab(label="Track 1 · Strategic Placement", value="tab-track1"),
            dcc.Tab(label="Track 2 · Dynamic Risk", value="tab-track2"),
        ],
        style={"marginBottom": "0"},
    ),

    # ── TRACK 1 PANEL ────────────────────────────────────────────────────
    html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Div("Placement Method", style={"fontWeight": "600", "marginBottom": "6px", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="perm-method", options=method_options, value=default_method,
                    clearable=False, disabled=not TRACK1_AVAILABLE,
                ),
            ], style={"width": "240px"}),
            html.Div([
                html.Div([
                    html.Span("Number of ERCs (k)", style={"fontWeight": "600", "fontSize": "13px"}),
                    html.Span(id="perm-k-display", style={
                        "marginLeft": "10px", "fontWeight": "700",
                        "color": COLORS["accent2"], "fontSize": "15px",
                    }),
                ], style={"marginBottom": "12px"}),
                dcc.Slider(
                    id="perm-k",
                    min=default_k_options[0]["value"] if default_k_options else 10,
                    max=default_k_options[-1]["value"] if default_k_options else 1000,
                    step=None,
                    marks={int(o["value"]): {"label": str(int(o["value"])), "style": {"fontSize": "12px"}}
                           for o in default_k_options} if default_k_options else {10: "10", 100: "100", 1000: "1000"},
                    value=default_k,
                    disabled=not TRACK1_AVAILABLE,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "1", "minWidth": "300px", "paddingBottom": "8px"}),
        ], style={**CARD_STYLE, "display": "flex", "gap": "20px", "marginBottom": "16px", "alignItems": "flex-end"}),

        # KPI cards
        html.Div(id="perm-kpis", style={
            "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "12px", "marginBottom": "16px",
        }),

        # All-configurations comparison table
        html.Div([
            html.H3("All Configurations", style={
                "margin": "0 0 12px 0", "color": COLORS["accent"],
                "fontSize": "16px", "fontWeight": "700",
            }),
            dash_table.DataTable(
                id="config-table",
                columns=[
                    {"name": "Method",          "id": "method_label"},
                    {"name": "k",               "id": "k"},
                    {"name": "Wtd Mean",        "id": "wtd_mean"},
                    {"name": "Wtd P95",         "id": "wtd_p95"},
                    {"name": "Coverage ≤10 min","id": "cov_10"},
                    {"name": "Coverage ≤15 min","id": "cov_15"},
                ],
                data=[],
                row_selectable="single",
                selected_rows=[],
                style_table={"overflowX": "auto"},
                style_cell={
                    "padding": "8px 12px", "textAlign": "center",
                    "fontSize": "13px", "fontFamily": "'Helvetica Neue', Arial, sans-serif",
                    "border": f"1px solid #d4dae4",
                    "cursor": "pointer",
                },
                style_header={
                    "fontWeight": "700", "backgroundColor": "#eef2f7",
                    "color": "#1c3d5a", "border": f"1px solid #d4dae4",
                    "fontSize": "12px",
                },
                style_data={"backgroundColor": "white", "color": "#1a2535"},
                style_data_conditional=[],
                tooltip_delay=0,
                tooltip_duration=None,
            ),
            html.Div(
                "Click a row to select that configuration",
                style={"fontSize": "11px", "color": "#6b7a90", "marginTop": "6px"},
            ),
            html.Div(id="metrics-comparison-table", style={"display": "none"}),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # Map
        html.Div([
            html.H3("Demand Cells & ERC Placement", style={
                "margin": "0 0 4px 0", "color": COLORS["accent"],
                "fontSize": "16px", "fontWeight": "700",
            }),
            html.Div(id="perm-map-legend", style={
                "fontSize": "12px", "color": COLORS["muted"],
                "marginBottom": "10px",
            }),
            dcc.Graph(id="perm-map", config={"displayModeBar": True}),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # Analytics row: Coverage curve + Distance distribution
        html.Div([
            html.Div([
                html.H3("Coverage vs. Number of ERCs", style={
                    "margin": "0 0 4px 0", "color": COLORS["accent"],
                    "fontSize": "15px", "fontWeight": "700",
                }),
                html.Div(
                    "How demand coverage improves as more ERCs are deployed. "
                    "Solid lines = ≤10 min threshold, dotted = ≤15 min.",
                    style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"},
                ),
                dcc.Graph(id="perm-coverage-curve",
                          config={"displayModeBar": False}, style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.H3("Travel Distance Distribution", style={
                    "margin": "0 0 4px 0", "color": COLORS["accent"],
                    "fontSize": "15px", "fontWeight": "700",
                }),
                html.Div(
                    "Box plots of demand-cell-to-ERC distance (km) across all k values "
                    "for the selected method. Selected k is highlighted.",
                    style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"},
                ),
                dcc.Graph(id="perm-dist-chart",
                          config={"displayModeBar": False}, style={"height": "300px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ERC load balance chart
        html.Div([
            html.H3("ERC Load Balance", style={
                "margin": "0 0 4px 0", "color": COLORS["accent"],
                "fontSize": "15px", "fontWeight": "700",
            }),
            html.Div(
                "Demand weight served per ERC for the selected configuration. "
                "Red bars = overloaded (>1.5× mean), amber = moderately high (>1.1×), blue = balanced. "
                "Available for k ≤ 100.",
                style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"},
            ),
            dcc.Graph(id="perm-load-chart",
                      config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

    ], id="track1-panel", style={"marginTop": "16px"}),

    # ── TRACK 2 PANEL ────────────────────────────────────────────────────
    html.Div([
        # Controls row
        html.Div([
            html.Div([
                html.Div("Day", style={"fontWeight": "600", "marginBottom": "6px", "fontSize": "13px"}),
                dcc.Dropdown(
                    id="dyn-day",
                    options=[{"label": d, "value": d} for d in DAY_OPTIONS],
                    value=DEFAULT_DAY, clearable=False,
                ),
            ], style={"width": "200px"}),
            html.Div([
                html.Div([
                    html.Span("Hour of Day", style={"fontWeight": "600", "fontSize": "13px"}),
                    html.Span(id="dyn-hour-label", style={
                        "marginLeft": "10px", "fontWeight": "700",
                        "color": COLORS["accent2"], "fontSize": "14px",
                    }),
                ], style={"marginBottom": "10px"}),
                dcc.Slider(
                    id="dyn-hour", min=0, max=23, step=1, value=17,
                    marks={v: str(v) for v in range(0, 24, 3)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "1", "padding": "0 10px", "paddingBottom": "8px"}),
            html.Div([
                html.Div("Show Top-K", style={"fontWeight": "600", "marginBottom": "6px", "fontSize": "13px"}),
                dcc.RadioItems(
                    id="dyn-topk",
                    options=[{"label": f" Top-{v}", "value": v} for v in TOPK_VALUES],
                    value=DEFAULT_TOPK,
                    labelStyle={"display": "block", "marginBottom": "4px"},
                    inputStyle={"marginRight": "6px"},
                ),
            ], style={"width": "140px"}),
        ], style={**CARD_STYLE, "display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "flex-end"}),

        # KPI cards
        html.Div(id="dyn-kpis", style={
            "display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "12px", "marginBottom": "16px",
        }),

        # Deployment Queue table
        html.Div([
            html.H3("Deployment Queue", style={"margin": "0 0 4px 0", "color": COLORS["accent"], "fontSize": "16px", "fontWeight": "700"}),
            html.Div(
                "Top-K highest-risk cells for the selected day and hour. Click a row to highlight on map.",
                style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "10px"},
            ),
            dash_table.DataTable(
                id="track2-table", columns=[], data=[],
                row_selectable="single",
                selected_rows=[],
                style_table={"overflowX": "auto", "borderRadius": "8px", "overflow": "hidden"},
                style_cell={
                    "padding": "9px 12px", "textAlign": "center",
                    "fontSize": "13px", "fontFamily": "Arial, sans-serif",
                    "border": "1px solid #edf1f5", "cursor": "pointer",
                },
                style_header={
                    "fontWeight": "700", "backgroundColor": "#eef2f7",
                    "color": COLORS["accent"], "border": "1px solid #d9dee7",
                },
                style_data={"backgroundColor": "white", "color": COLORS["text"]},
                style_data_conditional=[],
                page_size=10,
            ),
        ], style={**CARD_STYLE, "marginBottom": "16px"}),

        # Map + Hourly chart
        html.Div([
            html.Div([
                html.H3("High-Risk Cells Map", style={"margin": "0 0 4px 0", "color": COLORS["accent"], "fontSize": "16px", "fontWeight": "700"}),
                html.Div(id="dyn-map-legend", style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"}),
                dcc.Graph(id="track2-map", config={"displayModeBar": True, "scrollZoom": False}, style={"height": "580px"}),
            ], style={**CARD_STYLE, "marginBottom": "16px"}),
            html.Div([
                html.H3("Hourly Risk Pattern vs. Historical Accidents", style={
                    "margin": "0 0 4px 0", "color": COLORS["accent"],
                    "fontSize": "15px", "fontWeight": "700",
                }),
                html.Div(
                    "Bars = avg model risk for Top-K cells. Purple dotted line = historical accident count for this day.",
                    style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "6px"},
                ),
                dcc.Graph(id="dyn-hourly-chart", config={"displayModeBar": False}, style={"height": "320px", "width": "100%"}),
                html.Div(id="dyn-hourly-insight", style={"fontSize": "13px", "color": COLORS["text"], "marginTop": "10px"}),
            ], style={**CARD_STYLE, "marginBottom": "16px"}),

            # Persistent hotspots
            html.Div([
                html.H3("Persistent Hotspots", style={
                    "margin": "0 0 4px 0", "color": COLORS["accent"],
                    "fontSize": "15px", "fontWeight": "700",
                }),
                html.Div(
                    f"Cells most frequently in Top-10 across all {_MAX_HOW} day×hour slots. "
                    "Size = frequency, color = avg risk score.",
                    id="dyn-hotspot-subtitle",
                    style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"},
                ),
                html.Div([
                    dcc.Graph(id="dyn-hotspot-map",
                              config={"displayModeBar": False},
                              style={"height": "400px", "flex": "1.2"}),
                    html.Div(id="dyn-hotspot-table", style={"flex": "1", "overflowY": "auto", "maxHeight": "400px"}),
                ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
            ], style=CARD_STYLE),
        ], style={"marginTop": "0px"}),

        # Day × Hour heatmap (bottom)
        html.Div([
            html.H3("Risk Pattern — Day × Hour", style={
                "margin": "0 0 4px 0", "color": COLORS["accent"],
                "fontSize": "15px", "fontWeight": "700",
            }),
            html.Div(
                "Average model risk score across all cells. Click any cell to jump to that day and hour.",
                style={"fontSize": "12px", "color": COLORS["muted"], "marginBottom": "8px"},
            ),
            dcc.Store(id="dyn-heatmap-data", data=[]),
            html.Div(id="dyn-heatmap"),
        ], style={**CARD_STYLE, "marginTop": "16px"}),

        # Store for selected cell
        dcc.Store(id="dyn-selected-cell", data=None),

    ], id="track2-panel", style={"marginTop": "16px", "display": "none"}),

], style={
    "backgroundColor": COLORS["bg"],
    "minHeight": "100vh",
    "padding": "28px 32px",
    "fontFamily": "'Helvetica Neue', Arial, sans-serif",
    "maxWidth": "1400px",
    "margin": "0 auto",
})


# =========================
# CALLBACKS
# =========================
@app.callback(
    Output("track1-panel", "style"),
    Output("track2-panel", "style"),
    Input("main-tabs", "value"),
)
def toggle_panels(active_tab):
    visible = {"marginTop": "16px"}
    hidden = {"marginTop": "16px", "display": "none"}
    return (visible, hidden) if active_tab == "tab-track1" else (hidden, visible)


@app.callback(
    Output("perm-k", "marks"),
    Output("perm-k", "min"),
    Output("perm-k", "max"),
    Output("perm-k", "value"),
    Input("perm-method", "value"),
    State("perm-k", "value"),
)
def update_k_slider(method, current_k):
    if not TRACK1_AVAILABLE:
        return {10: "10", 100: "100", 1000: "1000"}, 10, 1000, 10
    opts = get_k_options_for_method(method)
    if not opts:
        return {10: "10", 100: "100", 1000: "1000"}, 10, 1000, 10
    marks = {int(o["value"]): {"label": str(int(o["value"])), "style": {"fontSize": "12px"}} for o in opts}
    vals = [int(o["value"]) for o in opts]
    # Keep current k if it's valid for the new method; otherwise default to first
    keep_k = int(current_k) if current_k is not None and int(current_k) in vals else vals[0]
    return marks, min(vals), max(vals), keep_k


@app.callback(
    Output("perm-k-display", "children"),
    Input("perm-k", "value"),
)
def update_k_display(k_val):
    if k_val is None:
        return ""
    return f"k = {int(k_val)}"

@app.callback(
    Output("perm-method", "value", allow_duplicate=True),
    Output("perm-k", "value",      allow_duplicate=True),
    Input("config-table", "active_cell"),
    Input("config-table", "selected_rows"),
    prevent_initial_call=True,
)
def table_row_click(active_cell, selected_rows):
    """When user clicks a config table row, update method+k selectors."""
    from dash import callback_context, no_update
    if not callback_context.triggered:
        return no_update, no_update
    # Prefer selected_rows (checkbox click) over active_cell (cell click)
    if not active_cell and not selected_rows:
        return no_update, no_update
    # We embed _method/_k as hidden data columns — fetch via clientside
    # Use active_cell row index
    row_idx = active_cell["row"] if active_cell else (selected_rows[0] if selected_rows else None)
    if row_idx is None:
        return no_update, no_update
    # Reconstruct from erc_metrics sorted the same way as table_data
    sorted_metrics = erc_metrics.sort_values(["method", "k"]).reset_index(drop=True)
    if row_idx >= len(sorted_metrics):
        return no_update, no_update
    row = sorted_metrics.iloc[row_idx]
    return row["method"], int(row["k"])




@app.callback(
    Output("perm-kpis", "children"),
    Output("perm-map", "figure"),
    Output("perm-map-legend", "children"),
    Output("config-table", "data"),
    Output("config-table", "style_data_conditional"),
    Output("config-table", "selected_rows"),
    Output("perm-coverage-curve", "figure"),
    Output("perm-dist-chart", "figure"),
    Output("perm-load-chart", "figure"),
    Input("main-tabs", "value"),
    Input("perm-method", "value"),
    Input("perm-k", "value"),
)
def update_permanent_tab(active_tab, method, k_value):
    if active_tab != "tab-track1":
        return [], go.Figure(), "", [], [], [], go.Figure(), go.Figure(), go.Figure()

    if not TRACK1_AVAILABLE:
        kpis = [make_kpi(t, "N/A") for t in ["Wtd Mean Travel Time", "Wtd P95 Travel Time", "Coverage ≤10 min", "Coverage ≤15 min"]]
        return kpis, blank_figure("Track 1 data not available."), "", [], [], [], go.Figure(), go.Figure(), go.Figure()

    # ── KPI cards ──
    row = pd.DataFrame()
    if not erc_metrics.empty and {"method", "k"}.issubset(erc_metrics.columns):
        row = erc_metrics[(erc_metrics["method"] == method) & (erc_metrics["k"] == k_value)].copy()

    if row.empty:
        kpis = [make_kpi(t, "N/A") for t in ["Wtd Mean Travel Time", "Wtd P95 Travel Time", "Coverage ≤10 min", "Coverage ≤15 min"]]
    else:
        sel = row.iloc[0]
        best = best_erc_config() or {}
        kpis = [
            make_kpi("Wtd Mean Travel Time", format_minutes(sel["wtd_mean_min"]),
                     subtitle="(lower is better)"),
            make_kpi("Wtd P95 Travel Time", format_minutes(sel["wtd_p95_min"]),
                     subtitle="(lower is better)"),
            make_kpi("Coverage ≤ 10 min", format_pct(sel["coverage_10_min"]),
                     subtitle="% demand covered",
                     highlight=(best.get("method") == method and best.get("k") == k_value)),
            make_kpi("Coverage ≤ 15 min", format_pct(sel["coverage_15_min"]),
                     subtitle="% demand covered",
                     highlight=(best.get("method") == method and best.get("k") == k_value)),
        ]

    # ── Map ──
    fig = build_permanent_map(method, k_value)

    # ── Map legend text ──
    k_int = int(k_value) if k_value is not None else 0
    if k_int <= 100:
        legend_text = (
            "● Demand cells colored by ERC service area.  "
            "Red ● = ERC location.  "
            "Lines = assignment.  "
            "Orange rings = underserved cells (nearest ERC >30 km)."
        )
    else:
        legend_text = (
            f"● Demand cells (sized by incident weight).  "
            f"Red ● = {k_int} ERC locations (labels hidden at this scale).  "
            "Orange rings = underserved cells (nearest ERC >30 km)."
        )

    # ── Comparison table (DataTable) ──
    table_data = []
    table_style_cond = []
    selected_row_idx = []
    if not erc_metrics.empty:
        sorted_metrics = erc_metrics.sort_values(["method", "k"]).reset_index(drop=True)
        for i, mrow in sorted_metrics.iterrows():
            is_selected = (mrow["method"] == method and mrow["k"] == k_value)
            if is_selected:
                selected_row_idx = [i]
            table_data.append({
                "method_label": METHOD_LABELS.get(mrow["method"], mrow["method"]),
                "k":      str(int(mrow["k"])),
                "wtd_mean": format_minutes(mrow["wtd_mean_min"]),
                "wtd_p95":  format_minutes(mrow["wtd_p95_min"]),
                "cov_10":   format_pct(mrow["coverage_10_min"]),
                "cov_15":   format_pct(mrow["coverage_15_min"]),
                # hidden cols for click callback
                "_method": mrow["method"],
                "_k":      str(int(mrow["k"])),
            })
        table_style_cond = [
            {"if": {"row_index": i},
             "backgroundColor": "#dbeafe", "fontWeight": "700",
             "borderLeft": "3px solid #2563a8"}
            for i in selected_row_idx
        ]

    coverage_fig = build_coverage_curve(method, k_value)
    dist_fig = build_distance_distribution(method, k_value)
    load_fig = build_erc_load_chart(method, k_value)

    return kpis, fig, legend_text, table_data, table_style_cond, selected_row_idx, coverage_fig, dist_fig, load_fig


@app.callback(
    Output("dyn-kpis", "children"),
    Output("track2-table", "columns"),
    Output("track2-table", "data"),
    Output("track2-table", "style_data_conditional"),
    Output("track2-map", "figure"),
    Output("dyn-map-legend", "children"),
    Output("dyn-hourly-chart", "figure"),
    Output("dyn-hourly-insight", "children"),
    Output("dyn-heatmap-data", "data"),
    Input("main-tabs", "value"),
    Input("dyn-day", "value"),
    Input("dyn-hour", "value"),
    Input("dyn-topk", "value"),
    State("dyn-selected-cell", "data"),
)
def update_dynamic_tab(active_tab, day_name, hour, top_k, selected_cell):
    if active_tab != "tab-track2":
        return [], [], [], [], go.Figure(), "", go.Figure(), "", []

    slot = get_track2_slot(day_name, hour, top_k)
    hourly_chart = build_hourly_chart_with_history(day_name, hour, top_k)
    dynamic_map = build_dynamic_map(day_name, hour, top_k, selected_cell_id=selected_cell)
    heatmap_data, _, _ = build_day_hour_heatmap_data(day_name, hour)

    avg_risk    = float(slot["risk_score"].mean()) if not slot.empty else np.nan
    highest_risk = float(slot["risk_score"].max()) if not slot.empty else np.nan
    hist_row = hourly_summary[(hourly_summary["day_name"] == day_name) & (hourly_summary["hour"] == int(hour))]
    hist_count = int(hist_row["historical_accident_count"].iloc[0]) if not hist_row.empty else None

    kpis = [
        make_kpi("Day / Hour", f"{day_name}  {_hour_label(hour)}"),
        make_kpi("Top-K Cells", str(int(top_k))),
        make_kpi("Avg Risk (Queue)", "N/A" if pd.isna(avg_risk) else f"{avg_risk:.4f}"),
        make_kpi("Peak Risk (Queue)", "N/A" if pd.isna(highest_risk) else f"{highest_risk:.4f}"),
        make_kpi("Historical Accidents", "N/A" if hist_count is None else f"{hist_count:,}", subtitle="this day×hour"),
    ]

    COL_LABELS = {
        "pred_rank":     "Rank",
        "cell_id":       "Cell ID",
        "cell_lat":      "Lat",
        "cell_lng":      "Lng",
        "risk_score":    "Risk Score",
        "cell_rate":     "Cell Rate",
        "cell_how_rate": "HOW Rate",
        "neighbor_rate": "Neighbor Rate",
    }
    display_cols = ["pred_rank", "cell_id", "cell_lat", "cell_lng", "risk_score"]
    for opt_col in ["cell_rate", "cell_how_rate", "neighbor_rate"]:
        if opt_col in slot.columns:
            display_cols.append(opt_col)

    table_columns = [{"name": COL_LABELS.get(c, c), "id": c} for c in display_cols]
    style_cond = []
    if slot.empty:
        table_data = []
    else:
        tf = slot[display_cols].copy()
        for col in tf.columns:
            if col != "cell_id":
                tf[col] = pd.to_numeric(tf[col], errors="coerce")
        table_data = tf.round(4).to_dict("records")
        # Highlight selected cell row
        if selected_cell:
            for i, row in enumerate(table_data):
                if str(row.get("cell_id")) == str(selected_cell):
                    style_cond = [{"if": {"row_index": i},
                                   "backgroundColor": "#dbeafe",
                                   "borderLeft": "3px solid #2563a8"}]
                    break

    # Map legend
    n_underserved = 0
    if not slot.empty:
        n_underserved = len(slot)  # all slots are "high risk" by definition
    sel_name = f"  ·  Selected: {selected_cell}" if selected_cell else ""
    map_legend = (
        f"Top-{int(top_k)} highest-risk cells · {day_name} {_hour_label(hour)}{sel_name}"
    )

    # Insight text
    topk_hourly = [
        float(get_track2_slot(day_name, h, top_k, log_debug=False)["risk_score"].mean())
        if not get_track2_slot(day_name, h, top_k, log_debug=False).empty else 0.0
        for h in range(24)
    ]
    if topk_hourly:
        day_mean = float(np.mean(topk_hourly))
        peak_hour = int(np.argmax(topk_hourly))
        sel_val = float(topk_hourly[int(hour)])
        pct = ((sel_val - day_mean) / day_mean * 100.0) if day_mean else 0.0
        insight = (
            f"Peak hour: {_hour_label(peak_hour)}  ·  "
            f"Selected hour is {pct:+.1f}% vs day-average Top-{int(top_k)} risk"
        )
    else:
        insight = "No Top-K hourly rows found for this day."

    return kpis, table_columns, table_data, style_cond, dynamic_map, map_legend, hourly_chart, insight, heatmap_data




@app.callback(
    Output("dyn-hour-label", "children"),
    Input("dyn-hour", "value"),
)
def update_hour_label(hour):
    return _hour_label(hour) if hour is not None else ""


@app.callback(
    Output("dyn-selected-cell", "data"),
    Input("track2-table", "active_cell"),
    Input("track2-table", "selected_rows"),
    Input("dyn-day", "value"),
    Input("dyn-hour", "value"),
    Input("dyn-topk", "value"),
    State("track2-table", "data"),
    prevent_initial_call=True,
)
def update_selected_cell(active_cell, selected_rows, day_name, hour, top_k, table_data):
    from dash import callback_context, no_update
    triggered = [t["prop_id"] for t in callback_context.triggered]
    # If day/hour/topk changed, clear selection
    if any("dyn-day" in t or "dyn-hour" in t or "dyn-topk" in t for t in triggered):
        return None
    if not table_data:
        return None
    row_idx = None
    if active_cell:
        row_idx = active_cell["row"]
    elif selected_rows:
        row_idx = selected_rows[0]
    if row_idx is None or row_idx >= len(table_data):
        return None
    return str(table_data[row_idx].get("cell_id", ""))




@app.callback(
    Output("dyn-heatmap", "children"),
    Input("dyn-heatmap-data", "data"),
    State("dyn-day", "value"),
    State("dyn-hour", "value"),
)
def render_heatmap_div(rows, selected_day, selected_hour):
    """Render the Day×Hour heatmap as a pure HTML/CSS grid — no plotly.js needed."""
    if not rows:
        return html.Div("Loading...", style={"color": COLORS["muted"], "padding": "20px"})

    days  = _DAY_ORDER
    hours = list(range(24))

    # YlOrRd color interpolation (5 stops)
    def risk_color(norm):
        stops = [
            (0.0,  (255, 255, 204)),
            (0.25, (254, 178,  76)),
            (0.5,  (253, 141,  60)),
            (0.75, (227,  26,  28)),
            (1.0,  (128,   0,  38)),
        ]
        for i in range(len(stops)-1):
            t0, c0 = stops[i]
            t1, c1 = stops[i+1]
            if norm <= t1:
                t = (norm - t0) / (t1 - t0)
                r = int(c0[0] + t*(c1[0]-c0[0]))
                g = int(c0[1] + t*(c1[1]-c0[1]))
                b = int(c0[2] + t*(c1[2]-c0[2]))
                return f"rgb({r},{g},{b})"
        return "rgb(128,0,38)"

    # Build lookup
    lookup = {(r["day"], r["hour"]): r for r in rows}

    CELL_W = "36px"
    CELL_H = "26px"

    # Header row (hour labels every 3h)
    header_cells = [html.Div("", style={"width": "36px", "flexShrink": "0"})]
    for h in hours:
        lbl = _hour_label(h).replace(":00 ", "").replace(" ", "") if h % 3 == 0 else ""
        header_cells.append(html.Div(lbl, style={
            "width": CELL_W, "flexShrink": "0", "textAlign": "center",
            "fontSize": "9px", "color": COLORS["muted"],
        }))

    grid_rows = [html.Div(header_cells, style={"display": "flex", "marginBottom": "2px"})]

    for day in days:
        cells = [html.Div(day, style={
            "width": "36px", "flexShrink": "0",
            "fontSize": "11px", "fontWeight": "600",
            "color": COLORS["accent"], "paddingTop": "4px",
        })]
        for h in hours:
            r = lookup.get((day, h), {})
            norm = r.get("norm", 0.0)
            val  = r.get("val", 0.0)
            is_sel = (day == selected_day and h == int(selected_hour or -1))
            bg = risk_color(norm)
            border = "2.5px solid #1c3d5a" if is_sel else "1px solid rgba(255,255,255,0.3)"
            cells.append(html.Div(
                title=f"{day} {_hour_label(h)}: {val:.4f}",
                style={
                    "width": CELL_W, "height": CELL_H, "flexShrink": "0",
                    "backgroundColor": bg,
                    "border": border,
                    "boxSizing": "border-box",
                    "cursor": "pointer",
                    "borderRadius": "1px",
                },
                id={"type": "heatmap-cell", "day": day, "hour": h},
            ))
        grid_rows.append(html.Div(cells, style={
            "display": "flex", "marginBottom": "2px",
        }))

    # Color legend bar
    legend_stops = ["rgb(255,255,204)", "rgb(254,178,76)", "rgb(253,141,60)", "rgb(227,26,28)", "rgb(128,0,38)"]
    gradient = f"linear-gradient(to right, {', '.join(legend_stops)})"
    all_vals = [r["val"] for r in rows]
    v_min = min(all_vals) if all_vals else 0.0
    v_max = max(all_vals) if all_vals else 1.0

    legend = html.Div([
        html.Div(style={
            "height": "10px", "flex": "1",
            "background": gradient, "borderRadius": "3px",
        }),
        html.Div(f"{v_min:.4f}", style={"fontSize": "9px", "color": COLORS["muted"], "marginLeft": "6px"}),
        html.Div("Avg Risk →", style={"fontSize": "9px", "color": COLORS["muted"], "margin": "0 6px"}),
        html.Div(f"{v_max:.4f}", style={"fontSize": "9px", "color": COLORS["muted"]}),
    ], style={"display": "flex", "alignItems": "center", "marginTop": "6px", "paddingLeft": "36px"})

    return html.Div([
        html.Div(grid_rows, style={"overflowX": "auto"}),
        legend,
    ])


@app.callback(
    Output("dyn-day",  "value", allow_duplicate=True),
    Output("dyn-hour", "value", allow_duplicate=True),
    Input({"type": "heatmap-cell", "day": ALL, "hour": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def heatmap_cell_click(n_clicks_list):
    from dash import callback_context, no_update, ALL
    if not callback_context.triggered:
        return no_update, no_update
    triggered = callback_context.triggered[0]
    if not triggered["value"]:
        return no_update, no_update
    import json as _json
    prop_id = triggered["prop_id"]
    # prop_id looks like: {"day":"Mon","hour":17,"type":"heatmap-cell"}.n_clicks
    try:
        id_part = prop_id.split(".n_clicks")[0]
        cell_id = _json.loads(id_part)
        return cell_id["day"], int(cell_id["hour"])
    except Exception:
        return no_update, no_update




@app.callback(
    Output("dyn-hotspot-map",   "figure"),
    Output("dyn-hotspot-table", "children"),
    Input("main-tabs", "value"),
)
def update_hotspot_panel(active_tab):
    if active_tab != "tab-track2":
        return go.Figure(), html.Div()

    fig = build_hotspot_map()

    # Build hotspot summary table
    _hs = _hotspot_freq.copy()
    _hs["pct"] = (_hs["times_in_top10"] / _MAX_HOW * 100).round(1)
    _hs["avg_risk_fmt"] = _hs["avg_risk"].round(4)

    th_s = {"padding": "7px 10px", "fontWeight": "700", "backgroundColor": "#eef2f7",
             "color": COLORS["accent"], "fontSize": "12px",
             "borderBottom": f"2px solid {COLORS['border']}", "textAlign": "center"}
    td_s = {"padding": "6px 10px", "fontSize": "12px",
             "borderBottom": f"1px solid {COLORS['border']}", "textAlign": "center"}

    rows = []
    for i, row in _hs.iterrows():
        rows.append(html.Tr([
            html.Td(f"#{i+1}", style={**td_s, "fontWeight": "700", "color": COLORS["accent2"]}),
            html.Td(row["cell_id"], style=td_s),
            html.Td(f"{int(row['times_in_top10'])} / {_MAX_HOW}", style=td_s),
            html.Td(f"{row['pct']}%", style=td_s),
            html.Td(f"{row['avg_risk_fmt']:.4f}", style=td_s),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Rank", style=th_s),
            html.Th("Cell ID", style=th_s),
            html.Th("In Top-10", style=th_s),
            html.Th("% of Slots", style=th_s),
            html.Th("Avg Risk", style=th_s),
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "12px"})

    return fig, table


@app.callback(
    Output("track2-map", "figure", allow_duplicate=True),
    Output("dyn-map-legend", "children", allow_duplicate=True),
    Input("dyn-selected-cell", "data"),
    State("dyn-day", "value"),
    State("dyn-hour", "value"),
    State("dyn-topk", "value"),
    prevent_initial_call=True,
)
def update_map_on_selection(selected_cell, day_name, hour, top_k):
    """Redraw the map when a table row is selected, zooming in on the cell."""
    from dash import no_update
    if day_name is None or hour is None or top_k is None:
        return no_update, no_update

    fig = build_dynamic_map(day_name, hour, top_k, selected_cell_id=selected_cell)

    if selected_cell:
        # Find the selected cell's coordinates
        slot = get_track2_slot(day_name, hour, top_k, log_debug=False)
        sel_row = slot[slot["cell_id"] == str(selected_cell)]
        if not sel_row.empty:
            sel_lat = float(sel_row["cell_lat"].iloc[0])
            sel_lng = float(sel_row["cell_lng"].iloc[0])
            # Zoom in close enough to see the neighborhood
            fig.update_layout(
                **_map_layout(
                    style="carto-positron",
                    center={"lat": sel_lat, "lon": sel_lng},
                    zoom=11,
                )
            )
        sel_name = f"  ·  Selected: {selected_cell}"
    else:
        # Zoom back out to CA view when deselected
        fig.update_layout(
            **_map_layout(
                style="carto-positron",
                center={"lat": 37.26, "lon": -121.5},
                zoom=5.6,
            )
        )
        sel_name = ""

    legend = f"Top-{int(top_k)} highest-risk cells · {day_name} {_hour_label(hour)}{sel_name}"
    return fig, legend


if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PORT", 8050))
    print(f"[dashboard] Launching on port {_port}. Track 1 available: {TRACK1_AVAILABLE}")
    print(f"[dashboard] Base dir: {BASE_DIR}")
    print(f"[dashboard] Output dir: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=_port, debug=False)
