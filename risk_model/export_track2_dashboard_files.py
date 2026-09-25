import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
GRID_DEG = 0.1
TRAIN_YEAR = 2022
DATA_PATH = "./CA_Accidents_Main.csv"
OUTPUT_DIR = Path("outputs")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def find_data_path() -> Path | None:
    configured = Path(DATA_PATH).expanduser()
    if configured.exists():
        return configured.resolve()

    script_dir = Path(__file__).resolve().parent
    search_roots = [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
        script_dir.parent.parent.parent,
    ]

    seen = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)

        direct = root / "CA_Accidents_Main.csv"
        if direct.exists():
            return direct.resolve()

        for found in root.rglob("CA_Accidents_Main.csv"):
            return found.resolve()
    return None


def stop_missing_input() -> None:
    print("Required input file `CA_Accidents_Main.csv` was not found.")
    print("Set DATA_PATH at the top of export_track2_dashboard_files.py or place the file in this folder or a nearby parent folder.")
    raise SystemExit(1)


def load_accidents(path: Path) -> pd.DataFrame:
    print(f"Loading accidents data from: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = {"Start_Time", "Lat", "Lng", "Severity"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"Input file is missing required columns: {missing}")
        raise SystemExit(1)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time", "Lat", "Lng", "Severity"]).copy()
    df["Start_Time"] = df["Start_Time"].dt.tz_localize(None)

    df["hour_bucket"] = df["Start_Time"].dt.floor("h")
    df["hour"] = df["Start_Time"].dt.hour
    df["dayofweek"] = df["Start_Time"].dt.dayofweek
    df["month"] = df["Start_Time"].dt.month
    df["year"] = df["Start_Time"].dt.year
    df["cell_lat_bin"] = (df["Lat"] // GRID_DEG).astype(int)
    df["cell_lng_bin"] = (df["Lng"] // GRID_DEG).astype(int)
    df["cell_id"] = df["cell_lat_bin"].astype(str) + "_" + df["cell_lng_bin"].astype(str)
    df["hour_of_week"] = df["dayofweek"] * 24 + df["hour"]
    return df


def compute_static_track2_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"] < TRAIN_YEAR].copy()
    test_df = df[df["year"] >= TRAIN_YEAR].copy()

    if train_df.empty or test_df.empty:
        print("Track 2 export requires both train rows (< 2022) and test rows (>= 2022).")
        raise SystemExit(1)

    all_cells = train_df[["cell_id", "cell_lat_bin", "cell_lng_bin"]].drop_duplicates().set_index("cell_id")
    if all_cells.empty:
        print("The train-only cell universe is empty after preprocessing.")
        raise SystemExit(1)

    n_train_hours = int(train_df["hour_bucket"].nunique())
    n_cells = int(len(all_cells))
    global_rate = len(train_df) / (n_train_hours * n_cells)

    cell_rate = train_df.groupby("cell_id").size() / n_train_hours
    how_hours = train_df.groupby("hour_of_week")["hour_bucket"].nunique()
    counts_cell_how = train_df.groupby(["cell_id", "hour_of_week"]).size()
    den = counts_cell_how.index.get_level_values("hour_of_week").map(how_hours).to_numpy()
    cell_how_rate = counts_cell_how / den

    rate_dict = cell_rate.to_dict()

    def neighbor_rate(clat: int, clng: int) -> float:
        neighbors = []
        for dlat in (-1, 0, 1):
            for dlng in (-1, 0, 1):
                if dlat == 0 and dlng == 0:
                    continue
                neighbors.append(rate_dict.get(f"{clat + dlat}_{clng + dlng}", global_rate))
        return float(np.mean(neighbors))

    cell_feat = pd.DataFrame(index=all_cells.index)
    cell_feat["cell_rate"] = cell_rate.reindex(cell_feat.index).fillna(global_rate)
    cell_feat["neighbor_rate"] = [neighbor_rate(row.cell_lat_bin, row.cell_lng_bin) for row in all_cells.itertuples()]

    cell_ids = all_cells.index.to_numpy()
    cell_rate_vec = cell_feat["cell_rate"].reindex(cell_ids).to_numpy()
    neighbor_vec = cell_feat["neighbor_rate"].reindex(cell_ids).to_numpy()
    cell_lat = (all_cells.loc[cell_ids, "cell_lat_bin"].to_numpy() + 0.5) * GRID_DEG
    cell_lng = (all_cells.loc[cell_ids, "cell_lng_bin"].to_numpy() + 0.5) * GRID_DEG

    risk_rows = []
    hourly_rows = []
    topk_rows = []

    # The full LightGBM model depends on timestamp-specific recency, which does not
    # collapse cleanly to a single hour-of-week table. For the dashboard export we
    # use a reproducible static Track 2 score: mean(cell_rate, neighbor_rate, cell_how_rate).
    for how in range(168):
        day_name = DAY_NAMES[how // 24]
        hour = how % 24

        if how in how_hours.index:
            how_series = cell_how_rate.xs(how, level="hour_of_week")
            how_vec = how_series.reindex(cell_ids).fillna(cell_feat["cell_rate"].reindex(cell_ids)).to_numpy()
        else:
            how_vec = cell_rate_vec.copy()

        risk_score = (cell_rate_vec + neighbor_vec + how_vec) / 3.0
        order = np.argsort(-risk_score, kind="stable")
        pred_rank = np.empty_like(order)
        pred_rank[order] = np.arange(1, len(order) + 1)

        frame = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "cell_lat": cell_lat,
                "cell_lng": cell_lng,
                "hour_of_week": how,
                "day_name": day_name,
                "hour": hour,
                "risk_score": risk_score,
                "cell_rate": cell_rate_vec,
                "cell_how_rate": how_vec,
                "neighbor_rate": neighbor_vec,
                "pred_rank": pred_rank,
                "is_top_10": (pred_rank <= 10).astype(int),
                "is_top_50": (pred_rank <= 50).astype(int),
                "is_top_100": (pred_rank <= 100).astype(int),
            }
        )
        risk_rows.append(frame)

        hourly_rows.append(
            {
                "hour_of_week": how,
                "day_name": day_name,
                "hour": hour,
                "avg_risk_score": float(risk_score.mean()),
                "historical_accident_count": int(train_df[train_df["hour_of_week"] == how].shape[0]),
            }
        )

        for k, label in ((10, "top10"), (50, "top50"), (100, "top100")):
            top_idx = order[: min(k, len(order))]
            snap = frame.iloc[top_idx][
                ["hour_of_week", "day_name", "hour", "cell_id", "cell_lat", "cell_lng", "risk_score", "pred_rank"]
            ].copy()
            snap.insert(3, "k_label", label)
            topk_rows.append(snap)

    dashboard_cell_risk = pd.concat(risk_rows, ignore_index=True)
    dashboard_hourly_summary = pd.DataFrame(hourly_rows).sort_values("hour_of_week").reset_index(drop=True)
    dashboard_topk_snapshot = pd.concat(topk_rows, ignore_index=True)
    return dashboard_cell_risk, dashboard_hourly_summary, dashboard_topk_snapshot


def main() -> None:
    np.random.seed(SEED)
    data_path = find_data_path()
    if data_path is None:
        stop_missing_input()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_accidents(data_path)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Start_Time'].min()} -> {df['Start_Time'].max()}")

    dashboard_cell_risk, dashboard_hourly_summary, dashboard_topk_snapshot = compute_static_track2_tables(df)

    cell_risk_path = OUTPUT_DIR / "dashboard_cell_risk.csv"
    hourly_summary_path = OUTPUT_DIR / "dashboard_hourly_summary.csv"
    topk_snapshot_path = OUTPUT_DIR / "dashboard_topk_snapshot.csv"

    dashboard_cell_risk.to_csv(cell_risk_path, index=False)
    dashboard_hourly_summary.to_csv(hourly_summary_path, index=False)
    dashboard_topk_snapshot.to_csv(topk_snapshot_path, index=False)

    print(f"\nWrote {cell_risk_path} {dashboard_cell_risk.shape}")
    print(dashboard_cell_risk.head(5).to_string(index=False))
    print(f"\nWrote {hourly_summary_path} {dashboard_hourly_summary.shape}")
    print(dashboard_hourly_summary.head(5).to_string(index=False))
    print(f"\nWrote {topk_snapshot_path} {dashboard_topk_snapshot.shape}")
    print(dashboard_topk_snapshot.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
