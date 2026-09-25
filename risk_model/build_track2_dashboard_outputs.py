import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
SEED = 42
GRID_DEG = 0.1
TRAIN_YEAR = 2022
NEG_RATIO = 20
TRAIN_HOUR_SAMPLE = 30000
N_PER_HOW = 50
OUTPUT_DIR = Path("outputs")
DATA_PATH = "./CA_Accidents_Main.csv"

DAY_NAMES_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURES = [
    "cell_rate",
    "neighbor_rate",
    "cell_how_rate",
    "rec_1h",
    "rec_6h",
    "rec_24h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "rush",
]


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
        candidate = root / "CA_Accidents_Main.csv"
        if candidate.exists():
            return candidate.resolve()
        for found in root.rglob("CA_Accidents_Main.csv"):
            return found.resolve()
    return None


def stop_missing_input() -> None:
    print("Required input file `CA_Accidents_Main.csv` was not found.")
    print("Set DATA_PATH at the top of build_track2_dashboard_outputs.py, or place the file in this folder or a nearby parent folder.")
    sys.exit(1)


def time_feats(ts: pd.Timestamp) -> dict[str, float]:
    hour = ts.hour
    dow = ts.dayofweek
    month = ts.month
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "weekend": int(dow >= 5),
        "rush": int((7 <= hour <= 9) or (16 <= hour <= 19)),
        "how": dow * 24 + hour,
    }


def load_data(data_path: Path) -> pd.DataFrame:
    print(f"Loading accidents data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    required_cols = {"Start_Time", "Lat", "Lng", "Severity"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        print(f"Input file is missing required columns: {missing}")
        sys.exit(1)

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Lat", "Lng", "Start_Time", "Severity"]).copy()
    df["Start_Time"] = df["Start_Time"].dt.tz_localize(None)

    df["hour_bucket"] = df["Start_Time"].dt.floor("h")
    df["hour"] = df["Start_Time"].dt.hour
    df["dayofweek"] = df["Start_Time"].dt.dayofweek
    df["month"] = df["Start_Time"].dt.month
    df["year"] = df["Start_Time"].dt.year
    df["cell_lat"] = (df["Lat"] // GRID_DEG).astype(int)
    df["cell_lng"] = (df["Lng"] // GRID_DEG).astype(int)
    df["cell_id"] = df["cell_lat"].astype(str) + "_" + df["cell_lng"].astype(str)
    df["hour_of_week"] = df["dayofweek"] * 24 + df["hour"]
    return df


def sample_hours_by_how(test_df: pd.DataFrame, rng: np.random.Generator) -> dict[int, np.ndarray]:
    sampled = {}
    grouped = (
        test_df[["hour_bucket", "hour_of_week"]]
        .drop_duplicates()
        .groupby("hour_of_week")["hour_bucket"]
    )

    for how in range(168):
        choices = np.array(sorted(pd.to_datetime(grouped.get_group(how)).to_numpy())) if how in grouped.groups else np.array([], dtype="datetime64[ns]")
        if len(choices) == 0:
            sampled[how] = np.array([], dtype="datetime64[ns]")
            continue
        replace = len(choices) < N_PER_HOW
        size = N_PER_HOW if replace else min(N_PER_HOW, len(choices))
        picked = rng.choice(choices, size=size, replace=replace)
        sampled[how] = np.array(sorted(pd.to_datetime(picked).to_numpy()))
    return sampled


def build_recency_maps(
    df: pd.DataFrame,
    all_cells_set: set[str],
    cell_to_idx: dict[str, int],
    n_cells: int,
    needed_hours: set[pd.Timestamp],
) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    min_h, max_h = min(needed_hours), max(needed_hours)
    all_hours_full = pd.date_range(min_h, max_h, freq="h")

    tmp = (
        df[df["cell_id"].isin(all_cells_set)]
        .drop_duplicates(subset=["hour_bucket", "cell_id"])[["hour_bucket", "cell_id"]]
    )

    hour_to_cellidx = {}
    for hb, grp in tmp.groupby("hour_bucket"):
        idxs = [cell_to_idx[cid] for cid in grp["cell_id"].values]
        if idxs:
            hour_to_cellidx[pd.Timestamp(hb)] = np.array(idxs, dtype=np.int32)

    buf24 = [np.zeros(n_cells, dtype=np.uint8) for _ in range(24)]
    sum6 = np.zeros(n_cells, dtype=np.int16)
    sum24 = np.zeros(n_cells, dtype=np.int16)
    prev_hour_vec = np.zeros(n_cells, dtype=np.uint8)

    recency_by_hour = {}
    for t_i, hb in enumerate(all_hours_full):
        cur = np.zeros(n_cells, dtype=np.uint8)
        idxs = hour_to_cellidx.get(pd.Timestamp(hb))
        if idxs is not None and len(idxs) > 0:
            cur[idxs] = 1

        rec_1h = prev_hour_vec.copy()
        rec_6h = sum6.copy()
        rec_24h = sum24.copy()
        if hb in needed_hours:
            recency_by_hour[pd.Timestamp(hb)] = (rec_1h, rec_6h, rec_24h)

        out24 = buf24[t_i % 24]
        out6 = buf24[(t_i - 6) % 24] if t_i >= 6 else np.zeros(n_cells, dtype=np.uint8)

        buf24[t_i % 24] = cur
        sum24 = sum24 + cur.astype(np.int16) - out24.astype(np.int16)
        sum6 = sum6 + cur.astype(np.int16) - out6.astype(np.int16)
        prev_hour_vec = cur

    return recency_by_hour


def train_model(
    train_df: pd.DataFrame,
    train_hours: np.ndarray,
    pos_cells_by_hour_train: dict[pd.Timestamp, set[str]],
    cell_ids: np.ndarray,
    all_cells_set: set[str],
    cell_to_idx: dict[str, int],
    cell_feat: pd.DataFrame,
    cell_how_rate: pd.Series,
    how_hours: pd.Series,
    recency_by_hour: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> lgb.LGBMClassifier:
    train_sample_n = min(TRAIN_HOUR_SAMPLE, len(train_hours))
    sampled_train_hours = rng.choice(train_hours, size=train_sample_n, replace=False)

    x_list = []
    y_list = []

    for hb_raw in sampled_train_hours:
        hb = pd.Timestamp(hb_raw)
        pos = [cid for cid in pos_cells_by_hour_train.get(hb, set()) if cid in all_cells_set]
        if not pos:
            continue

        tf = time_feats(hb)
        pos_set = set(pos)
        mask = ~np.isin(cell_ids, np.array(list(pos_set), dtype=object))
        neg_pool = cell_ids[mask]
        need = min(len(pos) * NEG_RATIO, len(neg_pool))
        if need <= 0:
            continue
        neg = list(rng.choice(neg_pool, size=need, replace=False))

        cells = pos + neg
        base = cell_feat.reindex(cells)

        if tf["how"] in how_hours.index:
            how_series = cell_how_rate.xs(tf["how"], level="hour_of_week")
        else:
            how_series = pd.Series(dtype=float)
        how_feat = how_series.reindex(cells).fillna(base["cell_rate"]).to_numpy()

        rec_1h_all, rec_6h_all, rec_24h_all = recency_by_hour[hb]
        cell_idx = np.array([cell_to_idx[cid] for cid in cells], dtype=np.int32)
        rec = np.column_stack([rec_1h_all[cell_idx], rec_6h_all[cell_idx], rec_24h_all[cell_idx]]).astype(float)

        x = np.column_stack(
            [
                base["cell_rate"].to_numpy(),
                base["neighbor_rate"].to_numpy(),
                how_feat,
                rec[:, 0],
                rec[:, 1],
                rec[:, 2],
                np.full(len(cells), tf["hour_sin"]),
                np.full(len(cells), tf["hour_cos"]),
                np.full(len(cells), tf["dow_sin"]),
                np.full(len(cells), tf["dow_cos"]),
                np.full(len(cells), tf["month_sin"]),
                np.full(len(cells), tf["month_cos"]),
                np.full(len(cells), tf["weekend"]),
                np.full(len(cells), tf["rush"]),
            ]
        )

        x_list.append(x)
        y_list.extend([1] * len(pos) + [0] * len(neg))

    if not x_list or not y_list:
        print("Unable to build Track 2 training matrix from the available input data.")
        sys.exit(1)

    x_train = np.vstack(x_list)
    y_train = np.array(y_list)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if pos <= 0 or neg <= 0:
        print("Training data did not contain both positive and negative samples.")
        sys.exit(1)

    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=neg / pos,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    print(f"Trained LightGBM on {len(y_train):,} rows with positive rate {y_train.mean():.5f}")
    return model


def build_full_x(
    hb: pd.Timestamp,
    cell_ids: np.ndarray,
    cell_feat: pd.DataFrame,
    cell_rate_vec: np.ndarray,
    cell_how_rate: pd.Series,
    how_hours: pd.Series,
    recency_by_hour: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    tf = time_feats(hb)
    base = cell_feat.reindex(cell_ids)

    if tf["how"] in how_hours.index:
        how_series = cell_how_rate.xs(tf["how"], level="hour_of_week")
    else:
        how_series = pd.Series(dtype=float)
    how_vec = how_series.reindex(cell_ids).fillna(base["cell_rate"]).to_numpy()

    rec_1h_all, rec_6h_all, rec_24h_all = recency_by_hour[hb]
    rec = np.column_stack([rec_1h_all, rec_6h_all, rec_24h_all]).astype(float)

    x = np.column_stack(
        [
            cell_rate_vec,
            base["neighbor_rate"].to_numpy(),
            how_vec,
            rec[:, 0],
            rec[:, 1],
            rec[:, 2],
            np.full(len(cell_ids), tf["hour_sin"]),
            np.full(len(cell_ids), tf["hour_cos"]),
            np.full(len(cell_ids), tf["dow_sin"]),
            np.full(len(cell_ids), tf["dow_cos"]),
            np.full(len(cell_ids), tf["month_sin"]),
            np.full(len(cell_ids), tf["month_cos"]),
            np.full(len(cell_ids), tf["weekend"]),
            np.full(len(cell_ids), tf["rush"]),
        ]
    )
    return x, how_vec


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    data_path = find_data_path()
    if data_path is None:
        stop_missing_input()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Start_Time'].min()} -> {df['Start_Time'].max()}")

    train_df = df[df["year"] < TRAIN_YEAR].copy()
    test_df = df[df["year"] >= TRAIN_YEAR].copy()
    if train_df.empty or test_df.empty:
        print("Track 2 requires both train rows (< 2022) and test rows (>= 2022).")
        sys.exit(1)

    all_cells = train_df[["cell_id", "cell_lat", "cell_lng"]].drop_duplicates().set_index("cell_id")
    if all_cells.empty:
        print("The train-only cell universe is empty after preprocessing.")
        sys.exit(1)

    cell_ids = all_cells.index.to_numpy()
    n_cells = len(all_cells)
    all_cells_set = set(all_cells.index)
    cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    print(f"Train accidents: {len(train_df):,}  Test accidents: {len(test_df):,}")
    print(f"Train cells: {n_cells:,}")

    train_pos = train_df.groupby(["hour_bucket", "cell_id"]).size().reset_index(name="cnt")
    test_pos = test_df.groupby(["hour_bucket", "cell_id"]).size().reset_index(name="cnt")

    pos_cells_by_hour_train = train_pos.groupby("hour_bucket")["cell_id"].apply(lambda x: set(x.values)).to_dict()
    pos_cells_by_hour_test = test_pos.groupby("hour_bucket")["cell_id"].apply(lambda x: set(x.values)).to_dict()

    train_hours = np.array(list(pos_cells_by_hour_train.keys()))
    test_hours = np.array(list(pos_cells_by_hour_test.keys()))
    if len(train_hours) == 0 or len(test_hours) == 0:
        print("No positive cell-hours were available after grouping the accidents data.")
        sys.exit(1)

    n_train_hours = train_df["hour_bucket"].nunique()
    global_rate = len(train_df) / (n_train_hours * n_cells)
    cell_rate = train_df.groupby("cell_id").size() / n_train_hours
    how_hours = train_df.groupby("hour_of_week")["hour_bucket"].nunique()
    counts_cell_how = train_df.groupby(["cell_id", "hour_of_week"]).size()
    den = counts_cell_how.index.get_level_values("hour_of_week").map(how_hours).values
    cell_how_rate = counts_cell_how / den

    rate_dict = cell_rate.to_dict()

    def neighbor_rate(clat: int, clng: int) -> float:
        nbrs = []
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                if d1 == 0 and d2 == 0:
                    continue
                nbrs.append(rate_dict.get(f"{clat + d1}_{clng + d2}", global_rate))
        return float(np.mean(nbrs))

    cell_feat = pd.DataFrame(index=all_cells.index)
    cell_feat["cell_rate"] = cell_rate.reindex(cell_feat.index).fillna(global_rate)
    cell_feat["neighbor_rate"] = [neighbor_rate(row.cell_lat, row.cell_lng) for row in all_cells.itertuples()]
    cell_rate_vec = cell_feat["cell_rate"].reindex(cell_ids).to_numpy()

    sampled_dashboard_hours = sample_hours_by_how(test_df, rng)
    sampled_dashboard_hours_flat = [pd.Timestamp(hb) for how in range(168) for hb in sampled_dashboard_hours[how]]
    needed_hours = set(pd.to_datetime(train_hours))
    needed_hours.update(sampled_dashboard_hours_flat)
    if not needed_hours:
        print("No hours were selected for recency feature construction.")
        sys.exit(1)

    recency_by_hour = build_recency_maps(df, all_cells_set, cell_to_idx, n_cells, needed_hours)
    print(f"Built recency vectors for {len(recency_by_hour):,} needed timestamps.")

    model = train_model(
        train_df=train_df,
        train_hours=train_hours,
        pos_cells_by_hour_train=pos_cells_by_hour_train,
        cell_ids=cell_ids,
        all_cells_set=all_cells_set,
        cell_to_idx=cell_to_idx,
        cell_feat=cell_feat,
        cell_how_rate=cell_how_rate,
        how_hours=how_hours,
        recency_by_hour=recency_by_hour,
        rng=rng,
    )

    cells_info = all_cells.copy()
    cells_info["cell_lat_center"] = (cells_info["cell_lat"] + 0.5) * GRID_DEG
    cells_info["cell_lng_center"] = (cells_info["cell_lng"] + 0.5) * GRID_DEG

    risk_rows = []
    hourly_rows = []
    topk_rows = []

    for how in range(168):
        sampled_hours = [pd.Timestamp(hb) for hb in sampled_dashboard_hours[how]]
        if not sampled_hours:
            continue

        score_sum = np.zeros(n_cells, dtype=float)
        cell_how_sum = np.zeros(n_cells, dtype=float)
        rec6_sum = np.zeros(n_cells, dtype=float)
        rec24_sum = np.zeros(n_cells, dtype=float)

        for hb in sampled_hours:
            x_full, how_vec = build_full_x(
                hb=hb,
                cell_ids=cell_ids,
                cell_feat=cell_feat,
                cell_rate_vec=cell_rate_vec,
                cell_how_rate=cell_how_rate,
                how_hours=how_hours,
                recency_by_hour=recency_by_hour,
            )
            proba = model.predict_proba(x_full)[:, 1]
            score_sum += proba
            cell_how_sum += how_vec
            rec_1h_all, rec_6h_all, rec_24h_all = recency_by_hour[hb]
            rec6_sum += rec_6h_all.astype(float)
            rec24_sum += rec_24h_all.astype(float)

        sample_count = float(len(sampled_hours))
        risk_score = score_sum / sample_count
        cell_how_avg = cell_how_sum / sample_count
        rec6_avg = rec6_sum / sample_count
        rec24_avg = rec24_sum / sample_count

        order = np.argsort(-risk_score)
        pred_rank = np.empty_like(order)
        pred_rank[order] = np.arange(1, n_cells + 1)

        day_name = DAY_NAMES_ABBR[how // 24]
        hour = how % 24

        df_how = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "cell_lat": cells_info.loc[cell_ids, "cell_lat_center"].to_numpy(),
                "cell_lng": cells_info.loc[cell_ids, "cell_lng_center"].to_numpy(),
                "hour_of_week": how,
                "day_name": day_name,
                "hour": hour,
                "risk_score": risk_score,
                "cell_rate": cell_rate_vec,
                "cell_how_rate": cell_how_avg,
                "neighbor_rate": cell_feat["neighbor_rate"].reindex(cell_ids).to_numpy(),
                "pred_rank": pred_rank,
                "is_top_10": (pred_rank <= 10).astype(int),
                "is_top_50": (pred_rank <= 50).astype(int),
                "is_top_100": (pred_rank <= 100).astype(int),
                "rec_6h": rec6_avg,
                "rec_24h": rec24_avg,
            }
        )
        risk_rows.append(df_how)

        hourly_rows.append(
            {
                "hour_of_week": how,
                "day_name": day_name,
                "hour": hour,
                "avg_risk_score": float(risk_score.mean()),
                "historical_accident_count": int(test_df[test_df["hour_of_week"] == how].shape[0]),
            }
        )

        for k, label in [(10, "top10"), (50, "top50"), (100, "top100")]:
            top_idx = order[: min(k, len(order))]
            snap = df_how.iloc[top_idx][
                ["hour_of_week", "day_name", "hour", "cell_id", "cell_lat", "cell_lng", "risk_score", "pred_rank"]
            ].copy()
            snap.insert(3, "k_label", label)
            topk_rows.append(snap)

    if not risk_rows or not hourly_rows or not topk_rows:
        print("Failed to generate dashboard outputs; no hour-of-week rows were produced.")
        sys.exit(1)

    dashboard_cell_risk = pd.concat(risk_rows, ignore_index=True)
    dashboard_hourly_summary = pd.DataFrame(hourly_rows).sort_values("hour_of_week").reset_index(drop=True)
    dashboard_topk_snapshot = pd.concat(topk_rows, ignore_index=True)

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
