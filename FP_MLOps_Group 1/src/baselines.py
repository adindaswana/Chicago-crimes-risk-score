"""
baselines.py - Tangga baseline non-ML, empat tingkat.

Tujuannya mendekomposisi sumber sinyal, bukan menyediakan satu pembanding lemah:

    1. Global mean                              - lantai absolut.
    2. Rata-rata per cell_id                     - pola spasial murni.
    3. Rata-rata per (dow, hour)                 - pola temporal murni.
    4. Rata-rata per (cell_id, hour) + shrinkage  - baseline TERKUAT, ini yang
       harus dikalahkan model. Konstanta shrinkage `m` dipilih lewat validasi
       INTERNAL dari train; test set tidak pernah disentuh untuk keputusan ini.

Seluruh statistik baseline dihitung hanya dari data train. Adaptasi dari
`fit_group_baseline` / `predict_group_baseline` (HO2_MLOps_IraAriantiAlawiah.ipynb),
bagian yang diadopsi sesuai KRITIK REFERENSI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from features import TARGET_COLUMN
from src.config import INNER_VAL_SIZE, RANDOM_STATE, SHRINKAGE_GRID


def fit_global_mean(train: pd.DataFrame) -> float:
    return float(train[TARGET_COLUMN].mean())


def predict_global_mean(global_mean: float, n: int) -> np.ndarray:
    return np.full(n, global_mean)


def fit_group_mean(train: pd.DataFrame, keys: list[str]) -> dict:
    """Tingkat 2/3 - rata-rata mentah per kelompok `keys`, tanpa shrinkage."""
    return {
        "keys": keys,
        "global_mean": fit_global_mean(train),
        "group_mean": train.groupby(keys)[TARGET_COLUMN].mean(),
    }


def predict_group_mean(stats: dict, data: pd.DataFrame) -> np.ndarray:
    """Prediksi tingkat 2/3, fallback ke global mean bila kombinasi tak pernah
    muncul di train."""
    keys = stats["keys"]
    idx = (pd.MultiIndex.from_frame(data[keys]) if len(keys) > 1
           else pd.Index(data[keys[0]]))
    pred = stats["group_mean"].reindex(idx).to_numpy()
    return np.where(np.isnan(pred), stats["global_mean"], pred)


def fit_shrinkage_baseline(train: pd.DataFrame, keys: list[str], m: float = 0.0) -> dict:
    """Tingkat 4 - rata-rata per `keys` ditarik ke rata-rata cell_id (shrinkage m).

    m=0  -> rata-rata kelompok mentah (fallback bertingkat "keras").
    m>0  -> pred = (jumlah + m * rata2_sel) / (n + m). Makin sedikit sampel
    kelompok, makin kuat ditarik ke rata-rata selnya sendiri; kelompok yang
    tak pernah muncul otomatis = rata-rata sel.
    """
    global_mean = fit_global_mean(train)
    cell_mean = train.groupby("cell_id")[TARGET_COLUMN].mean()
    agg = train.groupby(keys)[TARGET_COLUMN].agg(["sum", "count"])
    parent = cell_mean.reindex(agg.index.get_level_values("cell_id")).to_numpy()
    group_mean = (agg["sum"] + m * parent) / (agg["count"] + m)
    return {"keys": keys, "global_mean": global_mean, "cell_mean": cell_mean, "group_mean": group_mean}


def predict_shrinkage_baseline(stats: dict, data: pd.DataFrame) -> np.ndarray:
    """Prediksi tingkat 4 dengan rantai fallback: kombinasi -> rata-rata sel ->
    global mean. Pada protokol B, sel yang ditahan tidak punya rata-rata sel di
    train sehingga otomatis jatuh ke global mean - inilah bukti baseline
    berbasis sel melemah pada sel yang belum pernah dilihat."""
    idx = pd.MultiIndex.from_frame(data[stats["keys"]])
    pred = stats["group_mean"].reindex(idx).to_numpy()
    fallback_cell = data["cell_id"].map(stats["cell_mean"]).to_numpy()
    pred = np.where(np.isnan(pred), fallback_cell, pred)
    return np.where(np.isnan(pred), stats["global_mean"], pred)


def select_shrinkage_m(train: pd.DataFrame, keys: list[str],
                        m_grid=SHRINKAGE_GRID,
                        val_size: float = INNER_VAL_SIZE,
                        random_state: int = RANDOM_STATE) -> tuple[float, pd.DataFrame]:
    """Pilih konstanta shrinkage `m` lewat validasi INTERNAL dari train.

    Test set sama sekali tidak disentuh untuk keputusan ini. Mengembalikan
    (m_terbaik, tabel_mae_per_m) untuk didokumentasikan di notebook.
    """
    inner_train, inner_val = train_test_split(train, test_size=val_size, random_state=random_state)
    rows = []
    for m in m_grid:
        stats = fit_shrinkage_baseline(inner_train, keys, m=m)
        pred = predict_shrinkage_baseline(stats, inner_val)
        mae = mean_absolute_error(inner_val[TARGET_COLUMN], pred)
        rows.append({"m": m, "mae_val": mae})
    table = pd.DataFrame(rows)
    best_m = float(table.loc[table["mae_val"].idxmin(), "m"])
    return best_m, table


def fit_baseline_ladder(train: pd.DataFrame) -> dict:
    """Bangun keempat tingkat baseline sekaligus dari data train yang sama."""
    best_m, m_table = select_shrinkage_m(train, ["cell_id", "hour"])
    return {
        "global_mean": fit_global_mean(train),
        "cell": fit_group_mean(train, ["cell_id"]),
        "hour_dow": fit_group_mean(train, ["dow", "hour"]),
        "cell_hour_shrunk": fit_shrinkage_baseline(train, ["cell_id", "hour"], m=best_m),
        "best_m": best_m,
        "m_selection_table": m_table,
    }


BASELINE_LABELS = {
    "global": "Baseline: Global Mean",
    "cell": "Baseline: Rata-rata per Sel",
    "hour_dow": "Baseline: Rata-rata per Jam+Hari",
    "cell_hour_shrunk": "Baseline: Sel+Jam (shrinkage)",
}


def predict_baseline_ladder(ladder: dict, data: pd.DataFrame) -> dict:
    """Prediksi keempat tingkat pada `data` (test set protokol mana pun)."""
    return {
        BASELINE_LABELS["global"]: predict_global_mean(ladder["global_mean"], len(data)),
        BASELINE_LABELS["cell"]: predict_group_mean(ladder["cell"], data),
        BASELINE_LABELS["hour_dow"]: predict_group_mean(ladder["hour_dow"], data),
        BASELINE_LABELS["cell_hour_shrunk"]: predict_shrinkage_baseline(ladder["cell_hour_shrunk"], data),
    }


__all__ = [
    "fit_global_mean", "predict_global_mean",
    "fit_group_mean", "predict_group_mean",
    "fit_shrinkage_baseline", "predict_shrinkage_baseline", "select_shrinkage_m",
    "fit_baseline_ladder", "predict_baseline_ladder", "BASELINE_LABELS",
]
