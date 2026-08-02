"""
batch.py - Jalur fitur vektorized untuk endpoint batch.

`features.build_features()` mengembalikan dict skalar dan melakukan lookup
MultiIndex per panggilan lewat `grid_lookup.loc[key]` - benar untuk satu titik,
tapi lambat kalau dipanggil per baris pada endpoint batch (safe-route menilai
puluhan segmen, heatmap menilai ratusan sel).

Modul ini TIDAK mengubah `features.py`. Ia memakai ulang fungsi murninya
(`latlon_to_grid`, `build_time_features`) dan mengganti HANYA jalur lookup
agregat: satu operasi `merge` menggantikan N pemanggilan `.loc[key]`.
Kebenarannya dibuktikan lewat uji parity skalar-vs-batch (lihat tests/).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

AGGREGATE_COLS = ["crime_count", "violent_share", "arrest_rate",
                   "cell_total_count", "neighbor_count_mean"]


def _assert_all_in_region(lat_arr: np.ndarray, lon_arr: np.ndarray) -> None:
    """Lapis pertahanan kedua terhadap koordinat di luar wilayah model.

    `api/main.py` sudah memvalidasi bbox per titik sebelum memanggil jalur ini,
    jadi dalam operasi normal guard ini seharusnya tidak pernah menyala. Ia ada
    supaya jaminan "tidak ada skor untuk koordinat di luar wilayah model" tetap
    berlaku walau ada pemanggil baru di masa depan yang memanggil `predict_batch`
    langsung tanpa lewat route handler.
    """
    mask = F.in_chicago_bbox(lat_arr, lon_arr)
    if not np.all(mask):
        bad_idx = np.flatnonzero(~mask).tolist()
        raise ValueError(
            f"Titik pada indeks {bad_idx} di luar bounding box wilayah model "
            f"(lat {F.LAT_MIN} s/d {F.LAT_MAX}, lon {F.LON_MIN} s/d {F.LON_MAX})."
        )


def build_features_batch(lat, lon, dt, grid_lookup: pd.DataFrame) -> pd.DataFrame:
    """Rakit feature vector untuk N titik sekaligus lewat satu operasi merge.

    Parameter identik secara semantik dengan `features.build_features`, hanya
    menerima array/Series alih-alih skalar. Hasilnya, kolom demi kolom, harus
    identik dengan memanggil `build_features` satu per satu.
    """
    lat_arr = np.asarray(lat, dtype="float64")
    lon_arr = np.asarray(lon, dtype="float64")
    _assert_all_in_region(lat_arr, lon_arr)
    dt_series = pd.to_datetime(pd.Series(dt).reset_index(drop=True))

    lat_r, lon_r, cell_id = F.latlon_to_grid(lat_arr, lon_arr)
    time_feats = F.build_time_features(dt_series)

    request_df = pd.DataFrame({
        "lat_r": lat_r, "lon_r": lon_r, "cell_id": cell_id,
        "hour": time_feats["hour"], "dow": time_feats["dow"],
        "is_weekend": time_feats["is_weekend"],
        "hour_sin": time_feats["hour_sin"], "hour_cos": time_feats["hour_cos"],
        "dow_sin": time_feats["dow_sin"], "dow_cos": time_feats["dow_cos"],
    })

    lookup_flat = grid_lookup.reset_index()[["cell_id", "dow", "hour"] + AGGREGATE_COLS]
    merged = request_df.merge(lookup_flat, on=["cell_id", "dow", "hour"], how="left")
    merged[AGGREGATE_COLS] = merged[AGGREGATE_COLS].fillna(0.0)
    merged["data_coverage"] = np.where(
        merged["cell_total_count"] > 0, "historical_data", "no_historical_data"
    )
    return merged


def predict_batch(model, grid_lookup: pd.DataFrame, lat, lon, dt) -> pd.DataFrame:
    """Rakit fitur (vektorized) lalu prediksi sekaligus untuk N titik. Skor
    diclip ke [0, 100] sebelum dikembalikan - regresi tidak mengenal batas skala."""
    feat_df = build_features_batch(lat, lon, dt, grid_lookup)
    X = F.assemble_feature_matrix(feat_df)
    raw = model.predict(X)
    feat_df["risk_score_raw"] = raw
    feat_df["risk_score"] = np.clip(raw, 0.0, 100.0)
    return feat_df


__all__ = ["AGGREGATE_COLS", "build_features_batch", "predict_batch"]
