"""
evaluation.py - Metrik dan evaluasi bersegmen.

Empat metrik dilaporkan bersama, tidak satu pun berdiri sendiri:
    MAE      - metrik utama, satuannya sama dengan skala skor 0-100.
    RMSE     - sensitivitas terhadap error besar.
    R2       - proporsi varians, dibaca hati-hati sesuai protokol (lihat README).
    Spearman - kualitas ranking. Untuk safe-route, urutan relatif antar lokasi
               lebih menentukan pengalaman pengguna daripada nilai skor absolut.

Adaptasi evaluasi bersegmen dari HO2 (MAE per kuintil, per kelompok jam, porsi
error besar, sel terburuk) - bagian yang diadopsi sesuai KRITIK REFERENSI.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HOUR_GROUP_BINS = [-1, 5, 11, 17, 23]
HOUR_GROUP_LABELS = ["00-05", "06-11", "12-17", "18-23"]
LARGE_ERROR_THRESHOLDS = [5, 10, 20]


def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2, Spearman untuk satu pasang (y_true, y_pred).

    Spearman bernilai NaN bila prediksi konstan (mis. baseline global mean pada
    protokol B, saat seluruh sel test tak dikenal) - itu memang tidak
    terdefinisi secara matematis, bukan bug.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman = spearmanr(y_pred, y_true).statistic
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearman),
    }


def evaluate_row(y_true, y_pred, protocol: str, predictor: str, predictor_type: str,
                  extra: dict | None = None) -> dict:
    """Satu baris tabel perbandingan: protokol, prediktor, dan keempat metrik."""
    row = {"protocol": protocol, "predictor": predictor, "predictor_type": predictor_type}
    row.update(compute_metrics(y_true, y_pred))
    if extra:
        row.update(extra)
    return row


def add_improvement_column(results_df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom persentase perbaikan MAE model terhadap baseline TERKUAT
    pada protokol yang sama (baseline dengan MAE terendah)."""
    df = results_df.copy()
    strongest = (df[df["predictor_type"] == "baseline"]
                 .groupby("protocol")["MAE"].min()
                 .rename("strongest_baseline_mae"))
    df = df.merge(strongest, on="protocol", how="left")
    df["improvement_vs_strongest_baseline_pct"] = (
        (1 - df["MAE"] / df["strongest_baseline_mae"]) * 100
    )
    return df.drop(columns=["strongest_baseline_mae"])


def segmented_evaluation(test_df: pd.DataFrame, y_true: np.ndarray,
                          pred_model: np.ndarray, pred_baseline: np.ndarray) -> dict:
    """Evaluasi bersegmen: MAE per kuintil label, MAE per kelompok jam, porsi
    error besar, dan sel dengan error terburuk. Dievaluasi pada model VS
    baseline terkuat, supaya nilai tambah model terlihat jujur di tiap segmen."""
    seg = pd.DataFrame({
        "y": y_true,
        "err_model": np.abs(y_true - pred_model),
        "err_baseline": np.abs(y_true - pred_baseline),
        "hour": test_df["hour"].to_numpy(),
        "cell_id": test_df["cell_id"].to_numpy(),
    })

    seg["quintile"] = pd.qcut(seg["y"], 5, labels=["Q1 (aman)", "Q2", "Q3", "Q4", "Q5 (rawan)"])
    per_quintile = seg.groupby("quintile", observed=True)[["err_model", "err_baseline"]].mean()
    per_quintile["improvement_pct"] = (1 - per_quintile["err_model"] / per_quintile["err_baseline"]) * 100

    seg["hour_group"] = pd.cut(seg["hour"], HOUR_GROUP_BINS, labels=HOUR_GROUP_LABELS)
    per_hour_group = seg.groupby("hour_group", observed=True)[["err_model", "err_baseline"]].mean()

    large_error_fractions = {
        t: {
            "model_pct": float((seg["err_model"] > t).mean() * 100),
            "baseline_pct": float((seg["err_baseline"] > t).mean() * 100),
        }
        for t in LARGE_ERROR_THRESHOLDS
    }

    worst_cells = (seg.groupby("cell_id")["err_model"].agg(["mean", "count"])
                   .query("count >= 10").nlargest(5, "mean"))

    return {
        "per_quintile": per_quintile,
        "per_hour_group": per_hour_group,
        "large_error_fractions": large_error_fractions,
        "worst_cells": worst_cells,
        "spearman_model": float(spearmanr(pred_model, y_true).statistic),
        "spearman_baseline": float(spearmanr(pred_baseline, y_true).statistic),
    }


__all__ = [
    "compute_metrics", "evaluate_row", "add_improvement_column", "segmented_evaluation",
]
