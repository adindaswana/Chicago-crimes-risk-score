"""
modeling.py - Kandidat model, ablasi target encoding, permutation importance,
dan pemilihan champion multi-kriteria.

Kandidat (tanpa hyperparameter tuning, sesuai batasan dokumen soal):
    Ridge Regression   - referensi linear. Alpha default (1.0) dipakai apa
                         adanya sebagai parameter wajar yang di-hardcode.
    Random Forest      - n_estimators=200, min_samples_leaf=2 (nilai wajar,
                         bukan hasil tuning, dipakai juga di HO2).
    XGBoost            - parameter default library.

Pemilihan champion TIDAK hanya berdasarkan MAE: waktu training dan ukuran
artefak `.joblib` ikut dipertimbangkan karena CP3 memerlukan retrain berulang
dan API memerlukan waktu muat cepat saat startup.
"""

from __future__ import annotations

import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from features import FEATURE_COLUMNS, TARGET_COLUMN, assemble_feature_matrix
from src.config import MAE_TIE_MARGIN, RANDOM_STATE, TARGET_ENCODING_SMOOTHING_M
from src.evaluation import compute_metrics, evaluate_row

CANDIDATE_MODELS = {
    "Model: Ridge Regression": lambda: Ridge(random_state=RANDOM_STATE),
    "Model: Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Model: XGBoost": lambda: XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
}


def make_model(name: str):
    """Konstruktor model tunggal. Satu-satunya tempat algoritma champion
    didefinisikan - training bundle final dan (nanti) retrain CP3 memakai
    fungsi ini juga, supaya pilihan model konsisten di seluruh pipeline."""
    return CANDIDATE_MODELS[name]()


def _artifact_kb(model) -> float:
    """Ukuran artefak `.joblib` model dalam KB, dihitung tanpa menulis ke disk."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell() / 1024.0


def train_candidates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Latih seluruh kandidat pada satu split (protokol B) dan kumpulkan
    metrik akurasi SEKALIGUS waktu training dan ukuran artefak."""
    X_train = assemble_feature_matrix(train_df)
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test = assemble_feature_matrix(test_df)
    y_test = test_df[TARGET_COLUMN].to_numpy()

    rows, fitted = [], {}
    for name, ctor in CANDIDATE_MODELS.items():
        model = ctor()
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred)
        rows.append({
            "model": name, **metrics,
            "train_seconds": round(train_seconds, 4),
            "artifact_kb": round(_artifact_kb(model), 1),
        })
        fitted[name] = model
    return pd.DataFrame(rows), fitted


def select_champion(comparison_df: pd.DataFrame, mae_tie_margin: float = MAE_TIE_MARGIN) -> tuple[str, str]:
    """Pilih champion dengan kriteria multi-dimensi.

    Bila MAE kandidat-kandidat berada dalam margin `mae_tie_margin` dari MAE
    terbaik, mereka dianggap setara secara akurasi, dan pemenang ditentukan
    oleh waktu training tercepat lalu artefak terkecil (operasional: CP3
    perlu retrain berulang, API perlu waktu muat cepat). Bila hanya satu
    kandidat berada dalam margin itu, akurasi menjadi penentu tunggal.
    """
    best_mae = comparison_df["MAE"].min()
    tied = comparison_df[comparison_df["MAE"] <= best_mae + mae_tie_margin].copy()

    if len(tied) == 1:
        champion = str(tied.iloc[0]["model"])
        reason = (
            f"{champion} memiliki MAE terendah ({tied.iloc[0]['MAE']:.3f}) dengan selisih "
            f"lebih dari {mae_tie_margin} terhadap kandidat lain, sehingga akurasi menjadi "
            f"kriteria penentu tunggal."
        )
        return champion, reason

    tied = tied.sort_values(["train_seconds", "artifact_kb"])
    winner = tied.iloc[0]
    champion = str(winner["model"])
    others = ", ".join(f"{r['model']} (MAE {r['MAE']:.3f})" for _, r in tied.iterrows())
    reason = (
        f"MAE {len(tied)} kandidat setara dalam margin {mae_tie_margin}: {others}. "
        f"Di antaranya, {champion} dipilih karena waktu training tercepat "
        f"({winner['train_seconds']:.2f} detik) dan artefak terkecil ({winner['artifact_kb']:.1f} KB) - "
        f"relevan karena CP3 memerlukan retrain berulang dan API memerlukan waktu muat cepat saat startup."
    )
    return champion, reason


def fit_target_encoding(train_df: pd.DataFrame, m: float = TARGET_ENCODING_SMOOTHING_M) -> tuple[pd.Series, float]:
    """Target encoding `cell_risk_mean` (rata-rata target per cell_id, smoothed
    ke global mean). Dihitung HANYA dari train - dipakai khusus untuk ablasi,
    TIDAK dipakai pada champion final (lihat KRITIK REFERENSI)."""
    global_mean = float(train_df[TARGET_COLUMN].mean())
    agg = train_df.groupby("cell_id")[TARGET_COLUMN].agg(["sum", "count"])
    cell_risk_mean = (agg["sum"] + m * global_mean) / (agg["count"] + m)
    return cell_risk_mean, global_mean


def _apply_target_encoding(df: pd.DataFrame, cell_risk_mean: pd.Series, global_mean: float) -> pd.DataFrame:
    out = df.copy()
    out["cell_risk_mean"] = out["cell_id"].map(cell_risk_mean).fillna(global_mean)
    return out


def run_target_encoding_ablation(protocols: dict, algo_name: str = "Model: XGBoost") -> pd.DataFrame:
    """Ablasi terkontrol: latih `algo_name` DENGAN dan TANPA `cell_risk_mean`
    pada protokol A (acak) dan B (spasial), lalu bandingkan.

    Hipotesis yang diuji: keunggulan target encoding lenyap (atau berbalik
    merugikan) di protokol B, karena sel test tidak pernah dilihat saat fit
    encoding sehingga fiturnya kolaps ke global mean untuk seluruh baris.
    """
    rows = []
    for protocol_name in ("A", "B"):
        train_df = protocols[protocol_name]["train"]
        test_df = protocols[protocol_name]["test"]
        y_test = test_df[TARGET_COLUMN].to_numpy()

        # --- tanpa target encoding ---
        model_plain = make_model(algo_name)
        model_plain.fit(assemble_feature_matrix(train_df), train_df[TARGET_COLUMN].to_numpy())
        pred_plain = model_plain.predict(assemble_feature_matrix(test_df))
        rows.append(evaluate_row(y_test, pred_plain, protocol_name,
                                  f"{algo_name} (tanpa target encoding)", "ablation"))

        # --- dengan target encoding ---
        cell_risk_mean, global_mean = fit_target_encoding(train_df)
        train_te = _apply_target_encoding(train_df, cell_risk_mean, global_mean)
        test_te = _apply_target_encoding(test_df, cell_risk_mean, global_mean)
        cols_te = FEATURE_COLUMNS + ["cell_risk_mean"]

        model_te = make_model(algo_name)
        model_te.fit(assemble_feature_matrix(train_te, cols_te), train_te[TARGET_COLUMN].to_numpy())
        pred_te = model_te.predict(assemble_feature_matrix(test_te, cols_te))
        rows.append(evaluate_row(y_test, pred_te, protocol_name,
                                  f"{algo_name} (dengan target encoding)", "ablation"))

        unseen_pct = float((~test_df["cell_id"].isin(train_df["cell_id"].unique())).mean() * 100)
        rows[-1]["unseen_cell_pct"] = unseen_pct
        rows[-2]["unseen_cell_pct"] = unseen_pct

    return pd.DataFrame(rows)


def compute_permutation_importance(model, X_test: pd.DataFrame, y_test: np.ndarray,
                                     n_repeats: int = 5, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance pada protokol B, untuk memverifikasi champion
    tidak bergantung pada satu fitur yang hampir sirkular dengan label."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


__all__ = [
    "CANDIDATE_MODELS", "make_model", "train_candidates", "select_champion",
    "fit_target_encoding", "run_target_encoding_ablation",
    "compute_permutation_importance",
]
