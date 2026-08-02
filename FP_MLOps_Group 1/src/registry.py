"""
registry.py - Bundle model, registry, dan metrics.

Bundle `.joblib` mengunci seluruh kontrak training-serving dalam SATU objek:
estimator, urutan kolom fitur, parameter label beku, versi model, waktu latih,
ringkasan metrik, dan sidik jari dataset. Kalau kolom atau parameter di-fit
ulang diam-diam di sisi serving, representasi bergeser (training-serving skew) -
bundle mencegah itu karena serving tidak pernah men-training ulang apa pun,
hanya memuat objek ini.

`registry.json` memuat satu entri per versi model (CP2 dimulai dengan `v1`),
format JSON supaya bisa dibaca ulang tanpa menjalankan notebook. CP3 tinggal
menambah entri baru ke file yang sama - tidak ada perubahan skema.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS
from src.config import MODEL_BUNDLE_PATH, MODEL_VERSION, REGISTRY_PATH


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    """Sidik jari dataset training: jumlah baris + hash ringkas isinya."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
    return {"n_rows": int(len(df)), "hash": digest}


def build_bundle(model, algorithm: str, metrics_summary: dict, dataset_df: pd.DataFrame,
                  label_params: dict, model_version: str = MODEL_VERSION) -> dict:
    """Rakit bundle final. Ini SATU-SATUNYA objek yang dimuat REST API - tidak
    ada file model terpisah yang perlu disinkronkan manual."""
    return {
        "estimator": model,
        "algorithm": algorithm,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_params": label_params,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics_summary,
        "dataset_fingerprint": dataset_fingerprint(dataset_df),
    }


def save_bundle(bundle: dict, path: Path = MODEL_BUNDLE_PATH) -> Path:
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = MODEL_BUNDLE_PATH) -> dict:
    return joblib.load(path)


def build_registry_entry(bundle: dict, artifact_path: Path, status: str = "champion") -> dict:
    """Satu entri registry untuk versi model ini."""
    return {
        "version": bundle["model_version"],
        "trained_at": bundle["trained_at"],
        "algorithm": bundle["algorithm"],
        "evaluation_protocols": list(bundle["metrics"].keys()),
        "metrics": bundle["metrics"],
        "artifact_path": str(Path(artifact_path).relative_to(Path(artifact_path).parent.parent)),
        "dataset_fingerprint": bundle["dataset_fingerprint"],
        "status": status,
    }


def _json_default(obj):
    """Konversi tipe non-serializable (numpy, Timestamp) agar json.dump tidak gagal."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Tipe tidak didukung untuk serialisasi JSON: {type(obj)}")


def _sanitize_nan(obj):
    """Ganti NaN/Infinity float dengan None secara rekursif.

    `NaN` bukan token JSON yang sah (RFC 8259); Python `json` menulisnya apa
    adanya secara default sehingga konsumen non-Python (Front-End) bisa gagal
    parse. Spearman pada baseline dengan prediksi konstan sengaja NaN (lihat
    evaluation.compute_metrics), jadi harus disanitasi sebelum ditulis.
    """
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def write_json(data, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(data, default=_json_default))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_nan(serializable), f, indent=2, ensure_ascii=False)
    return path


def write_registry(entries: list, path: Path = REGISTRY_PATH) -> Path:
    return write_json(entries, path)


def write_metrics(metrics: dict, path: Path) -> Path:
    return write_json(metrics, path)


__all__ = [
    "dataset_fingerprint", "build_bundle", "save_bundle", "load_bundle",
    "build_registry_entry", "write_json", "write_registry", "write_metrics",
]
