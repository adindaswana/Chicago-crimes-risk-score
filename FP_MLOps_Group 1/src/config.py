"""
config.py - Konfigurasi terpusat Checkpoint 2.

Satu sumber kebenaran untuk konstanta yang dipakai lintas modul: notebook, skrip
training, dan REST API. Mengubah nilai di sini otomatis konsisten di semua pemakai,
sehingga tidak ada konfigurasi ganda yang bisa diam-diam saling menyimpang
(sumber training-serving skew paling umum).
"""

from __future__ import annotations

import bisect
from pathlib import Path

from features import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN

# =============================================================================
# REPRODUCIBILITY
# Satu konstanta terpusat untuk seluruh random_state di pipeline.
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# PATH
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "features_labels.csv"
LABEL_PARAMS_PATH = ROOT_DIR / "label_params.json"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODEL_BUNDLE_PATH = MODELS_DIR / "model_bundle_v1.joblib"
REGISTRY_PATH = MODELS_DIR / "registry.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
OUTPUT_JSON_PATH = OUTPUTS_DIR / "FP_MLOps_Kelompok 1_Output.json"

for _d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VERSI MODEL
# Registry CP3 tinggal menambah entri baru di file yang sama - tidak ada
# perubahan kontrak API maupun struktur bundle yang diperlukan.
# =============================================================================

MODEL_VERSION = "v1"

# =============================================================================
# WILAYAH MODEL
# Satu-satunya tempat yang menyebut "Chicago" secara harfiah di lapisan CP2.
# Pesan error, /meta, dan dokumentasi seluruhnya membaca dari sini, bukan
# menulis ulang stringnya - migrasi wilayah kelak (mis. Jabodetabek, bila
# dataset Indonesia tersedia) berarti mengganti nilai di sini, plus bounding
# box di features.py, plus artefak model, tanpa menyentuh kontrak API.
# Lihat docs/REGION_SCOPE.md untuk konteks lengkap keputusan ini.
# =============================================================================

REGION_NAME = "Chicago, IL, USA"
REGION_TIMEZONE = "America/Chicago"
REGION_CENTER_LAT = round((LAT_MIN + LAT_MAX) / 2, 4)
REGION_CENTER_LON = round((LON_MIN + LON_MAX) / 2, 4)

# =============================================================================
# PROTOKOL EVALUASI TIGA-LAPIS
# =============================================================================

TEST_SIZE = 0.2                        # protokol A (acak) dan B (spasial, by cell_id)
INNER_VAL_SIZE = 0.2                   # validasi internal dari train, untuk pilih m baseline
NIGHT_HOURS = (21, 22, 23, 0, 1, 2, 3)  # protokol C: subset baris jam 21.00-03.00 dari test B

# =============================================================================
# BASELINE - kandidat konstanta shrinkage untuk tangga baseline tingkat 4
# =============================================================================

SHRINKAGE_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# =============================================================================
# ABLASI TARGET ENCODING
# m dipakai HANYA pada varian "dengan target encoding" (ditolak sebagai fitur
# champion, lihat KRITIK REFERENSI). Nilai sama dengan m_cell HO2 agar
# perbandingan apple-to-apple.
# =============================================================================

TARGET_ENCODING_SMOOTHING_M = 10.0

# =============================================================================
# KRITERIA PEMILIHAN CHAMPION (multi-kriteria, bukan hanya MAE)
# CP3 memerlukan retrain berulang dan API memerlukan waktu muat cepat, sehingga
# waktu training dan ukuran artefak ikut dipertimbangkan, bukan hanya akurasi.
# =============================================================================

MAX_ACCEPTABLE_TRAIN_SECONDS = 60.0
MAE_TIE_MARGIN = 0.5   # selisih MAE di bawah ini dianggap setara -> tie-break ke efisiensi

# =============================================================================
# AMBANG LEVEL RISIKO
# Didefinisikan SATU KALI di server dan diekspos lewat GET /meta. Front-End
# tidak boleh menduplikasi ambang ini di sisi klien.
# Batas kelas dipilih dekat kuartil label (Q1~25, median~42, Q3~59) dibulatkan
# ke angka genap supaya mudah dijelaskan ke pengguna non-teknis.
# =============================================================================

RISK_LEVEL_BOUNDS = [25.0, 50.0, 75.0]
RISK_LEVEL_LABELS = ["Low", "Medium", "High", "Very High"]


def risk_level(score: float) -> str:
    """Petakan skor 0-100 ke label level risiko.

    Batas kelas eksklusif di ujung bawah (mis. skor 25 -> "Medium", bukan "Low").

    >>> risk_level(74)
    'High'
    >>> risk_level(0)
    'Low'
    >>> risk_level(100)
    'Very High'
    """
    s = min(max(float(score), 0.0), 100.0)
    idx = bisect.bisect_right(RISK_LEVEL_BOUNDS, s)
    return RISK_LEVEL_LABELS[idx]


def risk_level_table() -> list[dict]:
    """Representasi tabel ambang level untuk diekspos lewat /meta dan dokumen kontrak."""
    bounds = [0.0] + RISK_LEVEL_BOUNDS + [100.0]
    return [
        {"level": label, "min": bounds[i], "max": bounds[i + 1]}
        for i, label in enumerate(RISK_LEVEL_LABELS)
    ]


# =============================================================================
# DISCLAIMER (dipakai identik oleh /meta, /risk-score, dan /risk-score/batch)
# =============================================================================

DISCLAIMER = (
    "Skor ini adalah estimasi statistik berbasis pola kejahatan historis Kota Chicago, "
    "bukan prediksi kepastian terjadinya kejahatan di masa depan. Skor hanya berlaku "
    "untuk koordinat di dalam bounding box yang dikembalikan lewat GET /meta; data "
    "historis dengan detail koordinat dan waktu yang setara belum tersedia untuk "
    "wilayah lain. Model tidak memprediksi kejahatan, melainkan mengaproksimasi fungsi "
    "label historis secara cepat dan dapat digeneralisasi ke lokasi/waktu dengan data "
    "historis terbatas. Skor tidak boleh dipakai sebagai satu-satunya dasar keputusan "
    "keamanan."
)


__all__ = [
    "RANDOM_STATE",
    "ROOT_DIR", "DATA_PATH", "LABEL_PARAMS_PATH",
    "MODELS_DIR", "OUTPUTS_DIR", "FIGURES_DIR",
    "MODEL_BUNDLE_PATH", "REGISTRY_PATH", "METRICS_PATH", "OUTPUT_JSON_PATH",
    "MODEL_VERSION",
    "REGION_NAME", "REGION_TIMEZONE", "REGION_CENTER_LAT", "REGION_CENTER_LON",
    "TEST_SIZE", "INNER_VAL_SIZE", "NIGHT_HOURS",
    "SHRINKAGE_GRID", "TARGET_ENCODING_SMOOTHING_M",
    "MAX_ACCEPTABLE_TRAIN_SECONDS", "MAE_TIE_MARGIN",
    "RISK_LEVEL_BOUNDS", "RISK_LEVEL_LABELS", "risk_level", "risk_level_table",
    "DISCLAIMER",
]
