"""
features.py - Fondasi Pseudo-Labeling & Feature Engineering Risk Score Chicago Crimes.

Modul ini adalah SATU-SATUNYA sumber kebenaran untuk logika grid-binning, fitur waktu,
severity, decay, dan normalisasi skor. Notebook CP1 mengimpor dari sini (tidak menyalin
ulang), dan REST API di CP2 nanti WAJIB mengimpor dari file yang sama.

Alasannya: kalau training dan serving memakai logika yang berbeda sedikit saja - misalnya
pembulatan grid yang beda satu desimal - API akan mengeluarkan skor yang ngawur padahal
modelnya benar. Semua fungsi di sini murni (tanpa efek samping, tanpa variabel global
notebook) supaya bisa dipanggil dari mana pun.

Rangkuman pipeline Risk Score:
    severity(kejahatan) -> x bobot waktu (decay) -> agregasi spasial (kernel Gaussian)
    -> normalisasi ke 0-100

Grup 01 - Ira Arianti Alawiah & Dinda (SISTECH 2026 MLOps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# KONFIGURASI
# Semua parameter desain dikumpulkan di sini supaya mudah diaudit dan diubah.
# Justifikasi tiap angka ada di notebook 01_pseudolabeling_feature_engineering.ipynb.
# =============================================================================

GRID_DECIMALS = 2        # pembulatan lat/lon -> sel ~1.1 km
HALF_LIFE_DAYS = 365.0   # half-life temporal decay (hari)
SIGMA_KM = 0.7           # bandwidth kernel Gaussian spasial (km)
CUTOFF_KM = 3.0          # radius potong (>4 sigma; bobot di luar < 0.01%)
CLIP_PERCENTILE = 0.99   # persentil pemotongan ekor sebelum normalisasi
SCORE_EXPONENT = 0.5     # eksponen kurva skor (0.5 = akar kuadrat)

EARTH_RADIUS_KM = 6371.0088

# Bounding box kasar wilayah Chicago (menyingkirkan koordinat rusak / 0,0).
LAT_MIN, LAT_MAX = 41.60, 42.05
LON_MIN, LON_MAX = -87.95, -87.50


# =============================================================================
# 1. GRID SPASIAL
# =============================================================================

def latlon_to_grid(lat, lon, decimals: int = GRID_DECIMALS):
    """Ubah koordinat kontinu menjadi ID sel grid.

    Membulatkan lat/lon ke `decimals` desimal sehingga titik-titik berdekatan jatuh ke
    sel yang sama. Pada 2 desimal, satu sel berukuran ~1.1 km.

    Menerima skalar maupun array/Series. Mengembalikan (lat_r, lon_r, cell_id).

    >>> latlon_to_grid(41.8010, -87.6526)
    (41.8, -87.65, '41.8_-87.65')
    """
    lat_r = np.round(np.asarray(lat, dtype="float64"), decimals)
    lon_r = np.round(np.asarray(lon, dtype="float64"), decimals)

    if lat_r.ndim == 0:  # skalar -> kembalikan tipe Python biasa
        lat_f, lon_f = float(lat_r), float(lon_r)
        return lat_f, lon_f, f"{lat_f}_{lon_f}"

    cell_id = pd.Series(lat_r).astype(str) + "_" + pd.Series(lon_r).astype(str)
    return lat_r, lon_r, cell_id.to_numpy()


def grid_to_index(lat_r, lon_r, decimals: int = GRID_DECIMALS):
    """Ubah koordinat sel yang sudah dibulatkan menjadi indeks grid integer (gx, gy).

    Indeks integer memudahkan pencarian tetangga (gx+1, gy-1, dst.) tanpa masalah
    presisi floating point.
    """
    step = 10.0 ** (-decimals)
    gx = np.round(np.asarray(lat_r, dtype="float64") / step).astype(int)
    gy = np.round(np.asarray(lon_r, dtype="float64") / step).astype(int)
    return gx, gy


def in_chicago_bbox(lat, lon) -> np.ndarray:
    """True bila koordinat berada di dalam bounding box Chicago."""
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)


# =============================================================================
# 2. FITUR WAKTU
# =============================================================================

def cyclical_encode(values, period: float):
    """Petakan nilai siklikal ke lingkaran satuan lewat (sin, cos).

    Jam 23:00 sebenarnya berdekatan dengan 00:00, tapi sebagai angka mentah selisihnya
    23 (dianggap paling jauh). Encoding ini memperbaikinya.
    """
    radians = 2.0 * np.pi * np.asarray(values, dtype="float64") / period
    return np.sin(radians), np.cos(radians)


def build_time_features(dt) -> dict:
    """Ubah datetime menjadi kumpulan fitur waktu.

    Menerima satu datetime (skalar) maupun Series/DatetimeIndex.
    Mengembalikan dict berisi: hour, dow, is_weekend, hour_sin/cos, dow_sin/cos.

    >>> f = build_time_features(pd.Timestamp("2026-04-11 23:30"))
    >>> f["hour"], f["dow"], f["is_weekend"]
    (23, 5, 1)
    """
    ts = pd.to_datetime(dt)
    scalar = not isinstance(ts, (pd.Series, pd.DatetimeIndex))

    if scalar:
        hour, dow = int(ts.hour), int(ts.dayofweek)
    else:
        accessor = ts.dt if isinstance(ts, pd.Series) else ts
        hour = accessor.hour
        dow = accessor.dayofweek
        hour = np.asarray(hour, dtype="int64")
        dow = np.asarray(dow, dtype="int64")

    hour_sin, hour_cos = cyclical_encode(hour, 24)
    dow_sin, dow_cos = cyclical_encode(dow, 7)
    is_weekend = (np.asarray(dow) >= 5).astype(int)

    out = {
        "hour": hour,
        "dow": dow,
        "is_weekend": int(is_weekend) if scalar else is_weekend,
        "hour_sin": float(hour_sin) if scalar else hour_sin,
        "hour_cos": float(hour_cos) if scalar else hour_cos,
        "dow_sin": float(dow_sin) if scalar else dow_sin,
        "dow_cos": float(dow_cos) if scalar else dow_cos,
    }
    return out


# =============================================================================
# 3. SEVERITY (keparahan per kejadian)
#
# Skala 1-100 berjangkar pada hierarki dampak terhadap manusia (semangat crime harm
# index): kejahatan terhadap nyawa & tubuh selalu di atas kejahatan properti; senjata
# dan kerentanan korban menaikkan keparahan.
#
#   Nyawa                          90-100
#   Kekerasan bersenjata            70-89
#   Kekerasan tanpa senjata         45-69
#   Properti + interaksi korban     30-44
#   Properti tanpa interaksi        10-29
#   Ketertiban umum                  1-9
#
# Diterapkan berjenjang 3 lapis supaya tidak ada satu nilai default yang meratakan
# mayoritas data.
# =============================================================================

# --- LAPIS 1: anchor eksplisit (kombinasi paling sering / paling penting) -----------
# Lima kombinasi bertanda (*) adalah contoh di dokumen soal; skala kita sengaja
# dijangkarkan supaya kompatibel dengan ilustrasi penilai.
SEVERITY_ANCHOR = {
    # nyawa (90-100)
    ("HOMICIDE", "FIRST DEGREE MURDER"): 100,
    ("HOMICIDE", "SECOND DEGREE MURDER"): 92,
    ("CRIMINAL SEXUAL ASSAULT", "AGGRAVATED: HANDGUN"): 95,   # (*) contoh soal
    ("CRIMINAL SEXUAL ASSAULT", "AGGRAVATED - HANDGUN"): 95,
    # kekerasan bersenjata / ancaman nyawa (70-89)
    ("CRIMINAL SEXUAL ASSAULT", "NON-AGGRAVATED"): 88,
    ("BATTERY", "AGGRAVATED: HANDGUN"): 85,
    ("BATTERY", "AGGRAVATED - HANDGUN"): 85,
    ("ROBBERY", "ARMED: HANDGUN"): 82,          # (*) contoh soal
    ("ROBBERY", "ARMED - HANDGUN"): 82,
    ("KIDNAPPING", "KIDNAPPING"): 80,
    ("ROBBERY", "AGGRAVATED"): 78,
    ("ASSAULT", "AGGRAVATED: HANDGUN"): 75,
    ("ASSAULT", "AGGRAVATED - HANDGUN"): 75,
    ("ROBBERY", "ARMED: KNIFE / CUTTING INSTRUMENT"): 74,
    ("WEAPONS VIOLATION", "UNLAWFUL USE HANDGUN"): 72,
    # kekerasan tanpa senjata & ancaman (45-69)
    ("ROBBERY", "STRONG ARM - NO WEAPON"): 62,
    ("ROBBERY", "STRONGARM - NO WEAPON"): 62,
    ("WEAPONS VIOLATION", "UNLAWFUL POSSESSION - HANDGUN"): 60,
    ("WEAPONS VIOLATION", "UNLAWFUL POSS OF HANDGUN"): 60,
    ("BATTERY", "DOMESTIC BATTERY SIMPLE"): 55,  # korban rentan & pola berulang
    ("BATTERY", "SIMPLE"): 50,                   # (*) contoh soal
    ("ASSAULT", "SIMPLE"): 45,
    # properti dengan interaksi korban (30-44)
    ("BURGLARY", "FORCIBLE ENTRY"): 42,
    ("THEFT", "FROM PERSON"): 38,                # (*) contoh soal
    ("BURGLARY", "UNLAWFUL ENTRY"): 36,
    ("MOTOR VEHICLE THEFT", "AUTOMOBILE"): 35,
    ("THEFT", "POCKET-PICKING"): 33,
    # properti tanpa interaksi (10-29)
    ("THEFT", "OVER $500"): 28,
    ("THEFT", "FROM BUILDING"): 22,
    ("THEFT", "$500 AND UNDER"): 20,             # (*) contoh soal
    ("THEFT", "RETAIL THEFT"): 18,
    ("CRIMINAL DAMAGE", "TO PROPERTY"): 15,
    ("DECEPTIVE PRACTICE", "FINANCIAL IDENTITY THEFT $300 AND UNDER"): 15,
    ("DECEPTIVE PRACTICE", "CREDIT CARD FRAUD"): 15,
    ("CRIMINAL DAMAGE", "TO VEHICLE"): 12,       # (*) contoh soal
    # ketertiban umum (1-9)
    ("CRIMINAL TRESPASS", "TO LAND"): 6,
    ("CRIMINAL TRESPASS", "TO BUILDING"): 6,
}

# --- LAPIS 2a: skor dasar per Primary Type ------------------------------------------
TYPE_BASE = {
    "HOMICIDE": 95,
    "HUMAN TRAFFICKING": 90,
    "CRIMINAL SEXUAL ASSAULT": 88,
    "KIDNAPPING": 78,
    "ARSON": 70,
    "ROBBERY": 65,
    "OFFENSE INVOLVING CHILDREN": 65,
    "SEX OFFENSE": 60,
    "CRIMINAL SEXUAL ABUSE": 60,
    "WEAPONS VIOLATION": 55,
    "STALKING": 55,
    "DOMESTIC VIOLENCE": 55,
    "BATTERY": 50,
    "INTIMIDATION": 50,
    "ASSAULT": 45,
    "RITUALISM": 40,
    "BURGLARY": 38,
    "MOTOR VEHICLE THEFT": 35,
    "CONCEALED CARRY LICENSE VIOLATION": 35,
    "NARCOTICS": 25,
    "OTHER NARCOTIC VIOLATION": 25,
    "THEFT": 22,
    "DECEPTIVE PRACTICE": 15,
    "OTHER OFFENSE": 15,
    "CRIMINAL DAMAGE": 14,
    "PROSTITUTION": 12,
    "INTERFERENCE WITH PUBLIC OFFICER": 12,
    "OBSCENITY": 10,
    "PUBLIC INDECENCY": 10,
    "CRIMINAL TRESPASS": 8,
    "PUBLIC PEACE VIOLATION": 8,
    "GAMBLING": 5,
    "LIQUOR LAW VIOLATION": 5,
    "NON-CRIMINAL": 2,
    "NON - CRIMINAL": 2,
    "NON-CRIMINAL (SUBJECT SPECIFIED)": 2,
}

# LAPIS 3: hanya untuk Primary Type yang benar-benar tak dikenal.
GLOBAL_DEFAULT_SEVERITY = 25

# Pada tipe ini senjata adalah ESENSI kejahatannya, bukan pemberat atas kejahatan lain.
# Tanpa pengecualian ini, skor kepemilikan senjata melompat ke pita kekerasan bersenjata.
WEAPON_IS_ESSENCE = {"WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION"}


def keyword_adjust(description: str, apply_weapon_modifier: bool = True) -> int:
    """Hitung modifier severity dari kualifikasi hukum di kolom Description.

    Memanfaatkan struktur data: Description berisi kualifikasi hukum yang berulang
    lintas tipe, dan kualifikasi yang sama bermakna sama di tipe mana pun.
    """
    d = str(description).upper()
    adj = 0

    if apply_weapon_modifier:
        # senjata: ambil satu modifier saja (yang terberat yang cocok)
        if ("HANDGUN" in d) or ("FIREARM" in d) or ("GUN" in d):
            adj += 18
        elif ("KNIFE" in d) or ("CUTTING" in d) or ("DANGEROUS WEAPON" in d) or ("ARMED" in d):
            adj += 10

    # kualifikasi hukum lain (boleh menumpuk)
    if "AGGRAVATED" in d:
        adj += 10
    if "DOMESTIC" in d:
        adj += 5
    if ("CHILD" in d) or ("MINOR" in d):
        adj += 8
    if "ATTEMPT" in d:
        adj -= 8

    return adj


def compute_severity(primary_type: str, description: str) -> float:
    """Severity satu kejadian (skalar, murni) - dipakai juga untuk uji akal sehat.

    >>> compute_severity("ROBBERY", "ARMED: HANDGUN")
    82.0
    >>> compute_severity("BATTERY", "SIMPLE")
    50.0
    """
    ptype = str(primary_type).upper().strip()
    desc = str(description).upper().strip()

    anchor = SEVERITY_ANCHOR.get((ptype, desc))
    if anchor is not None:
        return float(anchor)

    base = TYPE_BASE.get(ptype, GLOBAL_DEFAULT_SEVERITY)
    adj = keyword_adjust(desc, apply_weapon_modifier=ptype not in WEAPON_IS_ESSENCE)
    return float(np.clip(base + adj, 1, 100))


def compute_severity_series(primary_types, descriptions) -> np.ndarray:
    """Versi vektorized dari `compute_severity` untuk jutaan baris.

    Hasilnya identik dengan memanggil `compute_severity` baris per baris, tapi
    modifier hanya dihitung sekali per string Description unik.
    """
    ptype = pd.Series(primary_types).astype(str).str.upper().str.strip()
    desc = pd.Series(descriptions).astype(str).str.upper().str.strip()
    ptype.index = desc.index = pd.RangeIndex(len(ptype))

    # Lapis 2: base per tipe + modifier kata kunci
    base = ptype.map(TYPE_BASE).fillna(GLOBAL_DEFAULT_SEVERITY).astype("float64")

    uniq = desc.unique()
    adj_with = {d: keyword_adjust(d, True) for d in uniq}
    adj_without = {d: keyword_adjust(d, False) for d in uniq}
    adj = np.where(
        ptype.isin(WEAPON_IS_ESSENCE),
        desc.map(adj_without).to_numpy(),
        desc.map(adj_with).to_numpy(),
    )
    layer2 = np.clip(base.to_numpy() + adj, 1, 100)

    # Lapis 1: anchor menimpa lapis 2 bila kombinasinya terdaftar
    anchor = pd.Series(list(zip(ptype, desc))).map(SEVERITY_ANCHOR)
    return np.where(anchor.notna(), anchor.to_numpy(dtype="float64"), layer2)


def severity_layer(primary_types, descriptions) -> np.ndarray:
    """Tandai lapis mana yang dipakai tiap baris (untuk diagnostik cakupan)."""
    ptype = pd.Series(primary_types).astype(str).str.upper().str.strip()
    desc = pd.Series(descriptions).astype(str).str.upper().str.strip()
    ptype.index = desc.index = pd.RangeIndex(len(ptype))
    anchor = pd.Series(list(zip(ptype, desc))).map(SEVERITY_ANCHOR)
    return np.where(
        anchor.notna(), "1-anchor",
        np.where(ptype.isin(TYPE_BASE), "2-base+modifier", "3-default"),
    )


# =============================================================================
# 4. PEMBERSIHAN LABEL TAKSONOMI (diadopsi dari pendekatan Dinda)
# =============================================================================

def canonicalize_primary_type(df: pd.DataFrame,
                              iucr_col: str = "IUCR",
                              type_col: str = "Primary Type",
                              year_col: str = "Year") -> pd.Series:
    """Samakan Primary Type yang bercabang, memakai IUCR sebagai sumber kebenaran.

    Satu kode IUCR seharusnya memetakan ke satu Primary Type. Di data mentah ada
    beberapa kode yang labelnya berubah antar tahun (mis. 'CRIM SEXUAL ASSAULT' vs
    'CRIMINAL SEXUAL ASSAULT'). Aturannya generik dan deterministik: label kanonik =
    Primary Type yang paling sering dipakai pada TAHUN TERAKHIR kode itu muncul.

    Ini penting untuk severity: tanpa kanonikalisasi, satu jenis kejahatan terpecah
    menjadi dua entri tabel hanya karena beda ejaan.
    """
    if iucr_col not in df.columns:
        return df[type_col].astype(str).str.upper().str.strip()

    valid = df[[iucr_col, type_col, year_col]].dropna(subset=[iucr_col])
    last_year = valid.groupby(iucr_col)[year_col].transform("max")
    recent = valid[valid[year_col] == last_year]
    canon = recent.groupby(iucr_col)[type_col].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    mapped = df[iucr_col].map(canon).fillna(df[type_col])
    return mapped.astype(str).str.upper().str.strip()


def normalize_text(series: pd.Series) -> pd.Series:
    """Rapikan spasi/kapitalisasi kolom teks agar pencocokan string tidak gagal
    hanya karena selisih format."""
    return (series.astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"\s*/\s*", " / ", regex=True))


# Kategori kejahatan kekerasan mengacu FBI/UCR + penculikan & perdagangan manusia.
VIOLENT_TYPES = {
    "HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "CRIM SEXUAL ASSAULT",
    "ROBBERY", "BATTERY", "ASSAULT", "KIDNAPPING", "HUMAN TRAFFICKING",
}


# =============================================================================
# 5. TEMPORAL DECAY
# =============================================================================

def temporal_weight(age_days, half_life_days: float = HALF_LIFE_DAYS):
    """Bobot recency eksponensial: w = 0.5 ** (umur / half_life).

    Ekuivalen exp(-lambda*t) dengan lambda = ln2/half_life, tapi ditulis dalam bentuk
    half-life karena lebih mudah dijelaskan ("bobot separuh setiap H hari").

    Sengaja TIDAK memakai cutoff keras: kejadian lama meluruh mulus tanpa pernah
    menyentuh nol, sehingga kejadian 2 tahun lalu tetap berkontribusi (kecil), sesuai
    tabel akal sehat dokumen soal.

    >>> round(float(temporal_weight(365.0, 365.0)), 3)
    0.5
    """
    age = np.asarray(age_days, dtype="float64")
    return np.power(0.5, age / float(half_life_days))


# =============================================================================
# 6. AGREGASI SPASIAL
# =============================================================================

def build_spatial_weight_matrix(lat_r, lon_r,
                                sigma_km: float = SIGMA_KM,
                                cutoff_km: float = CUTOFF_KM,
                                normalize_rows: bool = True):
    """Matriks bobot antar-sel: kernel Gaussian pada jarak haversine.

        w_dist = exp(-d^2 / (2 * sigma^2))

    Memakai jarak bumi sesungguhnya, bukan selisih indeks grid, karena grid derajat
    bersifat anisotropik di Chicago (0.01 lintang ~ 1.11 km, tapi 0.01 bujur ~ 0.83 km).

    `normalize_rows=True` membuat tiap baris berjumlah 1, sehingga hasilnya WEIGHTED
    MEAN, bukan weighted sum. Ini disengaja: base_value tiap sel sudah memuat volume
    kejadian, jadi kernel-sum akan menghitung volume dua kali (pusat kawasan padat
    menggelembung, sel tepi kota/danau tertekan hanya karena tetangganya sedikit).

    Jarak antar-sel tidak bergantung waktu, jadi matriks ini cukup dihitung SEKALI
    lalu dipakai untuk seluruh 168 slot waktu via satu perkalian matriks.
    """
    from scipy import sparse
    from sklearn.neighbors import BallTree

    coords = np.radians(np.column_stack([
        np.asarray(lat_r, dtype="float64"),
        np.asarray(lon_r, dtype="float64"),
    ]))
    n = len(coords)

    tree = BallTree(coords, metric="haversine")
    ind, dist = tree.query_radius(coords, r=cutoff_km / EARTH_RADIUS_KM,
                                  return_distance=True)

    rows = np.repeat(np.arange(n), [len(x) for x in ind])
    cols = np.concatenate(ind)
    d_km = np.concatenate(dist) * EARTH_RADIUS_KM
    w = np.exp(-(d_km ** 2) / (2.0 * sigma_km ** 2))

    W = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    if normalize_rows:
        W = sparse.csr_matrix(W.multiply(1.0 / W.sum(axis=1)))
    return W


def spatial_weight(distance_km, sigma_km: float = SIGMA_KM):
    """Bobot kernel Gaussian untuk satu jarak (dipakai di uji akal sehat).

    >>> round(float(spatial_weight(1.0, 1.0)), 3)
    0.607
    """
    d = np.asarray(distance_km, dtype="float64")
    return np.exp(-(d ** 2) / (2.0 * sigma_km ** 2))


# =============================================================================
# 7. NORMALISASI KE SKALA 0-100
# =============================================================================

def fit_score_scaler(risk_raw, percentile: float = CLIP_PERCENTILE) -> float:
    """Tentukan nilai jangkar 'skor 100' = persentil ke-`percentile` dari risk_raw.

    Nilai ini HARUS dibekukan dan disimpan bersama model. Kalau tiap rilis data
    dinormalisasi ulang dengan jangkar sendiri, arti "skor 80" ikut bergeser dan
    perbandingan antar waktu jadi tidak sah.
    """
    return float(np.quantile(np.asarray(risk_raw, dtype="float64"), percentile))


def normalize_risk_score(risk_raw, clip_value: float,
                         exponent: float = SCORE_EXPONENT) -> np.ndarray:
    """Ubah risk_raw menjadi Risk Score 0-100.

        score = 100 * (min(risk_raw, clip_value) / clip_value) ** exponent

    Dua jangkar eksplisit:
        risk_raw = 0          -> skor 0    (tidak ada kejahatan sama sekali)
        risk_raw >= clip_value-> skor 100  (level persentil-99 ke atas)

    `exponent` mengatur seberapa terangkat bagian tengah distribusi:
        1.0  = linear  -> mayoritas skor nyungsep di bawah, daya beda hilang
        0.5  = akar    -> DIPAKAI, tengah mendarat di kisaran "sedang"
        ->0  = seperti log -> mayoritas skor terinflasi ke atas

    Pilihan ini menggantikan pipeline `log1p -> min-max` pada versi sebelumnya, yang
    membuat median skor ~70 sehingga area biasa-biasa saja terbaca "berisiko tinggi"
    oleh konsumen API.
    """
    raw = np.asarray(risk_raw, dtype="float64")
    if clip_value <= 0:
        return np.zeros_like(raw)
    ratio = np.clip(raw, 0.0, clip_value) / clip_value
    return 100.0 * np.power(ratio, exponent)


# =============================================================================
# 8. PERAKITAN FEATURE VECTOR (dipakai training DAN serving)
# =============================================================================

# Kolom fitur yang dikonsumsi model. Urutan dikunci: perubahan urutan kolom adalah
# salah satu sumber bug training-serving skew paling umum.
FEATURE_COLUMNS = [
    "lat_r", "lon_r",
    "hour", "dow", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "crime_count", "violent_share", "arrest_rate",
    "cell_total_count", "neighbor_count_mean",
]

# Kolom yang merupakan BAHAN pembentuk label - tidak boleh masuk fitur model.
# Dipisahkan eksplisit supaya tidak ada yang tak sengaja memakainya (sumber leakage).
LABEL_INGREDIENT_COLUMNS = ["base_value", "risk_raw"]

TARGET_COLUMN = "risk_score"


def build_features(lat, lon, dt, grid_lookup: pd.DataFrame | None = None) -> dict:
    """Rakit feature vector lengkap untuk satu permintaan (lat, lon, waktu).

    Inilah fungsi yang akan dipanggil REST API di CP2:
        1. koordinat -> sel grid
        2. waktu     -> fitur temporal
        3. sel+waktu -> fitur agregat, diambil dari tabel grid hasil training

    `grid_lookup` adalah `features_labels.csv` hasil notebook CP1, ter-index oleh
    (cell_id, dow, hour). Fitur agregat sengaja TIDAK dihitung ulang saat serving:
    ia merangkum histori bertahun-tahun, jadi harus datang dari tabel yang sama
    dengan yang dipakai saat training.

    Bila sel/slot tidak ada di tabel, fitur agregat diisi 0 - artinya memang tidak
    ada kejahatan tercatat di sana.
    """
    lat_r, lon_r, cell_id = latlon_to_grid(lat, lon)
    feat = {"lat_r": lat_r, "lon_r": lon_r, "cell_id": cell_id}
    feat.update(build_time_features(dt))

    aggregate_cols = ["crime_count", "violent_share", "arrest_rate",
                      "cell_total_count", "neighbor_count_mean"]

    if grid_lookup is None:
        for c in aggregate_cols:
            feat[c] = 0.0
        return feat

    key = (cell_id, feat["dow"], feat["hour"])
    try:
        row = grid_lookup.loc[key]
        for c in aggregate_cols:
            feat[c] = float(row[c])
    except KeyError:
        for c in aggregate_cols:
            feat[c] = 0.0
    return feat


def make_grid_lookup(features_labels: pd.DataFrame) -> pd.DataFrame:
    """Siapkan tabel `features_labels` untuk dipakai `build_features` saat serving."""
    return features_labels.set_index(["cell_id", "dow", "hour"]).sort_index()


def assemble_feature_matrix(frame: pd.DataFrame,
                            feature_columns=None) -> pd.DataFrame:
    """Susun matriks fitur dengan set & urutan kolom TERKUNCI.

    Dipakai identik saat training, evaluasi, maupun serving -> jaminan tidak ada
    pergeseran representasi antar tahap.
    """
    cols = list(feature_columns or FEATURE_COLUMNS)
    return frame.reindex(columns=cols).astype("float64").fillna(0.0)


__all__ = [
    "GRID_DECIMALS", "HALF_LIFE_DAYS", "SIGMA_KM", "CUTOFF_KM",
    "CLIP_PERCENTILE", "SCORE_EXPONENT",
    "latlon_to_grid", "grid_to_index", "in_chicago_bbox",
    "cyclical_encode", "build_time_features",
    "SEVERITY_ANCHOR", "TYPE_BASE", "VIOLENT_TYPES",
    "keyword_adjust", "compute_severity", "compute_severity_series", "severity_layer",
    "canonicalize_primary_type", "normalize_text",
    "temporal_weight", "build_spatial_weight_matrix", "spatial_weight",
    "fit_score_scaler", "normalize_risk_score",
    "FEATURE_COLUMNS", "LABEL_INGREDIENT_COLUMNS", "TARGET_COLUMN",
    "build_features", "make_grid_lookup", "assemble_feature_matrix",
]
