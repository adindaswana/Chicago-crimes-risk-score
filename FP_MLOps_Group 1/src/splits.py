"""
splits.py - Protokol evaluasi tiga-lapis dan pemuatan dataset.

Baseline dan model WAJIB dievaluasi pada split yang identik, sehingga fungsi split
di sini adalah satu-satunya sumber pembagian train/test yang dipakai baseline,
model, dan notebook.

Protokol:
    A. Acak            - train_test_split 80/20, seed terkunci. Mengukur interpolasi
                          di dalam grid yang sudah dikenal, BUKAN generalisasi.
    B. Holdout spasial  - GroupShuffleSplit by cell_id, ~20% sel ditahan penuh.
                          Angka headline / klaim generalisasi utama.
    C. Slot malam       - subset baris jam 21.00-03.00 dari test protokol B.
                          Relevansi produk (jam paling relevan bagi pengguna).

Holdout temporal tidak mungkin di CP2 karena satu baris `features_labels.csv` adalah
profil agregat lintas tahun, bukan observasi bertanggal (lihat README, bagian
batasan metodologis).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import DATA_PATH, NIGHT_HOURS, RANDOM_STATE, TEST_SIZE


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Muat `features_labels.csv` hasil CP1. Dataset tidak diregenerasi di CP2."""
    return pd.read_csv(path)


def split_random(df: pd.DataFrame, test_size: float = TEST_SIZE,
                  random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Protokol A - split acak baris, seed terkunci."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_spatial(df: pd.DataFrame, test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Protokol B - holdout spasial. Seluruh 168 baris tiap `cell_id` yang terpilih
    jatuh ke test, sehingga sel tersebut benar-benar tidak pernah dilihat model."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["cell_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def night_subset(test_df_spatial: pd.DataFrame, night_hours=NIGHT_HOURS) -> pd.DataFrame:
    """Protokol C - subset baris jam malam (21.00-03.00) dari test protokol B."""
    return test_df_spatial[test_df_spatial["hour"].isin(night_hours)].reset_index(drop=True)


def unseen_cell_fraction(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Persentase baris test yang `cell_id`-nya tidak pernah muncul di train.

    Dipakai untuk membuktikan kontras protokol A (mendekati 0%) vs protokol B
    (harus 100%) - inilah bukti bahwa protokol A tidak mengukur generalisasi.
    """
    known_cells = set(train_df["cell_id"].unique())
    unseen = ~test_df["cell_id"].isin(known_cells)
    return float(unseen.mean() * 100.0)


def build_all_protocols(df: pd.DataFrame) -> dict:
    """Rakit ketiga protokol sekaligus dan verifikasi kontras sel tak dikenal.

    Mengembalikan dict berisi train/test tiap protokol plus statistik verifikasi,
    supaya notebook maupun skrip training memakai pembagian yang identik.
    """
    train_a, test_a = split_random(df)
    train_b, test_b = split_spatial(df)
    test_c = night_subset(test_b)

    stats = {
        "A_unseen_cell_pct": unseen_cell_fraction(train_a, test_a),
        "B_unseen_cell_pct": unseen_cell_fraction(train_b, test_b),
        "A_train_rows": len(train_a), "A_test_rows": len(test_a),
        "B_train_rows": len(train_b), "B_test_rows": len(test_b),
        "C_test_rows": len(test_c),
    }
    return {
        "A": {"train": train_a, "test": test_a},
        "B": {"train": train_b, "test": test_b},
        "C": {"train": train_b, "test": test_c},
        "stats": stats,
    }


__all__ = [
    "load_dataset", "split_random", "split_spatial", "night_subset",
    "unseen_cell_fraction", "build_all_protocols",
]
