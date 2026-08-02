"""Pastikan root proyek ada di sys.path supaya `import features` dan `import src.*`
selalu berhasil, tidak peduli dari direktori mana pytest dijalankan."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import pytest

import features as F
from src.config import DATA_PATH, MODEL_BUNDLE_PATH
from src.registry import load_bundle

# Anchor tiap `dow` (0=Senin .. 6=Minggu) ke tanggal riil dalam pekan
# `reference_date` (2026-04-11, Sabtu), supaya baris CSV (yang hanya menyimpan
# dow/hour) bisa direkonstruksi menjadi datetime nyata untuk uji parity.
_DOW_TO_DATE = {
    0: "2026-04-13", 1: "2026-04-14", 2: "2026-04-15", 3: "2026-04-16",
    4: "2026-04-17", 5: "2026-04-11", 6: "2026-04-12",
}


def dow_hour_to_datetime(dow: int, hour: int) -> pd.Timestamp:
    return pd.Timestamp(f"{_DOW_TO_DATE[int(dow)]} {int(hour):02d}:00:00")


@pytest.fixture(scope="session")
def to_datetime():
    """Fixture pembungkus `dow_hour_to_datetime`, diakses lewat DI pytest agar
    berkas uji tidak perlu mengimpor `tests.conftest` secara manual."""
    return dow_hour_to_datetime


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def grid_lookup(dataset) -> pd.DataFrame:
    return F.make_grid_lookup(dataset)


@pytest.fixture(scope="session")
def bundle() -> dict:
    return load_bundle(MODEL_BUNDLE_PATH)
