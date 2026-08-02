"""
Uji batas (WAJIB #3): koordinat di luar Chicago, datetime tidak valid, sel
tanpa data historis, nilai prediksi mentah di luar [0, 100].
"""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import features as F
from api.batch import build_features_batch, predict_batch
from api.main import _parse_datetime, _validate_bbox
from src.config import REGION_TIMEZONE

# Sebaran titik Jabodetabek (bukan hanya satu kota) - divisi PM/BAS/UI-UX/FE
# mendesain seluruh aplikasi untuk wilayah ini, sedangkan model hanya valid
# untuk Chicago. Titik-titik ini WAJIB selalu ditolak, karena kesalahan
# konfigurasi Front-End saat pengembangan/demo hampir pasti mengirim salah
# satu dari koordinat semacam ini.
JABODETABEK_COORDS = [
    (-6.2088, 106.8456),  # Jakarta
    (-6.5971, 106.8060),  # Bogor
    (-6.4025, 106.7942),  # Depok
    (-6.1783, 106.6319),  # Tangerang
    (-6.2383, 106.9756),  # Bekasi
]


class _DummyModel:
    """Model tiruan yang sengaja memprediksi di luar [0, 100], untuk
    membuktikan `predict_batch` selalu meng-clip hasil akhirnya."""

    def predict(self, X):
        n = len(X)
        return np.linspace(-50.0, 150.0, n)


def test_coordinates_outside_chicago_are_rejected():
    # Jakarta, jelas di luar bounding box Chicago.
    assert not bool(F.in_chicago_bbox(-6.2, 106.8))
    with pytest.raises(ValueError, match="di luar bounding box"):
        _validate_bbox(-6.2, 106.8)


def test_coordinates_inside_chicago_are_accepted():
    assert bool(F.in_chicago_bbox(41.88, -87.62))
    _validate_bbox(41.88, -87.62)  # tidak boleh melempar


@pytest.mark.parametrize("lat, lon", JABODETABEK_COORDS)
def test_jabodetabek_coordinates_are_rejected_at_route_layer(lat, lon):
    assert not bool(F.in_chicago_bbox(lat, lon))
    with pytest.raises(ValueError, match="di luar bounding box"):
        _validate_bbox(lat, lon)


@pytest.mark.parametrize("lat, lon", JABODETABEK_COORDS)
def test_jabodetabek_coordinates_are_rejected_even_bypassing_route_layer(lat, lon, grid_lookup):
    # Panggil predict_batch/build_features_batch LANGSUNG, melewati
    # _validate_bbox di api/main.py sepenuhnya - membuktikan lapis pertahanan
    # kedua di api/batch.py sendiri yang menolak, bukan kebetulan lolos ke
    # model dan menghasilkan skor yang tampak wajar tapi tidak bermakna.
    dt = [pd.Timestamp("2026-04-11 23:00:00")]
    with pytest.raises(ValueError, match="di luar bounding box"):
        build_features_batch([lat], [lon], dt, grid_lookup)


def test_invalid_datetime_string_raises_value_error():
    with pytest.raises(ValueError, match="Format datetime tidak valid"):
        _parse_datetime("bukan-tanggal")


def test_valid_datetime_string_is_parsed():
    ts = _parse_datetime("2026-04-11T23:00:00")
    assert ts == pd.Timestamp("2026-04-11 23:00:00", tz=ZoneInfo(REGION_TIMEZONE))


def test_cell_without_historical_data_falls_back_to_zero_and_flags_coverage(grid_lookup):
    # Titik di dalam bounding box tapi jauh dari grid manapun di dataset -> sel kosong.
    lat, lon = 41.61, -87.94
    dt = pd.Timestamp("2026-04-11 23:00:00")
    feat = F.build_features(lat, lon, dt, grid_lookup)
    assert feat["crime_count"] == 0.0
    assert feat["cell_total_count"] == 0.0

    batch_df = build_features_batch([lat], [lon], [dt], grid_lookup)
    assert batch_df.loc[0, "data_coverage"] == "no_historical_data"


def test_cell_with_historical_data_is_flagged_correctly(dataset, grid_lookup):
    row = dataset.iloc[0]
    dt = pd.Timestamp("2026-04-11 23:00:00")
    batch_df = build_features_batch([row["lat_r"]], [row["lon_r"]], [dt], grid_lookup)
    assert batch_df.loc[0, "data_coverage"] == "historical_data"


def test_raw_predictions_outside_0_100_are_clipped(grid_lookup):
    dummy = _DummyModel()
    lat = [41.88, 41.75, 41.70]
    lon = [-87.62, -87.68, -87.60]
    dt = [pd.Timestamp("2026-04-11 23:00:00")] * 3

    result = predict_batch(dummy, grid_lookup, lat, lon, dt)

    assert (result["risk_score_raw"] < 0).any() or (result["risk_score_raw"] > 100).any()
    assert (result["risk_score"] >= 0).all()
    assert (result["risk_score"] <= 100).all()
