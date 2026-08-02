"""
Uji smoke API (WAJIB #4): seluruh endpoint merespons dengan status dan skema
yang benar. Memakai FastAPI TestClient sebagai context manager supaya
lifespan handler (muat bundle + grid_lookup) benar-benar berjalan.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "v1"


def test_meta(client):
    res = client.get("/meta")
    assert res.status_code == 200
    body = res.json()
    for key in ("model_version", "last_updated", "algorithm", "risk_levels",
                "bounding_box", "region_name", "region_center", "timezone",
                "response_fields", "disclaimer"):
        assert key in body
    assert len(body["risk_levels"]) == 4
    assert "lat" in body["region_center"] and "lon" in body["region_center"]


def test_risk_score_valid(client):
    res = client.get("/risk-score", params={
        "lat": 41.8827, "lon": -87.6233, "datetime": "2026-04-11T23:00:00",
    })
    assert res.status_code == 200
    body = res.json()
    for key in ("risk_score", "level", "model_version", "last_updated",
                "data_coverage", "disclaimer", "cell_id", "resolved_local_datetime"):
        assert key in body
    assert 0.0 <= body["risk_score"] <= 100.0
    assert body["level"] in ("Low", "Medium", "High", "Very High")


def test_risk_score_outside_bbox_returns_422(client):
    res = client.get("/risk-score", params={
        "lat": 0.0, "lon": 0.0, "datetime": "2026-04-11T23:00:00",
    })
    assert res.status_code == 422
    assert "detail" in res.json()


def test_risk_score_jakarta_coordinate_returns_422(client):
    # Regresi langsung terhadap Kendala 3: Front-End Jabodetabek yang salah
    # konfigurasi hampir pasti mengirim koordinat semacam ini saat demo/dev.
    res = client.get("/risk-score", params={
        "lat": -6.2088, "lon": 106.8456, "datetime": "2026-04-11T23:00:00",
    })
    assert res.status_code == 422


def test_risk_score_invalid_datetime_returns_422(client):
    res = client.get("/risk-score", params={
        "lat": 41.88, "lon": -87.62, "datetime": "not-a-date",
    })
    assert res.status_code == 422


def test_risk_score_missing_param_returns_422(client):
    res = client.get("/risk-score", params={"lat": 41.88, "lon": -87.62})
    assert res.status_code == 422


def test_risk_score_batch_valid(client):
    payload = {"points": [
        {"lat": 41.8827, "lon": -87.6233, "datetime": "2026-04-11T23:00:00"},
        {"lat": 41.7500, "lon": -87.6800, "datetime": "2026-04-14T09:00:00"},
    ]}
    res = client.post("/risk-score/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 2
    for item in body["results"]:
        assert 0.0 <= item["risk_score"] <= 100.0


def test_risk_score_batch_one_invalid_point_rejects_whole_request(client):
    payload = {"points": [
        {"lat": 41.8827, "lon": -87.6233, "datetime": "2026-04-11T23:00:00"},
        {"lat": 90.0, "lon": -87.68, "datetime": "2026-04-14T09:00:00"},
    ]}
    res = client.post("/risk-score/batch", json=payload)
    assert res.status_code == 422


def test_risk_score_batch_empty_points_returns_422(client):
    res = client.post("/risk-score/batch", json={"points": []})
    assert res.status_code == 422


def test_docs_available(client):
    res = client.get("/docs")
    assert res.status_code == 200
