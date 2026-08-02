"""
main.py - REST API FastAPI, Risk Score serving.

Bundle model dan `grid_lookup` dimuat SEKALI saat startup lewat lifespan
handler (bukan per request). Seluruh logika fitur diimpor dari `features.py`
(root, tidak diubah) dan `api/batch.py` (wrapper vektorized untuk endpoint
batch) - tidak ada logika duplikat antara jalur skalar dan batch.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import features as F
from api.batch import predict_batch
from api.schemas import (
    BatchRiskScoreRequest, BatchRiskScoreResponse, BoundingBox, HealthResponse,
    MetaResponse, RegionCenter, RiskLevelEntry, RiskScoreResponse,
)
from src.config import (
    DATA_PATH, DISCLAIMER, MODEL_BUNDLE_PATH, REGION_CENTER_LAT, REGION_CENTER_LON,
    REGION_NAME, REGION_TIMEZONE, risk_level, risk_level_table,
)
from src.registry import load_bundle

APP_STATE: dict = {}
REGION_TZ = ZoneInfo(REGION_TIMEZONE)


def _parse_datetime(value: str) -> pd.Timestamp:
    """Parse string datetime ISO 8601 lalu resolusikan ke waktu dinding zona model.

    String tanpa offset diperlakukan sebagai waktu dinding `REGION_TIMEZONE` secara
    langsung. String berzona (mis. offset Asia/Jakarta yang dikirim Front-End) dikonversi
    ke `REGION_TIMEZONE` lewat `zoneinfo` sebelum jam/hari diturunkan - tanpa ini, jam
    dari zona pengirim akan salah dibaca sebagai jam Chicago apa adanya. Waktu yang jatuh
    pada celah/ambiguitas transisi DST Chicago ditolak (-> 422) alih-alih ditebak.
    Melempar ValueError (-> 422) bila parsing atau resolusi zona waktu gagal.
    """
    try:
        ts = pd.to_datetime(value)
    except Exception as exc:
        raise ValueError(
            f"Format datetime tidak valid: '{value}'. Gunakan format ISO 8601, "
            f"misalnya 2026-04-11T23:00:00."
        ) from exc
    if pd.isna(ts):
        raise ValueError(f"Format datetime tidak valid: '{value}'.")
    try:
        ts_local = ts.tz_localize(REGION_TZ) if ts.tzinfo is None else ts.tz_convert(REGION_TZ)
    except ValueError as exc:
        raise ValueError(
            f"Waktu '{value}' tidak valid pada zona waktu {REGION_TIMEZONE}: jatuh pada "
            f"celah atau jam ambigu akibat transisi waktu musim panas/dingin (DST). "
            f"Gunakan jam lain atau sertakan offset UTC eksplisit."
        ) from exc
    return ts_local


def _validate_bbox(lat: float, lon: float) -> None:
    """Melempar ValueError (-> 422) bila koordinat di luar bounding box wilayah model."""
    if not bool(F.in_chicago_bbox(lat, lon)):
        raise ValueError(
            f"Koordinat (lat={lat}, lon={lon}) di luar bounding box wilayah model yang "
            f"didukung saat ini ({REGION_NAME}: lat {F.LAT_MIN} s/d {F.LAT_MAX}, "
            f"lon {F.LON_MIN} s/d {F.LON_MAX})."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = load_bundle(MODEL_BUNDLE_PATH)
    df = pd.read_csv(DATA_PATH)
    APP_STATE["bundle"] = bundle
    APP_STATE["grid_lookup"] = F.make_grid_lookup(df)
    yield
    APP_STATE.clear()


app = FastAPI(
    title="Women Safety Risk Score API",
    description=(
        "Prediksi Risk Score (0-100) untuk suatu lokasi dan waktu, berbasis pola "
        "historis Chicago Crimes. Dikonsumsi Front-End untuk Safe Route "
        "Recommendation & Risk Prediction by Time/Location."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


def _to_response(lat: float, lon: float, dt: pd.Timestamp, row: pd.Series) -> RiskScoreResponse:
    bundle = APP_STATE["bundle"]
    return RiskScoreResponse(
        lat=lat, lon=lon,
        datetime=dt.tz_localize(None).isoformat(),
        resolved_local_datetime=dt.isoformat(),
        cell_id=str(row["cell_id"]),
        risk_score=round(float(row["risk_score"]), 2),
        level=risk_level(float(row["risk_score"])),
        data_coverage=str(row["data_coverage"]),
        model_version=bundle["model_version"],
        last_updated=bundle["label_params"]["reference_date"][:10],
        disclaimer=DISCLAIMER,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check. Front-End memakai ini untuk menentukan fallback UI."""
    loaded = "bundle" in APP_STATE
    return HealthResponse(
        status="ok" if loaded else "not_ready",
        model_loaded=loaded,
        model_version=APP_STATE["bundle"]["model_version"] if loaded else None,
    )


@app.get("/meta", response_model=MetaResponse)
def meta():
    """Metadata model dan ambang level. Front-End mengambilnya sekali saat
    startup dan tidak perlu hardcode ambang risiko di sisi klien."""
    bundle = APP_STATE["bundle"]
    return MetaResponse(
        model_version=bundle["model_version"],
        last_updated=bundle["label_params"]["reference_date"][:10],
        algorithm=bundle["algorithm"],
        risk_levels=[RiskLevelEntry(**entry) for entry in risk_level_table()],
        bounding_box=BoundingBox(
            lat_min=F.LAT_MIN, lat_max=F.LAT_MAX, lon_min=F.LON_MIN, lon_max=F.LON_MAX,
        ),
        region_name=REGION_NAME,
        region_center=RegionCenter(lat=REGION_CENTER_LAT, lon=REGION_CENTER_LON),
        timezone=REGION_TIMEZONE,
        response_fields=list(RiskScoreResponse.model_fields.keys()),
        disclaimer=DISCLAIMER,
    )


@app.get("/risk-score", response_model=RiskScoreResponse)
def get_risk_score(
    lat: float = Query(..., description="Lintang titik yang dinilai"),
    lon: float = Query(..., description="Bujur titik yang dinilai"),
    datetime: str = Query(..., description="Waktu, format ISO 8601 (mis. 2026-04-11T23:00:00)"),
):
    """Skor risiko untuk satu titik (lat, lon, datetime)."""
    try:
        _validate_bbox(lat, lon)
        ts = _parse_datetime(datetime)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    bundle = APP_STATE["bundle"]
    result = predict_batch(bundle["estimator"], APP_STATE["grid_lookup"], [lat], [lon], [ts])
    return _to_response(lat, lon, ts, result.iloc[0])


@app.post("/risk-score/batch", response_model=BatchRiskScoreResponse)
def post_risk_score_batch(payload: BatchRiskScoreRequest):
    """Skor risiko untuk banyak titik dalam satu request, jalur fitur
    vektorized (satu merge, bukan N lookup) - lihat api/batch.py."""
    errors: list[str] = []
    parsed: list[tuple[float, float, pd.Timestamp]] = []
    for i, point in enumerate(payload.points):
        try:
            _validate_bbox(point.lat, point.lon)
            ts = _parse_datetime(point.datetime)
            parsed.append((point.lat, point.lon, ts))
        except ValueError as exc:
            errors.append(f"points[{i}]: {exc}")
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    bundle = APP_STATE["bundle"]
    lats = [p[0] for p in parsed]
    lons = [p[1] for p in parsed]
    dts = [p[2] for p in parsed]
    result = predict_batch(bundle["estimator"], APP_STATE["grid_lookup"], lats, lons, dts)

    results = [
        _to_response(lats[i], lons[i], dts[i], result.iloc[i])
        for i in range(len(parsed))
    ]
    return BatchRiskScoreResponse(results=results)


__all__ = ["app"]
