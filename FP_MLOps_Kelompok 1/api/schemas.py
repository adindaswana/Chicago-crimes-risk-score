"""
schemas.py - Skema request/response Pydantic untuk REST API.

Response `RiskScoreResponse` adalah superset dari contoh dokumen soal
(`risk_score`, `level`, `model_version`, `last_updated`), ditambah
`data_coverage` (isu etis - menandai lokasi tanpa data historis, bukan
kosmetik), `disclaimer` (framing etis pada output, bukan hanya slide), dan
`resolved_local_datetime` (hasil resolusi zona waktu, lihat api/main.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskScoreResponse(BaseModel):
    lat: float = Field(..., description="Lintang yang diminta (input, bukan hasil pembulatan grid)")
    lon: float = Field(..., description="Bujur yang diminta (input, bukan hasil pembulatan grid)")
    datetime: str = Field(..., description="Waktu yang diminta, format ISO 8601")
    resolved_local_datetime: str = Field(
        ..., description="Waktu setelah dikonversi ke waktu dinding zona model (lihat "
                          "'timezone' di /meta), ISO 8601 dengan offset UTC eksplisit. "
                          "Wajib dicek oleh Front-End bila mengirim datetime berzona, "
                          "karena inilah waktu yang benar-benar dipakai untuk menghitung fitur."
    )
    cell_id: str = Field(..., description="ID sel grid ~1.1 km tempat titik ini jatuh")
    risk_score: float = Field(..., ge=0, le=100, description="Skor risiko 0-100, sudah di-clip")
    level: str = Field(..., description="Label level risiko sesuai ambang di /meta")
    data_coverage: str = Field(
        ..., description="'historical_data' bila sel punya data kejahatan historis, "
                          "'no_historical_data' bila jatuh ke fallback nol"
    )
    model_version: str
    last_updated: str
    disclaimer: str


class BatchPointRequest(BaseModel):
    lat: float
    lon: float
    datetime: str = Field(..., description="Format ISO 8601, mis. 2026-04-11T23:00:00")


class BatchRiskScoreRequest(BaseModel):
    points: list[BatchPointRequest] = Field(..., min_length=1, max_length=5000)


class BatchRiskScoreResponse(BaseModel):
    results: list[RiskScoreResponse]


class RiskLevelEntry(BaseModel):
    level: str
    min: float
    max: float


class BoundingBox(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class RegionCenter(BaseModel):
    lat: float
    lon: float


class MetaResponse(BaseModel):
    model_version: str
    last_updated: str
    algorithm: str
    risk_levels: list[RiskLevelEntry]
    bounding_box: BoundingBox
    region_name: str = Field(..., description="Label wilayah yang didukung model saat ini")
    region_center: RegionCenter = Field(..., description="Titik tengah bounding_box, dipakai FE sebagai pusat peta default tanpa hardcode koordinat")
    timezone: str = Field(..., description="Zona waktu IANA yang dipakai model untuk menurunkan fitur waktu, mis. 'America/Chicago'")
    response_fields: list[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ErrorResponse(BaseModel):
    detail: str


__all__ = [
    "RiskScoreResponse", "BatchPointRequest", "BatchRiskScoreRequest",
    "BatchRiskScoreResponse", "RiskLevelEntry", "BoundingBox", "RegionCenter", "MetaResponse",
    "HealthResponse", "ErrorResponse",
]
