"""
Uji ekuivalensi representasi zona waktu (WAJIB, bagian 8.4 dokumen revisi).

Model membaca datetime sebagai waktu dinding `REGION_TIMEZONE` (America/Chicago).
Front-End berada di Jabodetabek (Asia/Jakarta, UTC+7) - kalau datetime yang
dikirim membawa offset selain Chicago, server WAJIB mengonversinya lewat
`zoneinfo`, bukan membaca komponen jam apa adanya. Salah tafsir di sini
berakibat fitur waktu yang salah (mis. malam terbaca siang), yang untuk
aplikasi keselamatan adalah bug serius, bukan kosmetik.
"""

from zoneinfo import ZoneInfo

import pytest

from api.main import _parse_datetime
from src.config import REGION_TIMEZONE

CHICAGO = ZoneInfo(REGION_TIMEZONE)


def test_naive_datetime_is_localized_to_region_timezone():
    ts = _parse_datetime("2026-04-11T23:00:00")
    assert ts.tzinfo is not None
    assert ts.tzinfo.key == REGION_TIMEZONE
    assert (ts.hour, ts.day) == (23, 11)


def test_jakarta_offset_is_converted_to_chicago_wall_clock():
    # 06:00 Jakarta (+07:00) == 18:00 Chicago hari sebelumnya (CDT, UTC-5).
    ts = _parse_datetime("2026-04-12T06:00:00+07:00")
    assert ts.tzinfo.key == REGION_TIMEZONE
    assert (ts.hour, ts.day) == (18, 11)


def test_utc_and_chicago_offset_strings_resolve_to_identical_instant():
    ts_utc = _parse_datetime("2026-04-12T04:00:00Z")
    ts_chicago = _parse_datetime("2026-04-11T23:00:00-05:00")
    assert ts_utc == ts_chicago
    assert (ts_utc.hour, ts_utc.day) == (ts_chicago.hour, ts_chicago.day)


def test_dst_spring_forward_gap_is_rejected():
    # 2026-03-08 02:00-02:59 tidak pernah ada di jam dinding Chicago (loncat ke 03:00).
    with pytest.raises(ValueError, match="tidak valid"):
        _parse_datetime("2026-03-08T02:30:00")


def test_dst_fall_back_ambiguous_hour_is_rejected():
    # 2026-11-01 01:00-01:59 terjadi dua kali (CDT lalu CST) - ambigu tanpa offset eksplisit.
    with pytest.raises(ValueError, match="tidak valid"):
        _parse_datetime("2026-11-01T01:30:00")


def test_dst_ambiguous_hour_is_resolved_when_offset_given_explicitly():
    # Sertakan offset eksplisit -> tidak ambigu lagi, wajib berhasil diparse.
    ts = _parse_datetime("2026-11-01T01:30:00-05:00")
    assert ts.tzinfo.key == REGION_TIMEZONE
