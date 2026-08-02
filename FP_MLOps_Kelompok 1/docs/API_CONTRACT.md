# Kontrak API - Risk Score Serving

Dokumen ini adalah kontrak resmi untuk Risk Score API yang menyediakan fitur
rekomendasi rute aman dan prediksi risiko berdasarkan waktu dan lokasi.
Skema di sini dianggap sumber kebenaran; skema aktual dapat diverifikasi
langsung lewat Swagger UI di `GET /docs` saat API berjalan.

Base URL (lokal): `http://127.0.0.1:8000`

Cara menjalankan API: lihat `README.md` bagian "Cara menjalankan API".

Model saat ini hanya valid untuk koordinat di wilayah Chicago, Illinois,
Amerika Serikat. Alasan teknis, cara penanganan koordinat di luar wilayah
tersebut, dan rencana perluasan cakupan dijelaskan di `docs/REGION_SCOPE.md`.
Aplikasi klien tidak perlu menuliskan asumsi wilayah apa pun sebagai nilai
tetap di kodenya - seluruh informasi wilayah (nama, titik tengah, bounding
box, zona waktu) tersedia lewat `GET /meta`.

---

## Ringkasan endpoint

| Method | Path                  | Fungsi                                    |
| :----- | :-------------------- | :----------------------------------------- |
| GET    | `/health`              | Liveness check                             |
| GET    | `/meta`                | Metadata model, ambang level, bounding box |
| GET    | `/risk-score`          | Skor risiko satu titik                     |
| POST   | `/risk-score/batch`    | Skor risiko banyak titik sekaligus         |
| GET    | `/docs`                | Swagger UI (dokumentasi hidup)             |

Seluruh endpoint mengembalikan JSON. CORS aktif untuk seluruh origin agar
pengembangan lokal tidak terhambat preflight.

---

## GET /health

Dipakai untuk menentukan status kesiapan layanan (mis. menampilkan status
"skor tidak tersedia" bila API belum siap).

**Response 200**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1"
}
```

`status` bernilai `"ok"` bila bundle model berhasil dimuat, `"not_ready"` bila
belum (seharusnya tidak pernah terjadi setelah startup selesai, karena bundle
dimuat sinkron sebelum API menerima request).

---

## GET /meta

Diambil sekali oleh aplikasi klien saat dimuat. Ambang level risiko,
bounding box, nama wilayah, dan zona waktu dapat berubah antar versi model,
sehingga harus selalu diambil dari sini, bukan ditulis sebagai nilai tetap
di kode klien.

**Response 200**

```json
{
  "model_version": "v1",
  "last_updated": "2026-04-11",
  "algorithm": "Model: XGBoost",
  "risk_levels": [
    { "level": "Low", "min": 0.0, "max": 25.0 },
    { "level": "Medium", "min": 25.0, "max": 50.0 },
    { "level": "High", "min": 50.0, "max": 75.0 },
    { "level": "Very High", "min": 75.0, "max": 100.0 }
  ],
  "bounding_box": {
    "lat_min": 41.60,
    "lat_max": 42.05,
    "lon_min": -87.95,
    "lon_max": -87.50
  },
  "region_name": "Chicago, IL, USA",
  "region_center": { "lat": 41.825, "lon": -87.725 },
  "timezone": "America/Chicago",
  "response_fields": [
    "lat", "lon", "datetime", "resolved_local_datetime", "cell_id", "risk_score",
    "level", "data_coverage", "model_version", "last_updated", "disclaimer"
  ],
  "disclaimer": "Skor ini adalah estimasi statistik berbasis pola kejahatan historis Kota Chicago, ..."
}
```

`region_name` dan `region_center` dapat dipakai untuk memberi label dan
memusatkan peta pada layar prediksi risiko tanpa perlu menuliskan koordinat
tetap di kode klien. `timezone` adalah zona waktu IANA yang dipakai server
untuk menurunkan fitur waktu (lihat `resolved_local_datetime` pada
`GET /risk-score` di bawah).

### Tabel ambang level risiko

| Level     | Rentang skor    |
| :-------- | :--------------- |
| Low       | 0 <= skor < 25    |
| Medium    | 25 <= skor < 50   |
| High      | 50 <= skor < 75   |
| Very High | 75 <= skor <= 100 |

Batas kelas eksklusif di ujung bawah (skor tepat 25 masuk `Medium`, bukan
`Low`). Nilai kanonik selalu diambil dari `GET /meta`, tabel ini hanya
referensi cepat.

---

## GET /risk-score

Skor risiko untuk satu titik.

### Query parameter

| Parameter  | Tipe   | Wajib | Keterangan                                   |
| :--------- | :----- | :---- | :-------------------------------------------- |
| `lat`      | float  | ya    | Lintang, harus di dalam `bounding_box` dari `GET /meta` |
| `lon`      | float  | ya    | Bujur, harus di dalam `bounding_box` dari `GET /meta`   |
| `datetime` | string | ya    | ISO 8601, dengan atau tanpa offset zona waktu, mis. `2026-04-11T23:00:00` atau `2026-04-12T06:00:00+07:00` |

Server menafsirkan `datetime` sebagai waktu dinding pada zona `timezone`
dari `GET /meta` (`America/Chicago`). String tanpa offset diperlakukan
langsung sebagai waktu pada zona tersebut; string berzona lain dikonversi
secara otomatis. Hasil konversi selalu terlihat pada field
`resolved_local_datetime` di response - klien yang mengirim waktu lokal
selain zona model wajib membaca field ini, bukan mengasumsikan field
`datetime` pada response sama dengan waktu yang dipakai model.

### Response 200

```json
{
  "lat": 41.8827,
  "lon": -87.6233,
  "datetime": "2026-04-11T23:00:00",
  "resolved_local_datetime": "2026-04-11T23:00:00-05:00",
  "cell_id": "41.88_-87.62",
  "risk_score": 100.0,
  "level": "Very High",
  "data_coverage": "historical_data",
  "model_version": "v1",
  "last_updated": "2026-04-11",
  "disclaimer": "Skor ini adalah estimasi statistik berbasis pola kejahatan historis Kota Chicago, ..."
}
```

Field `risk_score` sudah di-clip ke rentang [0, 100]. Field `data_coverage`
bernilai `"historical_data"` bila sel grid tempat titik ini jatuh memiliki
catatan kejahatan historis, atau `"no_historical_data"` bila jatuh ke fallback
nol. Ini adalah indikator penting: aplikasi klien wajib menampilkan
tampilan berbeda untuk `"no_historical_data"`, karena skor rendah pada
kondisi ini berarti "tidak ada data", bukan "aman". Field
`resolved_local_datetime` adalah waktu (ISO 8601 dengan offset UTC
eksplisit) yang benar-benar dipakai untuk menghitung fitur - lihat
penjelasan zona waktu di atas.

### Response 422 (kesalahan input)

```json
{
  "detail": "Koordinat (lat=-6.2088, lon=106.8456) di luar bounding box wilayah model yang didukung saat ini (Chicago, IL, USA: lat 41.6 s/d 42.05, lon -87.95 s/d -87.5)."
}
```

Terjadi bila koordinat di luar bounding box wilayah model, `datetime` tidak
bisa diparse, atau `datetime` jatuh pada celah/jam ambigu akibat transisi
musim (DST) pada zona waktu model. Pesan `detail` layak ditampilkan
langsung ke pengguna akhir.

### Contoh curl

```bash
curl "http://127.0.0.1:8000/risk-score?lat=41.8827&lon=-87.6233&datetime=2026-04-11T23:00:00"
```

### Contoh fetch (JavaScript)

```javascript
async function getRiskScore(lat, lon, datetime) {
  const params = new URLSearchParams({ lat, lon, datetime });
  const res = await fetch(`http://127.0.0.1:8000/risk-score?${params}`);
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail);
  }
  return res.json();
}

getRiskScore(41.8827, -87.6233, "2026-04-11T23:00:00")
  .then((data) => console.log(data.risk_score, data.level))
  .catch((err) => console.error(err.message));
```

---

## POST /risk-score/batch

Skor risiko untuk banyak titik dalam satu request. Dirancang untuk kasus
penggunaan yang memerlukan banyak titik sekaligus dalam waktu singkat, mis.
menilai puluhan segmen rute atau ratusan sel pada peta panas. Jalur fiturnya
dioptimalkan di sisi server (satu operasi merge, bukan pencarian per titik),
sehingga request besar tetap cepat.

### Request body

```json
{
  "points": [
    { "lat": 41.8827, "lon": -87.6233, "datetime": "2026-04-11T23:00:00" },
    { "lat": 41.7500, "lon": -87.6800, "datetime": "2026-04-14T09:00:00" }
  ]
}
```

Maksimum 5000 titik per request.

### Response 200

```json
{
  "results": [
    {
      "lat": 41.8827,
      "lon": -87.6233,
      "datetime": "2026-04-11T23:00:00",
      "resolved_local_datetime": "2026-04-11T23:00:00-05:00",
      "cell_id": "41.88_-87.62",
      "risk_score": 100.0,
      "level": "Very High",
      "data_coverage": "historical_data",
      "model_version": "v1",
      "last_updated": "2026-04-11",
      "disclaimer": "Skor ini adalah estimasi statistik berbasis pola kejahatan historis Kota Chicago, ..."
    },
    {
      "lat": 41.75,
      "lon": -87.68,
      "datetime": "2026-04-14T09:00:00",
      "resolved_local_datetime": "2026-04-14T09:00:00-05:00",
      "cell_id": "41.75_-87.68",
      "risk_score": 47.09,
      "level": "Medium",
      "data_coverage": "historical_data",
      "model_version": "v1",
      "last_updated": "2026-04-11",
      "disclaimer": "Skor ini adalah estimasi statistik berbasis pola kejahatan historis Kota Chicago, ..."
    }
  ]
}
```

Urutan `results` mengikuti urutan `points` pada request, sehingga aplikasi
klien bisa memetakan hasil kembali ke indeks aslinya.

### Response 422

Bila satu atau lebih titik tidak valid (di luar bounding box atau format
`datetime` salah), seluruh request ditolak dengan daftar kesalahan per
indeks - tidak ada hasil parsial:

```json
{
  "detail": [
    "points[1]: Koordinat (lat=90.0, lon=-87.68) di luar bounding box wilayah model yang didukung saat ini (Chicago, IL, USA: lat 41.6 s/d 42.05, lon -87.95 s/d -87.5)."
  ]
}
```

### Contoh curl

```bash
curl -X POST "http://127.0.0.1:8000/risk-score/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      { "lat": 41.8827, "lon": -87.6233, "datetime": "2026-04-11T23:00:00" },
      { "lat": 41.7500, "lon": -87.6800, "datetime": "2026-04-14T09:00:00" }
    ]
  }'
```

### Contoh fetch (JavaScript)

```javascript
async function getRiskScoreBatch(points) {
  const res = await fetch("http://127.0.0.1:8000/risk-score/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(JSON.stringify(err.detail));
  }
  return res.json();
}

getRiskScoreBatch([
  { lat: 41.8827, lon: -87.6233, datetime: "2026-04-11T23:00:00" },
  { lat: 41.75, lon: -87.68, datetime: "2026-04-14T09:00:00" },
]).then((data) => console.log(data.results));
```

---

## GET /docs

Swagger UI bawaan FastAPI - dokumentasi hidup yang selalu sinkron dengan kode
aktual, dibuka langsung di browser saat API berjalan lokal.

---

## Kesiapan untuk versi model mendatang

`model_version` dan `last_updated` sudah tersedia di setiap response. Saat
model baru dipromosikan (mis. hasil pelatihan ulang berkala atau dataset
wilayah baru), aplikasi klien tidak perlu perubahan kontrak apa pun - field
yang sama akan mencerminkan versi terbaru. `registry.json` di sisi server
menyimpan riwayat versi dan tinggal menambah entri baru dengan skema yang
identik untuk setiap versi model.

## Framing etis (wajib dibaca sebelum mengintegrasikan)

`risk_score` adalah aproksimasi cepat atas fungsi label historis, bukan
prediksi kepastian kejahatan. Aplikasi klien wajib menampilkan field
`disclaimer` kepada pengguna di titik keputusan (bukan hanya di halaman
"tentang"), dan wajib membedakan tampilan untuk
`data_coverage = "no_historical_data"` - menyamakannya dengan skor rendah
dapat memberi rasa aman yang keliru.
