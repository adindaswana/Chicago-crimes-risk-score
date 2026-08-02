# Risk Score API - Women Safety & Smart Protection Platform

Layanan yang memprediksi skor risiko (0-100) untuk suatu lokasi pada waktu
tertentu, berdasarkan pola kejahatan historis, dan menyajikannya sebagai
REST API untuk mendukung fitur rekomendasi rute aman dan prediksi risiko
berdasarkan waktu/lokasi.

Model saat ini dilatih menggunakan data kejahatan historis Kota Chicago,
Illinois, Amerika Serikat, dan hanya valid untuk koordinat di wilayah
tersebut. Alasan teknis, penanganan koordinat di luar wilayah ini, dan
rencana perluasan cakupan dijelaskan di `docs/REGION_SCOPE.md`.

---

## Struktur folder dan alasannya

```
FP_MLOps_Kelompok 1/
├── features.py                       # sumber kebenaran feature engineering
├── features_labels.csv               # dataset training (122.976 x 16)
├── label_params.json                             
│
├── src/                               #
│   ├── config.py                      #   RANDOM_STATE, path, ambang level, versi model, wilayah
│   ├── viz.py                         #   palet brand + rcParams, satu tempat
│   ├── splits.py                      #   protokol evaluasi A/B/C + pemuatan dataset
│   ├── baselines.py                   #   tangga baseline 4 tingkat
│   ├── modeling.py                    #   kandidat model, ablasi, permutation importance
│   ├── evaluation.py                  #   metrik & evaluasi bersegmen
│   └── registry.py                    #   bundle .joblib, registry.json, metrics.json
│
├── api/                                # REST API - mengimpor src/ dan features.py, TIDAK menduplikasi logika
│   ├── main.py                         #   FastAPI app, lifespan, endpoint
│   ├── schemas.py                      #   skema request/response Pydantic
│   └── batch.py                        #   jalur fitur vektorized (satu merge) untuk /risk-score/batch
│
├── notebooks/
│   └── 01_pseudolabeling_feature_engineering.ipynb   
│   └── 02_model_training_baseline_comparison.ipynb   # narasi lengkap pelatihan model, mengimpor src/ yang sama
│
├── train.py                            # skrip training end-to-end (non-interaktif, sama isi dengan notebook)
│
├── models/
│   ├── model_bundle_v1.joblib          # estimator + feature_columns + label_params + metrics + fingerprint
│   ├── registry.json                   # riwayat versi model
│   └── metrics.json                    # seluruh hasil evaluasi terstruktur
│
├── outputs/
│   ├── FP_MLOps_Kelompok 1_Output.json # contoh prediksi representatif
│   └── figures/                        # visualisasi tersimpan sebagai berkas PNG
│
├── tests/                              # pengujian wajib (lihat bagian Pengujian)
├── docs/
│   ├── API_CONTRACT.md                 # kontrak API
│   └── REGION_SCOPE.md                 # cakupan wilayah model dan rencana perluasannya
├── requirements.txt
└── README.md
```

**Alasan pemilihan struktur**: `src/` memisahkan logika dari narasi (notebook)
dan dari transport (API) - keduanya mengimpor modul yang identik, sehingga
tidak ada duplikasi dan tidak ada risiko notebook dan API diam-diam
berperilaku berbeda (bug training-serving skew paling umum). `features.py`
tetap di root sebagai sumber kebenaran feature engineering, hanya diimpor,
tidak pernah ditulis ulang. `train.py` di root adalah entry point
non-interaktif yang menjalankan pipeline yang sama dengan notebook - berguna
untuk melatih ulang model secara cepat tanpa membuka Jupyter.

---

## Cara menjalankan training

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py
```

Skrip ini memuat dataset, membangun protokol evaluasi A/B/C, melatih tangga
baseline dan kandidat model, menjalankan ablasi target encoding dan
permutation importance, memilih champion, melatih ulang champion pada
seluruh dataset, lalu menulis `models/model_bundle_v1.joblib`,
`models/registry.json`, `models/metrics.json`, dan
`outputs/FP_MLOps_Kelompok 1_Output.json`.

Untuk narasi lengkap beserta visualisasi, buka dan jalankan
`notebooks/02_model_training_baseline_comparison.ipynb` - notebook ini
mengimpor modul `src/` yang sama dengan `train.py`, hanya versi naratif dan
tervisualisasi.

---

## Cara menjalankan API

```bash
source .venv/bin/activate
python train.py                        # pastikan models/model_bundle_v1.joblib ada
uvicorn api.main:app --reload --port 8000
```

Buka `http://127.0.0.1:8000/docs` untuk Swagger UI. Kontrak lengkap (skema
request/response, ambang level, contoh `curl` dan `fetch`) ada di
`docs/API_CONTRACT.md`.

Endpoint: `GET /health`, `GET /meta`, `GET /risk-score`,
`POST /risk-score/batch`, `GET /docs`.

---

## Ringkasan hasil evaluasi

Model dievaluasi lewat tiga protokol: pembagian data acak (A), holdout
spasial berdasarkan sel grid (B), dan subset slot malam dari protokol B (C).
Angka headline diambil dari **protokol B** karena ini yang mengukur
generalisasi ke lokasi yang belum pernah dilihat model, bukan protokol A
yang lebih banyak mengukur interpolasi pada lokasi yang sudah dikenal.
Detail lengkap ada di `models/metrics.json` dan notebook bagian 6.

| Protokol | Prediktor | MAE | R2 | Spearman | Perbaikan vs baseline terkuat |
| :-- | :-- | --: | --: | --: | --: |
| A (acak) | Baseline: Sel+Jam (shrinkage) | 4.841 | 0.923 | 0.961 | - |
| A (acak) | **Champion: XGBoost** | 2.889 | 0.974 | 0.987 | +40.3% |
| B (spasial) | Baseline terkuat: Rata-rata per Jam+Hari | 17.864 | 0.090 | 0.282 | - |
| B (spasial) | **Champion: XGBoost** | **3.292** | **0.966** | **0.983** | **+81.6%** |
| C (slot malam, subset B) | Champion: XGBoost | 3.349 | 0.967 | 0.983 | +81.7% |

**Kandidat model** (protokol B, tanpa hyperparameter tuning):

| Model | MAE | Waktu training | Ukuran artefak |
| :-- | --: | --: | --: |
| Ridge Regression | 6.188 | ~0.01 detik | ~1 KB |
| Random Forest (200 pohon) | 3.360 | ~11 detik | ~751 MB |
| **XGBoost (champion)** | **3.292** | **~0.3 detik** | **~478 KB** |

**Champion: XGBoost.** MAE-nya setara secara statistik dengan Random Forest
(selisih < 0.5), tetapi waktu training ~40x lebih cepat dan artefak ~1600x
lebih kecil - relevan karena sistem ini perlu mendukung pelatihan ulang
berkala dan API memerlukan waktu muat cepat saat startup.

**Ablasi target encoding** (`cell_risk_mean`): pada protokol A, penambahan
fitur ini nyaris tidak mengubah MAE (2.889 -> 2.889). Pada protokol B, MAE
justru **memburuk drastis** (3.292 -> 6.499) karena fitur ini kolaps ke
global mean untuk seluruh sel yang tidak dikenal - indikasi kebocoran
target tersamar. Kesimpulan: `cell_risk_mean` **tidak dipakai** pada
champion final.

**Permutation importance** (protokol B): dua kontributor terbesar adalah
`neighbor_count_mean` dan `crime_count` (fitur agregat historis, bukan
turunan langsung label), tidak ada satu fitur pun yang mendominasi >90%
importance.

**Verifikasi protokol**: persentase baris test dengan sel tak dikenal -
protokol A: 0,00%, protokol B: 100,00%. Kontras ini membuktikan protokol A
mengukur interpolasi, bukan generalisasi.

---

## Batasan metodologis

1. **Holdout temporal tidak dapat dilakukan pada versi data saat ini.** Satu
   baris `features_labels.csv` adalah profil agregat lintas tahun untuk
   satu (sel, hari, jam), bukan observasi kejahatan bertanggal - tidak ada
   sumbu waktu yang bisa dipotong untuk cutoff temporal yang jujur.
   Regenerasi snapshot pada cutoff tanggal yang lebih baru merupakan
   pekerjaan pengembangan lanjutan.
2. **Framing label wajib jujur.** `risk_score` adalah fungsi deterministik
   dari data kejahatan historis. Performa tinggi tidak boleh diklaim sebagai
   "model memprediksi kejahatan" - model ini adalah aproksimasi cepat yang
   dapat digeneralisasi atas fungsi label yang mahal dihitung. Nilai
   tambahnya adalah kemampuan menilai sel dan slot tanpa data historis
   memadai, bukan mereplikasi perhitungan yang sudah ada.
3. **Konteks geografis terbatas pada Chicago.** Pola sosial, geografis, dan
   penegakan hukum berbeda antar wilayah. Setiap response API menyertakan
   `disclaimer` yang menyatakan ini secara eksplisit, dan koordinat di luar
   bounding box yang didukung selalu ditolak (kode status 422) pada lebih
   dari satu lapis independen. Detail lengkap ada di `docs/REGION_SCOPE.md`.
4. **Tanpa hyperparameter tuning** - seluruh parameter model adalah default
   library atau nilai wajar yang di-hardcode dengan justifikasi (lihat
   `src/modeling.py`).
5. **R2 protokol A secara struktural lebih tinggi** daripada protokol B,
   bukan karena modelnya berbeda, melainkan karena soal protokol A lebih
   mudah (sel sudah "dihafal" sebagian lewat fitur agregat). R2 protokol B
   adalah rujukan generalisasi yang jujur.

---

## Arsitektur yang mendukung pengembangan lanjutan

Beberapa keputusan desain disiapkan agar pengembangan lanjutan (retraining
berkala, model versi baru, cakupan wilayah baru) tidak memerlukan perubahan
pada kontrak API:

- Setiap response API menyertakan `model_version` dan `last_updated`, sehingga
  aplikasi klien tidak perlu perubahan kontrak apa pun saat model baru
  dipromosikan.
- Bundle model berversi (`model_bundle_v1.joblib`) dan `registry.json`
  berformat entri-per-versi (satu entri `v1` saat ini) - versi berikutnya
  tinggal menambah entri baru ke file yang sama.
- Ambang level risiko dan informasi wilayah (nama, titik tengah, bounding
  box, zona waktu) didefinisikan satu kali di server (`src/config.py`,
  `features.py`) dan diekspos lewat `/meta` - perubahan pada nilai-nilai ini
  tidak memerlukan perubahan di sisi aplikasi klien.

Fitur di luar cakupan versi ini: pembelajaran berkelanjutan (continual
learning) otomatis, deteksi drift, dan dashboard pemantauan operasional.

---

## Pengujian

```bash
source .venv/bin/activate
pytest tests/ -v
```

Lima kategori pengujian wajib, seluruhnya lulus (38 pengujian):

1. **Parity training-serving** (`test_parity_training_serving.py`) - fitur
   dari jalur training (baris CSV) dan jalur serving (`build_features` dari
   lat/lon/datetime mentah) identik dalam toleransi numerik.
2. **Parity skalar vs batch** (`test_parity_scalar_batch.py`) - wrapper
   vektorized `api/batch.py` menghasilkan fitur dan prediksi identik dengan
   jalur skalar per titik.
3. **Uji batas** (`test_boundary.py`) - koordinat di luar wilayah model
   (termasuk sebaran titik nyata di luar bounding box, ditolak pada dua
   lapis independen: route handler dan `api/batch.py` langsung), datetime
   tidak valid, sel tanpa data historis, prediksi mentah di luar [0, 100]
   (dibuktikan selalu ter-clip).
4. **Ekuivalensi zona waktu** (`test_timezone.py`) - datetime naive dan
   berzona dikonversi konsisten ke waktu dinding zona model; jam yang
   jatuh pada celah/ambiguitas transisi musim (DST) ditolak, bukan ditebak.
5. **Smoke API** (`test_api_smoke.py`) - seluruh endpoint merespons dengan
   status dan skema yang benar, termasuk kasus error 422.
