# Chicago Crimes Risk Score - MLOps Pipeline (SISTECH 2026)

Proyek ini membangun sistem prediksi **Risk Score** (skor 0-100 tingkat risiko suatu lokasi pada waktu tertentu) dari data [Chicago Crimes (2001-Present)](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data), mulai dari rekayasa fitur & label (HO1) hingga pemodelan dan continual learning (HO2), sebagai fondasi menuju final project (API serving).

| Milestone | Fokus | Notebook |
|---|---|---|
| **Hands-On 1** | Feature & Label Engineering (EDA -> preprocessing -> feature engineering -> pseudo-labeling Risk Score) | `01`-`05` |
| **Hands-On 2** | Baseline, Modeling, Evaluasi & Continual Learning (drift detection, checkpoint retraining, registry) | `06`-`07` |

---

## Daftar Isi

- [Bagian I - Hands-On 1: Feature & Label Engineering](#bagian-i---hands-on-1-feature--label-engineering)
- [Bagian II - Hands-On 2: Baseline, Modeling & Continual Learning](#bagian-ii---hands-on-2-baseline-modeling--continual-learning)

---

# Bagian I - Hands-On 1: Feature & Label Engineering

Membangun fondasi dataset: dari 8,5 juta insiden mentah menjadi `training_table.parquet` (fitur ruang-waktu + label Risk Score) yang siap dimodelkan.

Repositori ini membangun fondasi sebuah sistem prediksi **Risk Score** (skor 0-100 yang menggambarkan tingkat risiko suatu lokasi pada waktu tertentu) dari data kejahatan historis Kota Chicago yang bersumber dari https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

Alur pengerjaan mengikuti lima notebook berurutan:

| Notebook | Tahap | Keluaran utama |
|---|---|---|
| `01_eda-chicago-crimes.ipynb` | Exploratory Data Analysis | Pemahaman pola waktu & lokasi |
| `02-data-check-diagnostic-chicago-crimes.ipynb` | Diagnostik kualitas data | Keputusan pembersihan yang terjustifikasi |
| `03-preprocessing-chicago-crimes.ipynb` | Preprocessing | `chicago_crimes_clean.parquet` |
| `04-feature-engineering-chicago-crimes.ipynb` | Feature Engineering temporal & spasial | `features_incident.parquet`, `features_gridmonth.parquet` |
| `05-pseudo-labeling-severity.ipynb` | Pseudo-Labeling (Risk Score) | `pseudo_labels_gridmonth.parquet`, `training_table.parquet` |

---

### 1. Penjelasan Singkat Dataset

Dataset yang digunakan adalah **Chicago Crimes (2001-Present)**, catatan insiden kejahatan yang dilaporkan Departemen Kepolisian Kota Chicago.

- **Volume:** 8.534.663 insiden.
- **Rentang waktu:** 1 Januari 2001 hingga 11 April 2026 (tahun berjalan belum lengkap).
- **Cakupan tipe:** 34 kategori `Primary Type` (menjadi 33 setelah kanonikalisasi) dan 78 `Community Area`.
- **Kolom kunci:** `Date`, `Primary Type`, `Description`, `Location Description`, `Arrest`, `Domestic`, `Latitude`/`Longitude`, serta kode `IUCR`.
- **Karakteristik umum:** tingkat penangkapan (arrest) 25,1%, insiden domestik 17,3%, dan 98,87% baris memiliki koordinat geografis yang valid.

Sesuai anjuran pada task, seluruh rentang tahun digunakan (bukan subset) karena komputasi tetap terkendali melalui agregasi berbasis grid dan bulan; keputusan ini justru memperkaya sinyal historis untuk pemodelan risiko.

---

### 2. Justifikasi Keputusan Desain

#### 2.1 Unit Analisis: Sel Grid x Bulan

Koordinat lintang/bujur bersifat kontinu dan berpresisi tinggi sehingga dua insiden di lingkungan yang sama nyaris tidak pernah memiliki koordinat identik. Untuk membuat lokasi dapat dibandingkan dan diagregasi, koordinat dibin ke **grid 0,01 derajat (kurang lebih 1,1 km)**. Dimensi waktu diagregasi ke tingkat **bulan** agar seimbang antara sensitivitas musiman dan kestabilan jumlah sampel per sel. Unit analisis final adalah **sel grid x bulan**, menghasilkan panel penuh 746 sel x 304 periode = 226.784 baris.

#### 2.2 Severity Scoring: Kombinasi Primary Type dan Description

Severity setiap insiden dihitung sebagai:

```
severity = tier(Primary Type) x modifier(Description)
```

**Tier dasar per Primary Type** merepresentasikan tingkat bahaya intrinsik, mengikuti prinsip keamanan publik (kejahatan terhadap nyawa dan integritas tubuh diberi bobot tertinggi):

| Tier | Contoh Primary Type |
|---|---|
| 16 | HOMICIDE, CRIMINAL SEXUAL ASSAULT, KIDNAPPING, HUMAN TRAFFICKING |
| 8 | ROBBERY, ARSON, WEAPONS VIOLATION, OFFENSE INVOLVING CHILDREN, SEX OFFENSE |
| 4 | BATTERY, ASSAULT, BURGLARY, NARCOTICS, INTIMIDATION, STALKING |
| 2 | THEFT, CRIMINAL DAMAGE, CRIMINAL TRESPASS, DECEPTIVE PRACTICE |
| 1 | tipe lain (default) |

**Modifier dari Description** menyesuaikan keparahan dalam satu tipe yang sama:

| Kondisi pada Description | Faktor |
|---|---|
| Senjata api (HANDGUN, FIREARM, ARMED, dll.) | x1,8 |
| AGGRAVATED | x1,4 |
| Senjata tajam / senjata berbahaya lain | x1,3 |
| Ringan (SIMPLE, PETTY, $500 AND UNDER) | x0,9 |

Faktor akhir dibatasi pada rentang [0,8, 2,2] untuk mencegah nilai ekstrem. Pendekatan ini menjawab langsung contoh pada task: `ROBBERY + ARMED: HANDGUN` memperoleh severity lebih tinggi (hingga 17,6) dibanding ROBBERY biasa, dan `BATTERY AGGRAVATED` melampaui `BATTERY SIMPLE`. Secara empiris, 36,31% insiden memperoleh modifier tidak sama dengan 1,0, yang membuktikan Description benar-benar menambah daya beda keparahan.

**Catatan skala:** tabel severity pada task bersifat ilustratif (skala 0-100). Di sini severity diperlakukan sebagai **bobot relatif** (tier x modifier), bukan skor absolut, karena skor akhir 0-100 dibentuk melalui normalisasi di tahap akhir. Yang dijaga adalah urutan dan rasio antar-tipe, bukan angka absolutnya.

#### 2.3 Memodelkan Relevansi Waktu (Temporal Decay)

Sebuah kejahatan tidak selamanya relevan secara seragam. Kontribusi setiap sel-bulan diluruhkan mengikuti fungsi **exponential decay dengan half-life 3 bulan** pada jendela 12 bulan terakhir:

```
harm_temporal[t] = sum_{k=0..12} harm[t-k] x 0,5^(k/3)
```

Bobot menurun dari 1,0 (bulan berjalan) menjadi 0,5 (bulan ke-3) hingga 0,062 (bulan ke-12). Half-life 3 bulan dipilih **moderat** karena EDA menunjukkan hotspot relatif stabil antar tahun; peluruhan yang terlalu agresif akan membuang informasi struktural, sedangkan yang terlalu lambat mengabaikan dinamika terkini.

#### 2.4 Memodelkan Relevansi Lokasi (Spatial Decay)

Risiko suatu area tidak hanya ditentukan oleh insiden di titik itu sendiri, tetapi juga oleh lingkungan sekitarnya. Risiko disebarkan antar sel tetangga menggunakan **kernel Gaussian** (bandwidth 1 sel, radius 2 sel):

```
harm_spatial[sel] = sum_{tetangga} harm_temporal[tetangga] x exp(-(di^2 + dj^2) / (2 x bandwidth^2))
```

Bobot melemah seiring jarak, sehingga sel yang lebih jauh memberi kontribusi lebih kecil. Total bobot kernel adalah 6,17.

#### 2.5 Pembentukan Risk Score 0-100

Skor akhir dibentuk melalui pipeline:

```
harm -> harm_temporal -> harm_spatial -> log1p -> winsorize (persentil 99,5) -> min-max [0, 100]
```

Transformasi `log1p` dan winsorisasi diperlukan karena diagnostik menemukan densitas per sel sangat condong ke kanan (skewness 2,02); tanpa penipisan ekor, segelintir sel ekstrem akan mendominasi skala. Skor kemudian dipetakan ke empat tingkat: LOW (0-50), MEDIUM (50-80), HIGH (80-95), dan VERY HIGH (95-100).

#### 2.6 Representasi Fitur yang Dipilih

File `features_gridmonth.parquet` (226.784 baris x 34 kolom, 0 nilai kosong) memisahkan **basis label** dari **fitur prediktor** untuk mencegah kebocoran target:

- **Temporal siklikal:** `pmonth_sin`, `pmonth_cos`. Waktu bersifat siklikal (Desember dan Januari berdekatan), sehingga representasi linear tidak memadai; encoding sin/cos menjaga kedekatan ujung-ujung siklus. Pilihan ini dijustifikasi langsung oleh EDA yang menunjukkan jam 23 dan jam 0 berdampingan namun berjarak jauh secara linear.
- **Spasial:** `gi`, `gj`, `glat`, `glon` sebagai representasi sel grid.
- **Tren & recency (kausal, tanpa kebocoran):** `lag_1`, `lag_3`, `lag_12`, `roll3_mean`, `roll6_mean`, `roll12_mean`, `cell_hist_mean`, `trend_3_12`.
- **Komposisi terlambat (lagged):** `n_violent_lag1`, `arrest_rate_lag1`, `domestic_rate_lag1`.
- **Spillover spasial:** `neighbor_lag1_mean` (rata-rata aktivitas sel tetangga pada periode sebelumnya).
- **Pola perilaku:** `share_night`, `share_weekend`, beserta versi lag-nya.

Seluruh fitur berbasis lag dihitung dengan pergeseran ke masa lalu per sel sehingga tidak ada kebocoran informasi masa depan.

---

### 3. Insight Singkat dari EDA

- **Distribusi tipe sangat timpang (long-tail):** `THEFT` (1,81 juta) dan `BATTERY` (1,55 juta) mendominasi, jauh di atas tipe berbahaya seperti HOMICIDE. Hal ini menegaskan bahwa Risk Score harus berbasis keparahan, bukan sekadar volume.
- **Arrest rate mencerminkan penegakan, bukan bahaya:** `NARCOTICS` memiliki arrest rate hampir 99% dan `CRIMINAL TRESPASS` sekitar 67%, jauh melampaui kejahatan kekerasan berat. Karena itu arrest rate sengaja tidak dijadikan dasar severity.
- **Tren menurun jangka panjang:** volume turun dari kurang lebih 486 ribu (2001) ke titik terendah kurang lebih 210 ribu (2021), lalu naik ringan pada 2022-2024.
- **Pola musiman dan harian jelas:** insiden memuncak pada bulan-bulan musim panas serta pada sore hingga malam hari, dan menurun tajam pada dini hari. Pola ini memotivasi encoding waktu siklikal.
- **Konsentrasi spasial:** kejahatan terkelompok di area pusat, barat, dan selatan kota, bukan tersebar merata, sehingga pendekatan spatial decay relevan.
- **Hotspot stabil antar tahun:** peringkat area berisiko relatif konsisten meski volume total menurun, yang menjadi dasar pemilihan half-life temporal yang moderat.

---

### 4. Refleksi Singkat (Kendala dan Solusi)

- **Ukuran data besar (kurang lebih 2,4 GB, 8,5 juta baris).** Solusi: pemuatan dengan tipe data hemat memori (kategori/boolean/float kecil) dan komputasi tervektorisasi, sehingga seluruh rentang tahun dapat diproses tanpa subsetting.
- **Nilai hilang yang ambigu.** Diagnostik menunjukkan koordinat hilang (1,13%) terkonsentrasi pada tipe dan lokasi sensitif, yang mengindikasikan redaksi privasi, bukan galat acak. Solusi: menandai (`coord_missing`) tanpa melakukan imputasi agar tidak memalsukan lokasi.
- **Duplikasi semu pada Case Number.** 512 kasus dengan lebih dari satu baris ternyata seluruhnya HOMICIDE yang dicatat per korban. Solusi: baris dipertahankan agar tingkat keparahan kejahatan terberat tidak diremehkan.
- **Inkonsistensi label taksonomi.** 13 kode IUCR memetakan ke lebih dari satu Primary Type akibat perbedaan ejaan. Solusi: kanonikalisasi Primary Type berdasarkan IUCR sebagai sumber kebenaran.
- **Distribusi target condong ke atas (skewness risk_score -1,39; median 82).** Wajar untuk kota berdensitas kejahatan tinggi dan efek akumulasi spasial. Hal ini dicatat sebagai bahan perbaikan lanjutan (misalnya normalisasi per-periode) pada modul berikutnya, dan bukan penghalang bagi kesiapan dataset.
- **Ketiadaan ground truth Risk Score.** Diatasi melalui pseudo-labeling berbasis domain knowledge (severity + space-time decay) yang setiap parameternya dijustifikasi, bukan dipilih sembarang.

---

### 5. Keluaran Akhir

| Berkas | Dimensi | Keterangan |
|---|---|---|
| `chicago_crimes_clean.parquet` | 8.534.663 x 24 | Data bersih hasil preprocessing |
| `features_incident.parquet` | 8.534.663 x 23 | Fitur level insiden (basis severity) |
| `features_gridmonth.parquet` | 226.784 x 34 | Fitur level sel grid x bulan |
| `pseudo_labels_gridmonth.parquet` | 226.784 x 13 | Label Risk Score inti |
| `training_table.parquet` | 226.784 x 41 | Fitur + label, siap untuk pemodelan (Hands-On 2) |

Statistik Risk Score akhir: rata-rata 78,0; median 82,1. Distribusi tingkat: HIGH 110.603, MEDIUM 87.473, VERY HIGH 16.946, LOW 11.762. Tidak terdapat nilai kosong pada target. Kolom `is_partial_period` menandai bulan berjalan yang belum lengkap agar dapat dikecualikan saat pelatihan.

---

### 6. Reproduksi

Jalankan notebook secara berurutan (01 hingga 05). Notebook 04 membaca keluaran preprocessing, sedangkan notebook 05 membaca `features_incident.parquet` dan `features_gridmonth.parquet` dari direktori keluaran notebook 04. 

---

# Bagian II - Hands-On 2: Baseline, Modeling & Continual Learning

Melanjutkan dari `training_table.parquet`: membangun baseline, melatih & mengevaluasi model regresi Risk Score, lalu mensimulasikan continual learning dengan drift detection, checkpoint retraining bersyarat, dan model registry.

Repositori ini melanjutkan Hands-On 1: dari dataset fitur ruang-waktu + label `risk_score` (0-100) untuk grid kejahatan Chicago, membangun mesin prediksi Risk Score, mengevaluasinya terhadap baseline, lalu mensimulasikan continual learning (drift detection, checkpoint retraining bersyarat, dan model registry ringan).


### 1. Ringkasan Data & Setup

- **Sumber**: `training_table.parquet` (output Hands-On 1) - 226.784 baris, 746 sel grid, periode 2001-01 s/d 2026-04.
- **Pembersihan**: membuang 746 baris `is_partial_period == 1` (bulan terpotong) -> **226.038 baris** bersih, 0 NaN pada fitur maupun target.
- **Target**: `risk_score` kontinu 0-100 (mean 78.03, std 15.47). Nilai relatif tinggi & mulus karena label HO1 dibangun dari akumulasi harm temporal-spasial.
- **Split kronologis** (mencegah kebocoran waktu):
  - Train: <= 2021-12 (187.992 baris)
  - Validation: 2022-01 s/d 2023-12 (17.904 baris)
  - Test: 2024-01 s/d 2026-03 (20.142 baris)
  - Tidak ada overlap periode antar split.

---

### 2. Baseline (tanpa training)

Pertanyaan yang dijawab baseline: *prediksi paling logis untuk Risk Score suatu sel tanpa melatih model?* Jawabannya adalah **melanjutkan pola historis terdekat**. Diuji tiga baseline pada test set:

| Baseline | MAE | RMSE | R2 | Ide |
|---|---|---|---|---|
| **Persistence** | **0.322** | **0.463** | **0.999** | Skor sel = skor bulan sebelumnya |
| CellMean | 3.026 | 4.051 | 0.914 | Rata-rata historis per sel |
| GlobalMean | 10.679 | 13.964 | -0.022 | Rata-rata global konstan |

**Justifikasi baseline terpilih**: **Persistence** dipakai sebagai patokan utama. `risk_score` sangat terautokorelasi antar bulan (nilai bulan ini nyaris sama dengan bulan lalu), sehingga "tebak nilai bulan lalu" sudah mencapai MAE 0.322 & R2 0.999. Ini sengaja dipilih sebagai baseline **tersulit** agar uji nilai-tambah model benar-benar ketat.

---

### 3. Feature Vector Assembly & Model

**Feature assembly**: seluruh fitur HO1 dirangkai lewat fungsi `assemble_features()` yang mengunci **urutan kolom**, memaksa `float32`, dan mengisi NaN = 0. Urutan fitur disimpan ke `feature_list*.json` agar **representasi saat training identik dengan saat serving** (mencegah training-serving skew). Kolom yang berpotensi bocor (mengandung info masa depan/target langsung) dikeluarkan.

**Justifikasi model**: dipilih model **berbasis pohon (ensemble)** karena hubungan fitur ruang-waktu terhadap Risk Score bersifat non-linear & penuh interaksi (lokasi x musim x tetangga), yang tidak tertangkap model linear. Terbukti pada eksperimen: Ridge (linear) jauh lebih buruk daripada RandomForest/HistGBR di kedua tahap.

---

### 4. Evaluasi Model vs Baseline (dengan analisis)

Evaluasi dilakukan di **test set** (data yang tak pernah dilihat model). Metrik: **MAE** (rata-rata galat absolut, mudah diinterpretasi pada skala 0-100), **RMSE** (menghukum galat besar), **R2** (proporsi variansi yang dijelaskan).

#### Iterasi 1 - Model v1 (fitur dasar): GAGAL mengalahkan baseline

| Model (test) | MAE | RMSE | R2 |
|---|---|---|---|
| **Persistence (baseline)** | **0.322** | 0.463 | 0.999 |
| HistGBR (v1) | 1.031 | 1.398 | 0.990 |

**Analisis (temuan jujur)**: model v1 justru **kalah** dari Persistence (MAE 1.031 vs 0.322; ~220% lebih buruk). Penyebabnya jelas dari permutation importance: fitur paling berpengaruh adalah `neighbor_lag1_mean` (7.74) dan koordinat spasial `glon`/`glat`, sementara **nilai `risk_score` historis sel itu sendiri belum tersedia sebagai fitur**. Padahal itulah sinyal yang membuat Persistence begitu kuat. Ini temuan penting, bukan kegagalan: mengungkap kebutuhan fitur lag-risk.

#### Iterasi 2 - Model v2 (+ fitur lag-risk): BERHASIL

Ditambahkan 3 fitur kausal `risk_lag_1`, `risk_lag_3`, `risk_lag_12` (nilai `risk_score` sel di 1/3/12 bulan lalu; hanya melihat masa lalu) -> total **26 fitur**. Retraining memilih **RandomForest** (MAE validasi 0.313, terbaik dibanding HistGBR 0.329 & Ridge 0.377).

| Model (test) | MAE | RMSE | R2 |
|---|---|---|---|
| **RandomForest + lag-risk (v2)** | **0.278** | **0.430** | 0.999 |
| Persistence (baseline) | 0.322 | 0.463 | 0.999 |
| HistGBR (v1) | 1.031 | 1.398 | 0.990 |

**Analisis**: model v2 akhirnya **mengalahkan Persistence sebesar +13.9%** (MAE 0.278 vs 0.322). Peningkatan datang murni dari memasukkan riwayat Risk Score sebagai fitur, sekaligus mengombinasikannya dengan konteks spasial-musiman yang tidak dimiliki Persistence.

#### Kualitas peringkat & kasus sulit

Karena penggunaan akhir adalah **memeringkat area paling berisiko**, dievaluasi pula:

- **Spearman rank correlation**: 0.9996 | **Akurasi tier** (LOW/MEDIUM/HIGH/VERY HIGH): 0.9888.
- Subset kasus sulit (model vs Persistence, MAE):

| Subset | n | MAE model | MAE Persistence |
|---|---|---|---|
| Semua test | 20.142 | 0.278 | 0.322 |
| Perubahan besar (top 25%) | 5.036 | **0.539** | 0.703 |
| Sel volatilitas tinggi | 5.049 | **0.334** | 0.371 |

**Analisis**: keunggulan model paling terasa justru pada kasus sulit, yaitu ketika skor **berubah tajam** dari bulan sebelumnya - persis di situ Persistence gagal (karena hanya menyalin nilai lama), dan model memberi nilai tambah nyata.

---

### 5. Continual Learning 
**Desain simulasi**: data test/masa depan dipecah kronologis menjadi batch tahunan (2022, 2023, 2024, 2025, 2026) yang "datang" berurutan. Champion awal = model v2 (RandomForest + lag-risk). Untuk tiap batch:

1. **Monitor drift** (PSI) + degradasi performa champion.
2. Jika drift terdeteksi -> latih **challenger** pada data diperluas (s/d akhir tahun batch, dikurangi holdout 6 bulan terakhir untuk uji adil).
3. **Bandingkan** champion vs challenger di holdout yang sama.
4. **Promote** challenger hanya jika MAE-nya <= champion; jika tidak -> **reject & rollback** (champion lama dipertahankan).
5. Catat keputusan ke registry (berhasil maupun gagal).

#### Perjalanan v1 -> v4

| Batch | PSI_max | Drift? | Challenger vs Champion (MAE) | Keputusan | Champion setelahnya |
|---|---|---|---|---|---|
| init | - | - | - (acuan MAE 0.278) | INIT | v1 |
| 2022 | 0.631 | Ya | 0.324 <= 0.326 | **PROMOTE** | **v2** |
| 2023 | 0.628 | Ya | 0.280 > 0.279 | **REJECT** (rollback) | v2 |
| 2024 | 0.450 | Ya | 0.271 > 0.268 | **REJECT** (rollback) | v2 |
| 2025 | 0.301 | Ya | 0.263 <= 0.273 | **PROMOTE** | **v3** |
| 2026 | 10.611 | Ya | 0.278 <= 0.279 | **PROMOTE** | **v4** |

**Narasi**: drift terdeteksi di **semua** batch, namun model **tidak** serta-merta diganti. Pada 2023 & 2024, challenger hasil retraining ternyata **sedikit lebih buruk** dari champion di holdout, sehingga **ditolak dan champion lama dipertahankan** (rollback) - keputusan ini dicatat sebagai bukti kriteria deployment bekerja dua arah, bukan sekadar "selalu pakai model terbaru". Promosi terjadi di 2022, 2025, dan 2026 ketika challenger benar-benar >= champion. Champion final = **v4**.

---

### 6. Justifikasi Threshold & Kriteria

- **Drift metric - PSI (Population Stability Index)** dengan ambang **>= 0.2**. Mengikuti konvensi industri (mis. Evidently AI): PSI < 0.1 = stabil, 0.1-0.2 = pergeseran moderat, **> 0.2 = pergeseran signifikan**. PSI dipilih karena sederhana, tak butuh label, dan bisa dihitung per fitur.
- **Fitur yang dipantau**: 21 dari 26 (mengecualikan identitas/koordinat statis `year_idx`, `glat`, `glon`, `gi`, `gj` yang bukan indikator perubahan pola).
- **Trigger sekunder - degradasi performa**: retraining juga dipicu bila MAE champion pada batch memburuk **> 15%** dibanding acuan. Ini menangkap concept drift yang tak selalu tampak dari distribusi fitur.
- **Holdout 6 bulan terakhir** tiap batch dipakai sebagai arena uji champion vs challenger yang identik -> perbandingan adil.
- **Kriteria promosi**: challenger dipromosikan **hanya jika MAE_challenger <= MAE_champion** pada holdout. Perbandingan ketat ini konservatif (menghindari mengganti model demi peningkatan yang tidak nyata).

---

### 7. Output Model & Registry

Rekam jejak disimpan dalam format sederhana namun informatif: **`model_registry.csv`** (satu baris per keputusan), plus **checkpoint `.joblib` tiap versi** dan **`production_meta.json`** untuk champion aktif. Kolom registry:

`version, batch, train_end, n_train, model_type, MAE_batch, R2_batch, psi_max, psi_mean, drift_detected, decision`

Contoh isi (mencakup versi yang DITOLAK, sesuai anjuran mendokumentasikan kegagalan):

| version | batch | train_end | n_train | MAE_batch | psi_max | drift | decision |
|---|---|---|---|---|---|---|---|
| 1 | init | 2021-12 | 187.992 | - | - | False | INIT |
| 2 | 2022 | 2022-12 | 192.468 | 0.332 | 0.631 | True | PROMOTED |
| 2 | 2023 | 2022-12 | 192.468 | 0.290 | 0.628 | True | REJECTED |
| 2 | 2024 | 2022-12 | 192.468 | 0.271 | 0.450 | True | REJECTED |
| 3 | 2025 | 2025-12 | 219.324 | 0.282 | 0.301 | True | PROMOTED |
| 4 | 2026 | 2026-12 | 221.562 | 0.290 | 10.611 | True | PROMOTED |

Dengan registry ini, siapa pun dapat memahami riwayat model kapan saja tanpa membongkar kode.

---

### 8. Refleksi - Kendala & Solusi

1. **Baseline terlalu kuat & model awal kalah.** Persistence mencapai MAE 0.322 karena label sangat autokorelatif. Model v1 kalah telak. *Solusi*: diagnosis lewat permutation importance -> menambahkan fitur lag-risk (riwayat `risk_score`), sehingga v2 akhirnya unggul +13.9%.
2. **Margin champion vs challenger sangat tipis** (mis. 0.280 vs 0.279). Karena label mulus, selisih MAE antar versi kecil sekali dan sebagian berada dalam rentang noise. *Solusi/catatan*: memakai perbandingan ketat + holdout konsisten; ke depan bisa ditambah *non-inferiority margin* (epsilon) agar keputusan lebih stabil.
3. **PSI ekstrem pada batch 2026 (10.611)** ternyata **artefak musiman**: batch 2026 hanya berisi bulan Jan-Mar, sehingga fitur siklik bulan berbeda jauh dari referensi setahun penuh - bukan drift pola nyata. *Solusi/catatan*: diinterpretasikan sebagai efek musiman; perbaikan lanjutan adalah membandingkan distribusi per musim atau mengecualikan fitur siklik dari PSI.
4. **Drift terdeteksi di semua batch** karena ambang sensitif + referensi hanya diperbarui saat promote. *Solusi*: memasangkan PSI dengan pengecekan performa aktual & keputusan berbasis holdout, sehingga retraining sering tidak otomatis berarti model sering diganti.

---
