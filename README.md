# Chicago-crimes-risk-score
HO1 MLOps SISTECH 2026 - Risk Score pipeline

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

## 1. Penjelasan Singkat Dataset

Dataset yang digunakan adalah **Chicago Crimes (2001-Present)**, catatan insiden kejahatan yang dilaporkan Departemen Kepolisian Kota Chicago.

- **Volume:** 8.534.663 insiden.
- **Rentang waktu:** 1 Januari 2001 hingga 11 April 2026 (tahun berjalan belum lengkap).
- **Cakupan tipe:** 34 kategori `Primary Type` (menjadi 33 setelah kanonikalisasi) dan 78 `Community Area`.
- **Kolom kunci:** `Date`, `Primary Type`, `Description`, `Location Description`, `Arrest`, `Domestic`, `Latitude`/`Longitude`, serta kode `IUCR`.
- **Karakteristik umum:** tingkat penangkapan (arrest) 25,1%, insiden domestik 17,3%, dan 98,87% baris memiliki koordinat geografis yang valid.

Sesuai anjuran pada task, seluruh rentang tahun digunakan (bukan subset) karena komputasi tetap terkendali melalui agregasi berbasis grid dan bulan; keputusan ini justru memperkaya sinyal historis untuk pemodelan risiko.

---

## 2. Justifikasi Keputusan Desain

### 2.1 Unit Analisis: Sel Grid x Bulan

Koordinat lintang/bujur bersifat kontinu dan berpresisi tinggi sehingga dua insiden di lingkungan yang sama nyaris tidak pernah memiliki koordinat identik. Untuk membuat lokasi dapat dibandingkan dan diagregasi, koordinat dibin ke **grid 0,01 derajat (kurang lebih 1,1 km)**. Dimensi waktu diagregasi ke tingkat **bulan** agar seimbang antara sensitivitas musiman dan kestabilan jumlah sampel per sel. Unit analisis final adalah **sel grid x bulan**, menghasilkan panel penuh 746 sel x 304 periode = 226.784 baris.

### 2.2 Severity Scoring: Kombinasi Primary Type dan Description

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

### 2.3 Memodelkan Relevansi Waktu (Temporal Decay)

Sebuah kejahatan tidak selamanya relevan secara seragam. Kontribusi setiap sel-bulan diluruhkan mengikuti fungsi **exponential decay dengan half-life 3 bulan** pada jendela 12 bulan terakhir:

```
harm_temporal[t] = sum_{k=0..12} harm[t-k] x 0,5^(k/3)
```

Bobot menurun dari 1,0 (bulan berjalan) menjadi 0,5 (bulan ke-3) hingga 0,062 (bulan ke-12). Half-life 3 bulan dipilih **moderat** karena EDA menunjukkan hotspot relatif stabil antar tahun; peluruhan yang terlalu agresif akan membuang informasi struktural, sedangkan yang terlalu lambat mengabaikan dinamika terkini.

### 2.4 Memodelkan Relevansi Lokasi (Spatial Decay)

Risiko suatu area tidak hanya ditentukan oleh insiden di titik itu sendiri, tetapi juga oleh lingkungan sekitarnya. Risiko disebarkan antar sel tetangga menggunakan **kernel Gaussian** (bandwidth 1 sel, radius 2 sel):

```
harm_spatial[sel] = sum_{tetangga} harm_temporal[tetangga] x exp(-(di^2 + dj^2) / (2 x bandwidth^2))
```

Bobot melemah seiring jarak, sehingga sel yang lebih jauh memberi kontribusi lebih kecil. Total bobot kernel adalah 6,17.

### 2.5 Pembentukan Risk Score 0-100

Skor akhir dibentuk melalui pipeline:

```
harm -> harm_temporal -> harm_spatial -> log1p -> winsorize (persentil 99,5) -> min-max [0, 100]
```

Transformasi `log1p` dan winsorisasi diperlukan karena diagnostik menemukan densitas per sel sangat condong ke kanan (skewness 2,02); tanpa penipisan ekor, segelintir sel ekstrem akan mendominasi skala. Skor kemudian dipetakan ke empat tingkat: LOW (0-50), MEDIUM (50-80), HIGH (80-95), dan VERY HIGH (95-100).

### 2.6 Representasi Fitur yang Dipilih

File `features_gridmonth.parquet` (226.784 baris x 34 kolom, 0 nilai kosong) memisahkan **basis label** dari **fitur prediktor** untuk mencegah kebocoran target:

- **Temporal siklikal:** `pmonth_sin`, `pmonth_cos`. Waktu bersifat siklikal (Desember dan Januari berdekatan), sehingga representasi linear tidak memadai; encoding sin/cos menjaga kedekatan ujung-ujung siklus. Pilihan ini dijustifikasi langsung oleh EDA yang menunjukkan jam 23 dan jam 0 berdampingan namun berjarak jauh secara linear.
- **Spasial:** `gi`, `gj`, `glat`, `glon` sebagai representasi sel grid.
- **Tren & recency (kausal, tanpa kebocoran):** `lag_1`, `lag_3`, `lag_12`, `roll3_mean`, `roll6_mean`, `roll12_mean`, `cell_hist_mean`, `trend_3_12`.
- **Komposisi terlambat (lagged):** `n_violent_lag1`, `arrest_rate_lag1`, `domestic_rate_lag1`.
- **Spillover spasial:** `neighbor_lag1_mean` (rata-rata aktivitas sel tetangga pada periode sebelumnya).
- **Pola perilaku:** `share_night`, `share_weekend`, beserta versi lag-nya.

Seluruh fitur berbasis lag dihitung dengan pergeseran ke masa lalu per sel sehingga tidak ada kebocoran informasi masa depan.

---

## 3. Insight Singkat dari EDA

- **Distribusi tipe sangat timpang (long-tail):** `THEFT` (1,81 juta) dan `BATTERY` (1,55 juta) mendominasi, jauh di atas tipe berbahaya seperti HOMICIDE. Hal ini menegaskan bahwa Risk Score harus berbasis keparahan, bukan sekadar volume.
- **Arrest rate mencerminkan penegakan, bukan bahaya:** `NARCOTICS` memiliki arrest rate hampir 99% dan `CRIMINAL TRESPASS` sekitar 67%, jauh melampaui kejahatan kekerasan berat. Karena itu arrest rate sengaja tidak dijadikan dasar severity.
- **Tren menurun jangka panjang:** volume turun dari kurang lebih 486 ribu (2001) ke titik terendah kurang lebih 210 ribu (2021), lalu naik ringan pada 2022-2024.
- **Pola musiman dan harian jelas:** insiden memuncak pada bulan-bulan musim panas serta pada sore hingga malam hari, dan menurun tajam pada dini hari. Pola ini memotivasi encoding waktu siklikal.
- **Konsentrasi spasial:** kejahatan terkelompok di area pusat, barat, dan selatan kota, bukan tersebar merata, sehingga pendekatan spatial decay relevan.
- **Hotspot stabil antar tahun:** peringkat area berisiko relatif konsisten meski volume total menurun, yang menjadi dasar pemilihan half-life temporal yang moderat.

---

## 4. Refleksi Singkat (Kendala dan Solusi)

- **Ukuran data besar (kurang lebih 2,4 GB, 8,5 juta baris).** Solusi: pemuatan dengan tipe data hemat memori (kategori/boolean/float kecil) dan komputasi tervektorisasi, sehingga seluruh rentang tahun dapat diproses tanpa subsetting.
- **Nilai hilang yang ambigu.** Diagnostik menunjukkan koordinat hilang (1,13%) terkonsentrasi pada tipe dan lokasi sensitif, yang mengindikasikan redaksi privasi, bukan galat acak. Solusi: menandai (`coord_missing`) tanpa melakukan imputasi agar tidak memalsukan lokasi.
- **Duplikasi semu pada Case Number.** 512 kasus dengan lebih dari satu baris ternyata seluruhnya HOMICIDE yang dicatat per korban. Solusi: baris dipertahankan agar tingkat keparahan kejahatan terberat tidak diremehkan.
- **Inkonsistensi label taksonomi.** 13 kode IUCR memetakan ke lebih dari satu Primary Type akibat perbedaan ejaan. Solusi: kanonikalisasi Primary Type berdasarkan IUCR sebagai sumber kebenaran.
- **Distribusi target condong ke atas (skewness risk_score -1,39; median 82).** Wajar untuk kota berdensitas kejahatan tinggi dan efek akumulasi spasial. Hal ini dicatat sebagai bahan perbaikan lanjutan (misalnya normalisasi per-periode) pada modul berikutnya, dan bukan penghalang bagi kesiapan dataset.
- **Ketiadaan ground truth Risk Score.** Diatasi melalui pseudo-labeling berbasis domain knowledge (severity + space-time decay) yang setiap parameternya dijustifikasi, bukan dipilih sembarang.

---

## 5. Keluaran Akhir

| Berkas | Dimensi | Keterangan |
|---|---|---|
| `chicago_crimes_clean.parquet` | 8.534.663 x 24 | Data bersih hasil preprocessing |
| `features_incident.parquet` | 8.534.663 x 23 | Fitur level insiden (basis severity) |
| `features_gridmonth.parquet` | 226.784 x 34 | Fitur level sel grid x bulan |
| `pseudo_labels_gridmonth.parquet` | 226.784 x 13 | Label Risk Score inti |
| `training_table.parquet` | 226.784 x 41 | Fitur + label, siap untuk pemodelan (Hands-On 2) |

Statistik Risk Score akhir: rata-rata 78,0; median 82,1. Distribusi tingkat: HIGH 110.603, MEDIUM 87.473, VERY HIGH 16.946, LOW 11.762. Tidak terdapat nilai kosong pada target. Kolom `is_partial_period` menandai bulan berjalan yang belum lengkap agar dapat dikecualikan saat pelatihan.

---

## 6. Reproduksi

Jalankan notebook secara berurutan (01 hingga 05). Notebook 04 membaca keluaran preprocessing, sedangkan notebook 05 membaca `features_incident.parquet` dan `features_gridmonth.parquet` dari direktori keluaran notebook 04. 
