# Cakupan Wilayah dan Keterbatasan Data

Dokumen ini menjelaskan wilayah geografis yang saat ini didukung oleh Risk
Score API, alasan teknis di baliknya, dan bagaimana cakupan tersebut dapat
diperluas di masa depan.

---

## 1. Cakupan saat ini

Model Risk Score dilatih menggunakan data kejahatan historis Kota Chicago,
Illinois, Amerika Serikat, dan hanya valid untuk koordinat di dalam wilayah
tersebut. Bounding box, nama wilayah, dan titik tengah yang didukung dapat
diambil kapan saja lewat `GET /meta`.

Dataset kejahatan Chicago dipakai karena tersedia secara terbuka dengan
detail koordinat dan waktu yang memadai untuk melatih dan mengevaluasi
model semacam ini. Data historis dengan detail setara belum tersedia secara
publik untuk banyak wilayah lain, termasuk kota-kota di Indonesia. Cakupan
wilayah dapat diperluas ketika dataset yang setara tersedia, tanpa mengubah
desain model maupun kontrak API (lihat bagian 4).

## 2. Mengapa koordinat di luar Chicago tidak dapat langsung digunakan

Risk Score dibentuk dari dua kelompok sinyal:

| Kelompok fitur | Contoh | Dapat digunakan di wilayah lain? |
| :-- | :-- | :-- |
| Spasial | posisi sel grid, jumlah kejadian historis di sekitar titik | Tidak. Nilai-nilai ini merepresentasikan lokasi fisik di Chicago dan tidak memiliki padanan langsung di kota lain |
| Temporal | jam, hari dalam minggu, akhir pekan | Sebagian besar dapat digeneralisasi, karena pola risiko yang meningkat pada malam hari cenderung berlaku lintas wilayah |

Analisis kepentingan fitur pada evaluasi model menunjukkan bahwa kontributor
terbesar terhadap skor adalah fitur spasial - kelompok sinyal yang justru
tidak dapat ditransfer ke wilayah lain. Memaksakan koordinat di luar Chicago
ke dalam model akan menghasilkan angka yang tampak valid tetapi tidak
memiliki dasar data yang benar. Untuk aplikasi yang berkaitan dengan
keselamatan, hal ini tidak dapat diterima.

Sebagai konsekuensinya, API ini tidak melakukan proyeksi, transformasi, atau
simulasi apa pun terhadap koordinat di luar wilayah yang didukung.
Permintaan dengan koordinat di luar bounding box akan selalu ditolak (kode
status 422) dengan pesan yang menjelaskan wilayah yang didukung. Penolakan
ini diterapkan pada lebih dari satu lapis di sisi server, sehingga tidak
ada jalur permintaan yang dapat secara tidak sengaja meloloskan koordinat
di luar wilayah model.

## 3. Waktu dan zona waktu

Model membaca waktu sebagai waktu dinding pada zona waktu wilayah yang
didukung (`America/Chicago` untuk saat ini, dapat diambil lewat `GET
/meta`). Permintaan dengan datetime berzona lain dikonversi secara otomatis
ke zona tersebut, dan hasil konversinya selalu disertakan pada response
(`resolved_local_datetime`) agar tidak terjadi kesalahan tafsir waktu yang
tidak terlihat.

## 4. Perluasan cakupan wilayah

Ketika dataset baru dengan granularitas setara tersedia untuk wilayah lain,
model dapat dilatih ulang atas dataset tersebut dan artefaknya diganti di
sisi server. Nama wilayah, titik tengah, bounding box, dan zona waktu yang
dikembalikan lewat `GET /meta` akan otomatis mencerminkan wilayah baru
begitu server dijalankan ulang dengan model yang baru dilatih.

Kontrak API (nama dan struktur field pada request/response) tidak berubah
akibat perluasan wilayah ini, sehingga aplikasi klien yang membaca informasi
wilayah secara dinamis dari `GET /meta` - bukan menuliskannya sebagai nilai
tetap di kodenya - tidak memerlukan penyesuaian kode apa pun.
