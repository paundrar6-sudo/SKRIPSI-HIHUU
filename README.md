# Analisis Real-Time BI-Rate dan Nilai Tukar Rupiah terhadap Dollar AS menggunakan LSTM untuk Prediksi IHSG

Kode pendamping skripsi Paundra Ramadhani Putra Ansori (NIM 230535607268, Teknik Informatika, Universitas Negeri Malang).
Rujukan "Subbab x.y" di bawah mengacu pada naskah skripsi (Bab III metode, Bab IV hasil, Bab V kesimpulan).

## Persiapan (sekali saja)

Buka PowerShell di folder `LSTM_BIRate_Kurs_IHSG`, lalu:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Folder `.venv` sudah dibuat dan terisi. Langkah di atas hanya perlu diulang jika folder `.venv` dihapus.

## Cara menjalankan (satu perintah)

    .\.venv\Scripts\python.exe run_all.py

Urutannya: pasang paket, unduh IHSG (Yahoo Finance ^JKSE), unduh kurs USD/IDR JISDOR (Bank Indonesia; Yahoo IDR=X hanya cadangan),
unduh Fed Funds Rate (FRED), siapkan BI-Rate (Bank Indonesia), gabungkan data, latih LSTM, lalu buka dashboard di
http://localhost:8501 (Ctrl+C untuk berhenti).

Opsi: `--skip-install`, `--skip-fetch`, `--skip-train`, `--no-app`, `--walkforward`, `--lengkap` (ikut menjalankan `experiments.py` dan `econometrics.py`).

Perintah lain:

| Tujuan | Perintah | Subbab |
|---|---|---|
| Hanya dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` | 3.12, 4.8 |
| Pilih panjang jendela & ukuran LSTM (loss validasi) | `.\.venv\Scripts\python.exe tune.py` | 3.9, 4.2 |
| Latih + evaluasi data uji | `.\.venv\Scripts\python.exe train.py` | 3.10, 4.3 |
| Evaluasi walk-forward 5 lipatan | `.\.venv\Scripts\python.exe evaluate_walkforward.py` | 3.11, 4.4 |
| Eksperimen M0-M4, horizon, permutation importance | `.\.venv\Scripts\python.exe experiments.py` | 3.10-3.11, 4.5 |
| OLS Newey-West, VAR/Granger, studi peristiwa RDG | `.\.venv\Scripts\python.exe econometrics.py` | 3.11, 4.6 |
| Simulasi pola 70/30 | `.\.venv\Scripts\python.exe simulasi.py` (tanpa LSTM: `--tanpa-lstm`) | 3.11, 4.7 |
| Prediksi H+1 lewat terminal | `.\.venv\Scripts\python.exe predict_realtime.py --bi-rate 6.25 --kurs 18300` | 3.12 |
| Catat prediksi harian (forward test) | `.\.venv\Scripts\python.exe daily_update.py` | 3.12, 4.8 |
| Jadwalkan otomatis Senin-Jumat 17:30 | `.\.venv\Scripts\python.exe daily_update.py --pasang-jadwal 17:30` | 3.12 |
| Grafik naskah (PNG 300 dpi + PDF) | `.\.venv\Scripts\python.exe figures.py` (hasil di `figures/`, daftar di `figures/DAFTAR_GAMBAR.md`) | Bab IV |

Versi paket di `requirements.txt` dikunci sesuai lingkungan saat hasil skripsi dibuat (Python 3.13).

## Data dan periode (Subbab 3.4-3.7)

| Peran | Variabel | Sumber |
|---|---|---|
| Y | IHSG harian (harga penutupan) | Yahoo Finance (^JKSE) |
| X1 | BI-Rate (72 keputusan RDG, Jan 2021 s.d. 18 Nov 2026, terakhir 6,00%) | Bank Indonesia (`data_bawaan/bi_rate.csv`) |
| X2 | Kurs rupiah per dolar AS | JISDOR Bank Indonesia (cadangan: Yahoo Finance IDR=X) |
| Kontrol | Fed Funds Rate (hanya skenario M4) | FRED |

Periode penelitian 4 Januari 2021 s.d. 20 November 2026. Dataset berisi 1.416 baris sampai 19 November 2026 (hari terakhir yang
sudah punya target H+1); setelah 10 baris awal pemanasan volatilitas dibuang tersisa 1.406 baris. Pembagian kronologis:

| Bagian | Jumlah sequence | Rentang target |
|---|---|---|
| Latih | 916 | 18 Mar 2021 s.d. 24 Des 2024 |
| Validasi (15% akhir data latih) | 169 | 27 Des 2024 s.d. 22 Sep 2025 |
| Uji (20% akhir) | 282 | 23 Sep 2025 s.d. 20 Nov 2026 |

Data Yahoo sebelumnya memuat 36 hari lonjakan kurs >1,5% yang tidak ada di JISDOR (JISDOR: 2 hari), karena itu JISDOR dipakai.
Tidak ada komponen NLP. "Kurs naik" berarti rupiah MELEMAH; "kurs turun" berarti rupiah MENGUAT.

## Model (Subbab 3.7-3.9)

Fitur stasioner (`features.py`), dihitung identik saat latih dan prediksi:

| Fitur | Isi |
|---|---|
| `ret_ihsg` | log return IHSG |
| `vol_ihsg` | simpangan baku return IHSG 10 hari |
| `ret_kurs` | log return kurs USD/IDR |
| `d_bi` | perubahan BI-Rate dalam satuan 25 bps |
| `bi_rate` | level BI-Rate |
| `selisih_bi_fed` | BI-Rate dikurangi Fed Funds Rate |

Skenario fitur (Tabel 3.2):

| Skenario | Fitur | Hipotesis |
|---|---|---|
| M0 | `ret_ihsg`, `vol_ihsg` | pembanding riwayat IHSG saja |
| M1 | M0 + `d_bi`, `bi_rate` | H1 (BI-Rate) |
| M2 | M0 + `ret_kurs` | H2 (kurs) |
| M3 | kelima fitur di atas (model utama) | H3 (gabungan) |
| M4 | M3 + `selisih_bi_fed` | uji tambahan |

Target: log return IHSG hari berikutnya. Standardisasi z-score di-fit hanya pada data latih.
Grid tuning (loss validasi, 3 seed, tanpa melihat data uji): jendela 5/10/20/40 hari × 16/32/64 unit; terpilih jendela 40, 32 unit.
LSTM 1 lapis, dropout 0,2, Adam (lr 0,001, weight decay 0,0001), batch 32, MSE, maks. 200 epoch, early stopping (kesabaran 20),
ensemble 5 seed (42, 7, 2024, 11, 99).
Pembanding (Subbab 3.8): naif (besok = hari ini), rata-rata historis, ARIMA (orde dipilih AIC data latih), regresi linear.
Uji (Subbab 3.10): RMSE, MAE, MAPE, akurasi arah, Diebold-Mariano dengan koreksi HLN, Pesaran-Timmermann, R² out-of-sample.

## Hasil (Bab IV)

### Data uji utama (Subbab 4.3), 282 hari, 23 Sep 2025 s.d. 20 Nov 2026

| Model | RMSE | MAE | MAPE | Arah benar | R² oos | p DM vs LSTM |
|---|---|---|---|---|---|---|
| LSTM (M3) | 116,51 | 79,27 | 1,172% | 46,45% | −0,61% | - |
| Naif | 116,15 | 78,86 | 1,166% | - | 0 | 0,135 |
| Rata-rata historis | 116,30 | 78,98 | 1,168% | 46,81% | −0,23% | 0,172 |
| ARIMA | 116,75 | 78,87 | 1,166% | 47,87% | −0,57% | 0,981 |
| Regresi linear | 117,57 | 80,02 | 1,185% | 46,10% | −2,62% | 0,190 |

Uji Pesaran-Timmermann untuk arah LSTM: p 0,756 (tidak lebih baik dari tebakan acak).

### Walk-forward (Subbab 4.4)

5 lipatan expanding window, 703 hari uji (4 Des 2023 s.d. 19 Nov 2026): RMSE LSTM 93,23 vs naif 92,99; p DM 0,183;
LSTM menang di 0 dari 5 lipatan. RMSE gabungan skenario: M0 93,13, M1 93,21, M2 93,20, M3 93,23, M4 93,28.

### Skenario fitur, horizon, dan kepentingan fitur (Subbab 4.5)

| Skenario | RMSE uji | p DM vs M0 |
|---|---|---|
| M0 | 116,63 | - |
| M1 | 116,59 | 0,282 |
| M2 | 116,67 | 0,847 |
| M3 | 116,51 | 0,388 |
| M4 | 116,70 | 0,724 |

H1, H2, dan H3 tidak didukung: menambah BI-Rate dan/atau kurs tidak menurunkan galat secara signifikan.
Horizon M3 vs naif: H+5 284,26 vs 281,12; H+20 640,43 vs 591,98. Permutation importance semua kelompok fitur ≈ 0.

### Ekonometrika (Subbab 4.6)

- OLS hari yang sama (Newey-West): koefisien kurs −0,952 (p 5,5×10⁻¹⁰); BI-Rate tidak signifikan.
- VAR lag 2 (AIC). Granger: kurs → IHSG p 0,721; BI-Rate → IHSG p 0,449; IHSG → kurs p 0,0004.
- Studi peristiwa 64 keputusan RDG dengan jendela estimasi lengkap (naik 11, turun 6, tetap 47) dari 72 keputusan di CSV;
  tidak ada CAR yang lolos koreksi Bonferroni.

### Simulasi 70/30 (Subbab 4.7) dan forward test (Subbab 4.8)

Simulasi 70/30: latih 19 Jan 2021 s.d. 12 Feb 2025 (985 hari), data uji 13 Feb 2025 s.d. 23 Nov 2026 (423 hari).
Forward test tercatat di `logs/prediksi_harian.csv` (prediksi 20 Nov 2026 untuk 23 Nov 2026: 5.149,30; aktual 5.215,85;
galat 66,55 vs naif 69,52).

### Kesimpulan (Bab V)

LSTM tidak mengungguli baseline naif; selisihnya tidak signifikan secara statistik. Kurs berhubungan kuat dengan IHSG pada hari
yang sama, tetapi tidak membantu meramal hari berikutnya. Ini konsisten dengan hipotesis pasar efisien (Fama, 1970).

## Isi dashboard (Subbab 3.12, 4.8)

1. Prediksi H+1: perkiraan IHSG hari bursa berikutnya, akurasi model di data uji, dan uji skenario BI-Rate/kurs baru.
2. Data pasar: grafik gabungan IHSG, BI-Rate, kurs JISDOR, dan Fed Funds Rate.
3. Hubungan X dan Y: korelasi, kombinasi BI-Rate × kurs terhadap IHSG pada H0 dan H+1, teori vs data.
4. Evaluasi model: tabel semua pembanding, uji Diebold-Mariano, keunggulan kumulatif, walk-forward.
5. Kontribusi variabel: skenario M0-M4, horizon, permutation importance.
6. Simulasi 70/30: pola kurs dan siklus BI-Rate terhadap kekuatan gerak IHSG, LSTM 70/30.
7. Rekam jejak: log forward test harian.
8. Metodologi: data, fitur, hyperparameter, metode uji beserta rujukan, batasan, status berkas.

Tampilan mengikuti `DESIGN.md` (IBM Carbon); tema ada di `.streamlit/config.toml`.

## Berkas

    config.py  features.py  modeling.py  lstm_model.py
    fetch_ihsg.py  fetch_kurs.py  fetch_interest_rate.py  build_dataset.py
    tune.py  train.py  evaluate_walkforward.py  experiments.py  econometrics.py  simulasi.py
    predict_realtime.py  daily_update.py  analysis.py  figures.py  app.py  run_all.py
    data_bawaan/bi_rate.csv     BI-Rate asli, diverifikasi di bi.go.id
    model/                      metrics.json, tuning.json, walkforward.json, experiments.json,
                                econometrics.json, simulasi.json, test_predictions.csv
    logs/prediksi_harian.csv    rekam jejak forward test
    figures/                    grafik naskah (PNG + PDF)

## Skill agen AI (folder `.opencode/skills/`)

antislop, antislop-ui, antislop-copywriting, antislop-code, antislop-human, antislop-layoutmobile, ui-ux-pro-max.
Arah desain: `DESIGN.md` (IBM Carbon, dari voltagent/awesome-design-md).

## Batasan (Subbab 3.14, 4.9)

Hasil bersifat statistik untuk penelitian, bukan saran investasi. Korelasi bukan sebab-akibat.
BI-Rate hanya berubah 19 kali selama periode (12 naik, 7 turun), sehingga kombinasi yang melibatkan BI-Rate naik/turun memiliki n kecil.
Studi peristiwa memakai model rata-rata konstan tanpa data ekspektasi pasar, sehingga kejutan kebijakan tidak dapat dipisahkan
dari keputusan yang sudah diantisipasi.
