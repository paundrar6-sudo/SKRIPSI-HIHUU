# Analisis Real-Time BI-Rate dan Nilai Tukar Rupiah terhadap Dollar AS menggunakan LSTM untuk Prediksi IHSG

## Persiapan (sekali saja)

Buka PowerShell di folder `LSTM_BIRate_Kurs_IHSG`, lalu:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Folder `.venv` sudah dibuat dan terisi. Langkah di atas hanya perlu diulang jika folder `.venv` dihapus.

## Cara menjalankan (satu perintah)

    .\.venv\Scripts\python.exe run_all.py

Urutannya: pasang paket, unduh IHSG (Yahoo Finance ^JKSE), unduh kurs USD/IDR (Yahoo IDR=X), unduh Fed Funds Rate (FRED),
siapkan BI-Rate (Bank Indonesia), gabungkan data, latih LSTM, lalu buka dashboard di http://localhost:8501 (Ctrl+C untuk berhenti).

Opsi: `--skip-install`, `--skip-fetch`, `--skip-train`, `--no-app`, `--walkforward`.

Perintah lain:

| Tujuan | Perintah |
|---|---|
| Hanya dashboard | `.\.venv\Scripts\python.exe -m streamlit run app.py` |
| Pilih panjang jendela & ukuran LSTM (loss validasi) | `.\.venv\Scripts\python.exe tune.py` |
| Latih + evaluasi data uji | `.\.venv\Scripts\python.exe train.py` |
| Evaluasi walk-forward 5 lipatan | `.\.venv\Scripts\python.exe evaluate_walkforward.py` |
| Prediksi H+1 lewat terminal | `.\.venv\Scripts\python.exe predict_realtime.py --bi-rate 6.25 --kurs 18300` |
| Eksperimen M0-M4, horizon, kepentingan fitur | `.\.venv\Scripts\python.exe experiments.py` |
| Granger/VAR, event study RDG | `.\.venv\Scripts\python.exe econometrics.py` |
| Catat prediksi harian (forward test) | `.\.venv\Scripts\python.exe daily_update.py` |
| Jadwalkan otomatis Senin-Jumat 17:30 | `.\.venv\Scripts\python.exe daily_update.py --pasang-jadwal 17:30` |

Kurs kini diambil dari **JISDOR Bank Indonesia** (sesuai proposal Bab III); Yahoo hanya cadangan.
Data Yahoo sebelumnya memuat 36 hari lonjakan >1,5% yang tidak ada di JISDOR (JISDOR: 2 hari).

## Variabel penelitian

| Peran | Variabel | Sumber |
|---|---|---|
| Y | IHSG harian | Yahoo Finance (^JKSE) |
| X1 | BI-Rate (72 keputusan RDG, Jan 2021 s.d. 18 Nov 2026, terakhir 6,00%) | Bank Indonesia (`data_bawaan/bi_rate.csv`) |
| X2 | Kurs rupiah per dolar AS | Yahoo Finance (IDR=X) |
| Kontrol | Fed Funds Rate | FRED |

Tidak ada komponen NLP. "Kurs naik" berarti rupiah MELEMAH; "kurs turun" berarti rupiah MENGUAT.

## Model

Fitur stasioner (`features.py`), dihitung identik saat latih dan prediksi:

| Fitur | Isi |
|---|---|
| `ret_ihsg` | log return IHSG |
| `vol_ihsg` | simpangan baku return IHSG 10 hari |
| `ret_kurs` | log return kurs USD/IDR |
| `d_bi` | perubahan BI-Rate dalam satuan 25 bps |
| `bi_rate` | level BI-Rate |
| `selisih_bi_fed` | BI-Rate dikurangi Fed Funds Rate |

Target: log return IHSG hari berikutnya. Standardisasi z-score di-fit hanya pada data latih.
LSTM 1 lapis, 16 unit, jendela 10 hari (dipilih `tune.py` dari loss validasi, tanpa melihat data uji),
dropout 0,2, weight decay 1e-4, early stopping (kesabaran 20) pada 15% validasi, ensemble 5 seed.
Pembanding: naif (besok = hari ini), rata-rata historis, ARIMA (orde dipilih AIC data latih), regresi linear.
Uji signifikansi: Diebold-Mariano dengan koreksi HLN.

## Hasil (data sampai 20 November 2026)

Data uji 282 hari (23 Sep 2025 s.d. 20 Nov 2026):

| Model | RMSE | MAE | MAPE | Arah benar | p DM vs LSTM |
|---|---|---|---|---|---|
| LSTM | 116,62 | 79,23 | 1,172% | 47,5% | - |
| Naif | 116,15 | 78,86 | 1,166% | - | 0,105 |
| Rata-rata historis | 116,30 | 78,98 | 1,168% | 46,8% | 0,117 |
| ARIMA(2,0,0) | 116,75 | 78,87 | 1,166% | 47,9% | 0,889 |
| Regresi linear | 117,91 | 80,17 | 1,187% | 49,6% | 0,144 |

Walk-forward 5 lipatan (703 hari uji, Des 2023 s.d. Nov 2026): RMSE LSTM 93,21 vs naif 92,99; LSTM menang di 0 dari 5 lipatan; p DM 0,132.

Kesimpulan: LSTM setara dengan baseline naif; selisihnya tidak signifikan secara statistik. Ini konsisten dengan
hipotesis pasar efisien (return harian mendekati random walk) dan merupakan temuan yang sah untuk dibahas di Bab IV.
Sebelum pembaruan, model lama (fitur level) jauh lebih buruk dari naif (RMSE 127,68 vs 115,88).

## Isi dashboard

1. Prediksi H+1: perkiraan IHSG hari bursa berikutnya, seberapa akurat model di data uji, dan uji skenario BI-Rate/kurs baru.
2. Data pasar: grafik gabungan IHSG, BI-Rate, kurs, dan Fed Rate.
3. Hubungan X dan Y: korelasi, kombinasi BI-Rate × kurs terhadap IHSG pada H0 dan H+1, teori vs data.
4. Evaluasi model: tabel semua pembanding, uji Diebold-Mariano, keunggulan kumulatif, walk-forward.
5. Metodologi: fitur, hyperparameter, teori, batasan, status berkas.

Tampilan mengikuti `DESIGN.md` (IBM Carbon); tema ada di `.streamlit/config.toml`.

## Berkas

    config.py  features.py  modeling.py  lstm_model.py
    fetch_ihsg.py  fetch_kurs.py  fetch_interest_rate.py  build_dataset.py
    tune.py  train.py  evaluate_walkforward.py  predict_realtime.py
    analysis.py  app.py  run_all.py
    data_bawaan/bi_rate.csv     BI-Rate asli, diverifikasi di bi.go.id
    model/                      metrics.json, walkforward.json, tuning.json, test_predictions.csv

## Skill agen AI (folder `.opencode/skills/`)

antislop, antislop-ui, antislop-copywriting, antislop-code, antislop-human, antislop-layoutmobile, ui-ux-pro-max.
Arah desain: `DESIGN.md` (IBM Carbon, dari voltagent/awesome-design-md).

## Batasan

Hasil bersifat statistik untuk penelitian, bukan saran investasi. Korelasi bukan sebab-akibat.
BI-Rate hanya berubah beberapa kali dalam lima tahun, sehingga kombinasi yang melibatkan BI-Rate naik/turun memiliki n kecil.
