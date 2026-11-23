"""
config.py
=========
Konfigurasi terpusat untuk proyek:
    "ANALISIS REAL-TIME BI-RATE DAN NILAI TUKAR RUPIAH TERHADAP DOLLAR AS
     MENGGUNAKAN LONG SHORT-TERM MEMORY (LSTM) UNTUK PREDIKSI IHSG"

Semua skrip lain mengimpor nilai dari file ini, supaya kalau Anda ingin
mengganti rentang tanggal, path folder, atau parameter model, cukup ubah
di SATU tempat ini saja.
"""
import os

# --------------------------------------------------------------------------
# Rentang waktu penelitian (5 tahun terakhir)
# --------------------------------------------------------------------------
START_DATE = "2021-01-01"
END_DATE = "2026-12-31"

# --------------------------------------------------------------------------
# Sumber data REAL (bukan simulasi)
# --------------------------------------------------------------------------
IHSG_TICKER = "^JKSE"                      # Yahoo Finance - IHSG
# Kurs rupiah per 1 dolar AS (USD/IDR) dari Yahoo Finance. "IDR=X" adalah
# simbol standar; "USDIDR=X" dipakai sebagai cadangan bila yang pertama kosong.
KURS_TICKERS = ["IDR=X", "USDIDR=X"]
# Fed Funds Rate (suku bunga acuan AS) - diambil langsung dari FRED
# (Federal Reserve Economic Data), tidak butuh API key.
FRED_FEDFUNDS_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
# BI 7-Day Reverse Repo Rate (suku bunga acuan Bank Indonesia)
BI_RATE_SOURCE_URL = "https://www.bi.go.id/id/statistik/indikator/bi-7day-rr.aspx"

# --------------------------------------------------------------------------
# Folder & file data
# --------------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IHSG_CSV = os.path.join(DATA_DIR, "ihsg.csv")
FEDFUNDS_CSV = os.path.join(DATA_DIR, "fed_funds_rate.csv")
KURS_CSV = os.path.join(DATA_DIR, "kurs_usd_idr.csv")
BI_RATE_CSV = os.path.join(DATA_DIR, "bi_rate.csv")
BI_RATE_MANUAL_TEMPLATE_CSV = os.path.join(DATA_DIR, "bi_rate_manual_template.csv")
DATASET_CSV = os.path.join(DATA_DIR, "dataset_final.csv")        # untuk training (ada target besok)
FEATURES_CSV = os.path.join(DATA_DIR, "dataset_features.csv")    # termasuk hari terbaru (untuk prediksi)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_ihsg.pt")            # ensemble: daftar state_dict
TRAIN_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.png")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
TEST_PRED_PATH = os.path.join(MODEL_DIR, "test_predictions.csv")
WALKFORWARD_PATH = os.path.join(MODEL_DIR, "walkforward.json")
TUNING_PATH = os.path.join(MODEL_DIR, "tuning.json")

# --------------------------------------------------------------------------
# Parameter model LSTM
# --------------------------------------------------------------------------
# Daftar & definisi fitur ada di features.py (fitur stasioner: return & selisih).
from features import FEATURE_COLUMNS  # noqa: E402

SEQUENCE_LENGTH = 10       # jumlah hari bursa historis yang dilihat LSTM (dipilih lewat tune.py)
HIDDEN_SIZE = 16           # dipilih lewat tune.py (loss validasi terendah)
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4        # regularisasi L2
BATCH_SIZE = 32
EPOCHS = 200               # batas atas; pelatihan berhenti lebih awal (early stopping)
PATIENCE = 20              # berhenti bila loss validasi tidak membaik selama N epoch

TRAIN_SPLIT = 0.8          # 80% awal = latih (+validasi), 20% akhir = uji
VAL_SPLIT = 0.15           # 15% terakhir dari bagian latih dipakai untuk validasi/early stopping
SEEDS = [42, 7, 2024, 11, 99]   # ensemble: rata-rata 5 model dengan seed berbeda

RANDOM_SEED = SEEDS[0]
WALKFORWARD_FOLDS = 5      # jumlah lipatan evaluasi walk-forward (expanding window)
