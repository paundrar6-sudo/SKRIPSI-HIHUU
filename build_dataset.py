"""
build_dataset.py
=================
Menggabungkan sumber data NYATA menjadi satu deret waktu per hari bursa:

    1. ihsg.csv            (Yahoo Finance)  -> Y  : IHSG
    2. bi_rate.csv         (Bank Indonesia) -> X1 : BI Rate
    3. kurs_usd_idr.csv    (JISDOR BI)      -> X2 : kurs rupiah per dolar AS (cadangan: Yahoo)
    4. fed_funds_rate.csv  (FRED)           -> variabel kontrol

Keluaran:
    data/dataset_features.csv -- semua hari bursa termasuk yang TERBARU
                                  (dipakai untuk prediksi H+1 dan dashboard)
    data/dataset_final.csv    -- hanya hari yang sudah punya "besok"
                                  (dipakai untuk melatih LSTM)

Jalankan (setelah fetch_ihsg.py, fetch_kurs.py, fetch_interest_rate.py):
    python build_dataset.py
"""
import os

import pandas as pd

import config


def _need(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"\n[ERROR] {path} belum ada. {hint}\n")


def load_real_sources():
    _need(config.IHSG_CSV, "Jalankan `python fetch_ihsg.py` (atau `python run_all.py`).")
    _need(config.FEDFUNDS_CSV, "Jalankan `python run_all.py` (unduh Fed Funds Rate dari FRED).")
    _need(config.KURS_CSV, "Jalankan `python fetch_kurs.py` (atau `python run_all.py`).")
    _need(config.BI_RATE_CSV, "Salin data_bawaan/bi_rate.csv ke data/bi_rate.csv (otomatis oleh run_all.py).")

    ihsg = pd.read_csv(config.IHSG_CSV, parse_dates=["date"])
    fed = pd.read_csv(config.FEDFUNDS_CSV, parse_dates=["date"])
    bi = pd.read_csv(config.BI_RATE_CSV, parse_dates=["date"])
    bi_col = [c for c in bi.columns if "rate" in c.lower()][0]
    bi = bi.rename(columns={bi_col: "bi_rate"})[["date", "bi_rate"]]
    kurs = pd.read_csv(config.KURS_CSV, parse_dates=["date"])[["date", "kurs_usd_idr"]]
    return ihsg, fed, bi, kurs


def build_daily_frame(ihsg, fed, bi, kurs):
    df = ihsg.copy().sort_values("date").reset_index(drop=True)
    # merge_asof (backward): tiap hari bursa memakai nilai dari keputusan/data
    # TERAKHIR pada atau sebelum hari itu. Penting karena tanggal keputusan
    # bank sentral bisa jatuh di hari libur bursa (merge biasa akan membuangnya).
    for other in (fed, bi, kurs):
        df = pd.merge_asof(df, other.sort_values("date"), on="date", direction="backward")

    # Pastikan numerik (mis. koma desimal "5,75" -> 5.75)
    df["bi_rate"] = pd.to_numeric(df["bi_rate"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    for col in ("fed_rate", "kurs_usd_idr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("bi_rate", "fed_rate", "kurs_usd_idr"):
        df[col] = df[col].ffill().bfill()
        if df[col].isna().all():
            raise SystemExit(f"\n[ERROR] Kolom '{col}' kosong total. Periksa berkas sumbernya di folder data/.\n")
    return df


def main():
    ihsg, fed, bi, kurs = load_real_sources()
    df = build_daily_frame(ihsg, fed, bi, kurs)

    required = ["ihsg_close", "ihsg_return", "bi_rate", "fed_rate", "kurs_usd_idr"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"[info] Membuang {before - len(df)} baris awal yang masih kosong (wajar untuk return hari pertama).")

    # Semua hari (termasuk hari bursa terbaru yang belum punya "besok")
    df.to_csv(config.FEATURES_CSV, index=False)

    df["ihsg_close_next"] = df["ihsg_close"].shift(-1)
    df = df.dropna(subset=["ihsg_close_next"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("[ERROR] Dataset akhir kosong. Periksa rentang tanggal data IHSG, kurs, dan BI Rate.")

    nan_report = df[required + ["ihsg_close_next"]].isna().sum()
    if nan_report.sum() > 0:
        raise SystemExit(f"[ERROR] Masih ada NaN setelah pembersihan:\n{nan_report}")

    df.to_csv(config.DATASET_CSV, index=False)
    print(f"Dataset latih tersimpan: {config.DATASET_CSV} ({len(df)} baris, 0 NaN)")
    print(f"Dataset fitur terbaru  : {config.FEATURES_CSV} (sampai {df['date'].max().date()} + 1 hari terbaru)")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
