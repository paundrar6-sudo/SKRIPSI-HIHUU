"""
fetch_kurs.py
=============
Mengambil kurs rupiah terhadap dolar AS (USD/IDR) yang NYATA dari Yahoo Finance.
Ini adalah variabel X2 skripsi (X1 = BI Rate). Nilai 16.000 berarti
Rp16.000 per 1 dolar AS; kurs naik = rupiah melemah.

Verifikasi manual:
    https://finance.yahoo.com/quote/IDR%3DX/history

Catatan: kurs resmi Bank Indonesia adalah JISDOR, tetapi BI tidak menyediakan
unduhan otomatis yang stabil. Yahoo Finance dipakai karena bisa diunduh
otomatis; selisihnya dengan JISDOR kecil. Sebutkan sumber ini di Bab III.

Jalankan:
    python fetch_kurs.py
"""
import pandas as pd
import yfinance as yf

import config

# Rentang wajar rupiah per dolar (2021-2026) untuk mendeteksi data terbalik/rusak
KURS_MIN, KURS_MAX = 10_000, 25_000


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["kurs_usd_idr"])
    df = df[df["kurs_usd_idr"] > 0].copy()
    # Yahoo kadang menyisipkan satu titik "salah cetak" pada data valas.
    # Buang titik yang menyimpang >5% dari median 5 hari di sekitarnya.
    med = df["kurs_usd_idr"].rolling(5, center=True, min_periods=1).median()
    outlier = (df["kurs_usd_idr"] / med - 1).abs() > 0.05
    if outlier.any():
        print(f"[info] Membuang {int(outlier.sum())} titik kurs yang menyimpang >5% dari sekitarnya.")
    return df[~outlier].reset_index(drop=True)


def fetch_kurs(start=config.START_DATE, end=config.END_DATE):
    print("Mengunduh kurs USD/IDR dari Yahoo Finance...")
    raw = None
    for ticker in config.KURS_TICKERS:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if not data.empty:
            print(f"Memakai simbol: {ticker}")
            raw = data
            break
    if raw is None:
        raise RuntimeError(
            "Data kurs kosong untuk simbol " + ", ".join(config.KURS_TICKERS) +
            ". Cek koneksi internet, atau unduh manual dari "
            "https://finance.yahoo.com/quote/IDR%3DX/history"
        )

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "kurs_usd_idr"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = _clean(df)

    med = df["kurs_usd_idr"].median()
    if not (KURS_MIN <= med <= KURS_MAX):
        raise RuntimeError(
            f"Median kurs {med:,.2f} di luar rentang wajar {KURS_MIN:,}-{KURS_MAX:,} "
            f"rupiah per dolar -- kemungkinan simbol salah atau data terbalik."
        )

    df.to_csv(config.KURS_CSV, index=False)
    print(f"Tersimpan {len(df)} baris kurs USD/IDR asli -> {config.KURS_CSV}")
    print(f"Rentang: Rp{df['kurs_usd_idr'].min():,.0f} - Rp{df['kurs_usd_idr'].max():,.0f} per USD")
    print("Sumber: https://finance.yahoo.com/quote/IDR%3DX/history")
    return df


if __name__ == "__main__":
    fetch_kurs()
