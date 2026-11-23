"""
fetch_ihsg.py
=============
Mengambil data historis IHSG (^JKSE) yang NYATA dari Yahoo Finance.
Dapat diverifikasi langsung di:
    https://finance.yahoo.com/quote/%5EJKSE/history

Instalasi:
    pip install yfinance pandas --break-system-packages

Jalankan:
    python fetch_ihsg.py
"""
import pandas as pd
import yfinance as yf

import config


def fetch_ihsg(start=config.START_DATE, end=config.END_DATE):
    print(f"Mengunduh data IHSG ({config.IHSG_TICKER}) dari Yahoo Finance...")
    df = yf.download(config.IHSG_TICKER, start=start, end=end, progress=False)

    if df.empty:
        raise RuntimeError(
            "Data IHSG kosong. Cek koneksi internet atau ticker '^JKSE' di Yahoo Finance."
        )

    df = df.reset_index()
    # yfinance kadang mengembalikan kolom MultiIndex -> ratakan
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "ihsg_close"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["ihsg_close"])
    df = df[df["ihsg_close"] > 0].reset_index(drop=True)  # buang baris rusak (harga kosong/0)
    df["ihsg_return"] = df["ihsg_close"].pct_change()

    df.to_csv(config.IHSG_CSV, index=False)
    print(f"Tersimpan {len(df)} baris data IHSG asli -> {config.IHSG_CSV}")
    print("Sumber: https://finance.yahoo.com/quote/%5EJKSE/history")
    return df


if __name__ == "__main__":
    fetch_ihsg()
