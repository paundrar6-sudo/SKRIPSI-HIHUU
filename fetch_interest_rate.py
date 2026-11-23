"""
fetch_interest_rate.py
=======================
Mengambil data suku bunga acuan yang NYATA dari dua sumber:

1. Fed Funds Rate (suku bunga acuan The Fed, Amerika Serikat)
   - Sumber: FRED (Federal Reserve Economic Data), tidak butuh API key,
     diunduh langsung sebagai CSV resmi.
   - URL: https://fred.stlouisfed.org/series/FEDFUNDS

2. BI 7-Day Reverse Repo Rate (suku bunga acuan Bank Indonesia)
   - Sumber: situs resmi Bank Indonesia (bi.go.id).
   - Karena struktur halaman BI dapat berubah sewaktu-waktu, skrip ini
     MENCOBA melakukan scraping otomatis; jika gagal, skrip akan
     menghasilkan file template (bi_rate_manual_template.csv) yang bisa
     Anda isi manual dari halaman resmi BI -- ini untuk menjamin data
     yang masuk ke model tetap 100% data resmi, bukan tebakan.

Instalasi:
    pip install pandas requests beautifulsoup4 --break-system-packages

Jalankan:
    python fetch_interest_rate.py
"""
import io

import pandas as pd
import requests

import config


def fetch_fed_funds_rate():
    """Unduh Fed Funds Rate asli dari FRED (CSV resmi, tanpa API key)."""
    print("Mengunduh Fed Funds Rate dari FRED...")
    resp = requests.get(config.FRED_FEDFUNDS_CSV, timeout=30)
    resp.raise_for_status()

    # FRED menandai data kosong dengan karakter "." (bukan string kosong),
    # yang TIDAK otomatis dikenali pandas sebagai NaN -> harus dinyatakan
    # eksplisit lewat na_values, kalau tidak kolom ini akan jadi tipe teks
    # dan menyebabkan error/NaN tersembunyi di tahap training.
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df.columns = ["date", "fed_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["fed_rate"] = pd.to_numeric(df["fed_rate"], errors="coerce")
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)]
    df["date"] = df["date"].dt.date

    # Isi kekosongan (hari libur/belum terbit) dengan nilai terakhir yang valid
    df["fed_rate"] = df["fed_rate"].ffill().bfill()

    n_missing = df["fed_rate"].isna().sum()
    if n_missing > 0:
        print(f"[peringatan] Masih ada {n_missing} baris fed_rate kosong setelah "
              f"ffill/bfill -- periksa data/fed_funds_rate.csv secara manual.")

    df.to_csv(config.FEDFUNDS_CSV, index=False)
    print(f"Tersimpan {len(df)} baris Fed Funds Rate asli -> {config.FEDFUNDS_CSV}")
    print("Sumber: https://fred.stlouisfed.org/series/FEDFUNDS")
    return df


def try_scrape_bi_rate():
    """Percobaan scraping tabel BI-7Day Reverse Repo Rate dari situs resmi BI.

    CATATAN: struktur HTML situs pemerintah dapat berubah sewaktu-waktu.
    Jika scraping ini gagal (mengembalikan None), gunakan
    generate_manual_template() di bawah dan isi datanya manual dari:
    https://www.bi.go.id/id/statistik/indikator/bi-7day-rr.aspx
    """
    try:
        resp = requests.get(config.BI_RATE_SOURCE_URL, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        # Situs BI biasanya memuat tabel "Tanggal" & "BI-7Day RR Rate (%)"
        for t in tables:
            cols_lower = [str(c).lower() for c in t.columns]
            if any("tanggal" in c or "date" in c for c in cols_lower) and \
               any("rate" in c or "suku bunga" in c for c in cols_lower):
                return t
        return None
    except Exception as ex:
        print(f"[info] Scraping otomatis BI rate gagal: {ex}")
        return None


def generate_manual_template():
    """Buat template CSV kosong yang bisa Anda isi manual dari situs resmi BI,
    supaya data suku bunga BI tetap 100% bersumber dari BI, bukan asumsi."""
    template = pd.DataFrame({
        "date": pd.date_range(config.START_DATE, config.END_DATE, freq="MS").date,
        "bi_rate": [None] * len(pd.date_range(config.START_DATE, config.END_DATE, freq="MS")),
    })
    template.to_csv(config.BI_RATE_MANUAL_TEMPLATE_CSV, index=False)
    print(f"Template dibuat: {config.BI_RATE_MANUAL_TEMPLATE_CSV}")
    print("Silakan isi kolom 'bi_rate' secara manual dari halaman resmi:")
    print(f"  {config.BI_RATE_SOURCE_URL}")
    print("Lalu simpan ulang file tersebut sebagai:", config.BI_RATE_CSV)


def fetch_bi_rate():
    print("Mencoba mengambil BI 7-Day Reverse Repo Rate dari situs resmi BI...")
    df = try_scrape_bi_rate()
    if df is not None and not df.empty:
        df.to_csv(config.BI_RATE_CSV, index=False)
        print(f"Tersimpan data BI Rate asli -> {config.BI_RATE_CSV}")
        print(f"Sumber: {config.BI_RATE_SOURCE_URL}")
        return df
    else:
        print("Scraping otomatis tidak berhasil (kemungkinan struktur halaman "
              "berubah atau diblokir). Membuat template manual sebagai gantinya.")
        generate_manual_template()
        return None


if __name__ == "__main__":
    fetch_fed_funds_rate()
    fetch_bi_rate()
