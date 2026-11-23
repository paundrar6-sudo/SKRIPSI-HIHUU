"""
fetch_kurs.py
=============
Mengambil kurs rupiah terhadap dolar AS (USD/IDR) = variabel X2 skripsi.
Nilai 16.000 berarti Rp16.000 per 1 dolar AS; kurs naik = rupiah melemah.

Sumber utama: JISDOR Bank Indonesia (kurs referensi resmi, sesuai proposal Bab III).
    https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx
    Diunduh lewat tombol "Unduh" (Excel) di halaman tersebut.

Cadangan: Yahoo Finance (IDR=X), hanya bila situs BI tidak bisa diakses.
    Data Yahoo memuat lonjakan semu (mis. -3,7% lalu kembali esok harinya) yang
    tidak ada di JISDOR, jadi cadangan ini ditandai jelas di kolom `sumber`.

Jalankan:
    python fetch_kurs.py
"""
import io
import re
from datetime import date
from html import unescape

import pandas as pd
import requests

import config

# Rentang wajar rupiah per dolar (2021-2026) untuk mendeteksi data terbalik/rusak
KURS_MIN, KURS_MAX = 10_000, 25_000
JISDOR_URL = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx"
_P = "ctl00$ctl51$g_8e393016_c37f_48c2_9aa2_ab195a9c1128$ctl00$"


def _form_fields(html: str) -> dict:
    out = {}
    for tag in re.findall(r"<input[^>]*>", html):
        name = re.search(r'name="([^"]+)"', tag)
        val = re.search(r'value="([^"]*)"', tag)
        typ = re.search(r'type="([^"]+)"', tag)
        if name and (not typ or typ.group(1) in ("hidden", "text")):
            out[unescape(name.group(1))] = unescape(val.group(1)) if val else ""
    return out


def _with_dates(fields: dict, start: str, end: str) -> dict:
    d1 = pd.Timestamp(start).strftime("%d/%m/%Y")
    d2 = min(pd.Timestamp(end), pd.Timestamp(date.today())).strftime("%d/%m/%Y")
    fields = dict(fields)
    fields[_P + "TextBoxFrom"] = fields[_P + "HiddenFieldDateFrom"] = d1
    fields[_P + "TextBoxDateTo"] = fields[_P + "HiddenFieldDateTo"] = d2
    return fields


def fetch_jisdor(start=config.START_DATE, end=config.END_DATE) -> pd.DataFrame:
    """Unduh seri JISDOR dari situs BI (form ASP.NET: Cari, lalu Unduh Excel)."""
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    page = s.get(JISDOR_URL, timeout=90).text
    form = _with_dates(_form_fields(page), start, end)
    form[_P + "ButtonCari"] = "Cari"
    page = s.post(JISDOR_URL, data=form, timeout=120).text

    form = _with_dates(_form_fields(page), start, end)
    form["__EVENTTARGET"], form["__EVENTARGUMENT"] = _P + "ButtonExport", ""
    resp = s.post(JISDOR_URL, data=form, timeout=180)
    if "spreadsheet" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError("Situs BI tidak mengembalikan berkas Excel JISDOR.")

    x = pd.read_excel(io.BytesIO(resp.content), header=None)
    head = x.index[x[1].astype(str).str.strip().str.lower() == "tanggal"][0]
    x = x.iloc[head + 1:, [1, 2]].dropna()
    kurs = (x[2].astype(str).str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False).astype(float))
    df = pd.DataFrame({"date": pd.to_datetime(x[1].astype(str), format="%m/%d/%Y %I:%M:%S %p").dt.date,
                       "kurs_usd_idr": kurs.values})
    df["sumber"] = "JISDOR"
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_yahoo(start=config.START_DATE, end=config.END_DATE) -> pd.DataFrame:
    import yfinance as yf
    for ticker in config.KURS_TICKERS:
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if not raw.empty:
            break
    else:
        raise RuntimeError("Data kurs Yahoo kosong untuk " + ", ".join(config.KURS_TICKERS))
    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "kurs_usd_idr"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["kurs_usd_idr"])
    # buang titik yang menyimpang >5% dari median 5 hari di sekitarnya (salah cetak Yahoo)
    med = df["kurs_usd_idr"].rolling(5, center=True, min_periods=1).median()
    df = df[(df["kurs_usd_idr"] / med - 1).abs() <= 0.05].copy()
    df["sumber"] = f"Yahoo {ticker}"
    return df.reset_index(drop=True)


def fetch_kurs(start=config.START_DATE, end=config.END_DATE):
    print("Mengunduh kurs JISDOR dari Bank Indonesia...")
    try:
        df = fetch_jisdor(start, end)
    except Exception as e:
        print(f"[peringatan] JISDOR gagal diunduh ({e}). Memakai cadangan Yahoo Finance.")
        df = fetch_yahoo(start, end)

    med = df["kurs_usd_idr"].median()
    if not (KURS_MIN <= med <= KURS_MAX):
        raise RuntimeError(f"Median kurs {med:,.2f} di luar rentang wajar {KURS_MIN:,}-{KURS_MAX:,}.")

    df.to_csv(config.KURS_CSV, index=False)
    print(f"Tersimpan {len(df)} hari kurs USD/IDR ({df['sumber'].iloc[-1]}) -> {config.KURS_CSV}")
    print(f"Rentang: {df['date'].min()} s.d. {df['date'].max()} | "
          f"Rp{df['kurs_usd_idr'].min():,.0f} - Rp{df['kurs_usd_idr'].max():,.0f} per USD")
    return df


if __name__ == "__main__":
    fetch_kurs()
