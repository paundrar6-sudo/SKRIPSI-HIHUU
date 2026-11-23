"""
analysis.py
===========
Fungsi analisis hubungan variabel X terhadap IHSG (Y). Dipisahkan dari
dashboard supaya logikanya bisa diuji dan dijelaskan sendiri.

Istilah:
    X1 = BI-Rate                     X2 = kurs USD/IDR (rupiah per 1 dolar AS)
    Kurs NAIK  = rupiah MELEMAH      Kurs TURUN = rupiah MENGUAT
    H0  = hari terjadinya perubahan  H+1 = hari bursa berikutnya

Catatan: semua ini adalah hubungan STATISTIK pada data historis, bukan bukti
sebab-akibat.
"""
import numpy as np
import pandas as pd
from scipy import stats

BLOK_BULANAN = 20          # 20 hari bursa ~ 1 bulan (blok tidak saling tumpang tindih)
MIN_N = 8                  # di bawah ini kesimpulan ditolak (sampel terlalu kecil)
BI_AMBANG = 0.001          # perubahan BI-Rate di atas ini dianggap naik/turun

URUT = ["naik", "turun", "tetap"]
BI_TEKS = {"naik": "BI-Rate naik", "turun": "BI-Rate turun", "tetap": "BI-Rate tetap"}
KURS_TEKS = {"naik": "kurs naik (rupiah melemah)", "turun": "kurs turun (rupiah menguat)",
             "tetap": "kurs relatif tetap"}


def arah(x, ambang):
    """Ubah angka perubahan menjadi 'naik' / 'turun' / 'tetap'."""
    x = np.asarray(x, dtype=float)
    return np.where(x > ambang, "naik", np.where(x < -ambang, "turun", "tetap"))


# ---------------------------------------------------------------------------
# 1. Korelasi perubahan harian
# ---------------------------------------------------------------------------
def perubahan_harian(df: pd.DataFrame) -> pd.DataFrame:
    ch = pd.DataFrame({
        "date": df["date"],
        "Δ IHSG (%)": df["ihsg_close"].pct_change() * 100,
        "Δ Kurs USD/IDR (%)": df["kurs_usd_idr"].pct_change() * 100,
        "Δ BI Rate (poin)": df["bi_rate"].diff(),
        "Δ Fed Rate (poin)": df["fed_rate"].diff(),
    })
    return ch.dropna().reset_index(drop=True)


def kekuatan(r: float) -> str:
    a = abs(r)
    return ("sangat lemah" if a < 0.1 else "lemah" if a < 0.3 else "sedang" if a < 0.5 else "kuat")


def tafsir_korelasi(nama: str, r: float, p: float, n: int) -> str:
    """Kalimat penjelasan dalam bahasa awam untuk satu korelasi X terhadap Δ IHSG."""
    if not np.isfinite(r):
        return (f"{nama}: korelasi tidak dapat dihitung karena nilainya hampir tidak pernah berubah "
                f"pada data ini.")
    signif = "signifikan (p < 0,05)" if p < 0.05 else "TIDAK signifikan (p ≥ 0,05), bisa jadi kebetulan"
    if "Kurs" in nama:
        arti = ("saat kurs naik (rupiah melemah), IHSG cenderung turun" if r < 0
                else "saat kurs naik (rupiah melemah), IHSG cenderung ikut naik")
    elif "BI" in nama:
        arti = ("saat BI-Rate naik, IHSG cenderung turun" if r < 0
                else "saat BI-Rate naik, IHSG cenderung naik")
    else:
        arti = ("saat variabel ini naik, IHSG cenderung turun" if r < 0
                else "saat variabel ini naik, IHSG cenderung naik")
    return f"{nama}: r = {r:+.3f} ({kekuatan(r)}, {'negatif' if r < 0 else 'positif'}), {signif}. Artinya: {arti} (n = {n} hari)."


def tabel_korelasi(ch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kolom, label in [("Δ Kurs USD/IDR (%)", "X2 Kurs USD/IDR"), ("Δ BI Rate (poin)", "X1 BI Rate"),
                         ("Δ Fed Rate (poin)", "Kontrol: Fed Rate")]:
        if ch[kolom].std() == 0:
            r, p = float("nan"), float("nan")
        else:
            r, p = stats.pearsonr(ch[kolom], ch["Δ IHSG (%)"])
        rows.append({"Variabel": label, "r": r, "p-value": p, "n": len(ch),
                     "Kekuatan": kekuatan(r) if np.isfinite(r) else "-",
                     "Penjelasan": tafsir_korelasi(label, r, p, len(ch))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Matriks kombinasi BI-Rate x Kurs -> IHSG (H0 dan H+1)
# ---------------------------------------------------------------------------
def siapkan_pengamatan(df: pd.DataFrame, skala: str = "harian", kurs_ambang: float | None = None) -> pd.DataFrame:
    """Ubah data menjadi pengamatan: perubahan BI-Rate, perubahan kurs, return IHSG
    pada H0 dan H+1.

    skala='harian'  : tiap hari bursa; H0 = return hari itu, H+1 = return hari berikutnya.
    skala='bulanan' : blok 20 hari bursa; H0 = return IHSG sepanjang blok, H+1 = blok berikutnya.
    """
    d = df[["date", "ihsg_close", "bi_rate", "kurs_usd_idr"]].sort_values("date").reset_index(drop=True)
    if skala == "harian":
        kurs_ambang = 0.1 if kurs_ambang is None else kurs_ambang
        obs = pd.DataFrame({
            "date": d["date"],
            "d_bi": d["bi_rate"].diff(),
            "d_kurs": d["kurs_usd_idr"].pct_change() * 100,
            "r0": d["ihsg_close"].pct_change() * 100,
        })
        obs["r1"] = obs["r0"].shift(-1)
    else:
        kurs_ambang = 0.5 if kurs_ambang is None else kurs_ambang
        L = BLOK_BULANAN
        rows = []
        for s in range(0, len(d) - L, L):
            e, nxt = s + L, s + 2 * L
            rows.append({
                "date": d["date"].iloc[e],
                "d_bi": d["bi_rate"].iloc[e] - d["bi_rate"].iloc[s],
                "d_kurs": (d["kurs_usd_idr"].iloc[e] / d["kurs_usd_idr"].iloc[s] - 1) * 100,
                "r0": (d["ihsg_close"].iloc[e] / d["ihsg_close"].iloc[s] - 1) * 100,
                "r1": ((d["ihsg_close"].iloc[nxt] / d["ihsg_close"].iloc[e] - 1) * 100) if nxt < len(d) else np.nan,
            })
        obs = pd.DataFrame(rows)
    obs = obs.dropna(subset=["d_bi", "d_kurs", "r0"]).reset_index(drop=True)
    obs["bi_arah"] = arah(obs["d_bi"], BI_AMBANG)
    obs["kurs_arah"] = arah(obs["d_kurs"], kurs_ambang)
    return obs


def _p(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < MIN_N or x.std() == 0:
        return float("nan")
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def _kesimpulan(mean: float, p: float, n: int, ihsg_ambang: float, hari: str) -> str:
    if n < MIN_N:
        return f"{hari}: sampel terlalu kecil (n={n}), belum bisa disimpulkan"
    if np.isfinite(p) and p < 0.05:
        return f"{hari}: IHSG cenderung {'NAIK' if mean > 0 else 'TURUN'} (signifikan)"
    if abs(mean) >= ihsg_ambang / 2:
        return f"{hari}: condong {'naik' if mean > 0 else 'turun'}, tetapi tidak signifikan"
    return f"{hari}: relatif tetap / tidak ada perubahan yang meyakinkan"


def _lanjutan(m0: float, m1: float, p0: float, p1: float, n1: int, ihsg_ambang: float) -> str:
    """Apakah arah H0 berlanjut, berbalik, atau hilang pada H+1?"""
    if n1 < MIN_N or not np.isfinite(m1):
        return "H+1: belum bisa dinilai"
    cond0 = abs(m0) >= ihsg_ambang / 2 or (np.isfinite(p0) and p0 < 0.05)
    cond1 = abs(m1) >= ihsg_ambang / 2 or (np.isfinite(p1) and p1 < 0.05)
    if not cond0 and not cond1:
        return "IHSG tidak ikut berubah baik di H0 maupun H+1"
    if cond0 and not cond1:
        return "Perubahan di H0 TIDAK berlanjut ke H+1 (hilang)"
    if not cond0 and cond1:
        return "Perubahan baru condong muncul di H+1 (mungkin efek tertunda; cek p-value)"
    return "Perubahan BERLANJUT di H+1" if np.sign(m0) == np.sign(m1) else "Perubahan BERBALIK di H+1"


def tabel_kombinasi(obs: pd.DataFrame, ihsg_ambang: float) -> pd.DataFrame:
    rows = []
    for bi in URUT:
        for ku in URUT:
            sub = obs[(obs["bi_arah"] == bi) & (obs["kurs_arah"] == ku)]
            if sub.empty:
                continue
            r0, r1 = sub["r0"], sub["r1"].dropna()
            m0, m1 = r0.mean(), (r1.mean() if len(r1) else float("nan"))
            p0, p1 = _p(r0), _p(r1)
            rows.append({
                "bi_arah": bi, "kurs_arah": ku,
                "Kondisi": f"{BI_TEKS[bi]} & {KURS_TEKS[ku]}",
                "n": len(sub),
                "Rata-rata ΔIHSG H0 (%)": m0,
                "% hari IHSG naik H0": float((r0 > ihsg_ambang).mean() * 100),
                "% hari IHSG turun H0": float((r0 < -ihsg_ambang).mean() * 100),
                "n H+1": len(r1),
                "Rata-rata ΔIHSG H+1 (%)": m1,
                "% hari IHSG naik H+1": float((r1 > ihsg_ambang).mean() * 100) if len(r1) else float("nan"),
                "% hari IHSG turun H+1": float((r1 < -ihsg_ambang).mean() * 100) if len(r1) else float("nan"),
                "p H0": p0, "p H+1": p1,
                "Kesimpulan H0": _kesimpulan(m0, p0, len(sub), ihsg_ambang, "H0"),
                "Kesimpulan H+1": _kesimpulan(m1, p1, len(r1), ihsg_ambang, "H+1"),
                "Berlanjut ke H+1?": _lanjutan(m0, m1, p0, p1, len(r1), ihsg_ambang),
            })
    return pd.DataFrame(rows)


def cari_pola(tabel: pd.DataFrame, bi: str, kurs: str):
    """Ambil baris pola historis untuk kombinasi tertentu (atau None)."""
    sub = tabel[(tabel["bi_arah"] == bi) & (tabel["kurs_arah"] == kurs)]
    return None if sub.empty else sub.iloc[0]
