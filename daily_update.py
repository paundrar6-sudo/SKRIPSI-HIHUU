"""
daily_update.py
===============
Pipeline near-real-time harian (proposal subbab 3.11):

    1. Ambil data terbaru: IHSG (Yahoo), JISDOR (BI), Fed Funds Rate (FRED)
    2. Catat status tiap sumber (berhasil / gagal / tanggal terakhir)
    3. Gabungkan data (build_dataset.py)
    4. Isi nilai aktual untuk prediksi lama yang tanggal H+1-nya sudah lewat
    5. Buat prediksi H+1 dengan model yang SUDAH DIBEKUKAN (tidak dilatih ulang),
       lalu simpan ke logs/prediksi_harian.csv beserta waktu pembuatannya

Karena model tidak dilatih ulang dan setiap prediksi dicatat sebelum IHSG
H+1 diketahui, log ini adalah uji maju (forward test) yang jujur.

Jalankan manual:          python daily_update.py
Jadwal otomatis (sekali): python daily_update.py --pasang-jadwal 17:30
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

import config  # noqa: E402

KOLOM = ["dibuat_pada", "tanggal_h0", "tanggal_h1", "ihsg_h0", "prediksi_h1", "perubahan_pct",
         "sebaran_pct", "bi_rate", "kurs", "status_data", "aktual_h1", "galat_poin", "galat_naif_poin",
         "arah_benar"]


def ambil(nama, modul, fungsi, path):
    try:
        mod = __import__(modul)
        getattr(mod, fungsi)()
        akhir = pd.read_csv(path, parse_dates=["date"])["date"].max().date()
        return f"{nama} ok s.d. {akhir}"
    except Exception as e:  # catat, jangan hentikan pipeline
        print(f"[peringatan] {nama} gagal: {e}")
        if os.path.exists(path):
            akhir = pd.read_csv(path, parse_dates=["date"])["date"].max().date()
            return f"{nama} GAGAL (pakai data lama s.d. {akhir})"
        return f"{nama} GAGAL"


def baca_log() -> pd.DataFrame:
    if os.path.exists(config.PREDICTION_LOG):
        log = pd.read_csv(config.PREDICTION_LOG, parse_dates=["tanggal_h0", "tanggal_h1"],
                          dtype={"status_data": "object", "dibuat_pada": "object"})
    else:
        log = pd.DataFrame(columns=KOLOM)
    # kolom yang masih kosong terbaca sebagai float; arah_benar perlu menampung True/False
    log["arah_benar"] = log["arah_benar"].astype("object")
    return log


def isi_aktual(log: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Pasangkan prediksi dengan IHSG aktual pada hari bursa pertama setelah H0."""
    for i, row in log[log["aktual_h1"].isna()].iterrows():
        sesudah = feat[feat["date"] > row["tanggal_h0"]]
        if sesudah.empty:
            continue
        nyata = sesudah.iloc[0]
        log.loc[i, "tanggal_h1"] = nyata["date"]
        log.loc[i, "aktual_h1"] = nyata["ihsg_close"]
        log.loc[i, "galat_poin"] = nyata["ihsg_close"] - row["prediksi_h1"]
        log.loc[i, "galat_naif_poin"] = nyata["ihsg_close"] - row["ihsg_h0"]
        arah_pred = np.sign(row["prediksi_h1"] - row["ihsg_h0"])
        log.loc[i, "arah_benar"] = bool(arah_pred == np.sign(nyata["ihsg_close"] - row["ihsg_h0"]))
    return log


def jalankan(tanpa_unduh=False):
    status = []
    if not tanpa_unduh:
        status.append(ambil("IHSG", "fetch_ihsg", "fetch_ihsg", config.IHSG_CSV))
        status.append(ambil("JISDOR", "fetch_kurs", "fetch_kurs", config.KURS_CSV))
        status.append(ambil("Fed", "fetch_interest_rate", "fetch_fed_funds_rate", config.FEDFUNDS_CSV))
    r = subprocess.run([sys.executable, "build_dataset.py"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build_dataset.py gagal:\n{r.stdout}\n{r.stderr}")

    from predict_realtime import load_artifacts, load_features, prediksi_h1
    feat = load_features()
    log = isi_aktual(baca_log(), feat)

    p = prediksi_h1(feat, load_artifacts())
    kurs = pd.read_csv(config.KURS_CSV, parse_dates=["date"])
    if kurs["date"].max() < p["tanggal_h0"]:
        status.append(f"kurs H0 belum terbit, memakai {kurs['date'].max().date()}")
    if (log["tanggal_h0"] == p["tanggal_h0"]).any():
        print(f"Prediksi untuk H0 {p['tanggal_h0'].date()} sudah tercatat; tidak ditimpa.")
    else:
        baru = {"dibuat_pada": datetime.now().isoformat(timespec="seconds"),
                "tanggal_h0": p["tanggal_h0"], "tanggal_h1": p["tanggal_h1"], "ihsg_h0": p["ihsg_h0"],
                "prediksi_h1": p["ihsg_h1"], "perubahan_pct": p["perubahan_pct"],
                "sebaran_pct": p["sebaran_pct"], "bi_rate": p["bi_rate"], "kurs": p["kurs"],
                "status_data": "; ".join(status) or "tanpa unduh"}
        log = pd.concat([log, pd.DataFrame([baru])], ignore_index=True)
        print(f"Prediksi baru: H0 {p['tanggal_h0'].date()} IHSG {p['ihsg_h0']:,.2f} -> "
              f"H+1 {p['tanggal_h1'].date()} {p['ihsg_h1']:,.2f} ({p['perubahan_pct']:+.3f}%)")

    os.makedirs(config.LOG_DIR, exist_ok=True)
    log[KOLOM].to_csv(config.PREDICTION_LOG, index=False)
    selesai = log.dropna(subset=["aktual_h1"])
    if len(selesai):
        print(f"Rekam jejak {len(selesai)} hari: MAE LSTM {selesai['galat_poin'].abs().mean():.2f} vs "
              f"naif {selesai['galat_naif_poin'].abs().mean():.2f} poin")
    print("Status data:", "; ".join(status) or "-")


def pasang_jadwal(jam: str):
    """Daftarkan tugas harian Senin-Jumat di Windows Task Scheduler."""
    py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        py = sys.executable
    os.makedirs(os.path.join(HERE, config.LOG_DIR), exist_ok=True)
    bat = os.path.join(HERE, "daily_update.bat")
    with open(bat, "w", encoding="utf-8") as f:
        f.write(f'@echo off\r\ncd /d "{HERE}"\r\nset PYTHONIOENCODING=utf-8\r\n'
                f'"{py}" daily_update.py >> "{os.path.join(HERE, config.LOG_DIR, "daily_update.log")}" 2>&1\r\n')
    cmd = ["schtasks", "/Create", "/F", "/TN", "SkripsiIHSG_PrediksiHarian", "/SC", "WEEKLY",
           "/D", "MON,TUE,WED,THU,FRI", "/ST", jam, "/TR", f'"{bat}"']
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout or r.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tanpa-unduh", action="store_true", help="pakai data yang sudah ada")
    ap.add_argument("--pasang-jadwal", metavar="HH:MM", help="daftarkan ke Windows Task Scheduler")
    a = ap.parse_args()
    if a.pasang_jadwal:
        pasang_jadwal(a.pasang_jadwal)
    else:
        jalankan(a.tanpa_unduh)
