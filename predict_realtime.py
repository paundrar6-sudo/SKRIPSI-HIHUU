"""
predict_realtime.py
====================
Prediksi IHSG H+1 (hari bursa berikutnya) memakai ensemble LSTM yang sudah
dilatih (train.py). "Real-time" berarti: data pasar terbaru diambil saat sistem
dijalankan, dan pengguna dapat MEMASUKKAN DATA BARU (mis. BI-Rate hasil rapat
hari ini atau kurs terkini) untuk melihat bagaimana prediksi H+1 berubah.

Dua cara memasukkan data baru:
    perbarui : mengganti nilai pada hari bursa TERAKHIR (H0)
    tambah   : menambahkan satu hari bursa baru sebagai H0

Contoh:
    python predict_realtime.py                                   # data terbaru apa adanya
    python predict_realtime.py --bi-rate 6.25 --kurs 18500       # skenario BI-Rate naik & rupiah melemah
    python predict_realtime.py --kurs 18500 --ihsg 6900 --tambah-hari
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

import config
import features as F
from lstm_model import LSTMPredictor


def load_artifacts():
    scalers = joblib.load(config.SCALER_PATH)
    if "feature_scaler" not in scalers:
        raise SystemExit("Model lama (fitur level) terdeteksi. Jalankan ulang `python train.py` "
                         "agar model dilatih dengan fitur stasioner terbaru.")
    states = torch.load(config.MODEL_PATH, map_location="cpu")
    models = []
    for s in states if isinstance(states, list) else [states]:
        m = LSTMPredictor()
        m.load_state_dict(s)
        m.eval()
        models.append(m)
    return models, scalers["feature_scaler"], scalers["target_std"]


def load_features() -> pd.DataFrame:
    # dataset_features.csv memuat hari bursa TERBARU (dataset_final.csv berhenti sehari lebih awal)
    path = config.FEATURES_CSV if os.path.exists(config.FEATURES_CSV) else config.DATASET_CSV
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def buat_riwayat(feat: pd.DataFrame, ihsg_close=None, bi_rate=None, kurs=None, fed_rate=None,
                 mode: str = "perbarui", tanggal=None) -> pd.DataFrame:
    """Ambil data mentah secukupnya, terapkan data baru (jika ada) pada H0, lalu
    hitung ulang fitur. Hasil: SEQUENCE_LENGTH baris terakhir lengkap dengan fitur."""
    L = config.SEQUENCE_LENGTH
    h = feat.sort_values("date").tail(L + F.WARMUP + 1).reset_index(drop=True)
    h = h[["date", "ihsg_close", "bi_rate", "kurs_usd_idr", "fed_rate"]].copy()
    if mode == "tambah":
        baru = h.iloc[[-1]].copy()
        baru["date"] = pd.Timestamp(tanggal) if tanggal is not None else h["date"].iloc[-1] + pd.tseries.offsets.BDay(1)
        h = pd.concat([h, baru], ignore_index=True)
    i = len(h) - 1                      # indeks H0
    for col, val in (("ihsg_close", ihsg_close), ("bi_rate", bi_rate),
                     ("kurs_usd_idr", kurs), ("fed_rate", fed_rate)):
        if val is not None:
            h.loc[i, col] = float(val)
    return F.add_features(h).tail(L).reset_index(drop=True)


def prediksi(hist: pd.DataFrame, models, scaler, target_std) -> dict:
    """Jalankan ensemble LSTM pada riwayat (SEQUENCE_LENGTH baris) -> prediksi IHSG H+1."""
    seq = scaler.transform(hist).reshape(1, config.SEQUENCE_LENGTH, -1)
    with torch.no_grad():
        xt = torch.tensor(seq)
        outs = np.array([float(m(xt)) for m in models]) * target_std
    ret = float(outs.mean())
    h0 = float(hist["ihsg_close"].iloc[-1])
    h1 = h0 * np.exp(ret)
    pct = (h1 - h0) / h0 * 100
    return {
        "tanggal_h0": hist["date"].iloc[-1],
        "awal_input": hist["date"].iloc[0],
        "tanggal_h1": hist["date"].iloc[-1] + pd.tseries.offsets.BDay(1),
        "ihsg_h0": h0, "ihsg_h1": h1, "selisih_poin": h1 - h0, "perubahan_pct": pct,
        "arah": "NAIK" if pct > 0 else "TURUN" if pct < 0 else "TETAP",
        # sebaran antar-anggota ensemble: ukuran (kasar) ketidakpastian model
        "sebaran_pct": float((np.exp(outs).std()) * 100),
        "bi_rate": float(hist["bi_rate"].iloc[-1]), "kurs": float(hist["kurs_usd_idr"].iloc[-1]),
        "fed_rate": float(hist["fed_rate"].iloc[-1]),
    }


def prediksi_h1(feat, artifacts, **data_baru) -> dict:
    models, scaler, target_std = artifacts
    return prediksi(buat_riwayat(feat, **data_baru), models, scaler, target_std)


def _cetak(judul, r):
    print(f"{judul}")
    print(f"  H0 ({r['tanggal_h0'].date()}): IHSG {r['ihsg_h0']:,.2f} | BI-Rate {r['bi_rate']:.2f}% | "
          f"Kurs Rp{r['kurs']:,.0f} | Fed {r['fed_rate']:.2f}%")
    print(f"  H+1 ({r['tanggal_h1'].date()}): prediksi IHSG {r['ihsg_h1']:,.2f} "
          f"({r['selisih_poin']:+,.2f} poin, {r['perubahan_pct']:+.2f}%, sebaran ensemble "
          f"+-{r['sebaran_pct']:.2f}%) -> {r['arah']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ihsg", type=float, help="IHSG penutupan H0 (opsional)")
    ap.add_argument("--bi-rate", type=float, help="BI-Rate (%%) H0")
    ap.add_argument("--kurs", type=float, help="Kurs USD/IDR (Rp per dolar) H0")
    ap.add_argument("--fed", type=float, help="Fed Funds Rate (%%) H0")
    ap.add_argument("--tambah-hari", action="store_true", help="tambahkan sebagai hari bursa baru (default: perbarui hari terakhir)")
    a = ap.parse_args()

    art = load_artifacts()
    feat = load_features()
    dasar = prediksi_h1(feat, art)
    _cetak("=== Tanpa data baru (data pasar terbaru apa adanya) ===", dasar)
    if any(v is not None for v in (a.ihsg, a.bi_rate, a.kurs, a.fed)) or a.tambah_hari:
        skenario = prediksi_h1(feat, art, ihsg_close=a.ihsg, bi_rate=a.bi_rate, kurs=a.kurs,
                               fed_rate=a.fed, mode="tambah" if a.tambah_hari else "perbarui")
        _cetak("\n=== Dengan data baru Anda ===", skenario)
        print(f"\nSelisih prediksi H+1 akibat data baru: {skenario['ihsg_h1'] - dasar['ihsg_h1']:+,.2f} poin")


if __name__ == "__main__":
    main()
