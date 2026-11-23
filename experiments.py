"""
experiments.py
==============
Eksperimen untuk menjawab rumusan masalah no. 3 dan hipotesis H1-H3 (proposal
Tabel 3.2): apakah BI-Rate dan/atau kurs USD/IDR memperbaiki prediksi IHSG
dibandingkan model yang hanya memakai riwayat IHSG?

Bagian:
    ablation     M0-M4 pada pembagian latih/uji yang sama seperti train.py
    walkforward  M0-M4 pada 5 lipatan walk-forward (lebih kuat dari satu pembagian)
    horizon      M0 dan M3 untuk H+1, H+5, H+20 hari bursa
    importance   permutation importance kelompok fitur M3 pada data uji

Semua konfigurasi (panjang jendela, ukuran LSTM, seed) sama dengan config.py dan
ditetapkan sebelum data uji dilihat. Hasil: model/experiments.json

Jalankan:
    python experiments.py                 # semua bagian
    python experiments.py ablation horizon
"""
import json
import os
import sys
import time

import numpy as np

import config
import features as F
import modeling as M

NAIF = "Naif (besok = hari ini)"
HORIZONS = [1, 5, 20]
N_PERM = 30


def _fit_predict(df, a, b, cols, horizon=1):
    sp = M.split_sequences(df, a, b, config.SEQUENCE_LENGTH, columns=cols, horizon=horizon)
    models, _, _ = M.fit_ensemble(sp)
    pred = M.predict_ensemble(models, sp["X_te"]) * sp["target_std"]
    return sp, models, pred


def _ringkas(pred, act, last, h=1, pembanding=None):
    r = M.score(pred, act, last)
    r["dm_vs_naif"] = M.diebold_mariano(act - pred, act, h)          # naif: return 0
    r["pesaran_timmermann"] = M.pesaran_timmermann(pred, act)
    if pembanding is not None:
        r["dm_vs_m0"] = M.diebold_mariano(act - pred, act - pembanding, h)
    return r


def ablation(df):
    n = len(df)
    a = int(n * config.TRAIN_SPLIT)
    hasil, preds = {}, {}
    for kode, (nama, cols) in F.SKENARIO.items():
        sp, _, pred = _fit_predict(df, a, n, cols)
        rows = sp["rows_te"]
        act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
        hasil[kode] = {"nama": nama, "fitur": cols,
                       **_ringkas(pred, act, last, pembanding=preds.get("M0"))}
        preds[kode] = pred
        r = hasil[kode]
        print(f"  {kode} {nama:<48} RMSE {r['rmse']:.2f} | DM p vs naif {r['dm_vs_naif']['p_value']:.3f}"
              + (f" | vs M0 {r['dm_vs_m0']['p_value']:.3f}" if "dm_vs_m0" in r else ""))
    naif = M.score(np.zeros(len(act)), act, last)
    return {"models": hasil, "naif": naif, "n_test": int(len(act)),
            "test_start": str(df["date"].iloc[rows[0]].date()), "test_end": str(df["date"].iloc[rows[-1]].date())}


def walkforward(df):
    n = len(df)
    edges = np.linspace(int(n * 0.5), n, config.WALKFORWARD_FOLDS + 1).astype(int)
    per = {k: {"pred": [], "rmse": []} for k in F.SKENARIO}
    act_all, last_all, folds = [], [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
        fold = {"fold": i}
        for kode, (_, cols) in F.SKENARIO.items():
            sp, _, pred = _fit_predict(df, a, b, cols)
            rows = sp["rows_te"]
            act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
            per[kode]["pred"].append(pred)
            fold[kode] = M.score(pred, act, last)["rmse"]
        fold["naif"] = M.score(np.zeros(len(act)), act, last)["rmse"]
        fold.update(start=str(df["date"].iloc[rows[0]].date()), end=str(df["date"].iloc[rows[-1]].date()),
                    n_test=int(len(rows)))
        act_all.append(act)
        last_all.append(last)
        folds.append(fold)
        print(f"  lipatan {i}: " + " | ".join(f"{k} {fold[k]:.2f}" for k in [*F.SKENARIO, "naif"]))
    act, last = np.concatenate(act_all), np.concatenate(last_all)
    m0 = np.concatenate(per["M0"]["pred"])
    gab = {}
    for kode in F.SKENARIO:
        p = np.concatenate(per[kode]["pred"])
        gab[kode] = _ringkas(p, act, last, pembanding=None if kode == "M0" else m0)
        gab[kode]["menang_vs_naif"] = int(sum(f[kode] < f["naif"] for f in folds))
        gab[kode]["menang_vs_m0"] = int(sum(f[kode] < f["M0"] for f in folds)) if kode != "M0" else None
    gab["naif"] = M.score(np.zeros(len(act)), act, last)
    return {"folds": folds, "combined": gab, "n_test": int(len(act))}


def horizon(_df_unused):
    out = {}
    for h in HORIZONS:
        df = M.load_frame(h)
        n = len(df)
        a = int(n * config.TRAIN_SPLIT)
        res, m0 = {}, None
        for kode in ("M0", "M3"):
            sp, _, pred = _fit_predict(df, a, n, F.SKENARIO[kode][1], horizon=h)
            rows = sp["rows_te"]
            act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
            res[kode] = _ringkas(pred, act, last, h, pembanding=m0)
            m0 = pred if kode == "M0" else m0
        res["naif"] = M.score(np.zeros(len(act)), act, last)
        res["n_test"] = int(len(act))
        out[f"H+{h}"] = res
        print(f"  H+{h:<2}: RMSE M0 {res['M0']['rmse']:.2f} | M3 {res['M3']['rmse']:.2f} | naif "
              f"{res['naif']['rmse']:.2f} | DM p M3 vs naif {res['M3']['dm_vs_naif']['p_value']:.3f}")
    return out


def importance(df):
    """Kenaikan RMSE (poin IHSG) saat satu kelompok fitur diacak antar-sampel uji.
    Nilai ~0 atau negatif = model praktis tidak memakai fitur itu."""
    n = len(df)
    a = int(n * config.TRAIN_SPLIT)
    cols = F.SKENARIO["M3"][1]
    sp, models, pred = _fit_predict(df, a, n, cols)
    rows = sp["rows_te"]
    act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
    base = M.score(pred, act, last)["rmse"]
    rng = np.random.default_rng(config.RANDOM_SEED)
    grup = {"Riwayat IHSG": F.GRUP["ihsg"], "BI-Rate": F.GRUP["bi"], "Kurs USD/IDR": F.GRUP["kurs"]}
    out = {}
    for nama, g in grup.items():
        idx = [cols.index(c) for c in g]
        naik = []
        for _ in range(N_PERM):
            X = sp["X_te"].copy()
            perm = rng.permutation(len(X))
            X[:, :, idx] = X[perm][:, :, idx]
            p = M.predict_ensemble(models, X) * sp["target_std"]
            naik.append(M.score(p, act, last)["rmse"] - base)
        out[nama] = {"rmse_naik_mean": float(np.mean(naik)), "rmse_naik_std": float(np.std(naik))}
        print(f"  {nama:<14}: RMSE naik {np.mean(naik):+.3f} +- {np.std(naik):.3f} poin")
    return {"base_rmse": base, "groups": out, "n_perm": N_PERM}


BAGIAN = {"ablation": ablation, "walkforward": walkforward, "horizon": horizon, "importance": importance}


def main():
    pilih = [a for a in sys.argv[1:] if a in BAGIAN] or list(BAGIAN)
    hasil = {}
    if os.path.exists(config.EXPERIMENTS_PATH):
        with open(config.EXPERIMENTS_PATH, encoding="utf-8") as f:
            hasil = json.load(f)
    df = M.load_frame()
    for nama in pilih:
        t = time.time()
        print(f"\n=== {nama} ===")
        hasil[nama] = BAGIAN[nama](df)
        print(f"  ({time.time() - t:.0f} detik)")
    hasil["config"] = {"sequence_length": config.SEQUENCE_LENGTH, "hidden_size": config.HIDDEN_SIZE,
                       "seeds": config.SEEDS, "data_end": str(df["date"].iloc[-1].date())}
    with open(config.EXPERIMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nTersimpan: {config.EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
