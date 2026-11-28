"""
simulasi.py
===========
Uji coba 70/30: membaca KEKUATAN pergerakan IHSG berdasarkan kondisi BI-Rate dan
kurs USD/IDR, lalu menguji apakah pola itu (dan model LSTM) tetap benar di data
yang belum pernah dilihat.

    70% data awal  = data LATIH  : pola dipelajari & LSTM dilatih (15% akhirnya validasi)
    30% data akhir = data UJIAN  : pola dicek ulang & LSTM dinilai (tidak disentuh saat latih)

Kelas pergerakan IHSG per hari bursa (perubahan harga penutupan):
    turun kuat  < -1%        naik kuat  > +1%
    turun lemah -1% s.d. -0,25%   naik lemah +0,25% s.d. +1%
    datar       -0,25% s.d. +0,25%

Kondisi variabel X pada hari t (H0):
    kurs   : naik (rupiah melemah, > +0,25%), turun (rupiah menguat, < -0,25%), tetap
    BI-Rate: siklus kebijakan = arah keputusan RDG terakhir (pengetatan / pelonggaran);
             hari keputusan RDG dilaporkan terpisah karena jumlahnya sedikit

Bagian:
    1. Distribusi kelas IHSG di data latih vs ujian
    2. Pola kondisi X -> IHSG pada H0 (hari yang sama) dan H+1 (besok), latih vs ujian
    3. Kekuatan lanjutan: setelah IHSG naik/turun kuat, apa yang terjadi besok?
    4. LSTM dilatih ulang dengan 70/30 (hyperparameter dipilih ulang dari validasi
       di dalam 70% latih), dinilai di 30% ujian: akurasi, kelas, kuintil prediksi,
       dan bacaan LSTM per kondisi X

Jalankan:
    python simulasi.py               # semua (LSTM ~10 menit)
    python simulasi.py --tanpa-lstm  # hanya analisis pola (beberapa detik)
Hasil: model/simulasi.json
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import config
import features as F
import modeling as M

PORSI_LATIH = 0.70
KUAT, DATAR = 1.0, 0.25          # ambang kelas IHSG (%)
KURS_AMBANG = 0.25               # ambang kurs naik/turun (%)
MIN_N = 8                        # di bawah ini tidak disimpulkan
HAC_LAG = 5                      # lag Newey-West (1 minggu bursa)
VOL_LSTM = 20                    # jendela volatilitas untuk diagnosis LSTM (hari)
KELAS = ["turun kuat", "turun lemah", "datar", "naik lemah", "naik kuat"]
KURS = ["naik", "tetap", "turun"]
SIKLUS = ["pengetatan", "pelonggaran"]
SEQ_GRID, HIDDEN_GRID, TUNE_SEEDS = [10, 20, 40], [16, 32], config.SEEDS[:3]
PATH = os.path.join(config.MODEL_DIR, "simulasi.json")


def kelas(r):
    r = np.asarray(r, dtype=float)
    return np.select([r < -KUAT, r < -DATAR, r <= DATAR, r <= KUAT], KELAS[:4], KELAS[4])


def siapkan():
    df = M.load_frame()
    d = pd.DataFrame({"date": df["date"], "ihsg": df["ihsg_close"],
                      "r0": np.expm1(df["ret_ihsg"]) * 100, "r1": np.expm1(df["target"]) * 100,
                      "kurs_pct": np.expm1(df["ret_kurs"]) * 100, "d_bi": df["d_bi"]})
    d["kelas0"], d["kelas1"] = kelas(d["r0"]), kelas(d["r1"])
    d["kurs"] = np.where(d["kurs_pct"] > KURS_AMBANG, "naik", np.where(d["kurs_pct"] < -KURS_AMBANG, "turun", "tetap"))
    arah = np.sign(d["d_bi"]).replace(0, np.nan).ffill()
    d["siklus"] = arah.map({1.0: "pengetatan", -1.0: "pelonggaran"})
    d["rdg"] = np.where(d["d_bi"] > 0, "naik", np.where(d["d_bi"] < 0, "turun", "-"))
    a = int(len(d) * PORSI_LATIH)
    d["bagian"] = np.where(np.arange(len(d)) < a, "latih", "ujian")
    return df, d, a


def _rentang(x):
    return {"mulai": str(x["date"].iloc[0].date()), "akhir": str(x["date"].iloc[-1].date()), "n": int(len(x))}


def distribusi(d):
    out = {}
    for b, x in d.groupby("bagian"):
        r = x["r0"]
        out[b] = {**_rentang(x), "std": float(r.std()),
                  "kelas_pct": {k: float((x["kelas0"] == k).mean() * 100) for k in KELAS},
                  "rata_naik": float(r[r > 0].mean()), "rata_turun": float(r[r < 0].mean()),
                  "pct_naik": float((r > 0).mean() * 100)}
        # asimetri: apakah hari turun lebih dalam daripada hari naik?
        out[b]["asimetri_t"] = stats.ttest_ind(r[r > 0], -r[r < 0], equal_var=False)._asdict()
        out[b]["asimetri_t"] = {k: float(v) for k, v in out[b]["asimetri_t"].items() if k in ("statistic", "pvalue")}
    return out


def _sel(x, kol, hari, dasar):
    """Ringkas satu kelompok hari: rata-rata return, % naik/turun kuat, uji vs sisa hari."""
    r = x[kol].dropna()
    kel = x["kelas0" if hari == "H0" else "kelas1"].loc[r.index]
    row = {"n": int(len(r)), "rata": float(r.mean()) if len(r) else None,
           "naik_kuat": float((kel == "naik kuat").mean() * 100) if len(r) else None,
           "turun_kuat": float((kel == "turun kuat").mean() * 100) if len(r) else None, "p": None}
    sisa = dasar[kol].drop(r.index, errors="ignore").dropna()
    if len(r) >= MIN_N and r.std() > 0:
        row["p"] = float(stats.ttest_ind(r, sisa, equal_var=False).pvalue)
    return row


def _bandingkan(lt, uj):
    """Pola dianggap TERBUKTI jika signifikan di latih dan arahnya sama (dan signifikan) di ujian."""
    if lt["p"] is None or uj["p"] is None:
        return "sampel kurang"
    sama = np.sign(lt["selisih"]) == np.sign(uj["selisih"])
    if lt["p"] < 0.05 and sama and uj["p"] < 0.05:
        return "terbukti"
    if lt["p"] < 0.05 and sama:
        return "arah sama, ujian tidak signifikan"
    if lt["p"] < 0.05:
        return "gagal (arah berbalik)"
    return "tidak ada pola di latih"


def pola(d):
    kondisi = [("kurs", k, None) for k in KURS] + [("siklus", None, s) for s in SIKLUS] + \
              [("kurs x siklus", k, s) for k, s in itertools.product(KURS, SIKLUS)] + \
              [("hari RDG", r, None) for r in ["naik", "turun"]]
    out = []
    for jenis, k, s in kondisi:
        if jenis == "hari RDG":
            m = d["rdg"] == k
            nama = f"Hari RDG: BI-Rate {k}"
        else:
            m = pd.Series(True, index=d.index)
            if k:
                m &= d["kurs"] == k
            if s:
                m &= d["siklus"] == s
            nama = " & ".join(filter(None, [f"kurs {k}" + (" (rupiah melemah)" if k == "naik" else
                                                            " (rupiah menguat)" if k == "turun" else "") if k else None,
                                            f"siklus {s}" if s else None]))
        row = {"jenis": jenis, "kondisi": nama}
        for hari, kol in [("H0", "r0"), ("H+1", "r1")]:
            for b in ["latih", "ujian"]:
                dasar = d[d["bagian"] == b]
                sel = _sel(dasar[m.loc[dasar.index]], kol, hari, dasar)
                sel["selisih"] = None if sel["rata"] is None else sel["rata"] - float(dasar[kol].mean())
                row[f"{hari} {b}"] = sel
            row[f"{hari} status"] = _bandingkan(row[f"{hari} latih"], row[f"{hari} ujian"])
        out.append(row)
    return out


def lanjutan(d):
    """Matriks transisi kelas H0 -> H+1 dan rata-rata return besok per kelas hari ini."""
    out = {}
    for b, x in d.groupby("bagian"):
        tab = pd.crosstab(x["kelas0"], x["kelas1"], normalize="index").reindex(index=KELAS, columns=KELAS) * 100
        ringkas = {}
        for k in KELAS:
            r = x.loc[x["kelas0"] == k, "r1"]
            p = float(stats.ttest_ind(r, x.loc[x["kelas0"] != k, "r1"], equal_var=False).pvalue) if len(r) >= MIN_N else None
            ringkas[k] = {"n": int(len(r)), "rata_besok": float(r.mean()), "p": p,
                          "lanjut_searah": float((np.sign(r) == (1 if "naik" in k else -1)).mean() * 100)
                          if k != "datar" else None}
        out[b] = {"transisi_pct": tab.round(2).fillna(0).to_dict(orient="index"), "per_kelas": ringkas}
    return out


# ---------------------------------------------------------------------------
# LSTM 70/30
# ---------------------------------------------------------------------------
def pilih_hyperparameter(df, a):
    rows = []
    for L, H in itertools.product(SEQ_GRID, HIDDEN_GRID):
        sp = M.split_sequences(df, a, a, L)
        _, vals, _ = M.fit_ensemble(sp, seeds=TUNE_SEEDS, hidden=H)
        rows.append({"sequence_length": L, "hidden_size": H, "val_loss": float(np.mean(vals))})
        print(f"    L={L:>3} H={H:>3} | loss validasi {rows[-1]['val_loss']:.4f}")
    rows.sort(key=lambda r: r["val_loss"])
    return rows


def lstm(df, d, a, tuning=None):
    if tuning is None:
        print("  Memilih hyperparameter dari validasi di dalam 70% latih...")
        tuning = pilih_hyperparameter(df, a)
    L, H = tuning[0]["sequence_length"], tuning[0]["hidden_size"]
    print(f"  Terpilih: L={L}, H={H}")
    hasil, preds = {}, {}
    for kode in ("M0", "M3"):
        cols = F.SKENARIO[kode][1]
        sp = M.split_sequences(df, a, len(df), L, columns=cols)
        models, _, _ = M.fit_ensemble(sp, hidden=H)
        preds[kode] = M.predict_ensemble(models, sp["X_te"]) * sp["target_std"]
        rows = sp["rows_te"]
        info = {"n_latih": int(len(sp["X_tr"])), "n_validasi": int(len(sp["X_va"])), "n_ujian": int(len(rows))}
    act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
    semua = {"LSTM M3": preds["M3"], "LSTM M0": preds["M0"], **M.baselines(df, a, rows, F.SKENARIO["M3"][1])}
    naif = semua["Naif (besok = hari ini)"]
    for nama, p in semua.items():
        r = M.score(p, act, last)
        if nama != "Naif (besok = hari ini)":
            r["dm_vs_naif"] = M.diebold_mariano(act - p, act - naif)
            r["pesaran_timmermann"] = M.pesaran_timmermann(p, act)
        hasil[nama] = r
    hasil["LSTM M3"]["dm_vs_m0"] = M.diebold_mariano(act - preds["M3"], act - preds["M0"])

    u = d.iloc[rows].copy()
    u["pred"] = np.expm1(preds["M3"]) * 100
    u["kelas_pred"] = kelas(u["pred"])
    konf = pd.crosstab(u["kelas1"], u["kelas_pred"]).reindex(index=KELAS, columns=KELAS, fill_value=0)
    u["kuintil"] = pd.qcut(u["pred"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    kuintil = [{"kuintil": int(q), "pred_rata": float(x["pred"].mean()), "aktual_rata": float(x["r1"].mean()),
                "naik_kuat": float((x["kelas1"] == "naik kuat").mean() * 100),
                "turun_kuat": float((x["kelas1"] == "turun kuat").mean() * 100), "n": int(len(x))}
               for q, x in u.groupby("kuintil")]
    rho, p_rho = stats.spearmanr(u["pred"], u["r1"])
    kuat = u["kelas1"].isin(["naik kuat", "turun kuat"])
    arah_kuat = float((np.sign(u.loc[kuat, "pred"]) == np.sign(u.loc[kuat, "r1"])).mean() * 100)
    per_kondisi = []
    for k, s in itertools.product(KURS, SIKLUS):
        x = u[(u["kurs"] == k) & (u["siklus"] == s)]
        if len(x):
            per_kondisi.append({"kurs": k, "siklus": s, "n": int(len(x)), "pred_rata": float(x["pred"].mean()),
                                "aktual_rata": float(x["r1"].mean()),
                                "arah_benar": float((np.sign(x["pred"]) == np.sign(x["r1"])).mean() * 100)})
    u[["date", "ihsg", "r1", "kelas1", "pred", "kelas_pred", "kurs", "siklus"]].to_csv(
        os.path.join(config.MODEL_DIR, "simulasi_prediksi_ujian.csv"), index=False)
    return {"hyperparameter": {"sequence_length": L, "hidden_size": H, "grid": tuning}, **info,
            "metrik": hasil, "kelas_prediksi_pct": {k: float((u["kelas_pred"] == k).mean() * 100) for k in KELAS},
            "konfusi": konf.to_dict(orient="index"), "kuintil": kuintil,
            "spearman": {"rho": float(rho), "p_value": float(p_rho)},
            "arah_hari_kuat": {"n": int(kuat.sum()), "benar_pct": arah_kuat},
            "per_kondisi": per_kondisi, "pred_std_pct": float(u["pred"].std()), "aktual_std_pct": float(u["r1"].std())}


def validitas(d, pola_res):
    """Uji ketahanan: koreksi Bonferroni, Mann-Whitney, regresi OLS dengan galat baku
    Newey-West (HAC), dan diagnosis prediksi LSTM terhadap volatilitas."""
    import statsmodels.api as sm
    n_uji = sum(1 for r in pola_res for h in ("H0", "H+1") if r[f"{h} latih"]["p"] is not None)
    batas = 0.05 / n_uji
    bonf = [{"kondisi": r["kondisi"], "hari": h, "p_latih": r[f"{h} latih"]["p"], "p_ujian": r[f"{h} ujian"]["p"],
             "lolos": bool(r[f"{h} latih"]["p"] < batas and (r[f"{h} ujian"]["p"] or 1) < batas)}
            for r in pola_res for h in ("H0", "H+1") if r[f"{h} status"] == "terbukti"]
    mw, ols = [], []
    for b in ("latih", "ujian"):
        x = d[d["bagian"] == b]
        for k in ("naik", "turun"):
            for h, kol in (("H0", "r0"), ("H+1", "r1")):
                g, sisa = x.loc[x["kurs"] == k, kol].dropna(), x.loc[x["kurs"] != k, kol].dropna()
                mw.append({"bagian": b, "kurs": k, "hari": h, "n": int(len(g)),
                           "p": float(stats.mannwhitneyu(g, sisa).pvalue)})
        for h, kol in (("H0", "r0"), ("H+1", "r1")):
            y = x[[kol, "kurs_pct", "d_bi"]].dropna()
            m = sm.OLS(y[kol], sm.add_constant(y[["kurs_pct", "d_bi"]])).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_LAG})
            ols.append({"bagian": b, "hari": h, "b_kurs": float(m.params["kurs_pct"]),
                        "p_kurs": float(m.pvalues["kurs_pct"]), "b_bi": float(m.params["d_bi"]),
                        "p_bi": float(m.pvalues["d_bi"]), "r2": float(m.rsquared), "n": int(m.nobs)})
    out = {"bonferroni": {"n_uji": n_uji, "batas": batas, "hasil": bonf}, "mann_whitney": mw,
           "ols_hac": {"maxlags": HAC_LAG, "hasil": ols}}
    f = os.path.join(config.MODEL_DIR, "simulasi_prediksi_ujian.csv")
    if os.path.exists(f):
        u = pd.read_csv(f, parse_dates=["date"])
        vol = pd.DataFrame({"date": d["date"], "vol20": d["r0"].abs().rolling(VOL_LSTM).mean()})
        u = u.merge(vol, on="date", how="left")
        u["q"] = pd.qcut(u["pred"].rank(method="first"), 5, labels=False)
        ct = pd.crosstab(u["q"], u["kelas1"].isin(["naik kuat", "turun kuat"]))
        sp = lambda a, b: dict(zip(("rho", "p"), map(float, stats.spearmanr(a, b, nan_policy="omit"))))  # noqa: E731
        out["lstm"] = {"pred_min": float(u["pred"].min()), "pred_max": float(u["pred"].max()),
                       "pred_std": float(u["pred"].std()), "pct_pred_positif": float((u["pred"] > 0).mean() * 100),
                       "pct_aktual_positif": float((u["r1"] > 0).mean() * 100),
                       "vs_volatilitas": sp(u["pred"], u["vol20"]), "vs_besar_gerak": sp(u["pred"], u["r1"].abs()),
                       "vs_arah": sp(u["pred"], u["r1"]), "chi2_kuintil_kuat_p": float(stats.chi2_contingency(ct)[1])}
    return out


def sensitivitas():
    """Ulangi analisis pola dengan ambang lain untuk memastikan kesimpulan tidak bergantung pada pilihan ambang."""
    global KUAT, DATAR, KURS_AMBANG
    asli = KUAT, DATAR, KURS_AMBANG
    out = []
    for kuat, datar, kurs in [(1.0, 0.25, 0.25), (1.5, 0.25, 0.25), (0.75, 0.2, 0.25), (1.0, 0.25, 0.10),
                              (1.0, 0.25, 0.50)]:
        KUAT, DATAR, KURS_AMBANG = kuat, datar, kurs
        p = pola(siapkan()[1])
        out.append({"kuat": kuat, "datar": datar, "kurs": kurs,
                    "h0": sum(r["H0 status"] == "terbukti" for r in p),
                    "h1": sum(r["H+1 status"] == "terbukti" for r in p)})
    KUAT, DATAR, KURS_AMBANG = asli
    return out


def cetak(res):
    print("\n=== 1. Distribusi kelas IHSG ===")
    for b, x in res["distribusi"].items():
        print(f"  {b:<5} {x['mulai']} s.d. {x['akhir']} ({x['n']} hari) std {x['std']:.2f}% | " +
              " | ".join(f"{k} {v:.1f}%" for k, v in x["kelas_pct"].items()))
        print(f"        rata hari naik {x['rata_naik']:+.2f}% vs hari turun {x['rata_turun']:+.2f}% "
              f"(uji asimetri p {x['asimetri_t']['pvalue']:.3f})")
    print("\n=== 2. Pola kondisi X -> IHSG ===")
    for r in res["pola"]:
        a, b, c, e = r["H0 latih"], r["H0 ujian"], r["H+1 latih"], r["H+1 ujian"]
        f = lambda v: "   -  " if v is None else f"{v:+.2f}"  # noqa: E731
        print(f"  {r['kondisi']:<52} H0 {f(a['rata'])}/{f(b['rata'])} [{r['H0 status']}] | "
              f"H+1 {f(c['rata'])}/{f(e['rata'])} [{r['H+1 status']}]  (n {a['n']}/{b['n']})")
    print("\n=== 3. Besok setelah hari ini (rata return H+1) ===")
    for b, x in res["lanjutan"].items():
        print(f"  {b:<5} " + " | ".join(f"{k} {v['rata_besok']:+.2f}%" for k, v in x["per_kelas"].items()))
    if "lstm" in res:
        L = res["lstm"]
        print("\n=== 4. LSTM 70/30 ===")
        for k, v in L["metrik"].items():
            print(f"  {k:<26} RMSE {v['rmse']:8.2f} | arah {v['direction_accuracy'] or 0:5.1f}% | "
                  f"DM p vs naif {v.get('dm_vs_naif', {}).get('p_value', float('nan')):.3f}")
        print("  kelas prediksi:", {k: round(v, 1) for k, v in L["kelas_prediksi_pct"].items()})
        print(f"  Spearman prediksi vs aktual {L['spearman']['rho']:+.3f} (p {L['spearman']['p_value']:.3f}) | "
              f"arah benar di hari kuat {L['arah_hari_kuat']['benar_pct']:.1f}% dari {L['arah_hari_kuat']['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tanpa-lstm", action="store_true")
    ap.add_argument("--tuning-ulang", action="store_true", help="abaikan hasil tuning 70/30 yang tersimpan")
    args = ap.parse_args()
    df, d, a = siapkan()
    lama = json.load(open(PATH, encoding="utf-8")) if os.path.exists(PATH) else {}
    res = {"pengaturan": {"porsi_latih": PORSI_LATIH, "ambang_kuat": KUAT, "ambang_datar": DATAR,
                          "ambang_kurs": KURS_AMBANG, "min_n": MIN_N, "kelas": KELAS},
           "distribusi": distribusi(d), "pola": pola(d), "lanjutan": lanjutan(d)}
    if not args.tanpa_lstm:
        t = time.time()
        grid = None if args.tuning_ulang else lama.get("lstm", {}).get("hyperparameter", {}).get("grid")
        res["lstm"] = lstm(df, d, a, grid)
        print(f"  ({time.time() - t:.0f} detik)")
    elif "lstm" in lama:
        res["lstm"] = lama["lstm"]
    res["validitas"] = validitas(d, res["pola"])
    res["validitas"]["sensitivitas"] = sensitivitas()
    cetak(res)
    v = res["validitas"]
    print(f"\n=== 5. Validitas ===\n  Bonferroni ({v['bonferroni']['n_uji']} uji, batas {v['bonferroni']['batas']:.4f}): "
          + ", ".join(f"{x['kondisi']} {x['hari']}" for x in v["bonferroni"]["hasil"] if x["lolos"]))
    for o in v["ols_hac"]["hasil"]:
        print(f"  OLS-HAC {o['bagian']:<5} {o['hari']:<3} kurs {o['b_kurs']:+.3f} (p {o['p_kurs']:.4f}) | "
              f"d_bi p {o['p_bi']:.3f} | R2 {o['r2']:.4f}")
    for s in v["sensitivitas"]:
        print(f"  ambang kuat {s['kuat']} kurs {s['kurs']}: H0 terbukti {s['h0']} | H+1 terbukti {s['h1']}")
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nTersimpan: {PATH}")


if __name__ == "__main__":
    main()
