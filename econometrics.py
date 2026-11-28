"""
econometrics.py
===============
Analisis pelengkap untuk menafsirkan hasil LSTM: hubungan antara BI-Rate, kurs
USD/IDR, dan IHSG. Bagian ini membahas keterkaitan, bukan klaim kausalitas
struktural (skripsi Subbab 3.11; hasil di Subbab 4.6).

1. Efek kontemporer  : ret_ihsg_t = a + b1 ret_kurs_t + b2 d_bi_t + e
                       (OLS, galat HAC Newey-West; Newey & West, 1987; Pers. 3.4-3.5)
2. Granger & VAR     : apakah kurs / BI-Rate hari ini membantu menjelaskan IHSG hari-hari berikutnya?
                       VAR(p) pada [ret_ihsg, ret_kurs, d_bi] (Sims, 1980), p dipilih AIC (Akaike, 1974;
                       maks 10), uji Granger (Granger, 1969), ditambah impulse response 10 hari (Pers. 3.9-3.11)
3. Event study RDG   : abnormal return IHSG di sekitar keputusan RDG Bank Indonesia (Brown & Warner, 1985;
                       Patell, 1976; Pers. 3.12-3.13). Dari 72 keputusan di bi_rate.csv, 64 yang punya
                       jendela estimasi [-130, -11] hari bursa lengkap (naik 11, turun 6, tetap 47);
                       model rata-rata konstan, p dikoreksi Bonferroni

Jalankan:
    python econometrics.py
Hasil: model/econometrics.json
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.api import VAR

import config
import features as F

ESTIMASI = (-130, -11)
JENDELA = {"[0]": (0, 0), "[-1,+1]": (-1, 1), "[0,+5]": (0, 5)}
IRF_HARI = 10


def _data():
    raw = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    df = F.add_features(raw)
    df["ret_ihsg"] *= 100
    df["ret_kurs"] *= 100
    df["d_bi"] = df["d_bi"] * 100       # basis poin
    return df


def kontemporer(df):
    X = sm.add_constant(df[["ret_kurs", "d_bi"]])
    res = sm.OLS(df["ret_ihsg"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out = {k: {"koef": float(res.params[k]), "p_value": float(res.pvalues[k])} for k in ("ret_kurs", "d_bi")}
    out["r2"] = float(res.rsquared)
    out["n"] = int(res.nobs)
    print(f"  ret_kurs: koef {out['ret_kurs']['koef']:+.3f} (p {out['ret_kurs']['p_value']:.4f}) | "
          f"d_bi (per bps): {out['d_bi']['koef']:+.4f} (p {out['d_bi']['p_value']:.3f}) | R2 {out['r2']:.3f}")
    return out


def granger_var(df):
    y = df[["ret_ihsg", "ret_kurs", "d_bi"]]
    model = VAR(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = max(1, int(model.select_order(10).aic))
        res = model.fit(p)
    uji = {}
    for sebab, akibat in [("ret_kurs", "ret_ihsg"), ("d_bi", "ret_ihsg"), ("ret_ihsg", "ret_kurs"),
                          ("d_bi", "ret_kurs")]:
        t = res.test_causality(akibat, [sebab], kind="f")
        uji[f"{sebab} -> {akibat}"] = {"F": float(t.test_statistic), "p_value": float(t.pvalue),
                                      "df": [int(d) for d in t.df]}
        print(f"  Granger {sebab:>8} -> {akibat:<8}: F {t.test_statistic:6.2f}, p {t.pvalue:.4f}")
    irf = res.irf(IRF_HARI)
    # respons ret_ihsg (%) terhadap guncangan 1 simpangan baku (ortogonal, Cholesky
    # dengan urutan d_bi, ret_kurs, ret_ihsg: kebijakan & kurs dianggap lebih dulu)
    res_o = VAR(df[["d_bi", "ret_kurs", "ret_ihsg"]]).fit(p)
    irf_o = res_o.irf(IRF_HARI)
    lo, hi = irf_o.errband_mc(orth=True, repl=500, signif=0.05, seed=config.RANDOM_SEED)
    respons = {}
    for j, nama in enumerate(["d_bi", "ret_kurs"]):
        respons[nama] = {"irf": irf_o.orth_irfs[:, 2, j].tolist(),
                         "lo": lo[:, 2, j].tolist(), "hi": hi[:, 2, j].tolist()}
    del irf
    return {"lag_aic": p, "granger": uji, "irf_ihsg": respons, "irf_hari": IRF_HARI,
            "urutan_cholesky": ["d_bi", "ret_kurs", "ret_ihsg"]}


def event_study(df):
    bi = pd.read_csv(config.BI_RATE_CSV, parse_dates=["date"])
    bi = bi.rename(columns={[c for c in bi.columns if "rate" in c.lower()][0]: "bi_rate"})
    bi["perubahan_bps"] = (bi["bi_rate"].diff() * 100).round()
    dates = df["date"].values
    r = df["ret_ihsg"].values
    kejadian = []
    for _, row in bi.dropna(subset=["perubahan_bps"]).iterrows():
        t0 = int(np.searchsorted(dates, np.datetime64(row["date"])))   # hari bursa pertama >= tanggal RDG
        if t0 + ESTIMASI[0] < 0 or t0 + max(b for _, b in JENDELA.values()) >= len(r):
            continue
        est = r[t0 + ESTIMASI[0]: t0 + ESTIMASI[1] + 1]
        mu, sd = est.mean(), est.std(ddof=1)
        e = {"tanggal": str(row["date"].date()), "perubahan_bps": int(row["perubahan_bps"]),
             "jenis": "naik" if row["perubahan_bps"] > 0 else "turun" if row["perubahan_bps"] < 0 else "tetap"}
        for nama, (a, b) in JENDELA.items():
            ar = r[t0 + a: t0 + b + 1] - mu
            e[f"CAR {nama}"] = float(ar.sum())
            e[f"SCAR {nama}"] = float(ar.sum() / (sd * np.sqrt(b - a + 1)))
        kejadian.append(e)
    ev = pd.DataFrame(kejadian)
    ringkas = {}
    for jenis in ["naik", "turun", "tetap", "semua"]:
        sub = ev if jenis == "semua" else ev[ev["jenis"] == jenis]
        row = {"n": int(len(sub))}
        for nama in JENDELA:
            car = sub[f"CAR {nama}"]
            if len(sub) >= 2:
                t, p = stats.ttest_1samp(car, 0.0)
                # uji Patell-style dengan SCAR (lebih tahan terhadap beda volatilitas)
                z = sub[f"SCAR {nama}"].sum() / np.sqrt(len(sub))
                row[nama] = {"car_mean": float(car.mean()), "t": float(t), "p_value": float(p),
                             "z_scar": float(z), "p_scar": float(2 * (1 - stats.norm.cdf(abs(z))))}
            else:
                row[nama] = {"car_mean": float(car.mean()) if len(sub) else None}
        ringkas[jenis] = row
        if len(sub) >= 2:
            j = row["[-1,+1]"]
            print(f"  RDG {jenis:<6} n={len(sub):>2}: CAR[-1,+1] {j['car_mean']:+.3f}% (p {j['p_value']:.3f})")
    return {"ringkas": ringkas, "kejadian": kejadian, "estimasi": list(ESTIMASI),
            "catatan": "Model rata-rata konstan; tanpa data ekspektasi pasar, sehingga 'kejutan' kebijakan "
                       "tidak dapat dipisahkan dari keputusan yang sudah diantisipasi."}


def main():
    df = _data()
    out = {"periode": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())], "n": int(len(df))}
    print("\n=== Efek kontemporer (OLS, HAC) ===")
    out["kontemporer"] = kontemporer(df)
    print("\n=== Granger & VAR ===")
    out["var"] = granger_var(df)
    print("\n=== Event study RDG ===")
    out["event_study"] = event_study(df)
    with open(config.ECONOMETRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nTersimpan: {config.ECONOMETRICS_PATH}")


if __name__ == "__main__":
    main()
