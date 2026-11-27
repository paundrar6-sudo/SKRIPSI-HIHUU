"""
figures.py
==========
Grafik resolusi tinggi untuk naskah skripsi (Bab III-IV).

Setiap grafik disimpan dua kali di folder figures/:
    .png  300 dpi (untuk Word)
    .pdf  vektor (tetap tajam saat diperbesar; untuk LaTeX atau cetak)

Lebar grafik 16 cm (pas untuk lebar teks A4 dengan margin 4-3-3-3 cm), huruf
Times New Roman 10 pt, angka berformat Indonesia (koma desimal, titik ribuan).
Semua angka dibaca dari hasil yang sudah tersimpan; skrip ini tidak melatih model.

Jalankan (setelah train.py, evaluate_walkforward.py, experiments.py, econometrics.py):
    python figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402
import features as F  # noqa: E402

OUT = "figures"
CM = 1 / 2.54
LEBAR = 16 * CM
BIRU, MERAH, HIJAU, ABU, HITAM = "#0f62fe", "#da1e28", "#198038", "#8d8d8d", "#161616"
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
DAFTAR = []

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e0e0e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def angka(x, d=0):
    """Format Indonesia: 18.278 / 1,17."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt(d=0, suffix=""):
    return FuncFormatter(lambda x, _: angka(x, d) + suffix)


def sumbu_tanggal(ax, interval=6):
    if interval >= 12:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(mdates.num2date(x).year)))
        return
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, interval)))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{BULAN[mdates.num2date(x).month - 1]} {mdates.num2date(x).year}"))


def legenda_atas(ax, ncol=3, **kw):
    """Legenda di atas area plot agar tidak menutupi data."""
    return ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, borderaxespad=0, **kw)


def bln(ts):
    ts = pd.Timestamp(ts)
    return f"{BULAN[ts.month - 1]} {ts.year % 100:02d}"


def simpan(fig, nama, judul):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{nama}.{ext}"), dpi=300)
    plt.close(fig)
    DAFTAR.append((nama, judul))
    print(f"  {nama}: {judul}")


def baca(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def g_data(df, m):
    fig, ax = plt.subplots(3, 1, figsize=(LEBAR, 13 * CM), sharex=True,
                           gridspec_kw={"height_ratios": [1.4, 1, 0.8]})
    ax[0].plot(df["date"], df["ihsg_close"], color=HITAM, lw=0.8)
    ax[0].set_ylabel("IHSG (poin)")
    ax[0].yaxis.set_major_formatter(fmt())
    ax[1].plot(df["date"], df["kurs_usd_idr"], color=BIRU, lw=0.8)
    ax[1].set_ylabel("Rp per USD")
    ax[1].yaxis.set_major_formatter(fmt())
    ax[2].step(df["date"], df["bi_rate"], where="post", color=MERAH, lw=1)
    ax[2].set_ylabel("BI-Rate (%)")
    ax[2].yaxis.set_major_formatter(fmt(2))
    if m:
        uji = pd.Timestamp(m["test_start"])
        for a in ax:
            a.axvspan(uji, df["date"].max(), color=ABU, alpha=0.15, lw=0)
        ax[0].text(uji, ax[0].get_ylim()[1], " periode uji", va="top", fontsize=8, color="#525252")
    sumbu_tanggal(ax[2], 12)
    fig.align_ylabels(ax)
    simpan(fig, "g01_data_penelitian", "IHSG, kurs JISDOR, dan BI-Rate selama periode penelitian")


def g_kurs_sumber():
    """JISDOR vs Yahoo IDR=X (alasan penggantian sumber kurs). Butuh internet untuk Yahoo."""
    try:
        import fetch_kurs
        y = fetch_kurs.fetch_yahoo()
    except Exception as e:
        print(f"  [lewati] g02: data Yahoo tidak dapat diunduh ({e})")
        return
    j = pd.read_csv(config.KURS_CSV, parse_dates=["date"])
    y["date"] = pd.to_datetime(y["date"])
    rj = np.log(j.set_index("date")["kurs_usd_idr"]).diff().dropna() * 100
    ry = np.log(y.set_index("date")["kurs_usd_idr"]).diff().dropna() * 100
    fig, ax = plt.subplots(2, 1, figsize=(LEBAR, 10 * CM), sharex=True, sharey=True)
    for a, r, nama, w in [(ax[0], ry, "Yahoo Finance IDR=X", MERAH), (ax[1], rj, "JISDOR Bank Indonesia", BIRU)]:
        a.plot(r.index, r.values, color=w, lw=0.6)
        a.axhline(1.5, color=ABU, lw=0.6, ls="--")
        a.axhline(-1.5, color=ABU, lw=0.6, ls="--")
        a.set_title(f"{nama}: {int((r.abs() > 1.5).sum())} hari dengan perubahan lebih dari ±1,5%", loc="left")
        a.set_ylabel("Perubahan harian (%)")
        a.yaxis.set_major_formatter(fmt(0))
    sumbu_tanggal(ax[1], 12)
    simpan(fig, "g02_kurs_yahoo_vs_jisdor", "Perubahan harian kurs USD/IDR: Yahoo Finance dibanding JISDOR")


def g_prediksi(m):
    p = pd.read_csv(config.TEST_PRED_CSV if hasattr(config, "TEST_PRED_CSV")
                    else os.path.join(config.MODEL_DIR, "test_predictions.csv"), parse_dates=["date"])
    fig, ax = plt.subplots(2, 1, figsize=(LEBAR, 12 * CM), gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(p["date"], p["aktual"], color=HITAM, lw=0.9, label="IHSG aktual")
    ax[0].plot(p["date"], p["prediksi"], color=BIRU, lw=0.8, ls="--", label="Prediksi LSTM (M3)")
    ax[0].yaxis.set_major_formatter(fmt())
    ax[0].set_ylabel("IHSG (poin)")
    sumbu_tanggal(ax[0], 2)
    ax[0].text(0, 1.13, f"(a) Seluruh data uji ({len(p)} hari)", transform=ax[0].transAxes, fontsize=10)
    z = p.tail(40)
    ax[1].plot(z["date"], z["aktual"], color=HITAM, lw=1, marker="o", ms=2.5, label="Aktual")
    ax[1].plot(z["date"], z["prediksi"], color=BIRU, lw=1, ls="--", marker="s", ms=2.5, label="LSTM")
    ax[1].plot(z["date"], z["hari_ini"], color=ABU, lw=0.8, ls=":", label="Naif (= hari sebelumnya)")
    ax[1].yaxis.set_major_formatter(fmt())
    ax[1].set_ylabel("IHSG (poin)")
    ax[1].xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{mdates.num2date(x).day} {BULAN[mdates.num2date(x).month - 1]}"))
    ax[1].text(0, 1.04, "(b) 40 hari uji terakhir", transform=ax[1].transAxes, fontsize=10)
    h, lab = ax[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=3, bbox_to_anchor=(1, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2)
    simpan(fig, "g03_prediksi_data_uji", "Prediksi LSTM dibanding IHSG aktual pada data uji")


def g_galat(m):
    p = pd.read_csv(os.path.join(config.MODEL_DIR, "test_predictions.csv"), parse_dates=["date"])
    e_l, e_n = p["aktual"] - p["prediksi"], p["aktual"] - p["hari_ini"]
    fig, ax = plt.subplots(1, 2, figsize=(LEBAR, 6.5 * CM))
    ax[0].scatter(e_n, e_l, s=6, color=BIRU, alpha=0.6, lw=0)
    lim = max(e_n.abs().max(), e_l.abs().max()) * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], color=ABU, lw=0.8, ls="--")
    ax[0].set_xlabel("Galat naif (poin)")
    ax[0].set_ylabel("Galat LSTM (poin)")
    ax[0].set_title("Galat harian hampir identik", loc="left")
    ax[0].xaxis.set_major_formatter(fmt())
    ax[0].yaxis.set_major_formatter(fmt())
    nama = ["LSTM", "Naif", "Rata-rata\nhistoris", "ARIMA", "Regresi\nlinear"]
    kunci = ["LSTM", "Naif (besok = hari ini)", "Rata-rata historis", "ARIMA", "Regresi linear"]
    rmse = [m["models"][k]["rmse"] for k in kunci]
    warna = [BIRU] + [ABU] * 4
    ax[1].bar(nama, rmse, color=warna, width=0.6)
    lo = min(rmse) - 1.5
    ax[1].set_ylim(lo, max(rmse) + 0.6)
    for i, v in enumerate(rmse):
        ax[1].text(i, v + 0.08, angka(v, 2), ha="center", fontsize=8)
    ax[1].set_ylabel("RMSE (poin IHSG)")
    ax[1].set_title("RMSE data uji (sumbu tegak tidak dari nol)", loc="left")
    ax[1].yaxis.set_major_formatter(fmt(0))
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    simpan(fig, "g04_galat_dan_rmse", "Galat LSTM dibanding naif dan RMSE semua model pada data uji")


def g_walkforward(wf):
    f = pd.DataFrame(wf["folds"])
    rel = (f["lstm_rmse"] / f["naive_rmse"] - 1) * 100
    fig, ax = plt.subplots(figsize=(LEBAR, 6 * CM))
    lab = [f"Lipatan {r.fold}\n{bln(r.start)}–\n{bln(r.end)}" for r in f.itertuples()]
    ax.bar(lab, rel, color=[HIJAU if v < 0 else MERAH for v in rel], width=0.55)
    ax.axhline(0, color=HITAM, lw=0.8)
    for i, (v, p) in enumerate(zip(rel, f["dm_p_vs_naive"])):
        ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{angka(v, 2)}%\np = {angka(p, 2)}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylabel("RMSE LSTM relatif naif (%)")
    ax.yaxis.set_major_formatter(fmt(1))
    ax.set_ylim(min(-0.1, rel.min() - 0.1), rel.max() + 0.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Positif = LSTM lebih buruk dari naif; p = uji Diebold-Mariano", loc="left")
    simpan(fig, "g05_walkforward", "Selisih RMSE LSTM terhadap naif di setiap lipatan walk-forward")


def g_ablation(ex):
    ab, wfx = ex["ablation"], ex.get("walkforward")
    kode = list(ab["models"])
    fig, ax = plt.subplots(figsize=(LEBAR, 6.5 * CM))
    x = np.arange(len(kode))
    w = 0.38
    r1 = [(ab["models"][k]["rmse"] / ab["naif"]["rmse"] - 1) * 100 for k in kode]
    ax.bar(x - w / 2, r1, w, color=BIRU, label=f"Pembagian latih/uji ({ab['n_test']} hari)")
    if wfx:
        r2 = [(wfx["combined"][k]["rmse"] / wfx["combined"]["naif"]["rmse"] - 1) * 100 for k in kode]
        ax.bar(x + w / 2, r2, w, color="#78a9ff", label=f"Walk-forward ({wfx['n_test']} hari)")
    ax.axhline(0, color=HITAM, lw=0.8)
    ket = {"M0": "M0\nIHSG", "M1": "M1\n+BI-Rate", "M2": "M2\n+kurs", "M3": "M3\n+BI-Rate+kurs",
           "M4": "M4\nM3+selisih Fed"}
    ax.set_xticks(x, [ket.get(k, k) for k in kode])
    for i, k in enumerate(kode):
        p = ab["models"][k].get("dm_vs_m0", {}).get("p_value")
        if p is not None:
            ax.text(i - w / 2, r1[i] + 0.01, f"p={angka(p, 2)}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("RMSE relatif naif (%)")
    ax.yaxis.set_major_formatter(fmt(2))
    ax.grid(axis="x", visible=False)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0), hi + (hi - min(lo, 0)) * 0.12)
    legenda_atas(ax, ncol=2)
    ax.text(1, -0.36, "Angka p = uji Diebold-Mariano terhadap M0 (pembagian latih/uji)", transform=ax.transAxes,
            ha="right", fontsize=8, color="#525252")
    simpan(fig, "g06_skenario_m0_m4", "RMSE skenario fitur M0-M4 relatif terhadap baseline naif")


def g_horizon(ex):
    hz = ex["horizon"]
    hs = list(hz)
    fig, ax = plt.subplots(figsize=(LEBAR * 0.75, 6 * CM))
    x = np.arange(len(hs))
    w = 0.36
    for j, (k, c) in enumerate([("M0", ABU), ("M3", BIRU)]):
        v = [(hz[h][k]["rmse"] / hz[h]["naif"]["rmse"] - 1) * 100 for h in hs]
        ax.bar(x + (j - 0.5) * w, v, w, color=c, label=k + (" (riwayat IHSG)" if k == "M0" else " (model inti)"))
        for i, vv in enumerate(v):
            ax.text(x[i] + (j - 0.5) * w, vv + 0.1, angka(vv, 1) + "%", ha="center", fontsize=8)
    ax.axhline(0, color=HITAM, lw=0.8)
    ax.set_xticks(x, hs)
    ax.set_ylabel("RMSE relatif naif (%)")
    ax.yaxis.set_major_formatter(fmt(0))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    simpan(fig, "g07_horizon", "RMSE M0 dan M3 relatif naif pada horizon H+1, H+5, dan H+20")


def g_importance(ex):
    im = ex["importance"]["groups"]
    nama = list(im)
    fig, ax = plt.subplots(figsize=(LEBAR * 0.75, 4.5 * CM))
    ax.barh(nama, [im[k]["rmse_naik_mean"] for k in nama], xerr=[im[k]["rmse_naik_std"] for k in nama],
            color=BIRU, height=0.5, error_kw={"ecolor": ABU, "lw": 0.8, "capsize": 3})
    ax.axvline(0, color=HITAM, lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Kenaikan RMSE saat variabel diacak (poin IHSG)")
    ax.xaxis.set_major_formatter(fmt(2))
    ax.grid(axis="y", visible=False)
    simpan(fig, "g08_permutation_importance",
           f"Permutation importance model M3 ({ex['importance']['n_perm']} pengacakan, rata-rata ± simpangan baku)")


def g_kontemporer(df, eko):
    d = F.add_features(df)
    x, y = d["ret_kurs"] * 100, d["ret_ihsg"] * 100
    fig, ax = plt.subplots(figsize=(LEBAR * 0.75, 7 * CM))
    ax.scatter(x, y, s=5, color=BIRU, alpha=0.45, lw=0)
    b = eko["kontemporer"]["ret_kurs"]["koef"]
    a = y.mean() - b * x.mean()
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, a + b * xs, color=MERAH, lw=1.2,
            label=f"OLS: koefisien {angka(b, 2)} (p < 0,001), R² {angka(eko['kontemporer']['r2'], 3)}")
    ax.set_xlabel("Perubahan kurs JISDOR hari t (%)  [positif = rupiah melemah]")
    ax.set_ylabel("Return IHSG hari t (%)")
    ax.xaxis.set_major_formatter(fmt(1))
    ax.yaxis.set_major_formatter(fmt(0))
    legenda_atas(ax, ncol=1)
    simpan(fig, "g09_kurs_ihsg_hari_sama", "Hubungan perubahan kurs dan return IHSG pada hari yang sama")


def g_irf(eko):
    v = eko["var"]
    fig, ax = plt.subplots(1, 2, figsize=(LEBAR, 6 * CM), sharey=True)
    for a, (k, judul) in zip(ax, [("d_bi", "Guncangan perubahan BI-Rate"), ("ret_kurs", "Guncangan kurs USD/IDR")]):
        r = v["irf_ihsg"][k]
        h = np.arange(len(r["irf"]))
        a.fill_between(h, r["lo"], r["hi"], color=BIRU, alpha=0.15, lw=0, label="Selang 95%")
        a.plot(h, r["irf"], color=BIRU, marker="o", ms=3, lw=1.2, label="Respons IHSG")
        a.axhline(0, color=HITAM, lw=0.8)
        a.set_title(judul, loc="left")
        a.set_xlabel("Hari bursa setelah guncangan")
        a.set_xticks(h)
        a.yaxis.set_major_formatter(fmt(2))
    ax[0].set_ylabel("Respons return IHSG (poin %)")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    simpan(fig, "g10_impulse_response",
           f"Impulse response return IHSG dari VAR({v['lag_aic']}), guncangan 1 simpangan baku (Cholesky)")


def g_event(eko):
    ev = pd.DataFrame(eko["event_study"]["kejadian"])
    jendela = ["[0]", "[-1,+1]", "[0,+5]"]
    jenis = [("naik", MERAH), ("tetap", ABU), ("turun", HIJAU)]
    fig, ax = plt.subplots(figsize=(LEBAR, 6.5 * CM))
    x = np.arange(len(jendela))
    w = 0.26
    for j, (jn, c) in enumerate(jenis):
        sub = ev[ev["jenis"] == jn]
        mean = [sub[f"CAR {w_}"].mean() for w_ in jendela]
        ci = [stats.t.ppf(0.975, len(sub) - 1) * sub[f"CAR {w_}"].std(ddof=1) / np.sqrt(len(sub))
              for w_ in jendela]
        ax.bar(x + (j - 1) * w, mean, w, yerr=ci, color=c, label=f"BI-Rate {jn} (n = {len(sub)})",
               error_kw={"ecolor": HITAM, "lw": 0.7, "capsize": 2.5})
    ax.axhline(0, color=HITAM, lw=0.8)
    ax.set_xticks(x, [f"Jendela {w_}" for w_ in jendela])
    ax.set_ylabel("CAR IHSG (%)")
    ax.yaxis.set_major_formatter(fmt(1))
    ax.grid(axis="x", visible=False)
    legenda_atas(ax, ncol=3)
    ax.text(1, -0.16, "Garis galat = selang kepercayaan 95%", transform=ax.transAxes, ha="right",
            fontsize=8, color="#525252")
    simpan(fig, "g11_event_study_rdg", "Cumulative abnormal return IHSG di sekitar keputusan RDG Bank Indonesia")


KELAS_WARNA = {"turun kuat": "#a2191f", "turun lemah": "#ff8389", "datar": "#c6c6c6",
               "naik lemah": "#6fdc8c", "naik kuat": "#0e6027"}


def g_sim_kelas(sim):
    ds, kelas = sim["distribusi"], sim["pengaturan"]["kelas"]
    fig, ax = plt.subplots(figsize=(LEBAR, 4.2 * CM))
    for b, y in [("ujian", 0), ("latih", 1)]:
        kiri = 0
        for k in kelas:
            v = ds[b]["kelas_pct"][k]
            ax.barh(y, v, left=kiri, color=KELAS_WARNA[k], height=0.6, label=k if b == "latih" else None)
            if v >= 6:
                ax.text(kiri + v / 2, y, angka(v, 0) + "%", ha="center", va="center", fontsize=8,
                        color="white" if k in ("turun kuat", "naik kuat") else HITAM)
            kiri += v
    ax.set_yticks([0, 1], [f"Ujian 30%\n({ds['ujian']['n']} hari)", f"Latih 70%\n({ds['latih']['n']} hari)"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Persentase hari bursa (%)")
    ax.grid(False)
    legenda_atas(ax, ncol=5)
    simpan(fig, "g12_simulasi_kelas_ihsg", "Sebaran kelas pergerakan harian IHSG pada data latih (70%) dan ujian (30%)")


def g_sim_pola(sim):
    pola = [r for r in sim["pola"] if r["jenis"] in ("kurs", "kurs x siklus")]
    singkat = {"kurs naik (rupiah melemah)": "Rupiah melemah", "kurs turun (rupiah menguat)": "Rupiah menguat",
               "kurs tetap": "Kurs tetap"}

    def label(k):
        bagian = k.split(" & siklus ")
        return singkat[bagian[0]] + (f"\n+ {bagian[1]}" if len(bagian) > 1 else "")

    nama = [label(r["kondisi"]) for r in pola]
    fig, ax = plt.subplots(1, 2, figsize=(LEBAR, 11 * CM), sharey=True)
    y = np.arange(len(pola))
    for a, h in zip(ax, ["H0", "H+1"]):
        for j, (b, c) in enumerate([("latih", ABU), ("ujian", BIRU)]):
            v = [r[f"{h} {b}"]["rata"] for r in pola]
            a.barh(y + (j - 0.5) * 0.38, v, 0.38, color=c, label=f"Data {b}")
        a.axvline(0, color=HITAM, lw=0.8)
        a.set_title(f"IHSG {h} ({'hari yang sama' if h == 'H0' else 'hari bursa berikutnya'})", loc="left")
        a.set_xlabel("Rata-rata perubahan IHSG (%)")
        a.xaxis.set_major_formatter(fmt(1))
        a.grid(axis="y", visible=False)
    lim = max(abs(v) for r in pola for h in ("H0", "H+1") for b in ("latih", "ujian")
              for v in [r[f"{h} {b}"]["rata"]]) * 1.15
    for a in ax:
        a.set_xlim(-lim, lim)
    for i, r in enumerate(pola):
        if r["H0 status"] == "terbukti":
            ax[0].text(lim * 0.98, i, "terbukti", ha="right", va="center", fontsize=7.5, color=HIJAU)
        if r["H+1 status"] == "gagal (arah berbalik)":
            ax[1].text(lim * 0.98, i, "berbalik", ha="right", va="center", fontsize=7.5, color=MERAH)
    ax[0].set_yticks(y, nama, fontsize=8)
    ax[0].invert_yaxis()
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    simpan(fig, "g13_simulasi_pola_kurs_birate",
           "Rata-rata perubahan IHSG menurut kondisi kurs dan siklus BI-Rate, data latih dibanding ujian")


def g_sim_lanjutan(sim):
    lj, kelas = sim["lanjutan"], sim["pengaturan"]["kelas"]
    fig, ax = plt.subplots(figsize=(LEBAR, 6 * CM))
    x = np.arange(len(kelas))
    for j, (b, c) in enumerate([("latih", ABU), ("ujian", BIRU)]):
        ax.bar(x + (j - 0.5) * 0.38, [lj[b]["per_kelas"][k]["rata_besok"] for k in kelas], 0.38, color=c,
               label=f"Data {b}")
    ax.axhline(0, color=HITAM, lw=0.8)
    ax.set_xticks(x, [k.replace(" ", "\n") for k in kelas])
    ax.set_xlabel("Kelas IHSG hari ini (H0)")
    ax.set_ylabel("Rata-rata IHSG besok (%)")
    ax.yaxis.set_major_formatter(fmt(2))
    ax.grid(axis="x", visible=False)
    legenda_atas(ax, ncol=2)
    simpan(fig, "g14_simulasi_lanjutan", "Rata-rata perubahan IHSG hari berikutnya menurut kelas pergerakan hari ini")


def g_sim_lstm(sim):
    kq = pd.DataFrame(sim["lstm"]["kuintil"])
    fig, ax = plt.subplots(figsize=(LEBAR * 0.8, 6 * CM))
    x = kq["kuintil"].values
    ax.bar(x - 0.19, kq["naik_kuat"], 0.38, color=HIJAU, label="Besok naik kuat (> +1%)")
    ax.bar(x + 0.19, kq["turun_kuat"], 0.38, color=MERAH, label="Besok turun kuat (< −1%)")
    ax.set_xticks(x, ["1\npaling\npesimis", "2", "3", "4", "5\npaling\noptimis"])
    ax.set_xlabel("Kuintil prediksi LSTM (70/30) di data ujian")
    ax.set_ylabel("Persentase hari (%)")
    ax.yaxis.set_major_formatter(fmt(0))
    ax.grid(axis="x", visible=False)
    legenda_atas(ax, ncol=2)
    simpan(fig, "g15_simulasi_lstm_kuintil", "Kejadian naik kuat dan turun kuat IHSG menurut kuintil prediksi LSTM")


def main():
    print("Membuat grafik di folder figures/ ...")
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    m, wf = baca(config.METRICS_PATH), baca(config.WALKFORWARD_PATH)
    ex, eko = baca(config.EXPERIMENTS_PATH), baca(config.ECONOMETRICS_PATH)
    g_data(df, m)
    g_kurs_sumber()
    if m:
        g_prediksi(m)
        g_galat(m)
    if wf:
        g_walkforward(wf)
    if ex:
        if "ablation" in ex:
            g_ablation(ex)
        if "horizon" in ex:
            g_horizon(ex)
        if "importance" in ex:
            g_importance(ex)
    if eko:
        g_kontemporer(df, eko)
        g_irf(eko)
        g_event(eko)
    sim = baca(os.path.join(config.MODEL_DIR, "simulasi.json"))
    if sim:
        g_sim_kelas(sim)
        g_sim_pola(sim)
        g_sim_lanjutan(sim)
        if "lstm" in sim:
            g_sim_lstm(sim)
    with open(os.path.join(OUT, "DAFTAR_GAMBAR.md"), "w", encoding="utf-8") as f:
        f.write("# Daftar gambar\n\nPNG 300 dpi untuk Word, PDF vektor untuk LaTeX/cetak.\n\n"
                "| Berkas | Usulan judul gambar |\n|---|---|\n")
        f.writelines(f"| `{n}` | {j} |\n" for n, j in DAFTAR)
    print(f"Selesai: {len(DAFTAR)} grafik (PNG 300 dpi + PDF) di {OUT}/")


if __name__ == "__main__":
    main()
