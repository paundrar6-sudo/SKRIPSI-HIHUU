"""
app.py  --  Dashboard interaktif
================================
ANALISIS REAL-TIME BI-RATE DAN NILAI TUKAR RUPIAH TERHADAP DOLLAR AS
MENGGUNAKAN LONG SHORT-TERM MEMORY (LSTM) UNTUK PREDIKSI IHSG

Tampilan mengikuti DESIGN.md (IBM Carbon): kanvas putih, teks #161616, satu
aksen biru, sudut persegi, IBM Plex Sans. Tema dasar ada di .streamlit/config.toml.

Jalankan:  streamlit run app.py
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A
import config

JUDUL = ("Analisis Real-Time BI-Rate dan Nilai Tukar Rupiah terhadap Dollar AS "
         "menggunakan LSTM untuk Prediksi IHSG")

st.set_page_config(page_title="Prediksi IHSG H+1 · LSTM", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Warna (DESIGN.md). Satu warna per variabel, dipakai konsisten di semua grafik.
# ---------------------------------------------------------------------------
INK, MUTED, SUBTLE, HAIR, SURFACE = "#161616", "#525252", "#8c8c8c", "#e0e0e0", "#f4f4f4"
BLUE = "#0f62fe"
C_IHSG, C_BI, C_KURS, C_FED = INK, BLUE, "#9f1853", SUBTLE
C_NAIK, C_TURUN = "#198038", "#da1e28"
FONT = "IBM Plex Sans, sans-serif"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1320px; }}
h1 {{ font-weight: 300; letter-spacing: 0; }}
.stMarkdown p.eyebrow {{ font-size: 0.875rem; color: {MUTED}; letter-spacing: 0.16px; margin: 0 0 0.25rem 0; }}
.stMarkdown p.judul {{ font-size: 2rem; font-weight: 300; line-height: 1.25; color: {INK}; margin: 0 0 0.5rem 0; max-width: 60ch; }}
.stMarkdown p.meta {{ font-size: 0.875rem; color: {MUTED}; margin: 0 0 1.5rem 0; }}
.stMarkdown p.angka-besar {{ font-size: 3.25rem; font-weight: 300; line-height: 1.1; color: {INK}; margin: 0; font-variant-numeric: tabular-nums; }}
.stMarkdown p.label {{ font-size: 0.875rem; color: {MUTED}; letter-spacing: 0.16px; margin: 0 0 0.25rem 0; }}
.stMarkdown .naik {{ color: {C_NAIK}; }}
.stMarkdown .turun {{ color: {C_TURUN}; }}
.stMarkdown .catatan, .stMarkdown p.catatan {{ font-size: 0.875rem; color: {MUTED}; line-height: 1.5; max-width: 85ch; }}
@media (max-width: 640px) {{
  .stMarkdown p.judul {{ font-size: 1.5rem; }}
  .stMarkdown p.angka-besar {{ font-size: 2.5rem; }}
}}
div[data-testid="stTabs"] button p {{ font-size: 0.95rem; }}
</style>
""", unsafe_allow_html=True)


def layout_dasar(**kw) -> dict:
    """Layout Plotly bersama: datar, garis kisi tipis, font IBM Plex."""
    base = dict(
        template="plotly_white", font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=SURFACE, linecolor=HAIR, showline=True),
        yaxis=dict(gridcolor=SURFACE, linecolor=HAIR, zerolinecolor=HAIR),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    )
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Data & artefak
# ---------------------------------------------------------------------------
def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


# Argumen _mtime/versi membuat cache dibaca ulang setiap kali file hasil diperbarui.
@st.cache_data
def _load_dataset(path: str, versi: float) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_dataset() -> pd.DataFrame:
    path = config.FEATURES_CSV if os.path.exists(config.FEATURES_CSV) else config.DATASET_CSV
    return _load_dataset(path, _mtime(path))


@st.cache_resource
def _load_model_artifacts(versi: float):
    from predict_realtime import load_artifacts
    return load_artifacts()


def load_model_artifacts():
    return _load_model_artifacts(_mtime(config.MODEL_PATH))


@st.cache_data
def _load_json(path: str, versi: float):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json(path: str):
    return _load_json(path, _mtime(path))


BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def tgl(d, hari=False, singkat=False) -> str:
    """Tanggal berbahasa Indonesia, mis. 'Senin, 23 November 2026' atau '23 Nov 2026'."""
    d = pd.Timestamp(d)
    b = BULAN[d.month - 1]
    s = f"{d.day} {b[:3] if singkat else b} {d.year}"
    return f"{HARI[d.weekday()]}, {s}" if hari else s


def angka_id(x, nd=2) -> str:
    """Angka format Indonesia: koma desimal, titik ribuan (mis. 1.234,56)."""
    return f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:+.{nd}f}%"


def kelas_arah(x: float) -> str:
    return "naik" if x > 0 else "turun" if x < 0 else ""


if not os.path.exists(config.DATASET_CSV):
    st.error("Berkas `data/dataset_final.csv` belum ada. Jalankan `python run_all.py` "
             "untuk mengunduh data dan melatih model.")
    st.stop()

df = load_dataset()
last, prev = df.iloc[-1], df.iloc[-2]
m = load_json(config.METRICS_PATH)
wf = load_json(config.WALKFORWARD_PATH)
ex = load_json(config.EXPERIMENTS_PATH)
sim = load_json(os.path.join(config.MODEL_DIR, "simulasi.json"))
eko = load_json(config.ECONOMETRICS_PATH)
model_siap = os.path.exists(config.MODEL_PATH) and m is not None
NAIF = "Naif (besok = hari ini)"

# ---------------------------------------------------------------------------
# Kepala halaman
# ---------------------------------------------------------------------------
st.markdown(f"<p class='eyebrow'>Skripsi · Prediksi IHSG hari bursa berikutnya</p>"
            f"<p class='judul'>{JUDUL}</p>"
            f"<p class='meta'>Data pasar terakhir: {tgl(last['date'])} · "
            f"{len(df):,} hari bursa sejak {BULAN[df['date'].min().month - 1]} {df['date'].min().year}</p>",
            unsafe_allow_html=True)

tab_pred, tab_data, tab_xy, tab_eval, tab_kontrib, tab_sim, tab_log, tab_met = st.tabs(
    ["Prediksi H+1", "Data pasar", "Hubungan X dan Y", "Evaluasi model", "Kontribusi variabel",
     "Simulasi 70/30", "Rekam jejak", "Metodologi"])

# ===========================================================================
# TAB 1  PREDIKSI H+1
# ===========================================================================
with tab_pred:
    if not model_siap:
        st.warning("Model belum dilatih. Jalankan `python train.py` lalu muat ulang halaman.")
    else:
        from predict_realtime import prediksi_h1
        art = load_model_artifacts()
        dasar = prediksi_h1(df, art)

        kiri, kanan = st.columns([3, 2], gap="large")
        with kiri:
            arah = kelas_arah(dasar["perubahan_pct"])
            st.markdown(
                f"<p class='label'>Perkiraan IHSG penutupan {tgl(dasar['tanggal_h1'], hari=True)}</p>"
                f"<p class='angka-besar'>{dasar['ihsg_h1']:,.2f}</p>"
                f"<p class='{arah}' style='font-size:1.125rem; margin:0.25rem 0 1rem 0;'>"
                f"{dasar['selisih_poin']:+,.2f} poin ({dasar['perubahan_pct']:+.3f}%) dari "
                f"{dasar['ihsg_h0']:,.2f} pada {tgl(dasar['tanggal_h0'], singkat=True)}</p>",
                unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("BI-Rate", f"{last['bi_rate']:.2f}%",
                      f"{last['bi_rate'] - prev['bi_rate']:+.2f} poin" if last["bi_rate"] != prev["bi_rate"] else None,
                      delta_color="inverse")
            c2.metric("Kurs USD/IDR", f"Rp{last['kurs_usd_idr']:,.0f}",
                      f"{(last['kurs_usd_idr'] / prev['kurs_usd_idr'] - 1) * 100:+.2f}%",
                      delta_color="inverse", help="Kurs naik berarti rupiah melemah.")
            c3.metric("Fed Funds Rate", f"{last['fed_rate']:.2f}%", help="Variabel kontrol (FRED).")

        with kanan:
            with st.container(border=True):
                st.markdown("**Seberapa jauh angka ini bisa dipercaya**")
                naif = m["models"][NAIF]
                st.markdown(
                    f"<div class='catatan'>"
                    f"Pada {m['n_test']} hari uji ({tgl(m['test_start'], singkat=True)} s.d. {tgl(m['test_end'], singkat=True)}), rata-rata "
                    f"meleset <b>{m['mae']:,.1f} poin</b> (MAPE {m['mape']:.2f}%). Tebakan paling sederhana, "
                    f"&ldquo;besok sama dengan hari ini&rdquo;, meleset {naif['mae']:,.1f} poin."
                    f"<br><br>{m['verdict']} Uji Diebold-Mariano: p = {m['dm_p_vs_naive']:.3f} "
                    f"({'berbeda signifikan' if m['dm_p_vs_naive'] < 0.05 else 'selisihnya tidak signifikan'}). "
                    f"Arah naik/turun benar {m['direction_accuracy']:.1f}% "
                    f"(pasar naik pada {m['test_up_share']:.1f}% hari uji)."
                    f"<br><br>Kelima anggota ensemble berbeda pendapat sekitar "
                    f"&plusmn;{dasar['sebaran_pct']:.2f}% untuk prediksi ini.</div>",
                    unsafe_allow_html=True)

        # Grafik 60 hari terakhir + titik prediksi
        hist = df.tail(60)
        pf = go.Figure()
        pf.add_trace(go.Scatter(x=hist["date"], y=hist["ihsg_close"], name="IHSG aktual",
                                line=dict(color=C_IHSG, width=2)))
        band = np.array([-1, 1]) * m["mae"]
        pf.add_trace(go.Scatter(
            x=[dasar["tanggal_h1"]] * 2, y=dasar["ihsg_h1"] + band, mode="lines",
            line=dict(color=BLUE, width=6), opacity=0.25, name=f"± MAE uji ({m['mae']:,.0f} poin)",
            hoverinfo="skip"))
        pf.add_trace(go.Scatter(x=[dasar["tanggal_h0"], dasar["tanggal_h1"]],
                                y=[dasar["ihsg_h0"], dasar["ihsg_h1"]], name="Prediksi LSTM H+1",
                                mode="lines+markers", line=dict(color=BLUE, width=2, dash="dot"),
                                marker=dict(size=[0, 9], color=BLUE)))
        pf.update_layout(**layout_dasar(height=340, title=dict(text="IHSG 60 hari bursa terakhir"),
                                        yaxis=dict(title="Level IHSG")))
        st.plotly_chart(pf, width="stretch")

        # ---------------- Skenario ----------------
        st.subheader("Uji skenario")
        st.markdown("<p class='catatan'>Masukkan BI-Rate hasil RDG terbaru atau kurs hari ini, "
                    "lalu bandingkan prediksinya dengan prediksi di atas.</p>", unsafe_allow_html=True)

        with st.form("skenario"):
            mode_label = st.radio("Data baru dipakai untuk",
                                  [f"Mengganti hari terakhir ({tgl(last['date'], singkat=True)})",
                                   "Menambah satu hari bursa baru"], horizontal=True)
            mode = "perbarui" if mode_label.startswith("Mengganti") else "tambah"
            i1, i2, i3, i4 = st.columns(4)
            bi_in = i1.number_input("BI-Rate (%)", value=float(last["bi_rate"]), step=0.25, format="%.2f")
            kurs_in = i2.number_input("Kurs USD/IDR (Rp)", value=float(last["kurs_usd_idr"]), step=50.0, format="%.0f")
            ihsg_in = i3.number_input("IHSG penutupan", value=float(last["ihsg_close"]), step=10.0, format="%.2f")
            fed_in = i4.number_input("Fed Funds Rate (%)", value=float(last["fed_rate"]), step=0.25, format="%.2f")
            tanggal_in = st.date_input("Tanggal hari bursa baru (hanya untuk mode tambah)",
                                       value=(last["date"] + pd.tseries.offsets.BDay(1)).date())
            kirim = st.form_submit_button("Hitung prediksi skenario", type="primary")

        if kirim:
            try:
                sk = prediksi_h1(df, art, ihsg_close=ihsg_in, bi_rate=bi_in, kurs=kurs_in, fed_rate=fed_in,
                                 mode=mode, tanggal=tanggal_in if mode == "tambah" else None)
                st.session_state["hasil_skenario"] = dict(sk=sk, mode=mode, bi_in=bi_in, kurs_in=kurs_in)
            except Exception as ex:
                st.error(f"Prediksi skenario gagal: {ex}")

        hasil = st.session_state.get("hasil_skenario")
        if hasil:
            sk = hasil["sk"]
            # pembanding: hari sebelum H0 skenario
            ref = df.iloc[-2] if hasil["mode"] == "perbarui" else df.iloc[-1]
            d_bi = hasil["bi_in"] - float(ref["bi_rate"])
            d_kurs = (hasil["kurs_in"] / float(ref["kurs_usd_idr"]) - 1) * 100
            bi_arah, kurs_arah = str(A.arah([d_bi], A.BI_AMBANG)[0]), str(A.arah([d_kurs], 0.1)[0])
            geser = sk["ihsg_h1"] - dasar["ihsg_h1"]

            s1, s2, s3 = st.columns(3)
            s1.metric("Prediksi tanpa data baru", f"{dasar['ihsg_h1']:,.2f}", f"{dasar['perubahan_pct']:+.3f}%")
            s2.metric(f"Prediksi skenario ({tgl(sk['tanggal_h1'], singkat=True)})", f"{sk['ihsg_h1']:,.2f}",
                      f"{sk['perubahan_pct']:+.3f}%")
            s3.metric("Pergeseran akibat skenario", f"{geser:+,.2f} poin",
                      f"{sk['perubahan_pct'] - dasar['perubahan_pct']:+.3f} poin persen", delta_color="off")

            kecil = abs(geser) < m["mae"] * 0.1
            st.markdown(
                f"<p class='catatan'>Skenario: {A.BI_TEKS[bi_arah]} ({d_bi:+.2f} poin), "
                f"{A.KURS_TEKS[kurs_arah]} ({d_kurs:+.2f}%). "
                + ("Pergeserannya jauh lebih kecil dari rata-rata kesalahan model, jadi praktis model "
                   "tidak bereaksi terhadap skenario ini." if kecil else
                   f"Pergeseran ini setara {abs(geser) / m['mae'] * 100:.0f}% dari rata-rata kesalahan "
                   "harian model.")
                + "</p>", unsafe_allow_html=True)

            tbl = A.tabel_kombinasi(A.siapkan_pengamatan(df, "harian"), 0.2)
            pola = A.cari_pola(tbl, bi_arah, kurs_arah)
            if pola is not None:
                st.markdown(
                    f"<p class='catatan'>Di data historis, kondisi <i>{pola['Kondisi']}</i> terjadi "
                    f"{int(pola['n'])} hari. Rata-rata IHSG hari itu {fmt_pct(pola['Rata-rata ΔIHSG H0 (%)'], 3)} "
                    f"({pola['Kesimpulan H0']}); hari berikutnya {fmt_pct(pola['Rata-rata ΔIHSG H+1 (%)'], 3)} "
                    f"({pola['Kesimpulan H+1']}).</p>", unsafe_allow_html=True)

        st.caption("Hasil ini estimasi statistik untuk penelitian, bukan rekomendasi jual-beli saham.")

# ===========================================================================
# TAB 2  DATA PASAR
# ===========================================================================
with tab_data:
    cp, cs = st.columns([2, 3])
    periode = cp.radio("Rentang", ["Semua", "2 tahun", "1 tahun", "6 bulan", "3 bulan"],
                       horizontal=True, index=2)
    seri = cs.multiselect("Variabel", ["IHSG", "BI-Rate", "Kurs USD/IDR", "Fed Funds Rate"],
                          default=["IHSG", "BI-Rate", "Kurs USD/IDR"])
    hari = {"Semua": None, "2 tahun": 730, "1 tahun": 365, "6 bulan": 182, "3 bulan": 91}[periode]
    view = df if hari is None else df[df["date"] >= df["date"].max() - pd.Timedelta(days=hari)]

    fig = go.Figure()
    lay = layout_dasar(height=480, xaxis=dict(domain=[0, 0.86]))
    if "IHSG" in seri:
        fig.add_trace(go.Scatter(x=view["date"], y=view["ihsg_close"], name="IHSG (sumbu kiri)",
                                 line=dict(color=C_IHSG, width=2), yaxis="y"))
        lay["yaxis"] = {**lay["yaxis"], "title": "Level IHSG"}
    if "BI-Rate" in seri or "Fed Funds Rate" in seri:
        lay["yaxis2"] = dict(title="Suku bunga (%)", overlaying="y", side="right", showgrid=False)
        if "BI-Rate" in seri:
            fig.add_trace(go.Scatter(x=view["date"], y=view["bi_rate"], name="BI-Rate (%)",
                                     line=dict(color=C_BI, width=2, shape="hv"), yaxis="y2"))
            ubah = view[view["bi_rate"].diff().fillna(0) != 0]
            fig.add_trace(go.Scatter(x=ubah["date"], y=ubah["bi_rate"], mode="markers", yaxis="y2",
                                     name="Keputusan RDG yang mengubah BI-Rate",
                                     marker=dict(color=C_BI, size=7, symbol="square"),
                                     hovertemplate="RDG: %{y:.2f}%<extra></extra>"))
        if "Fed Funds Rate" in seri:
            fig.add_trace(go.Scatter(x=view["date"], y=view["fed_rate"], name="Fed Funds Rate (%)",
                                     line=dict(color=C_FED, width=1.5, dash="dot", shape="hv"), yaxis="y2"))
    if "Kurs USD/IDR" in seri:
        lay["yaxis3"] = dict(title="Rp per USD", overlaying="y", side="right", anchor="free",
                             position=0.94, showgrid=False)
        fig.add_trace(go.Scatter(x=view["date"], y=view["kurs_usd_idr"], name="Kurs USD/IDR",
                                 line=dict(color=C_KURS, width=1.5), yaxis="y3"))
    fig.update_layout(**lay)
    st.plotly_chart(fig, width="stretch")
    st.markdown("<p class='catatan'>IHSG memakai sumbu kiri. BI-Rate dan Fed Funds Rate memakai sumbu kanan "
                "pertama (%), dan titik persegi menandai RDG yang mengubah BI-Rate. Kurs memakai sumbu kanan "
                "terluar; garis kurs yang naik berarti rupiah melemah.</p>", unsafe_allow_html=True)

    with st.expander("Tabel 50 hari bursa terakhir"):
        tabel = df.tail(50).sort_values("date", ascending=False)[
            ["date", "ihsg_close", "ihsg_return", "bi_rate", "kurs_usd_idr", "fed_rate"]]
        st.dataframe(tabel, hide_index=True, width="stretch", column_config={
            "date": st.column_config.DateColumn("Tanggal", format="DD MMM YYYY"),
            "ihsg_close": st.column_config.NumberColumn("IHSG", format="%.2f"),
            "ihsg_return": st.column_config.NumberColumn("Return IHSG", format="percent"),
            "bi_rate": st.column_config.NumberColumn("BI-Rate (%)", format="%.2f"),
            "kurs_usd_idr": st.column_config.NumberColumn("Kurs USD/IDR", format="%.0f"),
            "fed_rate": st.column_config.NumberColumn("Fed Rate (%)", format="%.2f"),
        })

# ===========================================================================
# TAB 3  HUBUNGAN X DAN Y
# ===========================================================================
with tab_xy:
    ch = A.perubahan_harian(df)

    st.subheader("1. Korelasi perubahan harian")
    kor = A.tabel_korelasi(ch)
    st.dataframe(kor[["Variabel", "r", "p-value", "Kekuatan", "n"]], hide_index=True, width="stretch",
                 column_config={"r": st.column_config.NumberColumn(format="%+.3f"),
                                "p-value": st.column_config.NumberColumn(format="%.4f")})
    for _, row in kor.iterrows():
        st.markdown(f"<p class='catatan' style='margin:0 0 0.25rem 0'>{row['Penjelasan']}</p>",
                    unsafe_allow_html=True)

    vals = ch.drop(columns="date")
    ca, cb = st.columns(2)
    with ca:
        cm = vals.corr()
        hm = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1,
                                  colorscale=[[0, C_TURUN], [0.5, "#ffffff"], [1, BLUE]],
                                  text=cm.round(2).values, texttemplate="%{text}",
                                  colorbar=dict(title="r", thickness=10)))
        hm.update_layout(**layout_dasar(height=380, title=dict(text="Matriks korelasi Pearson"),
                                        hovermode="closest"))
        st.plotly_chart(hm, width="stretch")
    with cb:
        x, y = vals["Δ Kurs USD/IDR (%)"], vals["Δ IHSG (%)"]
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        sc = go.Figure()
        sc.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Hari bursa",
                                marker=dict(size=4, color=C_IHSG, opacity=0.35)))
        sc.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, name=f"Garis regresi (kemiringan {slope:+.2f})",
                                line=dict(color=C_KURS, width=2)))
        sc.update_layout(**layout_dasar(height=380, hovermode="closest",
                                        title=dict(text="Return kurs vs return IHSG"),
                                        xaxis=dict(title="Δ Kurs USD/IDR (%)"),
                                        yaxis=dict(title="Δ IHSG (%)")))
        st.plotly_chart(sc, width="stretch")

    roll = vals["Δ IHSG (%)"].rolling(60).corr(vals["Δ Kurs USD/IDR (%)"])
    rf = go.Figure(go.Scatter(x=ch["date"], y=roll, line=dict(color=C_KURS, width=1.5), name="Korelasi 60 hari"))
    rf.add_hline(y=0, line_color=SUBTLE, line_width=1)
    rf.update_layout(**layout_dasar(height=260, showlegend=False,
                                    title=dict(text="Korelasi bergulir 60 hari: kurs vs IHSG"),
                                    yaxis=dict(range=[-1, 1], title="r")))
    st.plotly_chart(rf, width="stretch")

    st.subheader("2. Kombinasi BI-Rate × kurs terhadap IHSG (H0 dan H+1)")
    k1, k2, k3 = st.columns(3)
    skala = k1.radio("Skala pengamatan", ["harian", "bulanan"], horizontal=True)
    kurs_amb = k2.slider("Batas kurs dianggap tetap (±%)", 0.0, 2.0, 0.1 if skala == "harian" else 0.5, 0.05)
    ihsg_amb = k3.slider("Batas IHSG dianggap tetap (±%)", 0.0, 3.0, 0.2 if skala == "harian" else 1.0, 0.05)

    tbl = A.tabel_kombinasi(A.siapkan_pengamatan(df, skala, kurs_amb), ihsg_amb)
    kol = ["Kondisi", "n", "Rata-rata ΔIHSG H0 (%)", "% hari IHSG naik H0", "Rata-rata ΔIHSG H+1 (%)",
           "% hari IHSG naik H+1", "p H0", "p H+1", "Berlanjut ke H+1?"]
    st.dataframe(tbl[kol], hide_index=True, width="stretch", column_config={
        "Rata-rata ΔIHSG H0 (%)": st.column_config.NumberColumn(format="%+.3f"),
        "Rata-rata ΔIHSG H+1 (%)": st.column_config.NumberColumn(format="%+.3f"),
        "% hari IHSG naik H0": st.column_config.NumberColumn(format="%.0f%%"),
        "% hari IHSG naik H+1": st.column_config.NumberColumn(format="%.0f%%"),
        "p H0": st.column_config.NumberColumn(format="%.3f"),
        "p H+1": st.column_config.NumberColumn(format="%.3f"),
    })
    st.markdown(f"<p class='catatan'>Kondisi dengan n di bawah {A.MIN_N} tidak disimpulkan karena "
                "sampelnya terlalu kecil. BI-Rate hanya berubah pada sebagian kecil RDG, jadi kombinasi "
                "&ldquo;BI-Rate naik/turun&rdquo; jarang terjadi.</p>", unsafe_allow_html=True)

    label = [f"{k} (n={n})" for k, n in zip(tbl["Kondisi"], tbl["n"])]
    bar = go.Figure()
    bar.add_trace(go.Bar(x=label, y=tbl["Rata-rata ΔIHSG H0 (%)"], name="Hari kejadian (H0)", marker_color=INK))
    bar.add_trace(go.Bar(x=label, y=tbl["Rata-rata ΔIHSG H+1 (%)"], name="Hari berikutnya (H+1)",
                         marker_color=BLUE))
    bar.update_layout(**layout_dasar(barmode="group", height=420, hovermode="closest",
                                     title=dict(text="Rata-rata perubahan IHSG per kondisi"),
                                     yaxis=dict(title="Δ IHSG (%)"), xaxis=dict(tickangle=-25)))
    st.plotly_chart(bar, width="stretch")

    st.subheader("3. Teori dibandingkan data")
    teori = {
        ("naik", "naik"): "Cenderung turun: biaya dana naik dan rupiah melemah (modal keluar).",
        ("naik", "turun"): "Campuran: bunga naik menekan valuasi, rupiah menguat menarik modal.",
        ("turun", "naik"): "Campuran: bunga turun mendorong ekspansi, rupiah melemah memicu modal keluar.",
        ("turun", "turun"): "Cenderung naik: pelonggaran moneter dan rupiah menguat.",
        ("tetap", "naik"): "Kurs dominan: rupiah melemah cenderung menekan IHSG.",
        ("tetap", "turun"): "Kurs dominan: rupiah menguat cenderung menopang IHSG.",
    }
    rows = []
    for (bi, ku), harapan in teori.items():
        p = A.cari_pola(tbl, bi, ku)
        rows.append({
            "BI-Rate": bi, "Kurs USD/IDR": f"{ku} ({'rupiah melemah' if ku == 'naik' else 'rupiah menguat'})",
            "Harapan teori": harapan, "n": 0 if p is None else int(p["n"]),
            "Data H0": "belum terjadi" if p is None else p["Kesimpulan H0"].replace("H0: ", ""),
            "Data H+1": "-" if p is None else p["Kesimpulan H+1"].replace("H+1: ", ""),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ===========================================================================
# TAB 4  EVALUASI MODEL
# ===========================================================================
with tab_eval:
    if m is None:
        st.info("Metrik belum ada. Jalankan `python train.py`.")
    else:
        st.markdown(f"<p class='label'>Kesimpulan data uji ({tgl(m['test_start'])} s.d. {tgl(m['test_end'])}, "
                    f"{m['n_test']} hari)</p><p style='font-size:1.25rem; margin:0 0 1rem 0'>{m['verdict']}</p>",
                    unsafe_allow_html=True)

        st.subheader("Perbandingan dengan model pembanding")
        rows = []
        for nama, r in m["models"].items():
            dm = r.get("dm_vs_lstm", {})
            rows.append({"Model": nama, "RMSE": r["rmse"], "MAE": r["mae"], "MAPE (%)": r["mape"],
                         "Arah benar (%)": r["direction_accuracy"], "R² oos (%)": r["r2_oos"] * 100,
                         "p DM vs LSTM": dm.get("p_value")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "MAPE (%)": st.column_config.NumberColumn(format="%.3f"),
            "Arah benar (%)": st.column_config.NumberColumn(format="%.1f"),
            "R² oos (%)": st.column_config.NumberColumn(format="%+.2f"),
            "p DM vs LSTM": st.column_config.NumberColumn(format="%.3f"),
        })
        st.markdown(
            "<p class='catatan'>RMSE, MAE, dan MAPE dalam poin/persen level IHSG; makin kecil makin baik. "
            "R² oos membandingkan kesalahan return terhadap tebakan naif: positif berarti lebih baik dari naif. "
            "p DM adalah p-value uji Diebold-Mariano (koreksi HLN) antara LSTM dan model tersebut; "
            f"p &lt; 0,05 berarti bedanya signifikan. Arah naik/turun dibandingkan dengan proporsi hari naik "
            f"di data uji ({m['test_up_share']:.1f}%).</p>", unsafe_allow_html=True)

        if os.path.exists(config.TEST_PRED_PATH):
            tp = pd.read_csv(config.TEST_PRED_PATH, parse_dates=["date"])
            naif_col = f"pred_{NAIF}"
            # selisih kumulatif kuadrat galat: naik = LSTM lebih baik dari naif
            cum = ((tp["aktual"] - tp[naif_col]) ** 2 - (tp["aktual"] - tp["prediksi"]) ** 2).cumsum()
            e1, e2 = st.columns(2)
            with e1:
                ef = go.Figure()
                ekor = tp.tail(60)
                ef.add_trace(go.Scatter(x=ekor["date"], y=ekor["aktual"], name="Aktual",
                                        line=dict(color=INK, width=2)))
                ef.add_trace(go.Scatter(x=ekor["date"], y=ekor["prediksi"], name="LSTM",
                                        line=dict(color=BLUE, width=1.5)))
                ef.add_trace(go.Scatter(x=ekor["date"], y=ekor[naif_col], name="Naif",
                                        line=dict(color=SUBTLE, width=1.5, dash="dot")))
                ef.update_layout(**layout_dasar(height=360, title=dict(text="60 hari uji terakhir"),
                                                yaxis=dict(title="Level IHSG")))
                st.plotly_chart(ef, width="stretch")
            with e2:
                cf = go.Figure(go.Scatter(x=tp["date"], y=cum, line=dict(color=BLUE, width=1.5),
                                          name="Selisih kumulatif"))
                cf.add_hline(y=0, line_color=SUBTLE, line_width=1)
                cf.update_layout(**layout_dasar(height=360, showlegend=False,
                                                title=dict(text="Keunggulan kumulatif LSTM atas naif"),
                                                yaxis=dict(title="Σ (galat² naif − galat² LSTM)")))
                st.plotly_chart(cf, width="stretch")
            st.markdown("<p class='catatan'>Grafik kanan naik saat LSTM lebih tepat daripada tebakan naif "
                        "dan turun saat sebaliknya. Garis yang berakhir di bawah nol berarti, secara total, "
                        "naif lebih akurat.</p>", unsafe_allow_html=True)

        st.subheader("Walk-forward (uji pada beberapa periode)")
        if wf is None:
            st.markdown("<p class='catatan'>Belum dijalankan. Jalankan "
                        "`python evaluate_walkforward.py`.</p>", unsafe_allow_html=True)
        else:
            fold = pd.DataFrame([{
                "Lipatan": f["fold"], "Periode uji": f"{f['start']} s.d. {f['end']}", "Hari uji": f["n_test"],
                "RMSE LSTM": f["lstm_rmse"], "RMSE naif": f["naive_rmse"],
                "Arah benar (%)": f["lstm_direction"], "p DM": f["dm_p_vs_naive"],
                "LSTM lebih baik": "ya" if f["lstm_wins"] else "tidak"} for f in wf["folds"]])
            st.dataframe(fold, hide_index=True, width="stretch", column_config={
                "RMSE LSTM": st.column_config.NumberColumn(format="%.2f"),
                "RMSE naif": st.column_config.NumberColumn(format="%.2f"),
                "Arah benar (%)": st.column_config.NumberColumn(format="%.1f"),
                "p DM": st.column_config.NumberColumn(format="%.3f"),
            })
            gl, gn = wf["combined"]["LSTM"], wf["combined"][NAIF]
            st.markdown(
                f"<p class='catatan'>Gabungan {sum(f['n_test'] for f in wf['folds'])} hari uji: RMSE LSTM "
                f"{gl['rmse']:.2f} vs naif {gn['rmse']:.2f}; LSTM lebih baik di {wf['lstm_fold_wins']} dari "
                f"{wf['n_folds']} lipatan (p DM gabungan {gn['dm_vs_lstm']['p_value']:.3f}). Setiap lipatan "
                "melatih ulang model hanya dengan data sebelum periode ujinya.</p>", unsafe_allow_html=True)

        if os.path.exists(config.TRAIN_HISTORY_PATH):
            with st.expander("Kurva loss pelatihan"):
                st.image(config.TRAIN_HISTORY_PATH, width="stretch")
                st.markdown(f"<p class='catatan'>Pelatihan berhenti di epoch {m['epochs_stopped']} "
                            f"(early stopping, kesabaran {config.PATIENCE} epoch).</p>", unsafe_allow_html=True)


def p_teks(p):
    return "-" if p is None else f"{p:.3f}"


# ===========================================================================
# TAB 5  KONTRIBUSI VARIABEL (rumusan masalah 3, hipotesis H1-H3)
# ===========================================================================
with tab_kontrib:
    if ex is None or "ablation" not in ex:
        st.info("Belum ada hasil. Jalankan `python experiments.py`.")
    else:
        ab = ex["ablation"]
        uji = (m["test_start"], m["test_end"]) if m else (ab["test_start"], ab["test_end"])
        st.markdown(
            f"<p class='catatan'>Setiap skenario memakai tanggal latih, validasi, dan uji yang sama "
            f"({tgl(uji[0], singkat=True)} s.d. {tgl(uji[1], singkat=True)}, "
            f"{ab['n_test']} hari uji) serta hyperparameter yang sama. Yang berubah hanya fitur masukan "
            "(proposal Tabel 3.2).</p>", unsafe_allow_html=True)
        st.subheader("Skenario M0 sampai M4 pada data uji")
        rows = [{"Kode": k, "Fitur": r["nama"], "RMSE": r["rmse"], "MAE": r["mae"], "MAPE (%)": r["mape"],
                 "Arah benar (%)": r["direction_accuracy"],
                 "p arah (PT)": r["pesaran_timmermann"].get("p_value"),
                 "p DM vs M0": r.get("dm_vs_m0", {}).get("p_value"),
                 "p DM vs naif": r["dm_vs_naif"]["p_value"]} for k, r in ab["models"].items()]
        rows.append({"Kode": "-", "Fitur": NAIF, "RMSE": ab["naif"]["rmse"], "MAE": ab["naif"]["mae"],
                     "MAPE (%)": ab["naif"]["mape"]})
        fmt = {c: st.column_config.NumberColumn(format="%.3f") for c in
               ["MAPE (%)", "p arah (PT)", "p DM vs M0", "p DM vs naif"]}
        fmt.update({c: st.column_config.NumberColumn(format="%.2f") for c in ["RMSE", "MAE"]})
        fmt["Arah benar (%)"] = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=fmt)
        st.markdown("<p class='catatan'>p DM vs M0 menguji apakah menambah BI-Rate dan/atau kurs mengubah "
                    "akurasi dibanding model yang hanya memakai riwayat IHSG (H1 sampai H3). p arah (PT) adalah "
                    "uji Pesaran-Timmermann: p &lt; 0,05 berarti arah naik/turun lebih tepat daripada tebakan "
                    "acak.</p>", unsafe_allow_html=True)

        if "walkforward" in ex:
            st.subheader("Walk-forward 5 lipatan per skenario")
            wfx = ex["walkforward"]
            kode = [k for k in wfx["combined"] if k != "naif"]
            fr = pd.DataFrame([{"Lipatan": f["fold"], "Periode": f"{f['start']} s.d. {f['end']}",
                                **{k: f[k] for k in kode}, "Naif": f["naif"]} for f in wfx["folds"]])
            st.dataframe(fr, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.2f") for k in [*kode, "Naif"]})
            g = wfx["combined"]
            st.markdown(
                f"<p class='catatan'>Gabungan {wfx['n_test']} hari: " + "; ".join(
                    f"{k} RMSE {g[k]['rmse']:.2f}" + (f" (p DM vs M0 {p_teks(g[k].get('dm_vs_m0', {}).get('p_value'))})"
                                                       if k != "M0" else "") for k in kode)
                + f"; naif {g['naif']['rmse']:.2f}.</p>", unsafe_allow_html=True)

        if "horizon" in ex:
            st.subheader("Horizon lebih panjang")
            hz = ex["horizon"]
            st.dataframe(pd.DataFrame([{
                "Horizon": h, "Hari uji": r["n_test"], "RMSE M0": r["M0"]["rmse"], "RMSE M3": r["M3"]["rmse"],
                "RMSE naif": r["naif"]["rmse"], "Arah M3 (%)": r["M3"]["direction_accuracy"],
                "p DM M3 vs M0": r["M3"].get("dm_vs_m0", {}).get("p_value"),
                "p DM M3 vs naif": r["M3"]["dm_vs_naif"]["p_value"]} for h, r in hz.items()]),
                hide_index=True, width="stretch", column_config={
                    **{c: st.column_config.NumberColumn(format="%.2f") for c in ["RMSE M0", "RMSE M3", "RMSE naif"]},
                    "Arah M3 (%)": st.column_config.NumberColumn(format="%.1f"),
                    "p DM M3 vs M0": st.column_config.NumberColumn(format="%.3f"),
                    "p DM M3 vs naif": st.column_config.NumberColumn(format="%.3f")})
            st.markdown("<p class='catatan'>Target H+5 dan H+20 adalah perubahan IHSG kumulatif 5 dan 20 hari "
                        "bursa. Uji DM memakai koreksi untuk galat yang saling tumpang tindih.</p>",
                        unsafe_allow_html=True)

        if "importance" in ex:
            st.subheader("Seberapa bergantung model M3 pada tiap variabel")
            im = ex["importance"]
            nama = list(im["groups"])
            val = [im["groups"][k]["rmse_naik_mean"] for k in nama]
            err = [im["groups"][k]["rmse_naik_std"] for k in nama]
            fi = go.Figure(go.Bar(x=val, y=nama, orientation="h", marker_color=BLUE,
                                  error_x=dict(type="data", array=err, color=MUTED)))
            fi.add_vline(x=0, line_color=SUBTLE, line_width=1)
            fi.update_layout(**layout_dasar(height=240, hovermode="closest", showlegend=False,
                                            title=dict(text="Kenaikan RMSE saat variabel diacak (poin IHSG)")))
            st.plotly_chart(fi, width="stretch")
            st.markdown(f"<p class='catatan'>Permutation importance, {im['n_perm']} kali pengacakan. Makin besar "
                        "kenaikannya, makin bergantung model pada variabel itu. Nilai di sekitar nol berarti model "
                        "praktis tidak memakainya.</p>", unsafe_allow_html=True)

    if eko is not None:
        st.subheader("Analisis pelengkap: keterkaitan tanpa model LSTM")
        kt = eko["kontemporer"]
        gr = eko["var"]["granger"]
        st.dataframe(pd.DataFrame([
            {"Uji": "Kurs hari yang sama terhadap IHSG (OLS, HAC)",
             "Hasil": f"koefisien {kt['ret_kurs']['koef']:+.3f}", "p": kt["ret_kurs"]["p_value"]},
            {"Uji": "Perubahan BI-Rate hari yang sama terhadap IHSG (per bps)",
             "Hasil": f"koefisien {kt['d_bi']['koef']:+.4f}", "p": kt["d_bi"]["p_value"]},
            *[{"Uji": f"Granger: {k.replace('ret_kurs', 'kurs').replace('ret_ihsg', 'IHSG').replace('d_bi', 'BI-Rate')}",
               "Hasil": f"F = {v['F']:.2f}", "p": v["p_value"]} for k, v in gr.items()],
        ]), hide_index=True, width="stretch", column_config={"p": st.column_config.NumberColumn(format="%.4f")})
        st.markdown(f"<p class='catatan'>VAR dengan {eko['var']['lag_aic']} lag (AIC). Granger &ldquo;A &rarr; B&rdquo; "
                    "menguji apakah nilai A pada hari-hari sebelumnya membantu menjelaskan B; ini bukan bukti "
                    "sebab-akibat.</p>", unsafe_allow_html=True)

        es = eko["event_study"]["ringkas"]
        st.markdown("**Event study keputusan RDG Bank Indonesia**")
        st.dataframe(pd.DataFrame([{
            "Keputusan": j, "n": r["n"], "CAR [0] (%)": r["[0]"].get("car_mean"),
            "CAR [-1,+1] (%)": r["[-1,+1]"].get("car_mean"), "p [-1,+1]": r["[-1,+1]"].get("p_value"),
            "CAR [0,+5] (%)": r["[0,+5]"].get("car_mean"), "p [0,+5]": r["[0,+5]"].get("p_value")}
            for j, r in es.items()]), hide_index=True, width="stretch", column_config={
            **{c: st.column_config.NumberColumn(format="%+.3f") for c in
               ["CAR [0] (%)", "CAR [-1,+1] (%)", "CAR [0,+5] (%)"]},
            "p [-1,+1]": st.column_config.NumberColumn(format="%.3f"),
            "p [0,+5]": st.column_config.NumberColumn(format="%.3f")})
        st.markdown(f"<p class='catatan'>CAR = abnormal return kumulatif IHSG di sekitar tanggal RDG, terhadap "
                    f"rata-rata return hari bursa ke {eko['event_study']['estimasi'][0]} s.d. "
                    f"{eko['event_study']['estimasi'][1]}. {eko['event_study']['catatan']}</p>",
                    unsafe_allow_html=True)

# ===========================================================================
# TAB 6  SIMULASI 70/30: kekuatan naik/turun IHSG menurut kondisi BI-Rate & kurs
# ===========================================================================
KELAS_WARNA = {"turun kuat": "#a2191f", "turun lemah": "#ff8389", "datar": "#c6c6c6",
               "naik lemah": "#6fdc8c", "naik kuat": "#0e6027"}
STATUS_TEKS = {"terbukti": "Terbukti di data uji", "arah sama, ujian tidak signifikan": "Arah sama, belum signifikan",
               "gagal (arah berbalik)": "Gagal: berbalik di data uji", "tidak ada pola di latih": "Tidak ada pola",
               "sampel kurang": "Sampel terlalu sedikit"}

with tab_sim:
    if sim is None:
        st.info("Belum ada hasil. Jalankan `python simulasi.py`.")
    else:
        ds, ps = sim["distribusi"], sim["pengaturan"]
        lt, uj = ds["latih"], ds["ujian"]
        st.markdown(
            f"<p class='catatan'>Data dibagi menurut waktu: <b>70% latih</b> ({tgl(lt['mulai'], singkat=True)} s.d. "
            f"{tgl(lt['akhir'], singkat=True)}, {lt['n']} hari) untuk menemukan pola dan melatih LSTM, dan "
            f"<b>30% data uji</b> ({tgl(uj['mulai'], singkat=True)} s.d. {tgl(uj['akhir'], singkat=True)}, {uj['n']} hari) "
            "yang tidak disentuh sampai penilaian. Sebuah pola dianggap <b>terbukti</b> bila signifikan di data latih "
            "dan muncul lagi dengan arah sama serta signifikan di data uji.</p>", unsafe_allow_html=True)

        st.subheader("1. Seberapa kuat IHSG bergerak?")
        fk = go.Figure()
        for k in ps["kelas"]:
            fk.add_trace(go.Bar(y=["Uji 30%", "Latih 70%"], x=[uj["kelas_pct"][k], lt["kelas_pct"][k]], name=k,
                                orientation="h", marker_color=KELAS_WARNA[k],
                                hovertemplate=f"{k}: %{{x:.1f}}% hari<extra></extra>"))
        fk.update_layout(**layout_dasar(barmode="stack", height=200, hovermode="closest",
                                        xaxis=dict(title="% hari bursa", range=[0, 100])))
        st.plotly_chart(fk, width="stretch")
        st.markdown(
            f"<p class='catatan'>Kuat = perubahan harian lebih dari ±{angka_id(ps['ambang_kuat'], 0)}%, datar = di "
            f"dalam ±{angka_id(ps['ambang_datar'], 2)}%. Di data uji hari kuat jauh lebih sering "
            f"({angka_id(uj['kelas_pct']['naik kuat'] + uj['kelas_pct']['turun kuat'], 0)}% vs "
            f"{angka_id(lt['kelas_pct']['naik kuat'] + lt['kelas_pct']['turun kuat'], 0)}% hari), karena pasar "
            f"2025-2026 dua kali lebih bergejolak (simpangan baku {angka_id(uj['std'], 2)}% vs "
            f"{angka_id(lt['std'], 2)}%). Rata-rata hari turun ({angka_id(uj['rata_turun'], 2)}%) sedikit lebih "
            f"dalam daripada hari naik (+{angka_id(uj['rata_naik'], 2)}%), tetapi bedanya tidak signifikan "
            f"(p {angka_id(uj['asimetri_t']['pvalue'], 2)}).</p>", unsafe_allow_html=True)

        st.subheader("2. Kondisi BI-Rate dan kurs: apa yang terjadi pada IHSG?")
        hari_pilih = st.radio("Lihat IHSG pada", ["H0 (hari yang sama)", "H+1 (hari bursa berikutnya)"],
                              horizontal=True, key="sim_hari")
        h = "H0" if hari_pilih.startswith("H0") else "H+1"
        rows = []
        for r in sim["pola"]:
            a, b = r[f"{h} latih"], r[f"{h} ujian"]
            rows.append({"Kondisi": r["kondisi"], "n latih": a["n"], "Rata latih (%)": a["rata"],
                         "Naik kuat latih (%)": a["naik_kuat"], "Turun kuat latih (%)": a["turun_kuat"],
                         "n uji": b["n"], "Rata uji (%)": b["rata"], "Naik kuat uji (%)": b["naik_kuat"],
                         "Turun kuat uji (%)": b["turun_kuat"], "Hasil": STATUS_TEKS.get(r[f"{h} status"])})
        pct = st.column_config.NumberColumn(format="%.1f")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Rata latih (%)": st.column_config.NumberColumn(format="%+.2f"),
            "Rata uji (%)": st.column_config.NumberColumn(format="%+.2f"),
            **{c: pct for c in ["Naik kuat latih (%)", "Turun kuat latih (%)", "Naik kuat uji (%)",
                                "Turun kuat uji (%)"]}})
        terbukti = [r for r in sim["pola"] if r[f"{h} status"] == "terbukti"]
        if h == "H0":
            n_uji = sum(1 for r in sim["pola"] for hh in ("H0", "H+1") if r[f"{hh} latih"]["p"] is not None)
            batas = 0.05 / n_uji
            kuat_bonf = [r["kondisi"] for r in terbukti if r["H0 latih"]["p"] < batas and r["H0 ujian"]["p"] < batas]
            st.markdown(
                f"<p class='catatan'>{len(terbukti)} kondisi terbukti, semuanya berhubungan dengan kurs. Saat rupiah "
                "melemah lebih dari 0,25% IHSG turun pada hari yang sama, dan saat rupiah menguat IHSG naik. Setelah "
                f"koreksi uji berganda (Bonferroni, {n_uji} uji, batas p {angka_id(batas, 4)}), yang tetap lolos di "
                f"latih dan data uji hanya {len(kuat_bonf)}: {'; '.join(kuat_bonf)}. Kombinasi dengan siklus BI-Rate "
                "hanya mewarisi efek kurs, karena siklus BI-Rate sendiri tidak berpola. Hari keputusan RDG terlalu "
                "sedikit untuk disimpulkan.</p>", unsafe_allow_html=True)
        else:
            gagal = [r["kondisi"] for r in sim["pola"] if r["H+1 status"] == "gagal (arah berbalik)"]
            st.markdown(
                "<p class='catatan'>" + (f"{len(terbukti)} kondisi terbukti untuk hari berikutnya. " if terbukti else
                                         "Tidak ada kondisi yang terbukti untuk hari berikutnya. ")
                + "Pola kurs pada H0 tidak berlanjut ke H+1, sehingga kondisi BI-Rate dan kurs hari ini tidak memberi "
                "petunjuk arah IHSG besok."
                + (f" Pola yang tampak di data latih ({'; '.join(gagal)}) justru berbalik arah di data uji."
                   if gagal else "") + "</p>", unsafe_allow_html=True)
        st.markdown("<p class='catatan'>Siklus pengetatan = keputusan RDG terakhir menaikkan BI-Rate; pelonggaran = "
                    "menurunkan. Rata = rata-rata perubahan IHSG (%) pada kelompok hari itu. Uji: t-test Welch "
                    "terhadap hari lainnya.</p>", unsafe_allow_html=True)

        st.subheader("3. Apakah gerak kuat berlanjut besok?")
        lj = sim["lanjutan"]
        fl = go.Figure()
        for b, warna in [("latih", SUBTLE), ("ujian", BLUE)]:
            fl.add_trace(go.Bar(x=ps["kelas"], y=[lj[b]["per_kelas"][k]["rata_besok"] for k in ps["kelas"]],
                                name="Data latih" if b == "latih" else "Data uji", marker_color=warna,
                                customdata=[lj[b]["per_kelas"][k]["n"] for k in ps["kelas"]],
                                hovertemplate="%{x}: besok rata-rata %{y:+.2f}% (n %{customdata})<extra></extra>"))
        fl.add_hline(y=0, line_color=INK, line_width=1)
        fl.update_layout(**layout_dasar(barmode="group", height=300, hovermode="closest",
                                        xaxis=dict(title="Kelas IHSG hari ini"),
                                        yaxis=dict(title="Rata-rata IHSG besok (%)")))
        st.plotly_chart(fl, width="stretch")
        sig = [k for b in lj for k, v in lj[b]["per_kelas"].items() if v["p"] is not None and v["p"] < 0.05]
        st.markdown("<p class='catatan'>" + (
            "Tidak ada kelas yang diikuti perubahan besok yang berbeda signifikan dari hari lain, baik di latih "
            "maupun data uji. Naik kuat hari ini tidak berarti naik lagi besok, dan turun kuat tidak berarti "
            "berbalik naik." if not sig else f"Kelas dengan beda signifikan: {', '.join(sorted(set(sig)))}.")
            + "</p>", unsafe_allow_html=True)

        if "lstm" in sim:
            L = sim["lstm"]
            mt = L["metrik"]
            hp = L["hyperparameter"]
            st.subheader("4. LSTM yang dilatih 70% dan diuji 30%")
            st.markdown(
                f"<p class='catatan'>Model dilatih ulang hanya dengan 70% data (validasi {L['n_validasi']} hari di "
                f"ujungnya), panjang jendela {hp['sequence_length']} hari dan {hp['hidden_size']} unit dipilih dari "
                f"loss validasi, lalu dinilai pada {L['n_ujian']} hari data uji.</p>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Model": k, "RMSE": v["rmse"], "MAE": v["mae"], "MAPE (%)": v["mape"],
                "Arah benar (%)": v["direction_accuracy"], "p arah (PT)": v.get("pesaran_timmermann", {}).get("p_value"),
                "p DM vs naif": v.get("dm_vs_naif", {}).get("p_value")} for k, v in mt.items()]),
                hide_index=True, width="stretch", column_config={
                    "RMSE": st.column_config.NumberColumn(format="%.2f"),
                    "MAE": st.column_config.NumberColumn(format="%.2f"),
                    "MAPE (%)": st.column_config.NumberColumn(format="%.3f"),
                    "Arah benar (%)": st.column_config.NumberColumn(format="%.1f"),
                    "p arah (PT)": st.column_config.NumberColumn(format="%.3f"),
                    "p DM vs naif": st.column_config.NumberColumn(format="%.3f")})

            c1, c2 = st.columns(2)
            with c1:
                kq = pd.DataFrame(L["kuintil"])
                fq = go.Figure()
                fq.add_trace(go.Bar(x=kq["kuintil"], y=kq["naik_kuat"], name="Besok naik kuat", marker_color=C_NAIK))
                fq.add_trace(go.Bar(x=kq["kuintil"], y=kq["turun_kuat"], name="Besok turun kuat", marker_color=C_TURUN))
                fq.update_layout(**layout_dasar(barmode="group", height=320, hovermode="closest",
                                                title=dict(text="Hari data uji dikelompokkan menurut prediksi LSTM"),
                                                xaxis=dict(title="Kuintil prediksi (1 = paling pesimis, 5 = paling optimis)",
                                                           tickmode="array", tickvals=[1, 2, 3, 4, 5]),
                                                yaxis=dict(title="% hari")))
                st.plotly_chart(fq, width="stretch")
                st.markdown(
                    f"<p class='catatan'>Bila LSTM bisa membaca arah, kuintil 5 akan banyak naik kuat dan sedikit "
                    f"turun kuat. Yang terjadi, keduanya naik bersama (kuintil 5: naik kuat "
                    f"{angka_id(kq['naik_kuat'].iloc[-1], 0)}%, turun kuat {angka_id(kq['turun_kuat'].iloc[-1], 0)}%; "
                    f"kuintil 1: {angka_id(kq['naik_kuat'].iloc[0], 0)}% dan {angka_id(kq['turun_kuat'].iloc[0], 0)}%). "
                    "Prediksi LSTM lebih mencerminkan pasar yang sedang bergejolak daripada arah IHSG.</p>",
                    unsafe_allow_html=True)
            with c2:
                kp = L["kelas_prediksi_pct"]
                st.markdown("<p class='label'>Kelas yang diprediksi LSTM di data uji</p>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([{"Kelas": k, "Prediksi LSTM (%)": kp[k],
                                            "Aktual (%)": uj["kelas_pct"][k]} for k in ps["kelas"]]),
                             hide_index=True, width="stretch", column_config={
                                 "Prediksi LSTM (%)": st.column_config.NumberColumn(format="%.1f"),
                                 "Aktual (%)": st.column_config.NumberColumn(format="%.1f")})
                st.markdown(
                    f"<p class='catatan'>Simpangan baku prediksi LSTM {angka_id(L['pred_std_pct'], 3)}% per hari, "
                    f"sedangkan IHSG aktual {angka_id(L['aktual_std_pct'], 2)}%. Korelasi peringkat prediksi dengan "
                    f"aktual {angka_id(L['spearman']['rho'], 3)} (p {angka_id(L['spearman']['p_value'], 2)}). Pada "
                    f"{L['arah_hari_kuat']['n']} hari data uji yang bergerak kuat, arah LSTM benar "
                    f"{angka_id(L['arah_hari_kuat']['benar_pct'], 1)}%.</p>", unsafe_allow_html=True)

        if "validitas" in sim:
            v = sim["validitas"]
            st.subheader("5. Uji ketahanan hasil")
            bf = v["bonferroni"]
            st.markdown(
                f"<p class='catatan'>Ada {bf['n_uji']} uji sekaligus, jadi batas signifikansi dikoreksi Bonferroni "
                f"menjadi p &lt; {angka_id(bf['batas'], 4)} (Dunn, 1961). Yang tetap lolos di latih dan data uji: "
                f"{'; '.join(x['kondisi'] + ' (' + x['hari'] + ')' for x in bf['hasil'] if x['lolos']) or 'tidak ada'}."
                "</p>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{
                "Data": {"ujian": "uji"}.get(o["bagian"], o["bagian"]), "IHSG pada": o["hari"], "Koef. kurs": o["b_kurs"], "p kurs": o["p_kurs"],
                "Koef. ΔBI-Rate": o["b_bi"], "p ΔBI-Rate": o["p_bi"], "R²": o["r2"], "n": o["n"]}
                for o in v["ols_hac"]["hasil"]]), hide_index=True, width="stretch", column_config={
                    "Koef. kurs": st.column_config.NumberColumn(format="%+.3f"),
                    "p kurs": st.column_config.NumberColumn(format="%.4f"),
                    "Koef. ΔBI-Rate": st.column_config.NumberColumn(format="%+.3f"),
                    "p ΔBI-Rate": st.column_config.NumberColumn(format="%.3f"),
                    "R²": st.column_config.NumberColumn(format="%.4f")})
            st.markdown(
                f"<p class='catatan'>Regresi OLS perubahan IHSG (%) terhadap perubahan kurs (%) dan perubahan BI-Rate "
                f"(poin persen), galat baku Newey-West dengan {v['ols_hac']['maxlags']} lag (Newey &amp; West, 1987). "
                "Uji Mann-Whitney (Mann &amp; Whitney, 1947) yang tidak mengasumsikan distribusi normal memberi "
                "kesimpulan sama.</p>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Ambang kuat (±%)": s["kuat"], "Ambang kurs (±%)": s["kurs"],
                                        "Pola H0 terbukti": s["h0"], "Pola H+1 terbukti": s["h1"]}
                                       for s in v["sensitivitas"]]), hide_index=True, width="stretch")
            if "lstm" in v:
                lv = v["lstm"]
                st.markdown(
                    f"<p class='catatan'>Diagnosis LSTM: prediksi berkisar {angka_id(lv['pred_min'], 3)}% s.d. "
                    f"{angka_id(lv['pred_max'], 3)}%, {angka_id(lv['pct_pred_positif'], 0)}% di antaranya positif "
                    f"(IHSG aktual naik {angka_id(lv['pct_aktual_positif'], 0)}% hari). Korelasi Spearman prediksi "
                    f"dengan volatilitas 20 hari terakhir {angka_id(lv['vs_volatilitas']['rho'], 2)}, dengan besar "
                    f"gerak besok {angka_id(lv['vs_besar_gerak']['rho'], 2)}, tetapi dengan arah besok "
                    f"{angka_id(lv['vs_arah']['rho'], 2)} (p {angka_id(lv['vs_arah']['p'], 2)}). Prediksi LSTM "
                    "mengikuti tingkat gejolak, bukan arah IHSG.</p>", unsafe_allow_html=True)

# ===========================================================================
# TAB 7  REKAM JEJAK (forward test)
# ===========================================================================
with tab_log:
    if not os.path.exists(config.PREDICTION_LOG):
        st.info("Log belum ada. Jalankan `python daily_update.py` setiap hari bursa, atau pasang jadwal "
                "otomatis dengan `python daily_update.py --pasang-jadwal 17:30`.")
    else:
        lg = pd.read_csv(config.PREDICTION_LOG, parse_dates=["tanggal_h0", "tanggal_h1"])
        sel = lg.dropna(subset=["aktual_h1"])
        st.markdown("<p class='catatan'>Setiap prediksi dicatat sebelum IHSG H+1 diketahui, memakai model yang "
                    "tidak dilatih ulang. Ini uji maju yang benar-benar di luar sampel.</p>",
                    unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediksi tercatat", f"{len(lg)}", help=f"{len(sel)} sudah punya nilai aktual")
        if len(sel):
            r2.metric("MAE LSTM vs naif", f"{sel['galat_poin'].abs().mean():,.1f} poin",
                      f"naif {sel['galat_naif_poin'].abs().mean():,.1f} poin", delta_color="off")
            r3.metric("Arah benar", f"{sel['arah_benar'].astype(bool).mean() * 100:.0f}%",
                      f"dari {len(sel)} hari", delta_color="off")
        st.dataframe(lg.sort_values("tanggal_h0", ascending=False), hide_index=True, width="stretch",
                     column_config={
                         "tanggal_h0": st.column_config.DateColumn("H0", format="DD MMM YYYY"),
                         "tanggal_h1": st.column_config.DateColumn("H+1", format="DD MMM YYYY"),
                         "ihsg_h0": st.column_config.NumberColumn("IHSG H0", format="%.2f"),
                         "prediksi_h1": st.column_config.NumberColumn("Prediksi H+1", format="%.2f"),
                         "aktual_h1": st.column_config.NumberColumn("Aktual H+1", format="%.2f"),
                         "galat_poin": st.column_config.NumberColumn("Galat (poin)", format="%+.2f"),
                         "galat_naif_poin": st.column_config.NumberColumn("Galat naif", format="%+.2f"),
                         "perubahan_pct": st.column_config.NumberColumn("Prediksi Δ%", format="%+.3f"),
                         "sebaran_pct": st.column_config.NumberColumn("Sebaran ±%", format="%.3f"),
                         "versi_model": st.column_config.TextColumn("Versi model"),
                         "awal_input": st.column_config.TextColumn("Input sejak"),
                     })
        st.caption("Sesuai Subbab 3.12, setiap baris menyimpan waktu prediksi, tanggal observasi terakhir (H0), "
                   "versi model, awal rentang input, dan status kualitas data. Baris lama sebelum kolom versi "
                   "model ditambahkan dibiarkan kosong.")

# ===========================================================================
# TAB 8  METODOLOGI (ringkasan Bab III naskah)
# ===========================================================================
with tab_met:
    st.subheader("Data dan periode (Subbab 3.4 sampai 3.7)")
    st.markdown("""
| Variabel | Sumber | Catatan |
|---|---|---|
| IHSG (Y) | Yahoo Finance `^JKSE` | harga penutupan harian |
| BI-Rate (X1) | siaran pers RDG Bank Indonesia | 72 keputusan, berlaku sejak tanggal pengumuman |
| Kurs USD/IDR (X2) | JISDOR Bank Indonesia | Yahoo `IDR=X` hanya cadangan bila JISDOR gagal diunduh |
| Fed Funds Rate | FRED | hanya untuk skenario M4 |

Periode penelitian 4 Januari 2021 s.d. 20 November 2026: 1.416 hari bursa, tersisa 1.406 baris fitur setelah 10 baris
pemanasan. Data latih 916 sequence (18 Mar 2021 s.d. 24 Des 2024), validasi 169 (27 Des 2024 s.d. 22 Sep 2025), dan
data uji 282 (23 Sep 2025 s.d. 20 Nov 2026), dibagi urut waktu tanpa pengacakan.
""")

    st.subheader("Fitur dan target")
    st.markdown("""
| Fitur | Isi | Peran |
|---|---|---|
| `ret_ihsg` | log return IHSG hari t | riwayat Y |
| `vol_ihsg` | simpangan baku `ret_ihsg` 10 hari | risiko pasar |
| `ret_kurs` | log return kurs JISDOR (Bank Indonesia) | X2 (positif = rupiah melemah) |
| `d_bi` | perubahan BI-Rate, satuan 25 bps | X1, kejutan kebijakan |
| `bi_rate` | level BI-Rate | X1, rezim suku bunga |

| Skenario (Tabel 3.2) | Fitur | Hipotesis |
|---|---|---|
| M0 | `ret_ihsg`, `vol_ihsg` | pembanding riwayat IHSG saja |
| M1 | M0 + `d_bi`, `bi_rate` | H1: BI-Rate menambah informasi |
| M2 | M0 + `ret_kurs` | H2: kurs menambah informasi |
| M3 | kelima fitur (model inti) | H3: keduanya bersama |
| M4 | M3 + selisih BI-Rate dan Fed Funds Rate | uji tambahan |

Hasil tiap skenario ada di tab Kontribusi variabel.

Target: log return IHSG dari hari t ke t+1, lalu dikembalikan ke level:
""")
    st.latex(r"\widehat{\text{IHSG}}_{t+1} = \text{IHSG}_t \cdot e^{\hat r_{t+1}}")
    st.markdown("Fitur memakai perubahan, bukan level, karena level IHSG dan kurs terus bergeser; "
                "model yang diberi level cenderung hanya menyalin harga kemarin. Standardisasi z-score "
                "di-fit hanya pada data latih.")

    st.subheader("Model dan pelatihan (Subbab 3.9)")
    tun = load_json(config.TUNING_PATH)
    grid_l = sorted({g["sequence_length"] for g in tun["grid"]}) if tun else [5, 10, 20, 40]
    grid_h = sorted({g["hidden_size"] for g in tun["grid"]}) if tun else [16, 32, 64]
    st.markdown(f"""
| Parameter | Nilai |
|---|---|
| Arsitektur | LSTM {config.NUM_LAYERS} lapis, {config.HIDDEN_SIZE} unit tersembunyi |
| Panjang jendela | {config.SEQUENCE_LENGTH} hari bursa |
| Grid tuning | jendela {"/".join(map(str, grid_l))} × unit {"/".join(map(str, grid_h))}, 3 seed, dipilih dari loss validasi (bukan data uji) |
| Dropout / weight decay | {config.DROPOUT} / {config.WEIGHT_DECAY} |
| Optimizer / loss | Adam (Kingma & Ba, 2014), learning rate {config.LEARNING_RATE} / MSE |
| Batch / epoch | {config.BATCH_SIZE} / maksimal {config.EPOCHS}, early stopping kesabaran {config.PATIENCE} |
| Pembagian data | {int(config.TRAIN_SPLIT * 100)}% latih (15% terakhirnya validasi), {int(round((1 - config.TRAIN_SPLIT) * 100))}% uji, urut waktu |
| Ensemble | rata-rata {len(config.SEEDS)} model, seed {", ".join(map(str, config.SEEDS))} |
| Pembanding (Subbab 3.8) | naif, rata-rata historis, ARIMA (orde dipilih AIC data latih), regresi linear |
""")
    st.markdown("Model untuk prediksi H+1 di dashboard memakai bobot yang dibekukan setelah evaluasi dan tidak "
                "dilatih ulang setiap hari, sehingga rekam jejak forward test tetap jujur.")

    st.subheader("Metode pengujian dan rujukan (Subbab 3.10 dan 3.11)")
    st.markdown("""
| Tujuan | Metode | Rujukan |
|---|---|---|
| Galat prediksi | RMSE, MAE, MAPE (Tabel 3.3) | |
| LSTM vs pembanding | Diebold-Mariano dengan koreksi HLN | Diebold & Mariano (1995); Harvey dkk. (1997) |
| Akurasi arah | uji Pesaran-Timmermann | Pesaran & Timmermann (1992) |
| Daya prediksi vs rata-rata historis | R² out-of-sample | Campbell & Thompson (2008) |
| Kestabilan antarperiode | walk-forward 5 lipatan, expanding window | Tashman (2000) |
| Ketergantungan pada variabel | permutation importance, 30 pengacakan | Breiman (2001) |
| Efek hari yang sama | OLS, galat baku Newey-West | Newey & West (1987) |
| Efek hari berikutnya | VAR, lag dipilih AIC, uji Granger, impulse response | Sims (1980); Akaike (1974); Granger (1969) |
| Reaksi terhadap RDG | studi peristiwa CAR dan uji Patell | Brown & Warner (1985); Patell (1976) |
| Simulasi 70/30 | t-test Welch, Mann-Whitney, koreksi Bonferroni | Welch (1947); Mann & Whitney (1947); Dunn (1961) |
| Diagnosis LSTM | korelasi Spearman, uji khi-kuadrat | Spearman (1904); Pearson (1900) |
""")

    st.subheader("Hubungan teori (Subbab 2.1.5)")
    st.markdown("""
| BI-Rate | Kurs USD/IDR | Harapan terhadap IHSG | Mekanisme |
|---|---|---|---|
| Naik | Naik (rupiah melemah) | Cenderung turun | Biaya kredit naik dan imbal hasil instrumen lain lebih menarik; rupiah melemah menaikkan biaya impor dan kewajiban valas. |
| Turun | Turun (rupiah menguat) | Cenderung naik | Biaya kredit turun dan persepsi risiko membaik. |
| Naik | Turun (rupiah menguat) | Campuran | Tekanan bunga diimbangi stabilitas mata uang. |
| Turun | Naik (rupiah melemah) | Campuran | Stimulus bunga diimbangi risiko biaya impor. |

Bukti terdahulu tidak seragam: Kewal (2012) menemukan kurs signifikan tetapi suku bunga tidak, sedangkan Gery (2023)
menemukan keduanya signifikan. Hasil penelitian ini (Subbab 4.9): kurs berkaitan kuat dengan IHSG pada hari yang sama,
tetapi tidak membantu meramal hari berikutnya; BI-Rate tidak signifikan pada data harian.
""")

    st.subheader("Etika, validitas, dan batasan (Subbab 3.14 dan 4.9)")
    st.markdown("""
1. Prediksi bukan kepastian kondisi pasar dan bukan rekomendasi investasi.
2. Korelasi bukan kausalitas: angka di tab Hubungan X dan Y menunjukkan gerak bersama, bukan sebab-akibat.
3. Validitas dijaga dengan pembagian data menurut waktu, Standardisasi yang di-fit hanya pada data latih, pemilihan hyperparameter dari data validasi, dan pembandingan dengan model naif.
4. Return harian IHSG mendekati acak, sejalan dengan hipotesis pasar efisien bentuk lemah (Fama, 1970). LSTM yang setara dengan model naif adalah temuan yang sah, bukan kegagalan teknis.
5. BI-Rate hanya berubah 19 kali selama periode (12 naik, 7 turun) dan keputusan RDG umumnya sudah diantisipasi pasar; studi peristiwa tanpa data ekspektasi tidak dapat memisahkan kejutan kebijakan.
6. Frekuensi pembaruan IHSG, BI-Rate, dan JISDOR berbeda; kinerja model dapat turun saat terjadi perubahan rezim pasar.
""")

    st.subheader("Status berkas")
    berkas = [("IHSG", config.IHSG_CSV), ("BI-Rate", config.BI_RATE_CSV), ("Kurs USD/IDR", config.KURS_CSV),
              ("Fed Funds Rate", config.FEDFUNDS_CSV), ("Dataset latih", config.DATASET_CSV),
              ("Model LSTM", config.MODEL_PATH), ("Tuning", config.TUNING_PATH),
              ("Metrik evaluasi", config.METRICS_PATH), ("Walk-forward", config.WALKFORWARD_PATH),
              ("Eksperimen M0-M4", config.EXPERIMENTS_PATH), ("Ekonometrika", config.ECONOMETRICS_PATH),
              ("Simulasi 70/30", os.path.join(config.MODEL_DIR, "simulasi.json")),
              ("Log prediksi harian", config.PREDICTION_LOG)]
    st.dataframe(pd.DataFrame([{"Berkas": n, "Path": p, "Ada": "ya" if os.path.exists(p) else "belum"}
                               for n, p in berkas]), hide_index=True, width="stretch")
