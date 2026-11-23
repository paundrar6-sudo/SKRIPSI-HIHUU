"""
train.py
========
Melatih model LSTM final dan mengevaluasinya pada data uji (20% periode
terakhir), lalu membandingkannya dengan beberapa baseline secara jujur.

Yang dipelajari model: dari SEQUENCE_LENGTH hari terakhir fitur stasioner
(features.py) -> log return IHSG hari bursa berikutnya. Return itu lalu
diubah ke level IHSG:   IHSG_besok = IHSG_hari_ini x exp(return_prediksi)

Alur:
    1. Muat dataset + bentuk fitur (features.py)
    2. Bagi kronologis: latih | validasi (early stopping) | uji
       Scaler di-fit HANYA pada data latih (mencegah kebocoran data)
    3. Latih ensemble 5 LSTM (seed berbeda), tiap model berhenti saat loss
       validasi tidak membaik (early stopping)
    4. Evaluasi di skala IHSG asli + bandingkan dengan baseline:
       naif (besok = hari ini), rata-rata historis, ARIMA, regresi linear
    5. Uji Diebold-Mariano: apakah beda akurasi LSTM vs baseline signifikan?
    6. Latih ulang ensemble pada SELURUH data (latih + uji) untuk prediksi H+1
       ke depan, lalu simpan model, scaler, metrik, prediksi uji, dan grafik

Jalankan:
    python train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import modeling as M

NAIF = "Naif (besok = hari ini)"


def main():
    df = M.load_frame()
    L = config.SEQUENCE_LENGTH
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    print(f"Dataset: {n} hari bursa ({df['date'].min().date()} s.d. {df['date'].max().date()})")

    sp = M.split_sequences(df, train_end, n, L)
    rows_te = sp["rows_te"]
    print(f"Sequence latih {len(sp['X_tr'])} | validasi {len(sp['X_va'])} | uji {len(sp['X_te'])} "
          f"(uji {df['date'].iloc[rows_te[0]].date()} s.d. {df['date'].iloc[rows_te[-1]].date()})")

    print(f"\nMelatih ensemble {len(config.SEEDS)} LSTM (early stopping, patience {config.PATIENCE})...")
    models, val_losses, hists = M.fit_ensemble(sp)
    for s, v, h in zip(config.SEEDS, val_losses, hists):
        print(f"  seed {s:>4}: berhenti di epoch {len(h):3d}, loss validasi terbaik {v:.4f}")

    pred = M.predict_ensemble(models, sp["X_te"]) * sp["target_std"]
    actual = df["target"].values[rows_te]
    last = df["ihsg_close"].values[rows_te]

    hasil = {"LSTM": M.score(pred, actual, last)}
    preds = {"LSTM": pred}
    for name, p in M.baselines(df, train_end, rows_te).items():
        hasil[name] = M.score(p, actual, last)
        preds[name] = p

    err = {k: actual - v for k, v in preds.items()}
    for k in hasil:
        if k != "LSTM":
            hasil[k]["dm_vs_lstm"] = M.diebold_mariano(err["LSTM"], err[k])

    up_share = float(np.mean(actual > 0) * 100)
    print("\n=== Evaluasi data uji (skala IHSG asli) ===")
    print(f"{'Model':<26}{'RMSE':>9}{'MAE':>9}{'MAPE%':>8}{'Arah%':>8}{'R2oos%':>8}{'DM p':>8}")
    for k, r in hasil.items():
        dm = r.get("dm_vs_lstm", {}).get("p_value")
        arah = f"{r['direction_accuracy']:.1f}" if r["direction_accuracy"] is not None else "-"
        print(f"{k:<26}{r['rmse']:>9.2f}{r['mae']:>9.2f}{r['mape']:>8.3f}{arah:>8}"
              f"{r['r2_oos'] * 100:>8.2f}{(f'{dm:.3f}' if dm is not None else '-'):>8}")
    print(f"Proporsi hari naik di data uji: {up_share:.1f}%")
    print("DM p = p-value uji Diebold-Mariano LSTM vs model itu (p < 0.05 -> beda signifikan).")

    lstm, naif = hasil["LSTM"], hasil[NAIF]
    dm = hasil[NAIF]["dm_vs_lstm"]
    if lstm["rmse"] < naif["rmse"]:
        verdict = "LSTM lebih akurat daripada baseline naif"
        verdict += " dan bedanya signifikan (p < 0.05)." if dm["p_value"] is not None and dm["p_value"] < 0.05 \
            else ", tetapi bedanya TIDAK signifikan secara statistik."
    else:
        verdict = "LSTM belum mengungguli baseline naif pada data uji."
    print("\nKesimpulan:", verdict)

    # -----------------------------------------------------------------
    # Simpan hasil evaluasi
    # -----------------------------------------------------------------
    # tanggal H+1; baris terakhir tidak punya baris berikutnya di df -> hari bursa berikutnya
    tgl_h1 = df["date"].shift(-1).fillna(df["date"] + pd.tseries.offsets.BDay(1))
    dates_te = pd.to_datetime(tgl_h1.iloc[rows_te]).values
    out = pd.DataFrame({"date": dates_te, "aktual": last * np.exp(actual), "hari_ini": last})
    for k, p in preds.items():
        out["prediksi" if k == "LSTM" else f"pred_{k}"] = last * np.exp(p)
    out.to_csv(config.TEST_PRED_PATH, index=False)

    metrics = {
        "rmse": lstm["rmse"], "mae": lstm["mae"], "mape": lstm["mape"],
        "direction_accuracy": lstm["direction_accuracy"], "r2_oos": lstm["r2_oos"],
        "baseline_rmse": naif["rmse"], "baseline_mae": naif["mae"], "baseline_mape": naif["mape"],
        "dm_stat_vs_naive": dm["dm_stat"], "dm_p_vs_naive": dm["p_value"],
        "test_up_share": up_share, "verdict": verdict,
        "models": hasil,
        "n_train": int(len(sp["X_tr"])), "n_val": int(len(sp["X_va"])), "n_test": int(len(sp["X_te"])),
        "epochs": config.EPOCHS, "epochs_stopped": [len(h) for h in hists],
        "sequence_length": L, "hidden_size": config.HIDDEN_SIZE, "num_layers": config.NUM_LAYERS,
        "seeds": config.SEEDS, "features": config.FEATURE_COLUMNS,
        "test_start": str(pd.Timestamp(dates_te[0]).date()),
        "test_end": str(pd.Timestamp(dates_te[-1]).date()),
        "data_end": str(df["date"].iloc[-1].date()),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------
    # Model produksi: latih ulang pada seluruh data agar prediksi H+1 memakai
    # informasi terbaru. Early stopping memakai 15% periode paling akhir.
    # -----------------------------------------------------------------
    print("\nMelatih ulang ensemble pada seluruh data untuk prediksi H+1...")
    full = M.split_sequences(df, n, n, L)
    final_models, _, _ = M.fit_ensemble(full)
    torch.save([m.state_dict() for m in final_models], config.MODEL_PATH)
    joblib.dump({"feature_scaler": full["scaler"], "target_std": full["target_std"]}, config.SCALER_PATH)
    print(f"Model tersimpan: {config.MODEL_PATH}")

    # -----------------------------------------------------------------
    # Grafik untuk presentasi/sidang
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tr, va = zip(*hists[0])
    axes[0].plot(tr, color="#0f62fe", label="latih")
    axes[0].plot(va, color="#da1e28", label="validasi")
    axes[0].set_title(f"Kurva loss (seed {config.SEEDS[0]}, early stopping)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (return terstandar)")
    axes[0].legend()
    axes[1].plot(out["date"], out["aktual"], label="IHSG aktual", color="#161616")
    axes[1].plot(out["date"], out["prediksi"], label="Prediksi LSTM", color="#0f62fe", linestyle="--")
    axes[1].set_title(f"Data uji: MAPE LSTM {lstm['mape']:.3f}% vs naif {naif['mape']:.3f}%")
    axes[1].set_ylabel("IHSG")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.TRAIN_HISTORY_PATH, dpi=140)
    print(f"Grafik tersimpan: {config.TRAIN_HISTORY_PATH}")


if __name__ == "__main__":
    main()
