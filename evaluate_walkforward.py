"""
evaluate_walkforward.py
=======================
Evaluasi walk-forward (expanding window). Satu kali pemisahan latih/uji bisa
kebetulan menguntungkan atau merugikan model. Di sini 50% data terakhir
dibagi menjadi WALKFORWARD_FOLDS lipatan berurutan; untuk tiap lipatan model
dilatih ulang memakai SEMUA data sebelum lipatan itu, lalu diuji pada lipatan
tersebut. Dengan begitu model diuji di beberapa kondisi pasar yang berbeda.

Jalankan:
    python evaluate_walkforward.py
"""
import json

import numpy as np
import pandas as pd

import config
import modeling as M

NAIF = "Naif (besok = hari ini)"


def main():
    df = M.load_frame()
    n = len(df)
    edges = np.linspace(int(n * 0.5), n, config.WALKFORWARD_FOLDS + 1).astype(int)
    folds, all_pred, all_act, all_last, all_base = [], [], [], [], {}

    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]), 1):
        sp = M.split_sequences(df, a, b, config.SEQUENCE_LENGTH)
        models, _, _ = M.fit_ensemble(sp)
        rows = sp["rows_te"]
        pred = M.predict_ensemble(models, sp["X_te"]) * sp["target_std"]
        act, last = df["target"].values[rows], df["ihsg_close"].values[rows]
        base = M.baselines(df, a, rows)

        r_lstm, r_naif = M.score(pred, act, last), M.score(base[NAIF], act, last)
        dm = M.diebold_mariano(act - pred, act - base[NAIF])
        folds.append({
            "fold": k, "start": str(df["date"].iloc[rows[0]].date()), "end": str(df["date"].iloc[rows[-1]].date()),
            "n_train": int(a), "n_test": int(len(rows)),
            "lstm_rmse": r_lstm["rmse"], "naive_rmse": r_naif["rmse"],
            "lstm_mape": r_lstm["mape"], "naive_mape": r_naif["mape"],
            "lstm_direction": r_lstm["direction_accuracy"], "lstm_r2_oos": r_lstm["r2_oos"],
            "dm_p_vs_naive": dm["p_value"], "lstm_wins": r_lstm["rmse"] < r_naif["rmse"],
        })
        f = folds[-1]
        print(f"Lipatan {k}: {f['start']} s.d. {f['end']} | RMSE LSTM {f['lstm_rmse']:.2f} vs naif "
              f"{f['naive_rmse']:.2f} | arah {f['lstm_direction']:.1f}% | DM p {f['dm_p_vs_naive']:.3f}")
        all_pred.append(pred)
        all_act.append(act)
        all_last.append(last)
        for name, p in base.items():
            all_base.setdefault(name, []).append(p)

    pred, act, last = map(np.concatenate, (all_pred, all_act, all_last))
    gabungan = {"LSTM": M.score(pred, act, last)}
    for name, ps in all_base.items():
        p = np.concatenate(ps)
        gabungan[name] = M.score(p, act, last)
        gabungan[name]["dm_vs_lstm"] = M.diebold_mariano(act - pred, act - p)

    wins = sum(f["lstm_wins"] for f in folds)
    print(f"\nGabungan {len(act)} hari uji: RMSE LSTM {gabungan['LSTM']['rmse']:.2f} vs naif "
          f"{gabungan[NAIF]['rmse']:.2f} | LSTM menang di {wins}/{len(folds)} lipatan "
          f"| DM p {gabungan[NAIF]['dm_vs_lstm']['p_value']:.3f}")

    with open(config.WALKFORWARD_PATH, "w", encoding="utf-8") as fh:
        json.dump({"folds": folds, "combined": gabungan, "lstm_fold_wins": int(wins),
                   "n_folds": len(folds), "sequence_length": config.SEQUENCE_LENGTH,
                   "hidden_size": config.HIDDEN_SIZE}, fh, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"Tersimpan: {config.WALKFORWARD_PATH}")


if __name__ == "__main__":
    main()
