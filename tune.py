"""
tune.py
=======
Pencarian hyperparameter sederhana (grid) untuk SEQUENCE_LENGTH dan
HIDDEN_SIZE. Pemilihan HANYA berdasarkan loss validasi (bagian akhir data
latih); data uji tidak disentuh sama sekali, supaya hasil uji tetap jujur.

Jalankan:
    python tune.py
Lalu salin kombinasi terbaik ke config.py (SEQUENCE_LENGTH, HIDDEN_SIZE).
"""
import itertools
import json

import numpy as np

import config
import modeling as M

SEQ_GRID = [5, 10, 20, 40]
HIDDEN_GRID = [16, 32, 64]
TUNE_SEEDS = config.SEEDS[:3]


def main():
    df = M.load_frame()
    train_end = int(len(df) * config.TRAIN_SPLIT)
    rows = []
    for L, H in itertools.product(SEQ_GRID, HIDDEN_GRID):
        sp = M.split_sequences(df, train_end, train_end, L)
        _, vals, hists = M.fit_ensemble(sp, seeds=TUNE_SEEDS, hidden=H)
        rows.append({"sequence_length": L, "hidden_size": H, "val_loss": float(np.mean(vals)),
                     "val_loss_std": float(np.std(vals)), "epochs": [len(h) for h in hists]})
        print(f"L={L:>3} H={H:>3} | loss validasi {rows[-1]['val_loss']:.4f} +- {rows[-1]['val_loss_std']:.4f}")

    rows.sort(key=lambda r: r["val_loss"])
    best = rows[0]
    print(f"\nTerbaik (validasi): SEQUENCE_LENGTH={best['sequence_length']}, HIDDEN_SIZE={best['hidden_size']}")
    with open(config.TUNING_PATH, "w", encoding="utf-8") as f:
        json.dump({"best": best, "grid": rows, "note": "dipilih dari loss validasi, bukan data uji"}, f, indent=2)
    print(f"Tersimpan: {config.TUNING_PATH}")


if __name__ == "__main__":
    main()
