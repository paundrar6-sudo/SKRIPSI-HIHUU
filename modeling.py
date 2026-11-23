"""
modeling.py
===========
Inti pelatihan & evaluasi yang dipakai bersama oleh train.py, tune.py, dan
evaluate_walkforward.py, supaya ketiganya memakai prosedur yang persis sama.

Isi:
    load_frame()          dataset harian + fitur + target
    split_sequences()     potong latih/validasi/uji TANPA kebocoran data
    fit_ensemble()        latih beberapa LSTM (seed berbeda) + early stopping
    predict_ensemble()    rata-rata prediksi ensemble
    baselines()           pembanding: naif, rata-rata historis, ARIMA, regresi linear
    score()               RMSE/MAE/MAPE di skala IHSG + akurasi arah + R2 out-of-sample
    diebold_mariano()     uji signifikansi beda akurasi dua model (koreksi HLN)
"""
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

import config
import features as F
from lstm_model import LSTMPredictor

torch.set_num_threads(max(1, torch.get_num_threads()))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_frame() -> pd.DataFrame:
    """dataset_features.csv (semua hari) -> fitur + target (hari terakhir tanpa besok dibuang)."""
    raw = pd.read_csv(config.FEATURES_CSV, parse_dates=["date"])
    return F.add_target(F.add_features(raw))


def split_sequences(df: pd.DataFrame, train_end: int, test_end: int, seq_len: int,
                    val_split: float = config.VAL_SPLIT) -> dict:
    """Baris [0, train_end) = latih (bagian akhirnya validasi), [train_end, test_end) = uji.
    Scaler fitur & skala target di-fit HANYA pada baris latih (tanpa validasi)."""
    val_start = int(train_end * (1 - val_split))
    scaler = F.FeatureScaler().fit(df.iloc[:val_start])
    target_std = float(df["target"].iloc[:val_start].std())

    X_all = F.make_sequences(scaler.transform(df), seq_len)
    end_row = np.arange(seq_len - 1, len(df))            # baris terakhir tiap sequence
    y_all = (df["target"].values[end_row] / target_std).astype(np.float32)

    def pick(lo, hi):
        m = (end_row >= lo) & (end_row < hi)
        return X_all[m], y_all[m], end_row[m]

    X_tr, y_tr, _ = pick(0, val_start)
    X_va, y_va, _ = pick(val_start, train_end)
    X_te, y_te, rows_te = pick(train_end, test_end)
    return dict(scaler=scaler, target_std=target_std, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
                X_te=X_te, y_te=y_te, rows_te=rows_te, val_start=val_start)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
def _train_one(X_tr, y_tr, X_va, y_va, seed, hidden, layers, dropout, epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMPredictor(input_size=X_tr.shape[2], hidden_size=hidden, num_layers=layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1)),
                        batch_size=config.BATCH_SIZE, shuffle=True, generator=g)
    Xv, yv = torch.tensor(X_va), torch.tensor(y_va).unsqueeze(1)

    best, best_state, wait, hist = np.inf, None, 0, []
    for _ in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xv), yv).item()
        hist.append((tr_loss / len(X_tr), va_loss))
        if va_loss < best - 1e-6:
            best, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, hist


def fit_ensemble(split: dict, seeds=config.SEEDS, hidden=config.HIDDEN_SIZE, layers=config.NUM_LAYERS,
                 dropout=config.DROPOUT, epochs=config.EPOCHS, patience=config.PATIENCE):
    models, val_losses, hists = [], [], []
    for s in seeds:
        m, v, h = _train_one(split["X_tr"], split["y_tr"], split["X_va"], split["y_va"],
                             s, hidden, layers, dropout, epochs, patience)
        models.append(m)
        val_losses.append(v)
        hists.append(h)
    return models, val_losses, hists


def predict_ensemble(models, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0)
    with torch.no_grad():
        xt = torch.tensor(X)
        return np.mean([m(xt).numpy().ravel() for m in models], axis=0)


# ---------------------------------------------------------------------------
# Baseline (semua memprediksi log return besok; dilatih hanya pada data latih)
# ---------------------------------------------------------------------------
def baselines(df: pd.DataFrame, train_end: int, rows_te: np.ndarray) -> dict:
    y = df["target"].values
    out = {"Naif (besok = hari ini)": np.zeros(len(rows_te)),
           "Rata-rata historis": np.full(len(rows_te), y[:train_end].mean())}

    # ARIMA pada return: orde dipilih dengan AIC di data latih, parameter dibekukan,
    # lalu dipakai untuk prakiraan 1 langkah di data uji (tanpa melihat masa depan).
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # r[t+1] = target[t] (return dari hari t ke t+1), jadi r memuat juga return hari terakhir
        r = np.r_[df["ret_ihsg"].values[0], y]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits = {o: ARIMA(r[:train_end + 1], order=o, trend="c").fit()
                    for o in [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0)]}
            order, res = min(fits.items(), key=lambda kv: kv[1].aic)
            full = res.apply(r)
        # prediksi in-sample pada indeks t+1 = prakiraan return besok dari info s.d. hari t
        pred_next = np.asarray(full.predict(start=1, end=len(r) - 1))   # pred_next[t] -> target[t]
        out["ARIMA"] = pred_next[rows_te]
        print(f"[info] ARIMA: orde terpilih (AIC data latih) = {order}")
    except Exception as e:  # statsmodels opsional
        print(f"[info] ARIMA dilewati: {e}")

    # Regresi linear (OLS) dengan fitur yang sama dengan LSTM, hari terakhir saja
    Xf = np.column_stack([np.ones(len(df)), df[F.FEATURE_COLUMNS].values])
    beta, *_ = np.linalg.lstsq(Xf[:train_end], y[:train_end], rcond=None)
    out["Regresi linear"] = Xf[rows_te] @ beta
    return out


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def score(pred_ret, actual_ret, last_close) -> dict:
    pred_lvl = last_close * np.exp(pred_ret)
    act_lvl = last_close * np.exp(actual_ret)
    err = act_lvl - pred_lvl
    nz = pred_ret != 0
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err / act_lvl)) * 100),
        "direction_accuracy": float(np.mean(np.sign(pred_ret[nz]) == np.sign(actual_ret[nz])) * 100) if nz.any() else None,
        # R2 out-of-sample (Campbell & Thompson 2008) terhadap prakiraan return nol (= baseline naif).
        "r2_oos": float(1 - np.sum((actual_ret - pred_ret) ** 2) / np.sum(actual_ret ** 2)),
    }


def diebold_mariano(e_model, e_bench, h: int = 1) -> dict:
    """Uji Diebold-Mariano (loss kuadrat) dengan koreksi sampel kecil Harvey-Leybourne-Newbold.
    H0: akurasi sama. stat < 0 berarti model lebih akurat daripada pembanding."""
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    n = len(d)
    d_mean = d.mean()
    gamma = [np.sum((d[k:] - d_mean) * (d[:n - k] - d_mean)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return {"dm_stat": None, "p_value": None}
    dm = d_mean / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.cdf(-abs(dm), df=n - 1)
    return {"dm_stat": float(dm), "p_value": float(p)}
