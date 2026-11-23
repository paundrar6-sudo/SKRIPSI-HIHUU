"""
features.py
===========
Satu-satunya tempat fitur model dihitung. train.py, evaluate_walkforward.py,
dan predict_realtime.py semuanya memanggil fungsi di sini supaya fitur saat
pelatihan dan saat prediksi dijamin identik.

Kenapa fitur STASIONER (perubahan, bukan level)?
    Level IHSG dan kurs terus bergeser (tidak stasioner). Jika model diberi level
    mentah, ia cenderung menghafal "harga kemarin" dan gagal saat data uji berada
    di rentang harga yang belum pernah dilihat. Return dan selisih rata-ratanya
    stabil sepanjang waktu, sehingga pola yang dipelajari di data latih masih
    berlaku di data uji.

Fitur per hari bursa t:
    ret_ihsg        log return IHSG hari t                      (riwayat Y)
    vol_ihsg        simpangan baku ret_ihsg 10 hari terakhir    (risiko pasar)
    ret_kurs        log return kurs USD/IDR hari t              (X2; + = rupiah melemah)
    d_bi            perubahan BI-Rate hari t, satuan 25 bps     (X1: kejutan kebijakan)
    bi_rate         level BI-Rate                               (X1: rezim suku bunga)
    selisih_bi_fed  BI-Rate dikurangi Fed Funds Rate            (daya tarik aset rupiah)

Target: log return IHSG dari hari t ke t+1.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["ret_ihsg", "vol_ihsg", "ret_kurs", "d_bi", "bi_rate", "selisih_bi_fed"]
VOL_WINDOW = 10
WARMUP = VOL_WINDOW          # baris awal yang belum punya vol_ihsg lengkap
BI_STEP = 0.25               # d_bi dinyatakan dalam kelipatan 25 bps, tidak distandarkan
CLIP = 5.0                   # batas nilai fitur setelah standardisasi (redam outlier ekstrem)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom fitur ke frame harian (wajib: ihsg_close, kurs_usd_idr,
    bi_rate, fed_rate). Baris pemanasan awal dibuang."""
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["ret_ihsg"] = np.log(out["ihsg_close"]).diff()
    out["vol_ihsg"] = out["ret_ihsg"].rolling(VOL_WINDOW).std()
    out["ret_kurs"] = np.log(out["kurs_usd_idr"]).diff()
    out["d_bi"] = out["bi_rate"].diff()
    out["selisih_bi_fed"] = out["bi_rate"] - out["fed_rate"]
    return out.iloc[WARMUP:].reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = log return IHSG hari berikutnya. Baris terakhir (belum ada besok) dibuang."""
    out = df.copy()
    out["ihsg_close_next"] = out["ihsg_close"].shift(-1)
    out["target"] = np.log(out["ihsg_close_next"] / out["ihsg_close"])
    return out.dropna(subset=["target"]).reset_index(drop=True)


class FeatureScaler:
    """Standardisasi z-score yang di-fit HANYA pada data latih.
    d_bi tidak distandarkan (hampir selalu 0, sesekali +-1 langkah 25 bps)."""

    def fit(self, df: pd.DataFrame):
        self.mean_ = df[FEATURE_COLUMNS].mean()
        self.std_ = df[FEATURE_COLUMNS].std().replace(0, 1.0)
        self.mean_["d_bi"], self.std_["d_bi"] = 0.0, BI_STEP
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = (df[FEATURE_COLUMNS] - self.mean_) / self.std_
        return z.clip(-CLIP, CLIP).values.astype(np.float32)


def make_sequences(features: np.ndarray, seq_len: int) -> np.ndarray:
    """Jendela geser: sequence ke-i berisi baris [i, i+seq_len-1] dan
    dipasangkan dengan target baris i+seq_len-1."""
    n = len(features) - seq_len + 1
    return np.stack([features[i:i + seq_len] for i in range(n)]) if n > 0 else np.empty((0, seq_len, features.shape[1]))
