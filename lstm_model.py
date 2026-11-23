"""
lstm_model.py
=============
Arsitektur LSTM (Long Short-Term Memory) untuk memprediksi return IHSG hari
bursa berikutnya dari fitur stasioner (lihat features.py):

    [return IHSG, volatilitas IHSG, return kurs USD/IDR, perubahan BI-Rate,
     level BI-Rate, selisih BI-Rate - Fed Funds Rate]

Kenapa LSTM? -- LSTM adalah varian recurrent neural network yang dirancang
untuk menangkap pola dan ketergantungan jangka pendek-menengah dalam data
deret waktu, karena memiliki mekanisme "gate" (forget/input/output gate) yang
menjaga informasi penting dari beberapa langkah waktu sebelumnya tanpa
mengalami vanishing gradient seperti RNN sederhana.

Model sengaja dibuat kecil (1 layer, 32 unit). Data harian 2021-2026 hanya
~1.400 baris; model besar akan menghafal noise data latih (overfitting).
"""
import torch
import torch.nn as nn

import config


class LSTMPredictor(nn.Module):
    """LSTM many-to-one: menerima SEQUENCE_LENGTH hari fitur, menghasilkan satu
    nilai: return IHSG besok dalam satuan simpangan baku data latih."""

    def __init__(
        self,
        input_size: int = len(config.FEATURE_COLUMNS),
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        """
        x shape : (batch, sequence_length, num_features)
        output  : (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        return self.head(lstm_out[:, -1, :])


if __name__ == "__main__":
    model = LSTMPredictor()
    dummy = torch.randn(4, config.SEQUENCE_LENGTH, len(config.FEATURE_COLUMNS))
    print("Bentuk input :", dummy.shape)
    print("Bentuk output:", model(dummy).shape)
    print(model)
