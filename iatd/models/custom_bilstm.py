from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        pad_index: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_index,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # binario IA vs humano

    def forward(self, input_ids, lengths):
        """
        input_ids: (batch, seq_len)
        lengths: (batch,)
        """
        embedded = self.embedding(input_ids)  # (B, L, E)

        # opcional: empaquetar por longitudes reales
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, B, hidden_dim)
        # concatenamos direcciones de la última capa
        h_last = torch.cat(
            [h_n[-2], h_n[-1]], dim=1
        )  # (B, 2*hidden_dim) bidireccional

        h_last = self.dropout(h_last)
        logits = self.fc(h_last).squeeze(1)  # (B,)

        return logits  # luego aplicamos BCEWithLogitsLoss fuera
