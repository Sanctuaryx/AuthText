from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Clasificador binario IA vs humano basado en:
      - Embedding propio (inicializado aleatoriamente)
      - BiLSTM bidireccional
      - Capa de atención para hacer pooling de los estados ocultos
      - Capa final lineal que devuelve logits (no aplica sigmoide dentro)

    La atención aprende un peso por token, de modo que el modelo puede
    concentrarse en las partes más informativas del texto.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_index: int = 0,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_index,
        )
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained_embeddings tiene forma {pretrained_embeddings.shape}, "
                    f"pero se esperaba ({vocab_size}, {embed_dim})"
                )
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Atención: proyecta cada hidden state a un escalar (score)
        self.attn_w = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # binario IA vs humano

    def forward(self, input_ids, lengths):
        """
        input_ids: (batch, seq_len)
        lengths: (batch,) longitudes reales (sin padding)
        """
        embedded = self.embedding(input_ids)  # (B, L, E)

        # Empaquetamos según longitudes reales
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, L, 2*H)

        # Atención: score por token
        attn_scores = self.attn_w(out).squeeze(-1)  # (B, L)

        # Máscara de padding: donde length < posición → -inf
        mask = (
            torch.arange(out.size(1), device=lengths.device)[None, :]
            >= lengths[:, None]
        )
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # Pesos de atención normalizados
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)

        # Vector de texto = combinación lineal de hidden states
        text_vec = (out * attn_weights).sum(dim=1)  # (B, 2*H)

        text_vec = self.dropout(text_vec)
        logits = self.fc(text_vec).squeeze(1)  # (B,)
        return logits
