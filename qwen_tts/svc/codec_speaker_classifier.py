# coding=utf-8
"""Codec-level speaker classifier for cross-speaker SVC training.

Operates directly on 12Hz Mimi codec tokens (T, 16 codebooks) without needing
to decode to waveform. Supports both discrete token input (for pretraining on
real target_codec) and soft logits input (for cross-speaker supervision where
gradients must flow back to the mapper).
"""
import torch
from torch import nn
import torch.nn.functional as F


class CodecSpeakerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 3072, num_codebooks: int = 16,
                 emb_dim: int = 96, hidden: int = 512, num_heads: int = 4,
                 num_layers: int = 2, num_speakers: int = 13, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.codebook_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(num_codebooks)
        ])
        self.input_proj = nn.Linear(num_codebooks * emb_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, num_speakers)

    def _pool(self, x, mask):
        # x: (B, T, hidden); mask: (B, T) bool
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward_tokens(self, tokens: torch.Tensor, mask: torch.Tensor = None):
        """tokens: (B, T, num_codebooks) int64. Used for classifier pretraining on real codec."""
        embs = [self.codebook_embeds[i](tokens[..., i]) for i in range(self.num_codebooks)]
        x = torch.cat(embs, dim=-1)
        return self._forward_features(x, mask)

    def forward_logits(self, logits_list, mask: torch.Tensor = None):
        """logits_list: list of num_codebooks tensors of shape (B, T, V). Soft-embedded via softmax."""
        embs = []
        for i, logits in enumerate(logits_list):
            probs = F.softmax(logits, dim=-1)  # (B, T, V)
            emb = probs @ self.codebook_embeds[i].weight  # (B, T, emb_dim)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self._forward_features(x, mask)

    def _forward_features(self, x, mask):
        x = self.input_proj(x)
        key_pad = (~mask) if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=key_pad)
        pooled = self._pool(x, mask)
        return self.head(pooled)
