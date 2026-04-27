# coding=utf-8
"""SVC Mapper: non-autoregressive frame-level codec_0 prediction.

Replaces the failed AR approach. Instead of generating codec tokens one by one,
predicts ALL target codec_0 tokens at once via a small transformer encoder.

Architecture:
    Input per frame: source_codec_embeds (all 16 CB summed) + F0_embed + speaker_embed
    Model: N-layer bidirectional transformer encoder
    Output per frame: target_codec_0 logits (vocab_size)

Then sub-talker generates codec_1-15 from predicted codec_0 + hidden states.
"""

import torch
from torch import nn
import math


class SVCMapper(nn.Module):
    """Frame-level bidirectional transformer for codec_0 mapping."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 3072,
        num_codebooks: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks

        # Input projection
        self.input_norm = nn.LayerNorm(hidden_size)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4096, hidden_size))
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads: predict ALL codebooks (separate head per codebook)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_codebooks)
        ])

    def forward(
        self,
        source_embed: torch.Tensor,
        f0_embed: torch.Tensor,
        speaker_embed: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            source_embed: (B, T, D) summed codec embeddings from source audio
            f0_embed: (B, T, D) F0 embeddings from pitch reference
            speaker_embed: (B, D) target speaker embedding
            padding_mask: (B, T) True = valid, False = padding

        Returns:
            all_logits: list of (B, T, vocab_size) per codebook, or stacked (B, T, num_codebooks, vocab_size)
        """
        B, T, D = source_embed.shape

        # Information bottleneck on source during training
        if self.training:
            frame_mask = (torch.rand(B, T, 1, device=source_embed.device) > 0.5).float()
            source_embed = source_embed * frame_mask
            source_embed = source_embed + torch.randn_like(source_embed) * 0.5

        # Combine inputs
        x = source_embed + f0_embed + speaker_embed.unsqueeze(1).expand(-1, T, -1)
        x = self.input_norm(x)
        x = x + self.pos_encoding[:, :T]

        if padding_mask is not None:
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Predict ALL codebooks
        all_logits = [head(x) for head in self.output_heads]  # list of (B, T, V)
        return all_logits

    def predict(self, source_embed, f0_embed, speaker_embed, padding_mask=None, temperature=1.0):
        """Predict all codebook tokens. Returns (B, T, num_codebooks)."""
        all_logits = self.forward(source_embed, f0_embed, speaker_embed, padding_mask)
        tokens = []
        for logits in all_logits:
            if temperature <= 0:
                t = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:2])
            tokens.append(t)
        return torch.stack(tokens, dim=-1)  # (B, T, num_codebooks)
