# coding=utf-8
"""F0-to-embedding projector for SVC conditioning."""

import torch
from torch import nn


class F0Projector(nn.Module):
    """Projects F0 values to continuous embeddings for talker conditioning.

    Applies log(1 + F0) scaling, then a linear projection to hidden_size.
    Unvoiced frames (F0 == 0) use a separate learned embedding.
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(1, hidden_size)
        self.unvoiced_embed = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.unvoiced_embed, std=0.02)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """Convert F0 contour to embeddings.

        Args:
            f0: (T,) or (B, T) F0 values in Hz. 0.0 = unvoiced.

        Returns:
            (T, hidden_size) or (B, T, hidden_size) embeddings.
        """
        squeeze = False
        if f0.ndim == 1:
            f0 = f0.unsqueeze(0)
            squeeze = True

        voiced_mask = f0 > 0  # (B, T)

        # Log-scale transform, cast to model dtype
        log_f0 = torch.log1p(f0.float()).unsqueeze(-1).to(self.proj.weight.dtype)  # (B, T, 1)
        embed = self.proj(log_f0)  # (B, T, hidden_size)

        # Replace unvoiced frames with learned embedding (cast for autocast dtype match)
        unvoiced_mask = ~voiced_mask  # (B, T)
        embed[unvoiced_mask] = self.unvoiced_embed.to(embed.dtype)

        if squeeze:
            embed = embed.squeeze(0)
        return embed
