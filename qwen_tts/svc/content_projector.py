# coding=utf-8
"""Content projector: maps codec-space content signal to text-projected space."""

import torch
from torch import nn


class ContentProjector(nn.Module):
    """Projects source codec embeddings from codec-embedding space
    to the text-projected space expected by the talker transformer.

    Architecture matches text_projection (ResizeMLP pattern):
    hidden_size → intermediate → hidden_size
    """

    def __init__(self, hidden_size: int = 2048, intermediate_size: int = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size
        self.up = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act = nn.SiLU()
        self.down = nn.Linear(intermediate_size, hidden_size, bias=True)
        self._init_weights()

    def _init_weights(self):
        # Initialize close to identity to preserve information at start of training
        nn.init.eye_(self.up.weight[:self.up.out_features, :self.up.in_features].data[:min(self.up.out_features, self.up.in_features), :min(self.up.out_features, self.up.in_features)])
        nn.init.zeros_(self.up.bias)
        nn.init.eye_(self.down.weight[:self.down.out_features, :self.down.in_features].data[:min(self.down.out_features, self.down.in_features), :min(self.down.out_features, self.down.in_features)])
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., hidden_size) content signal in codec space.
        Returns:
            (..., hidden_size) projected to text-projected space.
        """
        return self.down(self.act(self.up(x)))
