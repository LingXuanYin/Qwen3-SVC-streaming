# coding=utf-8
"""SVC Mapper v3: HuBERT content + F0 + speaker → 16 codebooks.

Key design:
- Adversarial F0 removal via gradient reversal: content projection MUST NOT
  be able to predict F0. This forces true pitch-invariance (HuBERT alone has
  cosine 0.78 for ±5st shift, which is not enough).
- Non-AR frame-level prediction (no exposure bias).
- 16 separate codebook heads.
"""
import torch
from torch import nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SVCMapperHubert(nn.Module):
    """Content (HuBERT 768d) + F0 (cond_dim) + Speaker (cond_dim) → 16 codebooks.

    With adversarial F0 predictor on projected content to force pitch-invariance.
    """

    def __init__(
        self,
        content_dim: int = 768,
        cond_dim: int = 2048,
        hidden_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 3072,
        num_codebooks: int = 16,
        dropout: float = 0.1,
        f0_num_bins: int = 360,
        num_speakers: int = 0,  # 0 disables adversarial speaker head
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.f0_num_bins = f0_num_bins
        self.num_speakers = num_speakers

        self.content_proj = nn.Linear(content_dim, hidden_size)
        self.cond_proj = nn.Linear(cond_dim, hidden_size)

        self.input_norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 4096, hidden_size))
        nn.init.normal_(self.pos_encoding, std=0.02)

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

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_codebooks)
        ])

        # Adversarial F0 predictor: tries to predict F0 from projected content.
        # Via gradient reversal, this forces content_proj to produce F0-independent features.
        self.adv_f0_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, f0_num_bins),
        )

        # Adversarial speaker classifier on pooled content (one label per sample).
        # Forces content_proj to drop speaker identity so mapper must rely on speaker_embed.
        if num_speakers > 0:
            self.adv_spk_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, num_speakers),
            )
        else:
            self.adv_spk_predictor = None

    def forward(self, content, f0_embed, speaker_embed, padding_mask=None,
                adv_lambda=1.0, adv_spk_lambda=1.0):
        """
        Returns:
            codec_logits: list of (B, T, V) for 16 codebooks
            adv_f0_logits: (B, T, f0_num_bins) from gradient-reversed content projection
            adv_spk_logits: (B, num_speakers) or None — gradient-reversed pooled content
        """
        B, T, _ = content.shape

        c = self.content_proj(content)

        # Adversarial F0 on per-frame reversed content
        c_rev_f0 = grad_reverse(c, adv_lambda)
        adv_f0_logits = self.adv_f0_predictor(c_rev_f0)

        # Adversarial speaker on pooled reversed content (one sample label)
        adv_spk_logits = None
        if self.adv_spk_predictor is not None:
            c_rev_spk = grad_reverse(c, adv_spk_lambda)
            if padding_mask is not None:
                denom = padding_mask.sum(dim=1, keepdim=True).clamp(min=1).to(c_rev_spk.dtype)
                pooled = (c_rev_spk * padding_mask.unsqueeze(-1).to(c_rev_spk.dtype)).sum(dim=1) / denom
            else:
                pooled = c_rev_spk.mean(dim=1)
            adv_spk_logits = self.adv_spk_predictor(pooled)

        cond = self.cond_proj(f0_embed) + self.cond_proj(speaker_embed.unsqueeze(1).expand(-1, T, -1))
        x = self.input_norm(c + cond)
        x = x + self.pos_encoding[:, :T]

        if padding_mask is not None:
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        codec_logits = [head(x) for head in self.output_heads]
        return codec_logits, adv_f0_logits, adv_spk_logits

    def predict(self, content, f0_embed, speaker_embed, padding_mask=None, temperature=1.0):
        codec_logits, _, _ = self.forward(content, f0_embed, speaker_embed, padding_mask,
                                          adv_lambda=0.0, adv_spk_lambda=0.0)
        tokens = []
        for logits in codec_logits:
            if temperature <= 0:
                t = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:2])
            tokens.append(t)
        return torch.stack(tokens, dim=-1)

    @staticmethod
    def f0_to_bin(f0_hz, n_bins=360, f_min=32.7, f_max=1975.5):
        """Convert F0 in Hz to semitone bin index (for classification).

        Unvoiced (f0=0) → bin 0. Voiced → bin 1 to n_bins-1 (log-scale).
        """
        voiced = f0_hz > 0
        # Log-scale bin: evenly spaced in log2(f0)
        log_min = torch.log2(torch.tensor(f_min, device=f0_hz.device))
        log_max = torch.log2(torch.tensor(f_max, device=f0_hz.device))
        log_f0 = torch.log2(f0_hz.clamp(min=f_min))
        bins = ((log_f0 - log_min) / (log_max - log_min) * (n_bins - 1)).clamp(0, n_bins - 2).long() + 1
        bins = torch.where(voiced, bins, torch.zeros_like(bins))
        return bins
