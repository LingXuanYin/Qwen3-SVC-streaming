# coding=utf-8
"""Unit tests for F0 extraction, alignment, pitch shift, and projection."""

import numpy as np
import pytest
import torch

from qwen_tts.svc.f0_extractor import (
    align_f0_to_codec,
    extract_f0,
    pitch_shift,
    EXTRACTION_SR,
)
from qwen_tts.svc.f0_projector import F0Projector

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_sine(freq_hz: float, duration_s: float = 1.0, sr: int = EXTRACTION_SR) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (np.sin(2 * np.pi * freq_hz * t) * 0.8).astype(np.float32)


class TestExtractF0:
    def test_440hz_sine(self):
        audio = _make_sine(440.0)
        f0 = extract_f0(audio, EXTRACTION_SR, device=DEVICE)
        assert f0.ndim == 1
        voiced = f0[f0 > 0]
        assert len(voiced) > 0
        assert abs(voiced.mean().item() - 440.0) < 20.0

    def test_silent_audio(self):
        audio = np.zeros(EXTRACTION_SR, dtype=np.float32)
        f0 = extract_f0(audio, EXTRACTION_SR, device=DEVICE)
        assert (f0 == 0).all()

    def test_resample_from_48k(self):
        # 48kHz input should be resampled internally
        audio = _make_sine(440.0, sr=48000)
        f0 = extract_f0(audio, sr=48000, device=DEVICE)
        voiced = f0[f0 > 0]
        assert abs(voiced.mean().item() - 440.0) < 20.0


class TestAlignF0:
    def test_downsample_to_target_length(self):
        f0_100hz = torch.cat([torch.full((50,), 440.0), torch.zeros(50)])
        aligned = align_f0_to_codec(f0_100hz, target_length=12)
        assert aligned.shape == (12,)

    def test_preserves_voiced_mean(self):
        f0_100hz = torch.full((100,), 300.0)
        aligned = align_f0_to_codec(f0_100hz, target_length=12)
        assert abs(aligned.mean().item() - 300.0) < 1.0

    def test_unvoiced_stays_zero(self):
        f0_100hz = torch.zeros(100)
        aligned = align_f0_to_codec(f0_100hz, target_length=12)
        assert (aligned == 0).all()


class TestPitchShift:
    def test_octave_up(self):
        f0 = torch.tensor([440.0, 0.0, 220.0])
        shifted = pitch_shift(f0, 12.0)
        assert abs(shifted[0].item() - 880.0) < 0.1
        assert shifted[1].item() == 0.0
        assert abs(shifted[2].item() - 440.0) < 0.1

    def test_octave_down(self):
        f0 = torch.tensor([440.0])
        shifted = pitch_shift(f0, -12.0)
        assert abs(shifted[0].item() - 220.0) < 0.1

    def test_zero_shift_identity(self):
        f0 = torch.tensor([440.0, 0.0, 880.0])
        shifted = pitch_shift(f0, 0.0)
        assert torch.equal(shifted, f0)


class TestF0Projector:
    def test_output_shape_1d(self):
        proj = F0Projector(1024).to(DEVICE)
        f0 = torch.tensor([440.0, 0.0, 880.0]).to(DEVICE)
        embed = proj(f0)
        assert embed.shape == (3, 1024)

    def test_output_shape_batch(self):
        proj = F0Projector(1024).to(DEVICE)
        f0 = torch.tensor([[440.0, 0.0], [220.0, 880.0]]).to(DEVICE)
        embed = proj(f0)
        assert embed.shape == (2, 2, 1024)

    def test_unvoiced_frames_identical(self):
        proj = F0Projector(1024).to(DEVICE)
        f0 = torch.tensor([0.0, 440.0, 0.0]).to(DEVICE)
        embed = proj(f0)
        assert torch.equal(embed[0], embed[2])

    def test_voiced_frames_differ(self):
        proj = F0Projector(1024).to(DEVICE)
        f0 = torch.tensor([440.0, 880.0]).to(DEVICE)
        embed = proj(f0)
        assert not torch.equal(embed[0], embed[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
