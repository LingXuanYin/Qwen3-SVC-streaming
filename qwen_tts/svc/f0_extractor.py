# coding=utf-8
"""F0 (fundamental frequency) extraction, alignment, and pitch shifting for SVC."""

import math
from typing import Optional

import librosa
import numpy as np
import torch

# Lazy-loaded globals
_fcpe_model = None
_fcpe_device = None

EXTRACTION_SR = 16000
EXTRACTION_HOP = 160  # 16000 / 160 = 100 Hz output rate
CODEC_FRAME_RATE = 12  # 12 Hz codec


def _get_fcpe_model(device: str = "cuda:0") -> "torchfcpe.models_infer.InferCFNaiveMelPE":
    """Lazily load FCPE model to the specified device and cache it."""
    global _fcpe_model, _fcpe_device
    if _fcpe_model is None or _fcpe_device != device:
        import torchfcpe
        _fcpe_model = torchfcpe.spawn_bundled_infer_model(device)
        _fcpe_device = device
    return _fcpe_model


def extract_f0(
    audio: np.ndarray,
    sr: int,
    device: str = "cuda:0",
    threshold: float = 0.006,
) -> torch.Tensor:
    """Extract F0 contour from audio using FCPE (neural pitch estimator).

    Args:
        audio: 1-D float waveform.
        sr: Sample rate of the input audio.
        device: Device for inference.
        threshold: Voicing threshold (lower = more voiced frames detected).

    Returns:
        1-D float tensor of F0 values in Hz at 100 Hz frame rate.
        Unvoiced frames are 0.0.
    """
    if audio.ndim != 1:
        audio = audio.squeeze()
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")

    # Resample to 16 kHz if needed
    if sr != EXTRACTION_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=EXTRACTION_SR)

    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
    audio_tensor = audio_tensor.to(device)

    model = _get_fcpe_model(device)
    with torch.no_grad():
        f0 = model.infer(
            audio_tensor,
            sr=EXTRACTION_SR,
            decoder_mode="local_argmax",
            threshold=threshold,
        )
    # f0 shape: (1, T_frames, 1) → squeeze to (T_frames,)
    return f0.squeeze().cpu()


def align_f0_to_codec(
    f0_100hz: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Downsample 100 Hz F0 contour to 12 Hz codec frame rate.

    Uses segment-wise mean pooling over voiced frames. If all source frames
    in a segment are unvoiced, the output is 0.

    Args:
        f0_100hz: 1-D F0 tensor at 100 Hz.
        target_length: Exact number of 12 Hz codec frames to output.

    Returns:
        1-D F0 tensor of length target_length at 12 Hz.
    """
    src_len = f0_100hz.shape[0]
    out = torch.zeros(target_length, dtype=f0_100hz.dtype)

    for i in range(target_length):
        start = round(i * src_len / target_length)
        end = round((i + 1) * src_len / target_length)
        end = max(end, start + 1)  # at least one frame
        segment = f0_100hz[start:end]
        voiced_mask = segment > 0
        if voiced_mask.any():
            out[i] = segment[voiced_mask].mean()
        # else: stays 0.0

    return out


def pitch_shift(f0: torch.Tensor, semitones: float = 0.0) -> torch.Tensor:
    """Shift F0 contour by a number of semitones.

    Args:
        f0: F0 tensor (any shape). Unvoiced frames (0.0) are preserved.
        semitones: Number of semitones to shift. Positive = higher pitch.

    Returns:
        Shifted F0 tensor of the same shape.
    """
    if semitones == 0.0:
        return f0
    factor = 2.0 ** (semitones / 12.0)
    shifted = f0.clone()
    voiced = shifted > 0
    shifted[voiced] = shifted[voiced] * factor
    return shifted
