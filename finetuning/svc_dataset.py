# coding=utf-8
"""SVC training dataset: loads source/target/pitch audio pairs and preprocesses them."""

import json
import logging
import os
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec, pitch_shift as apply_pitch_shift, EXTRACTION_SR

logger = logging.getLogger(__name__)


class SVCDataset(Dataset):
    """Dataset for SVC training.

    Each JSONL line: {"source_audio": "...", "target_audio": "...", "pitch_audio": "..."(optional)}
    If pitch_audio is omitted, target_audio is used as pitch reference.
    """

    def __init__(
        self,
        data_list: list,
        speech_tokenizer,
        config: Qwen3TTSConfig,
        f0_device: str = "cuda:0",
    ):
        self.config = config
        self.speech_tokenizer = speech_tokenizer
        self.f0_device = f0_device

        # Filter valid entries
        self.data_list = []
        for item in data_list:
            if not os.path.exists(item.get("source_audio", "")):
                logger.warning(f"Skipping: source_audio not found: {item.get('source_audio')}")
                continue
            if not os.path.exists(item.get("target_audio", "")):
                logger.warning(f"Skipping: target_audio not found: {item.get('target_audio')}")
                continue
            pitch_path = item.get("pitch_audio", item["target_audio"])
            if not os.path.exists(pitch_path):
                logger.warning(f"Skipping: pitch_audio not found: {pitch_path}")
                continue
            self.data_list.append(item)

        logger.info(f"SVCDataset: {len(self.data_list)} valid samples from {len(data_list)} total")

    def __len__(self):
        return len(self.data_list)

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
        except Exception:
            audio, sr = librosa.load(path, sr=None, mono=True)
        return audio.astype(np.float32), int(sr)

    def _encode_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Encode audio to codec tokens via speech tokenizer. Returns (T, num_quantizers)."""
        enc = self.speech_tokenizer.encode(audio, sr=sr)
        return enc.audio_codes[0]  # (T, Q)

    def _extract_mel(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Extract mel spectrogram for speaker encoder (24kHz required)."""
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000
        ).transpose(1, 2)
        return mels  # (1, T_mel, 128)

    @torch.inference_mode()
    def __getitem__(self, idx):
        item = self.data_list[idx]
        source_path = item["source_audio"]
        target_path = item["target_audio"]
        pitch_path = item.get("pitch_audio", target_path)

        # Load audios
        source_wav, source_sr = self._load_audio(source_path)
        target_wav, target_sr = self._load_audio(target_path)
        pitch_wav, pitch_sr = self._load_audio(pitch_path)

        # Encode to codec tokens
        source_codes = self._encode_audio(source_wav, source_sr)  # (Ts, Q)
        target_codes = self._encode_audio(target_wav, target_sr)  # (Tt, Q)

        # Align lengths by truncation to shorter
        T = min(source_codes.shape[0], target_codes.shape[0])
        source_codes = source_codes[:T]
        target_codes = target_codes[:T]

        # Extract F0 from pitch reference, aligned to T frames
        f0_raw = extract_f0(pitch_wav, pitch_sr, device=self.f0_device)
        f0_aligned = align_f0_to_codec(f0_raw, target_length=T)

        # Apply pitch shift if specified in data entry
        pitch_shift_semitones = item.get("pitch_shift", 0.0)
        if pitch_shift_semitones != 0.0:
            f0_aligned = apply_pitch_shift(f0_aligned, float(pitch_shift_semitones))

        # Extract mel for speaker embedding (from target audio)
        ref_mel = self._extract_mel(target_wav, target_sr)

        return {
            "source_codes": source_codes,   # (T, Q)
            "target_codes": target_codes,   # (T, Q)
            "f0": f0_aligned,               # (T,)
            "ref_mel": ref_mel,             # (1, T_mel, 128)
        }

    @staticmethod
    def collate_fn(batch):
        """Collate variable-length samples with padding."""
        max_T = max(b["source_codes"].shape[0] for b in batch)
        B = len(batch)
        Q = batch[0]["source_codes"].shape[1]

        source_codes = torch.zeros(B, max_T, Q, dtype=torch.long)
        target_codes = torch.zeros(B, max_T, Q, dtype=torch.long)
        f0 = torch.zeros(B, max_T)
        mask = torch.zeros(B, max_T, dtype=torch.bool)

        for i, b in enumerate(batch):
            T = b["source_codes"].shape[0]
            source_codes[i, :T] = b["source_codes"]
            target_codes[i, :T] = b["target_codes"]
            f0[i, :T] = b["f0"]
            mask[i, :T] = True

        # Pad ref_mels to same length, cap at 400 frames (~4s) for speaker encoder memory
        MAX_MEL_FRAMES = 400
        mel_dim = batch[0]["ref_mel"].shape[-1]
        max_mel_T = min(max(b["ref_mel"].shape[1] for b in batch), MAX_MEL_FRAMES)
        ref_mels = torch.zeros(B, max_mel_T, mel_dim)
        for i, b in enumerate(batch):
            mel_T = min(b["ref_mel"].shape[1], MAX_MEL_FRAMES)
            ref_mels[i, :mel_T] = b["ref_mel"][0, :mel_T]

        return {
            "source_codes": source_codes,
            "target_codes": target_codes,
            "f0": f0,
            "mask": mask,
            "ref_mels": ref_mels,
        }
