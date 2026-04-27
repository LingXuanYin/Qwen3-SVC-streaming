# coding=utf-8
"""Content encoder based on HuBERT for SVC.

HuBERT features are speaker and pitch independent (trained for ASR-like tasks),
which is essential for SVC: the content input should not leak pitch/speaker info.
"""

import torch
import librosa
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HubertContentEncoder:
    """Wraps Chinese HuBERT to extract content features aligned to codec frame rate."""

    def __init__(self, model_path: str = "TencentGameMate/chinese-hubert-base",
                 device: str = "cuda:0", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.model = HubertModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        # HuBERT: 16kHz input, 20ms stride → 50Hz output
        self.sample_rate = 16000
        self.feature_rate = 50

    @torch.inference_mode()
    def encode(self, audio: np.ndarray, sr: int, target_frames: int = None,
               target_rate: int = 12) -> torch.Tensor:
        """Extract HuBERT content features and align to codec frame rate.

        Args:
            audio: 1-D waveform.
            sr: original sample rate.
            target_frames: number of output frames (match codec T).
            target_rate: target frame rate (12Hz for codec).

        Returns:
            (target_frames, hidden_size) content features on CPU float32.
        """
        if audio.ndim != 1:
            audio = audio.squeeze()
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        inputs = self.feature_extractor(audio, sampling_rate=self.sample_rate,
                                         return_tensors="pt").input_values
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        out = self.model(inputs, output_hidden_states=False)
        features = out.last_hidden_state[0]  # (T_hubert, hidden_size)

        # Downsample from 50Hz to target_rate (12Hz) via interpolation
        if target_frames is not None:
            T_hubert = features.shape[0]
            # Interpolate
            features_t = features.float().T.unsqueeze(0)  # (1, D, T_hubert)
            features_aligned = torch.nn.functional.interpolate(
                features_t, size=target_frames, mode='linear', align_corners=False
            ).squeeze(0).T  # (target_frames, D)
            return features_aligned.cpu()

        return features.float().cpu()

    @property
    def hidden_size(self):
        return self.model.config.hidden_size  # 768 for base
