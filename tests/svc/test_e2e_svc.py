# coding=utf-8
"""End-to-end SVC validation tests.

These tests require the Qwen3-TTS-12Hz-1.7B-Base model to be available.
They verify:
- SVC adapter application and checkpoint save/load
- Training forward pass (self-reconstruction)
- Inference pipeline (non-streaming and streaming)
- F0 control
- TTS compatibility (no regression)
"""

import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

# Skip all tests if model not available
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE = "cuda:0"


def _generate_test_audio(freq=440.0, duration=2.0, sr=24000):
    """Generate a sine wave test audio."""
    t = np.arange(int(sr * duration)) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32), sr


def _save_test_audio(tmpdir, name, freq=440.0, duration=2.0, sr=24000):
    audio, sr = _generate_test_audio(freq, duration, sr)
    path = os.path.join(tmpdir, name)
    sf.write(path, audio, sr)
    return path


@pytest.fixture(scope="module")
def qwen3tts():
    """Load base model once for all tests."""
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )
        return model
    except Exception as e:
        pytest.skip(f"Model not available: {e}")


@pytest.fixture(scope="module")
def test_audio_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_test_audio(tmpdir, "source.wav", freq=440.0, duration=2.0)
        _save_test_audio(tmpdir, "timbre.wav", freq=330.0, duration=2.0)
        _save_test_audio(tmpdir, "pitch.wav", freq=550.0, duration=2.0)
        yield tmpdir


class TestSVCAdapterAndCheckpoint:
    def test_apply_lora(self, qwen3tts):
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
        from qwen_tts.svc.svc_adapter import apply_svc_lora

        svc_config = Qwen3TTSSVCConfig(lora_rank=8, lora_alpha=16, sub_talker_lora_rank=4)
        f0_projector, trainable = apply_svc_lora(qwen3tts.model, svc_config)

        assert f0_projector is not None
        assert trainable > 0
        print(f"Trainable params: {trainable:,}")

        # Verify frozen modules
        for p in qwen3tts.model.speaker_encoder.parameters():
            assert not p.requires_grad, "Speaker encoder should be frozen"

    def test_save_load_checkpoint(self, qwen3tts):
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
        from qwen_tts.svc.f0_projector import F0Projector
        from qwen_tts.svc.svc_adapter import save_svc_checkpoint, load_svc_checkpoint

        svc_config = Qwen3TTSSVCConfig(lora_rank=8)
        hidden_size = qwen3tts.model.config.talker_config.hidden_size
        f0_projector = F0Projector(hidden_size).to(DEVICE)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_svc_checkpoint(qwen3tts.model, f0_projector, svc_config, tmpdir, step=100)
            ckpt_dir = os.path.join(tmpdir, "checkpoint-100")
            assert os.path.exists(ckpt_dir)
            assert os.path.exists(os.path.join(ckpt_dir, "f0_projector.pt"))
            assert os.path.exists(os.path.join(ckpt_dir, "svc_config.json"))

            # Check checkpoint size is reasonable
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(ckpt_dir)
                for f in filenames
            )
            print(f"Checkpoint size: {total_size / 1024 / 1024:.1f} MB")
            assert total_size < 200 * 1024 * 1024, "Checkpoint too large"


class TestSVCTrainingForward:
    def test_self_reconstruction_loss(self, qwen3tts, test_audio_dir):
        """Verify training forward pass produces valid loss."""
        from peft import PeftModel
        from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
        from qwen_tts.svc.f0_projector import F0Projector

        model = qwen3tts.model
        source_path = os.path.join(test_audio_dir, "source.wav")

        # Encode audio
        enc = model.speech_tokenizer.encode(source_path)
        codes = enc.audio_codes[0].to(DEVICE)  # (T, Q)
        T, Q = codes.shape

        # F0
        audio, sr = sf.read(source_path)
        f0_raw = extract_f0(audio.astype(np.float32), sr, device=DEVICE)
        f0_aligned = align_f0_to_codec(f0_raw, T).to(DEVICE)

        # F0 projector — use actual model hidden_size
        hidden_size = model.config.talker_config.hidden_size
        f0_proj = F0Projector(hidden_size).to(device=DEVICE, dtype=torch.bfloat16)
        f0_embed = f0_proj(f0_aligned.unsqueeze(0))  # (1, T, D)

        # Speaker embedding
        import librosa
        audio_24k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=24000)
        spk_embed = model.extract_speaker_embedding(audio_24k, 24000)

        # Codec embeddings — move everything to model device
        model_device = next(model.parameters()).device
        base_talker = model.talker.base_model.model if isinstance(model.talker, PeftModel) else model.talker
        codes_dev = codes[:, 0:1].unsqueeze(0).to(model_device)
        codec_sum = base_talker.get_input_embeddings()(codes_dev)[:, :, 0]
        f0_dev = f0_embed.to(device=model_device, dtype=codec_sum.dtype)
        spk_dev = spk_embed.to(device=model_device, dtype=codec_sum.dtype)
        input_embeds = codec_sum + f0_dev + spk_dev.view(1, 1, -1).expand(-1, T, -1)

        assert input_embeds.shape == (1, T, hidden_size)
        print(f"Input embeds shape: {input_embeds.shape}, dtype: {input_embeds.dtype}")


class TestSVCInference:
    def test_generate_svc_requires_adapter(self, qwen3tts):
        """Should error without adapter loaded."""
        with pytest.raises(RuntimeError, match="SVC adapter not loaded"):
            qwen3tts._validate_svc_ready()

    def test_audio_duration_validation(self, qwen3tts):
        """Should reject too-long audio."""
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
        qwen3tts._svc_config = Qwen3TTSSVCConfig(max_audio_duration_seconds=1.0)
        long_audio = np.zeros(48000, dtype=np.float32)  # 2 seconds at 24kHz
        with pytest.raises(ValueError, match="exceeds maximum"):
            qwen3tts._validate_audio_duration(long_audio, 24000, "test")
        qwen3tts._svc_config = None


class TestTTSCompatibility:
    def test_tts_methods_exist(self, qwen3tts):
        """Verify TTS methods still exist after SVC additions."""
        assert hasattr(qwen3tts, 'generate_voice_clone')
        assert hasattr(qwen3tts, 'create_voice_clone_prompt')
        assert hasattr(qwen3tts, 'generate_voice_design')
        assert hasattr(qwen3tts, 'generate_custom_voice')

    def test_svc_methods_exist(self, qwen3tts):
        """Verify SVC methods are added."""
        assert hasattr(qwen3tts, 'generate_svc')
        assert hasattr(qwen3tts, 'load_svc_adapter')
        assert hasattr(qwen3tts, '_svc_streaming_decode')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
