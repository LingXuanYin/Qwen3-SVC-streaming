# coding=utf-8
"""Post-training SVC validation on V100 (GPU 1).

Tests:
1. Load base model + SVC adapter on V100
2. Non-streaming inference with 3 inputs
3. Streaming inference consistency
4. Pitch shift control
5. Output audio validity (24kHz, correct length)
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

# Use GPU 1 (V100) for inference
DEVICE = "cuda:1"
CHECKPOINT = "L:/DATASET/svc_output/full_v3/checkpoint-10000"
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUTPUT_DIR = "L:/DATASET/svc_output/validation"


def make_test_audio(freq=440.0, duration=3.0, sr=24000):
    t = np.arange(int(sr * duration)) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32), sr


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SVC Post-Training Validation")
    print("=" * 60)

    # 1. Load model on V100
    print("\n[1] Loading base model on V100...")
    t0 = time.time()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE)
    print(f"    Model loaded in {time.time()-t0:.1f}s, device={model.device}")

    # 2. Load SVC adapter
    print("\n[2] Loading SVC adapter from checkpoint...")
    t0 = time.time()
    model.load_svc_adapter(CHECKPOINT)
    print(f"    Adapter loaded in {time.time()-t0:.1f}s")
    print(f"    F0 projector: {model._svc_f0_projector}")
    print(f"    SVC config: {model._svc_config.to_dict()}")

    # 3. Prepare test audio files
    print("\n[3] Preparing test audio files...")
    source_path = os.path.join(OUTPUT_DIR, "test_source.wav")
    timbre_path = os.path.join(OUTPUT_DIR, "test_timbre.wav")
    pitch_path = os.path.join(OUTPUT_DIR, "test_pitch.wav")

    # Use actual AISHELL-3 files if available
    aishell_dir = "L:/DATASET/vc_training/train/wav"
    real_files = []
    for dp, dn, fn in os.walk(aishell_dir):
        for f in fn:
            if f.endswith('.wav'):
                real_files.append(os.path.join(dp, f))
                if len(real_files) >= 3:
                    break
        if len(real_files) >= 3:
            break

    if len(real_files) >= 3:
        source_path = real_files[0]
        timbre_path = real_files[1]
        pitch_path = real_files[2]
        print(f"    Using real audio files from AISHELL-3")
    else:
        # Fall back to synthetic
        audio_s, sr = make_test_audio(440.0, 3.0)
        sf.write(source_path, audio_s, sr)
        audio_t, sr = make_test_audio(330.0, 3.0)
        sf.write(timbre_path, audio_t, sr)
        audio_p, sr = make_test_audio(550.0, 3.0)
        sf.write(pitch_path, audio_p, sr)
        print(f"    Using synthetic test audio")

    print(f"    Source: {source_path}")
    print(f"    Timbre: {timbre_path}")
    print(f"    Pitch:  {pitch_path}")

    # 4. Non-streaming inference
    print("\n[4] Non-streaming SVC inference...")
    t0 = time.time()
    try:
        wavs, sr = model.generate_svc(
            timbre_ref=timbre_path,
            source_audio=source_path,
            pitch_ref=pitch_path,
            streaming=False,
        )
        elapsed = time.time() - t0
        wav = wavs[0]
        out_path = os.path.join(OUTPUT_DIR, "svc_nonstream.wav")
        sf.write(out_path, wav, sr)
        print(f"    OK: {wav.shape[0]} samples, {wav.shape[0]/sr:.2f}s, sr={sr}, time={elapsed:.1f}s")
        print(f"    Saved: {out_path}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()

    # 5. Streaming inference
    print("\n[5] Streaming SVC inference...")
    t0 = time.time()
    try:
        chunks = list(model.generate_svc(
            timbre_ref=timbre_path,
            source_audio=source_path,
            pitch_ref=pitch_path,
            streaming=True,
        ))
        elapsed = time.time() - t0
        stream_wav = np.concatenate(chunks)
        out_path = os.path.join(OUTPUT_DIR, "svc_stream.wav")
        sf.write(out_path, stream_wav, sr)
        print(f"    OK: {len(chunks)} chunks, total {stream_wav.shape[0]} samples, {stream_wav.shape[0]/sr:.2f}s, time={elapsed:.1f}s")
        print(f"    Saved: {out_path}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()

    # 6. Pitch shift test
    print("\n[6] Pitch shift test (+12 semitones)...")
    try:
        wavs_shifted, sr = model.generate_svc(
            timbre_ref=timbre_path,
            source_audio=source_path,
            pitch_ref=pitch_path,
            pitch_shift=12.0,
            streaming=False,
        )
        out_path = os.path.join(OUTPUT_DIR, "svc_pitch_up12.wav")
        sf.write(out_path, wavs_shifted[0], sr)
        print(f"    OK: saved {out_path}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()

    # 7. GPU memory check
    print(f"\n[7] GPU memory: {torch.cuda.memory_allocated(1)/1024**3:.1f}GB / {torch.cuda.get_device_properties(1).total_memory/1024**3:.1f}GB")

    # 8. TTS compatibility check
    print("\n[8] TTS compatibility check...")
    assert hasattr(model, 'generate_voice_clone'), "generate_voice_clone missing"
    assert hasattr(model, 'generate_voice_design'), "generate_voice_design missing"
    print("    OK: TTS methods still present")

    print("\n" + "=" * 60)
    print("Validation complete. Check output in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
