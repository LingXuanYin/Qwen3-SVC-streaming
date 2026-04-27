# coding=utf-8
"""Preprocess SVC training data: precompute codec tokens, F0, speaker mels.

Saves all features as .pt files so training only needs tensor reads (no GPU inference in dataloader).

Usage:
    CUDA_VISIBLE_DEVICES=0 python finetuning/preprocess_svc.py \
        --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --train_jsonl L:/DATASET/svc_full.jsonl \
        --output_dir L:/DATASET/svc_preprocessed \
        --num_workers 4
"""

import argparse
import json
import os
import traceback

import librosa
import numpy as np
import soundfile as sf
import torch

from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec, pitch_shift as apply_pitch_shift


def load_audio(path):
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
    except Exception:
        audio, sr = librosa.load(path, sr=None, mono=True)
    return audio.astype(np.float32), int(sr)


def extract_mel(audio, sr, max_frames=400):
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000
    ).transpose(1, 2)  # (1, T, 128)
    if mels.shape[1] > max_frames:
        mels = mels[:, :max_frames]
    return mels.squeeze(0)  # (T, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model for tokenizer
    print("Loading model...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    tokenizer = qwen3tts.model.speech_tokenizer

    # Load JSONL
    with open(args.train_jsonl, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]
    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f"Processing {len(entries)} entries...")

    # Load existing manifest for incremental mode
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    existing = {}
    if os.path.exists(manifest_path):
        import json as _json
        with open(manifest_path) as _f:
            for item in _json.load(_f):
                existing[item["idx"]] = item
        print(f"Incremental mode: {len(existing)} already processed, skipping those.")

    manifest = list(existing.values())
    ok = len(existing)
    skip = 0

    for idx, entry in enumerate(entries):
        if idx in existing:
            continue
        if idx % 500 == 0:
            print(f"  [{idx}/{len(entries)}] ok={ok} skip={skip}")

        try:
            source_path = entry["source_audio"]
            target_path = entry["target_audio"]
            pitch_path = entry.get("pitch_audio", target_path)
            pitch_shift_val = entry.get("pitch_shift", 0.0)

            # Load audios
            source_wav, source_sr = load_audio(source_path)
            target_wav, target_sr = load_audio(target_path)
            pitch_wav, pitch_sr = load_audio(pitch_path) if pitch_path != target_path else (target_wav, target_sr)

            # Encode to codec tokens
            with torch.inference_mode():
                source_codes = tokenizer.encode(source_wav, sr=source_sr).audio_codes[0]  # (Ts, Q)
                target_codes = tokenizer.encode(target_wav, sr=target_sr).audio_codes[0]  # (Tt, Q)

            # Align by truncation
            T = min(source_codes.shape[0], target_codes.shape[0])
            if T < 3:
                skip += 1
                continue
            source_codes = source_codes[:T].cpu()
            target_codes = target_codes[:T].cpu()

            # F0
            f0_raw = extract_f0(pitch_wav, pitch_sr, device=args.device)
            f0_aligned = align_f0_to_codec(f0_raw, target_length=T)
            if pitch_shift_val != 0.0:
                f0_aligned = apply_pitch_shift(f0_aligned, float(pitch_shift_val))

            # Speaker mel (from target, capped at 400 frames)
            ref_mel = extract_mel(target_wav, target_sr, max_frames=400)

            # Save
            feat = {
                "source_codes": source_codes,   # (T, Q) int
                "target_codes": target_codes,   # (T, Q) int
                "f0": f0_aligned,               # (T,) float
                "ref_mel": ref_mel,             # (T_mel, 128) float
            }
            out_path = os.path.join(args.output_dir, f"{idx:06d}.pt")
            torch.save(feat, out_path)
            manifest.append({"idx": idx, "path": out_path, "T": T})
            ok += 1

        except Exception as e:
            skip += 1
            if skip <= 5:
                print(f"  Skip {idx}: {e}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f"\nDone: {ok} processed, {skip} skipped. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
