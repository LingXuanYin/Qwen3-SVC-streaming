# coding=utf-8
"""Preprocess SVC data: HuBERT content + pitch-shifted targets (IO-optimized).

Optimizations:
- F0 shifted analytically (no 5x re-extraction)
- Parallel audio loading + pitch shift via ProcessPool
- Batched GPU operations for codec encode and HuBERT
- HuBERT extracted once per sample (from original audio)
"""
import argparse, json, os, torch, soundfile as sf, librosa, numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec, pitch_shift as shift_f0
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def _load_and_shift(args_tuple):
    """CPU worker: load audio + generate pitch-shifted versions."""
    path, shifts, max_duration = args_tuple
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(-1)
        max_samples = int(sr * max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        if len(audio) < sr * 1.0:
            return None

        result = {"path": path, "sr": sr, "original": audio, "shifted": {}}
        for shift in shifts:
            if shift == 0:
                result["shifted"][0] = audio
            else:
                result["shifted"][shift] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
        return result
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--audio_list", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shifts", default="-6,-3,3,6")
    p.add_argument("--max_duration", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=16, help="CPU workers pre-fetch this many")
    args = p.parse_args()

    shifts = [int(s) for s in args.shifts.split(",")]
    all_shifts = [0] + shifts
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Shifts: {all_shifts}", flush=True)

    print("Loading models...", flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=args.device)
    tok = m.model.speech_tokenizer
    spk_enc = m.model.speaker_encoder
    hubert = HubertContentEncoder(device=args.device, dtype=torch.float16)

    with open(args.audio_list, encoding='utf-8') as f:
        entries = [json.loads(l) for l in f]
    entries = [e for e in entries if e.get("source_audio") == e.get("target_audio") and e.get("pitch_shift", 0) == 0]
    if args.max_samples:
        entries = entries[:args.max_samples]
    paths = [e["source_audio"] for e in entries]
    print(f"Processing {len(paths)} audios × {len(all_shifts)} shifts = {len(paths)*len(all_shifts)} pairs", flush=True)

    manifest = []
    ok = skip = 0

    # Process in chunks, parallel CPU pitch-shift + serial GPU
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk_idx, chunk in enumerate(_chunks(paths, args.prefetch)):
            if chunk_idx % 5 == 0:
                print(f"  chunk {chunk_idx} | ok={ok} skip={skip} | total target: {len(paths)*len(all_shifts)}", flush=True)

            # Parallel CPU: load + all shifts
            futures = [pool.submit(_load_and_shift, (p, all_shifts, args.max_duration)) for p in chunk]
            loaded = [f.result() for f in futures]

            for item in loaded:
                if item is None:
                    skip += len(all_shifts); continue
                try:
                    audio_orig = item["original"]
                    sr = item["sr"]

                    # Encode ORIGINAL for reference T, HuBERT, F0, mel (done ONCE)
                    with torch.inference_mode():
                        codes_orig = tok.encode(audio_orig, sr=sr).audio_codes[0].cpu()
                    T = codes_orig.shape[0]
                    if T < 10 or T > 90:
                        skip += len(all_shifts); continue

                    # HuBERT content from ORIGINAL (F0-invariant via HuBERT + adversarial training)
                    content = hubert.encode(audio_orig, sr, target_frames=T)

                    # F0 from ORIGINAL — we'll analytically shift for targets
                    f0_orig_raw = extract_f0(audio_orig, sr, device=args.device)

                    # Speaker embedding (precomputed from ORIGINAL — skips ref_mel padding + speaker_encoder forward in training loop)
                    a24 = librosa.resample(audio_orig, orig_sr=sr, target_sr=24000)
                    ref_mel = mel_spectrogram(
                        torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
                    ).transpose(1, 2).squeeze(0)[:400]
                    with torch.inference_mode():
                        spk_embed = spk_enc(
                            ref_mel.unsqueeze(0).to(device=args.device, dtype=torch.float16)
                        ).squeeze(0).float().cpu()

                    # For each shift: encode shifted audio (GPU) + analytical F0 shift
                    for shift, shifted_audio in item["shifted"].items():
                        try:
                            with torch.inference_mode():
                                target_codes = tok.encode(shifted_audio, sr=sr).audio_codes[0].cpu()
                            Tt = target_codes.shape[0]
                            mt = min(T, Tt)
                            if mt < 10:
                                skip += 1; continue

                            # F0 for this shift: analytically shift the original F0 (saves ~80% time vs re-extract)
                            f0_shifted = shift_f0(f0_orig_raw, float(shift))
                            f0_aligned = align_f0_to_codec(f0_shifted, mt)
                            f0_bins = SVCMapperHubert.f0_to_bin(f0_aligned, n_bins=360)

                            feat = {
                                "content": content[:mt],
                                "target_codes": target_codes[:mt],
                                "f0": f0_aligned,
                                "f0_bins": f0_bins,
                                "spk_embed": spk_embed,
                                "shift": shift,
                                "audio_path": item["path"],  # for cross-sample SVC validation
                            }
                            out_path = os.path.join(args.output_dir, f"{ok:06d}.pt")
                            torch.save(feat, out_path)
                            manifest.append({"idx": ok, "path": out_path, "T": mt, "shift": shift})
                            ok += 1
                        except Exception as e_shift:
                            skip += 1
                            if skip <= 5: print(f"  shift skip: {e_shift}")

                except Exception as e:
                    skip += len(all_shifts)
                    if skip <= 5: print(f"  item skip: {e}")

            # Free memory
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f)
    print(f"Done: {ok} pairs, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
