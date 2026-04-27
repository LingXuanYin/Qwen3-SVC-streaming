# coding=utf-8
"""Generate pitch-parallel training pairs.

For each audio: original + pitch-shifted version → parallel pair
Source: original codec tokens
Target: pitch-shifted codec tokens
F0: from pitch-shifted version
Speaker: from original (same person)

This provides ground truth for F0 control training.
"""
import json, os, random, torch, soundfile as sf, librosa, numpy as np, torchaudio
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--audio_list", required=True, help="JSONL with source_audio paths")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shifts", type=str, default="-5,-3,-1,1,3,5", help="Semitone shifts")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shifts = [int(s) for s in args.shifts.split(",")]

    print(f"Loading model...")
    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=args.device)
    tok = m.model.speech_tokenizer

    with open(args.audio_list, encoding='utf-8') as f:
        entries = [json.loads(l) for l in f]
    if args.max_samples:
        entries = entries[:args.max_samples]

    # Only use self-reconstruction entries (source==target)
    entries = [e for e in entries if e.get("source_audio") == e.get("target_audio")]
    print(f"Using {len(entries)} self-reconstruction entries, shifts={shifts}")

    manifest = []
    ok = skip = 0

    for idx, entry in enumerate(entries):
        if idx % 200 == 0:
            print(f"  [{idx}/{len(entries)}] ok={ok} skip={skip}", flush=True)
        try:
            path = entry["source_audio"]
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1: audio = audio.mean(-1)

            # Encode original
            with torch.inference_mode():
                source_codes = tok.encode(audio, sr=sr).audio_codes[0].cpu()
            T = source_codes.shape[0]
            if T < 10 or T > 90:
                skip += 1; continue

            # Speaker mel (from original)
            a24 = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            ref_mel = mel_spectrogram(
                torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
            ).transpose(1, 2).squeeze(0)[:400]

            # For each pitch shift
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            for shift in shifts:
                shifted = torchaudio.functional.pitch_shift(audio_t, sr, shift).squeeze().numpy()
                with torch.inference_mode():
                    target_codes = tok.encode(shifted, sr=sr).audio_codes[0].cpu()
                Tt = target_codes.shape[0]
                mt = min(T, Tt)
                if mt < 10: continue

                # F0 from shifted audio
                f0_raw = extract_f0(shifted, sr, device=args.device)
                f0_aligned = align_f0_to_codec(f0_raw, mt)

                feat = {
                    "source_codes": source_codes[:mt],
                    "target_codes": target_codes[:mt],
                    "f0": f0_aligned,
                    "ref_mel": ref_mel,
                }
                out_path = os.path.join(args.output_dir, f"{ok:06d}.pt")
                torch.save(feat, out_path)
                manifest.append({"idx": ok, "path": out_path, "T": mt, "shift": shift})
                ok += 1

        except Exception as e:
            skip += 1
            if skip <= 3: print(f"  Skip: {e}")

    with open(os.path.join(args.output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f)
    print(f"\nDone: {ok} pairs, {skip} skipped")

if __name__ == "__main__":
    main()
