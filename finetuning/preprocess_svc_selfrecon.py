# coding=utf-8
"""Self-reconstruction preprocessing for SVC: all features from same audio.

Training distribution == inference distribution at inference time:
  content: HuBERT from audio X
  F0: from audio X
  speaker: from audio X
  target_codec: from audio X

Model must learn disentanglement via adversarial F0 removal on content.
At inference, we swap components to do SVC (content from A, F0 from C, speaker from B).
"""
import argparse, json, os, torch, soundfile as sf, librosa, numpy as np
from concurrent.futures import ProcessPoolExecutor
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def _load_audio(args_tuple):
    path, max_duration = args_tuple
    try:
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1: audio = audio.mean(-1)
        max_samples = int(sr * max_duration)
        if len(audio) > max_samples: audio = audio[:max_samples]
        if len(audio) < sr * 1.0: return None
        return {"path": path, "sr": sr, "audio": audio}
    except: return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--audio_list", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_duration", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    print(f"Processing {len(paths)} audios (self-reconstruction)", flush=True)

    manifest = []
    ok = skip = 0

    def _chunks(lst, n):
        for i in range(0, len(lst), n): yield lst[i:i+n]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for chunk_idx, chunk in enumerate(_chunks(paths, args.prefetch)):
            if chunk_idx % 5 == 0:
                print(f"  chunk {chunk_idx} | ok={ok} skip={skip}", flush=True)
            futures = [pool.submit(_load_audio, (p, args.max_duration)) for p in chunk]
            loaded = [f.result() for f in futures]

            for item in loaded:
                if item is None: skip += 1; continue
                try:
                    audio = item["audio"]; sr = item["sr"]
                    with torch.inference_mode():
                        codes = tok.encode(audio, sr=sr).audio_codes[0].cpu()
                    T = codes.shape[0]
                    if T < 10 or T > 90:
                        skip += 1; continue

                    content = hubert.encode(audio, sr, target_frames=T)
                    f0 = align_f0_to_codec(extract_f0(audio, sr, device=args.device), T)
                    f0_bins = SVCMapperHubert.f0_to_bin(f0, n_bins=360)

                    a24 = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                    ref_mel = mel_spectrogram(
                        torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
                    ).transpose(1, 2).squeeze(0)[:400]
                    with torch.inference_mode():
                        spk_embed = spk_enc(
                            ref_mel.unsqueeze(0).to(device=args.device, dtype=torch.float16)
                        ).squeeze(0).float().cpu()

                    feat = {
                        "content": content,
                        "target_codes": codes,
                        "f0": f0,
                        "f0_bins": f0_bins,
                        "spk_embed": spk_embed,
                        "audio_path": item["path"],  # keep for cross-sample SVC validation
                    }
                    out_path = os.path.join(args.output_dir, f"{ok:06d}.pt")
                    torch.save(feat, out_path)
                    manifest.append({"idx": ok, "path": out_path, "T": T, "audio_path": item["path"]})
                    ok += 1
                except Exception as e:
                    skip += 1
                    if skip <= 3: print(f"  Skip: {e}")
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, "manifest.json"), 'w') as f:
        json.dump(manifest, f)
    print(f"Done: {ok} ok, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
