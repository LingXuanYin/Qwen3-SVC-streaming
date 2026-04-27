# coding=utf-8
"""Preprocess cross-speaker PARALLEL data from Qwen3-TTS synthesis output.

For each (source_A, synth_AB) pair in svc_synth_parallel/manifest.jsonl:
  - content     = HuBERT(source_A)          — real A speaker saying text X
  - spk_embed   = speaker_encoder(synth_AB) — synthesized B's voice
  - f0          = F0(synth_AB)              — matches target codec pitch
  - target_codes= codec(synth_AB)           — B saying X

This creates genuine cross-speaker parallel tuples where content speaker ≠ target
speaker. Training on these forces mapper to use spk_embed for output speaker.
"""
import argparse, json, os, sys, torch, librosa, numpy as np, soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def _load_wav(path, max_duration=10.0):
    try:
        audio, sr = sf.read(path, dtype='float32')
    except Exception:
        audio, sr = librosa.load(path, sr=None)
        audio = audio.astype(np.float32)
    if audio.ndim > 1: audio = audio.mean(-1)
    max_samples = int(sr * max_duration)
    if len(audio) > max_samples: audio = audio[:max_samples]
    if len(audio) < sr * 0.5: return None, None
    return audio, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth_manifest', default='L:/DATASET/svc_synth_parallel/manifest.jsonl')
    ap.add_argument('--output_dir', default='L:/DATASET/svc_synth_preprocessed')
    ap.add_argument('--model_path', default='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--max_duration', type=float, default=10.0)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--shifts', type=str, default='0', help='Comma-separated semitone shifts, e.g. "-6,-3,0,3,6"')
    args = ap.parse_args()
    shifts = [int(s) for s in args.shifts.split(',')]
    print(f'Pitch shifts per sample: {shifts}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading model on {args.device}...', flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=args.device)
    tok = m.model.speech_tokenizer
    spk_enc = m.model.speaker_encoder
    hubert = HubertContentEncoder(device=args.device, dtype=torch.float16)

    with open(args.synth_manifest, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]
    if args.max_samples:
        entries = entries[:args.max_samples]
    print(f'Processing {len(entries)} synthesis entries', flush=True)

    manifest = []
    ok = skip = 0

    for i, e in enumerate(entries):
        try:
            src_audio, src_sr = _load_wav(e['source_path'], args.max_duration)
            synth_audio_base, synth_sr = _load_wav(e['synth_path'], args.max_duration)
            if src_audio is None or synth_audio_base is None:
                skip += 1; continue

            for shift in shifts:
                try:
                    if shift == 0:
                        synth_audio = synth_audio_base
                    else:
                        synth_audio = librosa.effects.pitch_shift(synth_audio_base, sr=synth_sr, n_steps=float(shift))

                    # Target codec from SHIFTED SYNTH
                    with torch.inference_mode():
                        codes = tok.encode(synth_audio, sr=synth_sr).audio_codes[0].cpu()
                    T = codes.shape[0]
                    if T < 10 or T > 90:
                        skip += 1; continue

                    # Content from ORIGINAL SOURCE (A speaker) — unaffected by shift
                    content = hubert.encode(src_audio, src_sr, target_frames=T)

                    # F0 from shifted synth (matches codec pitch)
                    f0 = align_f0_to_codec(extract_f0(synth_audio, synth_sr, device=args.device), T)
                    f0_bins = SVCMapperHubert.f0_to_bin(f0, n_bins=360)

                    # Speaker embed from SHIFTED SYNTH
                    synth24 = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=24000)
                    ref_mel = mel_spectrogram(
                        torch.from_numpy(synth24).unsqueeze(0), n_fft=1024, num_mels=128,
                        sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
                    ).transpose(1, 2).squeeze(0)[:400]
                    with torch.inference_mode():
                        spk_embed = spk_enc(
                            ref_mel.unsqueeze(0).to(device=args.device, dtype=torch.float16)
                        ).squeeze(0).float().cpu()

                    feat = {
                        'content': content,
                        'target_codes': codes,
                        'f0': f0,
                        'f0_bins': f0_bins,
                        'spk_embed': spk_embed,
                        'audio_path': e['synth_path'],
                        'ref_speaker': e['ref_speaker'],
                        'src_speaker': e['src_speaker'],
                        'shift': shift,
                    }
                    out_path = os.path.join(args.output_dir, f'{ok:06d}.pt').replace('\\', '/')
                    torch.save(feat, out_path)
                    manifest.append({'idx': ok, 'path': out_path, 'T': T, 'shift': shift,
                                     'ref_speaker': e['ref_speaker'], 'src_speaker': e['src_speaker']})
                    ok += 1
                except Exception as ex_shift:
                    skip += 1
                    if skip <= 5: print(f'  skip idx={i} shift={shift}: {ex_shift}', flush=True)

        except Exception as ex:
            skip += len(shifts)
            if skip <= 5: print(f'  skip idx={i}: {ex}', flush=True)

        if (i + 1) % args.log_every == 0:
            print(f'  {i+1}/{len(entries)} ok={ok} skip={skip}', flush=True)

    with open(os.path.join(args.output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f)
    print(f'Done: {ok} ok, {skip} skipped', flush=True)


if __name__ == '__main__':
    main()
