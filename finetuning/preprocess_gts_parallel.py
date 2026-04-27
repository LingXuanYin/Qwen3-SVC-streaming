# coding=utf-8
"""Preprocess GTSinger's native cross-singer parallel data.

GTSinger has ~522 song clips that are sung by 2+ singers with matched index.
Each parallel pair (audio_A_singer1, audio_B_singer2) is a genuine singing
cross-speaker training tuple:
  - content = HuBERT(audio_A_singer1)
  - spk_embed = speaker_encoder(audio_B_singer2)
  - f0 = F0(audio_B_singer2)
  - target_codes = codec(audio_B_singer2)

Both directions are generated (A→B and B→A), plus optional pitch-shift aug.
"""
import argparse, json, os, sys, glob, torch, librosa, numpy as np, soundfile as sf
from collections import defaultdict

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


def find_parallel_pairs(root='L:/DATASET/GTSinger_repo'):
    """Returns list of (audio_A, audio_B, singer_A, singer_B) for index-matched cross-singer clips."""
    song_by_singer = defaultdict(dict)  # "Lang/Song" -> {singer: {clip_idx: path}}
    for wav in glob.glob(f'{root}/**/Control_Group/*.wav', recursive=True):
        parts = wav.replace(os.sep, '/').split('/')
        try:
            lang, singer, tech, song = parts[-6], parts[-5], parts[-4], parts[-3]
            clip_idx = os.path.basename(wav)
            song_by_singer[f'{lang}/{song}'].setdefault(singer, {})[clip_idx] = wav
        except IndexError:
            continue

    pairs = []
    for song, singers in song_by_singer.items():
        singer_list = sorted(singers.keys())
        for i in range(len(singer_list)):
            for j in range(i+1, len(singer_list)):
                a_clips = singers[singer_list[i]]
                b_clips = singers[singer_list[j]]
                shared = set(a_clips.keys()) & set(b_clips.keys())
                for idx in sorted(shared):
                    pairs.append((a_clips[idx], b_clips[idx], singer_list[i], singer_list[j]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default='L:/DATASET/svc_gts_parallel')
    ap.add_argument('--model_path', default='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--shifts', type=str, default='0', help='semitone shifts')
    ap.add_argument('--log_every', type=int, default=200)
    args = ap.parse_args()
    shifts = [int(s) for s in args.shifts.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)
    pairs = find_parallel_pairs()
    print(f'Found {len(pairs)} cross-singer parallel pairs', flush=True)
    # Double: generate both directions (A→B and B→A)
    directed = []
    for a, b, sa, sb in pairs:
        directed.append((a, b, sa, sb))
        directed.append((b, a, sb, sa))
    if args.max_samples:
        directed = directed[:args.max_samples]
    print(f'Expanded to {len(directed)} directed tuples × {len(shifts)} shifts = {len(directed)*len(shifts)} samples', flush=True)

    print(f'Loading model on {args.device}...', flush=True)
    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=args.device)
    tok = m.model.speech_tokenizer
    spk_enc = m.model.speaker_encoder
    hubert = HubertContentEncoder(device=args.device, dtype=torch.float16)

    manifest = []
    ok = skip = 0

    for i, (src_path, tgt_path, src_singer, tgt_singer) in enumerate(directed):
        try:
            src_audio, src_sr = _load_wav(src_path)
            tgt_audio_base, tgt_sr = _load_wav(tgt_path)
            if src_audio is None or tgt_audio_base is None:
                skip += len(shifts); continue

            for shift in shifts:
                try:
                    tgt_audio = tgt_audio_base if shift == 0 else librosa.effects.pitch_shift(
                        tgt_audio_base, sr=tgt_sr, n_steps=float(shift))

                    with torch.inference_mode():
                        codes = tok.encode(tgt_audio, sr=tgt_sr).audio_codes[0].cpu()
                    T = codes.shape[0]
                    if T < 10 or T > 90:
                        skip += 1; continue

                    content = hubert.encode(src_audio, src_sr, target_frames=T)
                    f0 = align_f0_to_codec(extract_f0(tgt_audio, tgt_sr, device=args.device), T)
                    f0_bins = SVCMapperHubert.f0_to_bin(f0, n_bins=360)

                    tgt24 = librosa.resample(tgt_audio, orig_sr=tgt_sr, target_sr=24000)
                    ref_mel = mel_spectrogram(
                        torch.from_numpy(tgt24).unsqueeze(0), n_fft=1024, num_mels=128,
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
                        'audio_path': tgt_path,
                        'ref_speaker': f'GTS-{tgt_singer.split("-")[0]}-{tgt_singer}',
                        'src_speaker': f'GTS-{src_singer.split("-")[0]}-{src_singer}',
                        'shift': shift,
                    }
                    out_path = os.path.join(args.output_dir, f'{ok:06d}.pt').replace('\\', '/')
                    torch.save(feat, out_path)
                    manifest.append({'idx': ok, 'path': out_path, 'T': T, 'shift': shift,
                                     'ref_speaker': feat['ref_speaker'], 'src_speaker': feat['src_speaker']})
                    ok += 1
                except Exception as ex_s:
                    skip += 1
                    if skip <= 5: print(f'  shift {shift} skip: {ex_s}', flush=True)

        except Exception as ex:
            skip += len(shifts)
            if skip <= 5: print(f'  pair {i} skip: {ex}', flush=True)

        if (i + 1) % args.log_every == 0:
            print(f'  pair {i+1}/{len(directed)} ok={ok} skip={skip}', flush=True)

    with open(os.path.join(args.output_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f)
    print(f'Done: {ok} ok, {skip} skipped', flush=True)


if __name__ == '__main__':
    main()
