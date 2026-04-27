"""Stress test: streaming on a 10-minute continuous singing input.

Seed-VC "streaming" = chunk-wise OUTPUT yielding; INPUT is fully pre-loaded.
A 10-min input exercises:
  - Whisper internal 30s splitting + concat logic (inference.py line ~286-307)
  - Multi-chunk DiT generation with crossfade boundaries (~20 chunks at 30s each)
  - Prolonged GPU memory usage — should be bounded per chunk, not grow

Pipeline:
  1. Concatenate N GTS singing clips from one singer → one long 10-min wav
  2. Pick timbre_ref + pitch_ref
  3. Stream-mode VC; save full output and yielded mp3 chunks
  4. Monitor RSS + reserved GPU memory per chunk yield
  5. Post-hoc metrics:
     - length ratio output/input (should be 1.0)
     - per-segment CER (segment-wise ASR on 30s windows of output vs input lyrics if available)
     - per-segment F0 pearson to pit (sliding)
     - any chunk-boundary artifacts (discontinuities in waveform or F0)
"""
import os, sys, argparse, glob, time, gc, random
import numpy as np, torch, librosa, soundfile as sf

# ffmpeg for pydub (streaming mp3 encode)
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
from pydub import AudioSegment as _AS
_AS.converter = ffmpeg_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
from seed_vc_wrapper import SeedVCWrapper
from qwen_tts.svc.f0_extractor import extract_f0

try:
    import psutil
except ImportError:
    psutil = None


def build_long_source(target_sec=600.0, out_path='output/longform_src.wav',
                      singer_glob='L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/*/Control_Group/*.wav',
                      sr=44100):
    """Concatenate GTS clips from one singer until >= target_sec."""
    clips = sorted(glob.glob(singer_glob))
    random.seed(0)
    random.shuffle(clips)
    out = []
    cur = 0.0
    for c in clips:
        try:
            a, csr = librosa.load(c, sr=sr)
        except Exception:
            continue
        if len(a) < sr * 1.0 or len(a) > sr * 30.0:
            continue
        out.append(a)
        # 0.3s silence gap between clips (so Whisper can reset phoneme context)
        out.append(np.zeros(int(sr * 0.3), dtype=np.float32))
        cur += len(a) / sr + 0.3
        if cur >= target_sec:
            break
    arr = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, arr, sr)
    print(f'Built {out_path}: {len(arr)/sr:.1f}s = {len(arr)/sr/60:.2f} min ({len(out)//2} clips)', flush=True)
    return out_path


def get_mem():
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9 if psutil else -1
    gpu = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    return rss, gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_minutes', type=float, default=10.0)
    ap.add_argument('--stream', action='store_true', default=True,
                    help='Always stream (goal of this test)')
    ap.add_argument('--diffusion_steps', type=int, default=30)
    ap.add_argument('--out_dir', default='output/longform')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build long source
    src_path = build_long_source(
        target_sec=args.target_minutes * 60.0,
        out_path=os.path.join(args.out_dir, 'src_long.wav'),
    )
    # Pick timbre + pitch (diff-gender speech + diff singer)
    tim = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[0]
    pit = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[0]
    print(f'tim={tim}\npit={pit}', flush=True)

    print('Loading SeedVCWrapper...', flush=True)
    wrapper = SeedVCWrapper()

    rss0, gpu0 = get_mem()
    print(f'Post-load RSS={rss0:.2f}GB GPU_reserved={gpu0:.2f}GB', flush=True)

    t0 = time.time()
    gen = wrapper.convert_voice(
        source=src_path, target=tim, pitch_ref=pit,
        diffusion_steps=args.diffusion_steps,
        length_adjust=1.0, inference_cfg_rate=0.7,
        f0_condition=True, auto_f0_adjust=False, pitch_shift=0,
        stream_output=True,
    )

    chunk_bytes_sizes = []
    full_audio = None; full_sr = 44100
    chunk_idx = 0
    for item in gen:
        if isinstance(item, tuple) and len(item) == 2:
            mp3_bytes, full = item
            chunk_idx += 1
            rss, gpu = get_mem()
            mp3_len = len(mp3_bytes) if mp3_bytes else 0
            chunk_bytes_sizes.append(mp3_len)
            print(f'  chunk {chunk_idx}: mp3={mp3_len/1024:.0f}KB '
                  f'RSS={rss:.2f}GB GPU={gpu:.2f}GB t={time.time()-t0:.1f}s',
                  flush=True)
            if full is not None:
                if isinstance(full, tuple):
                    full_sr, full_audio = full
                else:
                    full_audio = full
                break

    elapsed = time.time() - t0
    print(f'\nStreamed {chunk_idx} chunks in {elapsed:.1f}s (RTF={elapsed / (args.target_minutes * 60):.3f})', flush=True)

    if full_audio is None:
        print('ERROR: no full audio from stream')
        return

    if full_audio.ndim > 1:
        full_audio = full_audio.mean(-1) if full_audio.shape[1] > full_audio.shape[0] else full_audio.mean(0)
    out_wav_path = os.path.join(args.out_dir, 'out_long.wav')
    sf.write(out_wav_path, full_audio, full_sr)
    print(f'Saved {out_wav_path}: {len(full_audio)/full_sr:.1f}s', flush=True)

    # Post-hoc: length ratio, global F0 pearson to pit (nearest-stretch)
    src_dur = os.path.getsize(src_path)  # approx
    src_a, src_sr = sf.read(src_path, dtype='float32')
    out_a = full_audio
    len_ratio = (len(out_a) / full_sr) / (len(src_a) / src_sr)
    print(f'Length: in={len(src_a)/src_sr:.1f}s out={len(out_a)/full_sr:.1f}s ratio={len_ratio:.3f}', flush=True)

    # Per-segment F0 pearson (30s windows)
    WIN = 30.0
    n_wins = int(len(out_a) / full_sr / WIN)
    pit_a, pit_sr = sf.read(pit, dtype='float32')
    if pit_a.ndim > 1:
        pit_a = pit_a.mean(-1)
    pit16 = librosa.resample(pit_a, orig_sr=pit_sr, target_sr=16000)
    f0_pit = extract_f0(pit16, 16000, device='cuda:0').cpu().numpy()
    # Stretch pit F0 to output total length at 100Hz frame rate (nearest)
    out16 = librosa.resample(out_a, orig_sr=full_sr, target_sr=16000)
    f0_out = extract_f0(out16, 16000, device='cuda:0').cpu().numpy()
    idx = np.clip(np.round(np.linspace(0, len(f0_pit) - 1, len(f0_out))).astype(int), 0, len(f0_pit) - 1)
    f0_pit_stretched = f0_pit[idx]
    print(f'\nPer-30s-window F0 pearson (out vs pit-stretched):', flush=True)
    for w in range(n_wins):
        s = int(w * WIN * 100); e = int((w + 1) * WIN * 100)
        if e > len(f0_out):
            break
        fo = f0_out[s:e]; fp = f0_pit_stretched[s:e]
        v = (fo > 50) & (fp > 50)
        if v.sum() < 50:
            print(f'  win {w}: too few voiced'); continue
        r = float(np.corrcoef(np.log2(fo[v]), np.log2(fp[v]))[0, 1])
        print(f'  win {w} ({w*30}-{(w+1)*30}s): f0p={r:.3f} voiced={v.sum()}', flush=True)


if __name__ == '__main__':
    main()
