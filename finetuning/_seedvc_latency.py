"""Streaming latency benchmark.

Measures:
  - T_setup: time from convert_voice() call to first CFM call
            (whisper + prompt + length_regulator; per-invocation fixed cost)
  - T_first_chunk: time from call to first chunk yielded (audio starts playing)
  - T_interval: time between consecutive chunk yields (steady state)
  - C_k: audio duration of the k-th chunk
  - chunk_RTF: T_interval / C_k (should be < 1 for faster-than-realtime streaming)

Key question: for interactive use, how soon after calling can playback start?
"""
import os, sys, argparse, glob, time, random
import numpy as np, soundfile as sf, librosa, torch

import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
from pydub import AudioSegment as _AS
_AS.converter = ffmpeg_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
from seed_vc_wrapper import SeedVCWrapper


def bench_once(wrapper, src, tim, pit, steps, label):
    print(f'\n=== {label}: src={os.path.basename(src)} ===')
    src_dur = librosa.get_duration(path=src)
    print(f'  input duration: {src_dur:.2f}s')

    t0 = time.time()
    gen = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                diffusion_steps=steps, length_adjust=1.0,
                                inference_cfg_rate=0.7, f0_condition=True,
                                auto_f0_adjust=False, pitch_shift=0,
                                stream_output=True)

    chunk_times = []  # absolute wall-clock at each yield
    chunk_bytes = []
    first_chunk_time = None
    for i, item in enumerate(gen):
        if isinstance(item, tuple) and len(item) == 2:
            t = time.time() - t0
            mp3_bytes, full = item
            chunk_times.append(t)
            chunk_bytes.append(len(mp3_bytes) if mp3_bytes else 0)
            if first_chunk_time is None:
                first_chunk_time = t
            if full is not None:
                break

    total = time.time() - t0
    n_chunks = len(chunk_times)
    print(f'  first-chunk latency: {first_chunk_time*1000:.0f}ms  (user can start hearing audio this fast)')
    print(f'  total time: {total:.2f}s  chunks: {n_chunks}  RTF(total): {total/src_dur:.3f}')
    if n_chunks >= 2:
        intervals = [chunk_times[i+1] - chunk_times[i] for i in range(n_chunks - 1)]
        mean_iv = np.mean(intervals); max_iv = np.max(intervals); min_iv = np.min(intervals)
        print(f'  inter-chunk interval: mean={mean_iv*1000:.0f}ms min={min_iv*1000:.0f}ms max={max_iv*1000:.0f}ms')
        print(f'  per-chunk timeline (ms from start):')
        for i, t in enumerate(chunk_times[:10]):
            print(f'    chunk {i}: {t*1000:7.0f}ms mp3={chunk_bytes[i]/1024:.0f}KB')
        if n_chunks > 10:
            print(f'    ... (skip to last)')
            print(f'    chunk {n_chunks-1}: {chunk_times[-1]*1000:7.0f}ms mp3={chunk_bytes[-1]/1024:.0f}KB')
    return first_chunk_time, total, n_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diffusion_steps', type=int, default=30)
    args = ap.parse_args()

    # Pick tim + pit once
    tim = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[0]
    pit = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[0]

    # 3 sources of different lengths
    short_src = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))[0]
    # Use the pre-built long source from the 10-min test
    long_src = 'output/longform_10m/src_long.wav'

    # Build a ~30s medium source
    med_src = 'output/latency_src_30s.wav'
    if not os.path.exists(med_src):
        import soundfile as sf
        clips = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
        random.seed(0); random.shuffle(clips)
        out = []; cur = 0.0
        for c in clips:
            a, csr = librosa.load(c, sr=44100)
            out.append(a)
            out.append(np.zeros(int(44100 * 0.3), dtype=np.float32))
            cur += len(a)/44100 + 0.3
            if cur >= 30.0: break
        os.makedirs(os.path.dirname(med_src) or '.', exist_ok=True)
        sf.write(med_src, np.concatenate(out).astype(np.float32), 44100)

    print('Loading SeedVCWrapper (warmup)...')
    wrapper = SeedVCWrapper()

    # Warmup: first call is slower due to CUDA init / caches
    print('\n[warmup call — ignore timing]')
    bench_once(wrapper, short_src, tim, pit, args.diffusion_steps, 'warmup (short)')

    # Real benchmarks
    print('\n' + '=' * 70)
    print('LATENCY BENCHMARKS (post-warmup, wrapper loaded)')
    print('=' * 70)
    bench_once(wrapper, short_src, tim, pit, args.diffusion_steps, 'SHORT (~10s)')
    bench_once(wrapper, med_src, tim, pit, args.diffusion_steps, 'MEDIUM (~30s)')
    if os.path.exists(long_src):
        bench_once(wrapper, long_src, tim, pit, args.diffusion_steps, 'LONG (~10min)')
    else:
        print(f'\nLONG src not found: {long_src} (skipping)')


if __name__ == '__main__':
    main()
