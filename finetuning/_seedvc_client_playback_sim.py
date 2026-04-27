"""Client-side playback simulation.

Models what a real user experiences when connecting to a streaming VC service:
  - Server starts generating upon request; yields chunks as they're ready
  - Client buffers arriving chunks and plays them at real-time rate
  - User perceives: (a) time-to-first-audio (TTFA), (b) any underruns (silent gaps
    caused by client buffer exhausted before next chunk arrives)

Metrics:
  - TTFA: t of first chunk arrival (= user finally hears something)
  - Total generation time: t of last chunk
  - Total playback time: TTFA + audio_duration (if no underrun)
  - Underruns: list of (start, duration) pairs where buffer was empty
  - Safety margin: at each chunk arrival, compare cumulative audio available vs
    wall-clock needed to play it

For a healthy streaming service: TTFA < a few seconds, zero underruns.
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


def simulate_client(wrapper, src, tim, pit, diffusion_steps, label, sr=44100):
    """Run streaming inference and simulate client playback."""
    src_dur = librosa.get_duration(path=src)
    print(f'\n=== {label}: src={os.path.basename(src)} dur={src_dur:.1f}s ===', flush=True)

    t_wall_start = time.time()
    # chunks[i] = (t_arrival_s, audio_sample_count_in_this_chunk)
    chunks = []
    gen = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                diffusion_steps=diffusion_steps,
                                length_adjust=1.0, inference_cfg_rate=0.7,
                                f0_condition=True, auto_f0_adjust=False,
                                pitch_shift=0, stream_output=True)

    # Wrapper uses CBR 320 kbps for MP3 → bytes/sec = 320_000/8 = 40_000
    # But we prefer an exact sample count: record MP3 bytes per chunk, then
    # scale against the total full_audio length at the end.
    mp3_bytes_per_chunk = []
    full_audio = None
    for item in gen:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        mp3_bytes, full = item
        t_arrival = time.time() - t_wall_start
        mp3_bytes_per_chunk.append(len(mp3_bytes) if mp3_bytes else 0)
        chunks.append([t_arrival, 0])  # placeholder, filled below
        if full is not None:
            if isinstance(full, tuple):
                full_audio = full[1]
            else:
                full_audio = full
            break

    # Now allocate samples per chunk proportional to mp3 bytes (CBR ≈ proportional)
    if full_audio is not None:
        if full_audio.ndim > 1:
            full_audio = full_audio.mean(-1) if full_audio.shape[1] > full_audio.shape[0] else full_audio.mean(0)
        total_samples = len(full_audio)
        total_bytes = sum(mp3_bytes_per_chunk)
        if total_bytes > 0:
            for i, b in enumerate(mp3_bytes_per_chunk):
                chunks[i][1] = int(round(total_samples * b / total_bytes))
    cumulative_audio_samples = sum(c[1] for c in chunks)

    t_total = time.time() - t_wall_start
    total_audio_sec = cumulative_audio_samples / sr

    # Client playback simulation
    # At t_arrival_i, buffer adds chunks[i].samples seconds of playable audio.
    # Playback starts at t=chunks[0].t_arrival, consumes 1s audio per 1s wall-clock.
    # Underrun = wall_time > TTFA and available_audio (sum of chunks so far) - consumed < 0.
    ttfa = chunks[0][0] if chunks else None
    print(f'  TTFA: {ttfa*1000:.0f}ms', flush=True)
    print(f'  Total gen time: {t_total:.2f}s  chunks: {len(chunks)}  total_audio: {total_audio_sec:.2f}s', flush=True)
    print(f'  Gen RTF (wall/audio): {t_total/total_audio_sec:.3f}' if total_audio_sec else '', flush=True)

    # Walk through time: at each chunk arrival, compute how much audio is available
    # and compare to how much has been consumed by playback (= t_arrival - TTFA seconds of audio).
    underruns = []
    cum_samples_at = []
    cum_s = 0
    for t_arr, n_samp in chunks:
        cum_s += n_samp
        cum_samples_at.append((t_arr, cum_s))
    # Check margin at each chunk arrival
    min_margin_s = float('inf')
    for t_arr, cum_s in cum_samples_at:
        if t_arr < ttfa:
            continue
        available_audio_sec = cum_s / sr
        consumed_audio_sec = t_arr - ttfa
        margin_sec = available_audio_sec - consumed_audio_sec
        min_margin_s = min(min_margin_s, margin_sec)
        if margin_sec < 0:
            underruns.append((t_arr, -margin_sec))

    print(f'  Client min buffer margin: {min_margin_s:.2f}s  (>0 = safe, <0 = underrun)', flush=True)
    print(f'  Underruns: {len(underruns)}')
    if underruns:
        for t_u, dur in underruns[:5]:
            print(f'    at t={t_u:.2f}s: duration {dur*1000:.0f}ms')

    # Per-chunk timeline (first few + last)
    print(f'  Chunk arrival vs cumulative audio:', flush=True)
    for i, (t, c) in enumerate(cum_samples_at[:5]):
        print(f'    chunk {i}: arrived {t*1000:.0f}ms, cum_audio={c/sr:.2f}s')
    if len(cum_samples_at) > 5:
        i, (t, c) = len(cum_samples_at) - 1, cum_samples_at[-1]
        print(f'    chunk {i}: arrived {t*1000:.0f}ms, cum_audio={c/sr:.2f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diffusion_steps', type=int, default=30)
    args = ap.parse_args()

    tim = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[0]
    pit = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[0]
    short_src = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))[0]
    med_src = 'output/latency_src_30s.wav'
    long_src = 'output/longform_10m/src_long.wav'

    print('Loading SeedVCWrapper (warmup)...')
    wrapper = SeedVCWrapper()
    # Warmup
    print('\n[warmup]')
    simulate_client(wrapper, short_src, tim, pit, args.diffusion_steps, 'warmup')

    print('\n' + '=' * 70)
    print('CLIENT PLAYBACK SIMULATION (post-warmup)')
    print('=' * 70)
    simulate_client(wrapper, short_src, tim, pit, args.diffusion_steps, 'SHORT 10s')
    if os.path.exists(med_src):
        simulate_client(wrapper, med_src, tim, pit, args.diffusion_steps, 'MEDIUM 30s')
    if os.path.exists(long_src):
        simulate_client(wrapper, long_src, tim, pit, args.diffusion_steps, 'LONG 606s (10min)')


if __name__ == '__main__':
    main()
