"""Isolate variance sources on F3A_4 (problematic sample).

For the same (src, tim, pit), run 3 times in each mode and compute pairwise F0 pearson:
  - NS×NS: non-stream vs non-stream (CFM variance floor, path=non-stream)
  - ST×ST: stream vs stream        (CFM variance floor, path=stream)
  - NS×ST: non-stream vs stream    (cross-path)

If ST×ST ~= NS×NS ~= NS×ST → no stream-specific bug, everything is CFM noise.
If ST×ST << NS×NS OR ST×ST >> NS×ST → stream path has systematic issue.
"""
import os, sys, numpy as np, soundfile as sf, librosa, glob, random
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
from pydub import AudioSegment as _AS
_AS.converter = ffmpeg_path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
from seed_vc_wrapper import SeedVCWrapper
from qwen_tts.svc.f0_extractor import extract_f0


def run_once(wrapper, src, tim, pit, steps, stream=False):
    gen = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                diffusion_steps=steps, length_adjust=1.0,
                                inference_cfg_rate=0.7, f0_condition=True,
                                auto_f0_adjust=False, pitch_shift=0,
                                stream_output=stream)
    if not stream:
        audio = None; sr = 44100
        try:
            while True:
                next(gen)
        except StopIteration as e:
            audio = e.value
        if isinstance(audio, tuple):
            sr, audio = audio
    else:
        audio = None; sr = 44100
        for item in gen:
            if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
                full = item[1]
                if isinstance(full, tuple):
                    sr, audio = full
                else:
                    audio = full
                break
    if audio.ndim > 1:
        audio = audio.mean(-1) if audio.shape[1] > audio.shape[0] else audio.mean(0)
    return audio.astype(np.float32), sr


def f0_pearson(a, sr_a, b, sr_b):
    a16 = librosa.resample(a, orig_sr=sr_a, target_sr=16000)
    b16 = librosa.resample(b, orig_sr=sr_b, target_sr=16000)
    fa = extract_f0(a16, 16000, device='cuda:0').cpu().numpy()
    fb = extract_f0(b16, 16000, device='cuda:0').cpu().numpy()
    L = min(len(fa), len(fb))
    v = (fa[:L] > 50) & (fb[:L] > 50)
    if v.sum() < 5:
        return float('nan')
    return float(np.corrcoef(np.log2(fa[:L][v]), np.log2(fb[:L][v]))[0, 1])


def main():
    # Load combo F3A_4
    ALTOS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    TENORS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
    SPEECH_F = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))
    SPEECH_M = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0197/*.wav'))
    random.seed(42)
    combos = []
    for i in range(15):
        s = random.choice(ALTOS + TENORS)
        t = random.choice(SPEECH_M + SPEECH_F)
        p = random.choice(TENORS if s in ALTOS else ALTOS)
        combos.append((s, t, p))

    src, tim, pit = combos[4]
    print(f'Sample 4: src={os.path.basename(src)} tim={os.path.basename(tim)} pit={os.path.basename(pit)}')

    print('Loading SeedVCWrapper...')
    wrapper = SeedVCWrapper()

    N = 3
    ns = []; st = []
    for i in range(N):
        print(f'  NS run {i+1}/{N}...')
        ns.append(run_once(wrapper, src, tim, pit, 30, stream=False))
    for i in range(N):
        print(f'  ST run {i+1}/{N}...')
        st.append(run_once(wrapper, src, tim, pit, 30, stream=True))

    os.makedirs('output/seedvc_var', exist_ok=True)
    for i, (a, sr) in enumerate(ns):
        sf.write(f'output/seedvc_var/ns_{i}.wav', a, sr)
    for i, (a, sr) in enumerate(st):
        sf.write(f'output/seedvc_var/st_{i}.wav', a, sr)

    def pair_scores(xs, ys, label):
        scores = []
        for i, (xa, xs_sr) in enumerate(xs):
            for j, (ya, ys_sr) in enumerate(ys):
                if xs is ys and j <= i:
                    continue
                p = f0_pearson(xa, xs_sr, ya, ys_sr)
                scores.append(p)
                print(f'  {label}[{i}]×{label if xs is ys else "ST"}[{j}]: f0p={p:.3f}')
        return scores

    print('\n--- NS × NS ---')
    ns_ns = pair_scores(ns, ns, 'NS')
    print('\n--- ST × ST ---')
    st_st = pair_scores(st, st, 'ST')
    print('\n--- NS × ST ---')
    ns_st = []
    for i, (a, sr_a) in enumerate(ns):
        for j, (b, sr_b) in enumerate(st):
            p = f0_pearson(a, sr_a, b, sr_b)
            ns_st.append(p)
            print(f'  NS[{i}]×ST[{j}]: f0p={p:.3f}')

    def s(xs): return f'mean={np.mean(xs):.3f} min={min(xs):.3f} max={max(xs):.3f} n={len(xs)}'
    print('\n=== Summary (F3A_4) ===')
    print(f'NS×NS (path=non-stream variance): {s(ns_ns)}')
    print(f'ST×ST (path=stream variance):     {s(st_st)}')
    print(f'NS×ST (cross-path):                {s(ns_st)}')


if __name__ == '__main__':
    main()
