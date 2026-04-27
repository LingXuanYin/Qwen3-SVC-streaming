"""Diagnose why F0 pearson < 0.8 on 4 failing cases (F3A_0/9/10/13).

Hypotheses to distinguish:
  A) Time misalignment — pit duration != source duration, nearest stretch misaligns events
  B) Out-of-range — pit F0 outside timbre speaker's natural range, model pulled back
  C) Pit F0 noisy — multi-voice / vibrato / octave jumps in pitch reference
  D) Voicing mismatch — source unvoiced where pit voiced (or vice-versa) confuses model
  E) Model actually follows mean contour but with slow lag / smoothing → lower frame pearson

For each failing case, print per-sample diagnostics:
  - durations, voiced rates
  - shift-optimized pearson (allow +/- N-frame lag)
  - segment-wise pearson (chunk output to find bad regions)
  - correlations with other plausible targets (source F0, constant mean)
"""
import os, sys, glob, random, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.svc.f0_extractor import extract_f0


def nearest_resize(a, tl):
    sl = len(a)
    if sl == tl:
        return a.copy()
    idx = np.clip(np.round(np.linspace(0, sl - 1, tl)).astype(int), 0, sl - 1)
    return a[idx]


def pearson_voiced(a, b, min_n=5):
    v = (a > 50) & (b > 50)
    if v.sum() < min_n:
        return None, 0
    return float(np.corrcoef(np.log2(a[v]), np.log2(b[v]))[0, 1]), int(v.sum())


def main():
    # Replay combos — same seed as acceptance
    ALTOS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    TENORS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
    SPEECH_F = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))
    SPEECH_M = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0197/*.wav'))
    random.seed(42)
    combos = []
    for i in range(15):
        src = random.choice(ALTOS + TENORS)
        tim = random.choice(SPEECH_M + SPEECH_F)
        pit = random.choice(TENORS if src in ALTOS else ALTOS)
        combos.append((src, tim, pit, f'F3A_{i}'))

    failing = {'F3A_0', 'F3A_9', 'F3A_10', 'F3A_13'}
    passing_sample = {'F3A_1', 'F3A_12'}

    device = 'cuda:0'

    def load_mono(p):
        a, sr = sf.read(p, dtype='float32')
        return (a.mean(-1) if a.ndim > 1 else a), sr

    for src, tim, pit, label in combos:
        if label not in failing and label not in passing_sample:
            continue
        out_wavs = glob.glob(f'output/seedvc_accept_v2/{label}/*.wav')
        if not out_wavs:
            print(f'{label}: no wav'); continue

        src_a, src_sr = load_mono(src)
        pit_a, pit_sr = load_mono(pit)
        out_a, out_sr = load_mono(out_wavs[0])

        src16 = librosa.resample(src_a, orig_sr=src_sr, target_sr=16000)
        pit16 = librosa.resample(pit_a, orig_sr=pit_sr, target_sr=16000)
        out16 = librosa.resample(out_a, orig_sr=out_sr, target_sr=16000)
        f0_src = extract_f0(src16, 16000, device=device).cpu().numpy()
        f0_pit = extract_f0(pit16, 16000, device=device).cpu().numpy()
        f0_out = extract_f0(out16, 16000, device=device).cpu().numpy()

        # Durations (100Hz frame rate)
        d_src = len(f0_src) / 100.0
        d_pit = len(f0_pit) / 100.0
        d_out = len(f0_out) / 100.0

        # Align pit to output length (nearest — same as what model did internally)
        pit_on_out = nearest_resize(f0_pit, len(f0_out))
        src_on_out = nearest_resize(f0_src, len(f0_out))

        # Base correlation (reported metric)
        r_pit, n_pit = pearson_voiced(f0_out, pit_on_out)
        r_src, n_src = pearson_voiced(f0_out, src_on_out)

        # Voiced rates
        vr_out = (f0_out > 50).mean()
        vr_pit = (pit_on_out > 50).mean()
        vr_src = (src_on_out > 50).mean()

        # Mean F0 (Hz) on voiced frames
        m_out = f0_out[f0_out > 50].mean() if (f0_out > 50).any() else 0
        m_pit = pit_on_out[pit_on_out > 50].mean() if (pit_on_out > 50).any() else 0
        m_src = src_on_out[src_on_out > 50].mean() if (src_on_out > 50).any() else 0

        # Shift-optimized pearson (allow ±30 frames = ±0.3s lag)
        best_r = r_pit; best_lag = 0
        if r_pit is not None:
            for lag in range(-30, 31, 2):
                if lag == 0: continue
                if lag > 0:
                    a = f0_out[lag:]; b = pit_on_out[:-lag]
                else:
                    a = f0_out[:lag]; b = pit_on_out[-lag:]
                r, _ = pearson_voiced(a, b)
                if r is not None and r > best_r:
                    best_r = r; best_lag = lag

        # Segment-wise pearson (10 chunks)
        seg_rs = []
        T = len(f0_out)
        for k in range(10):
            a = f0_out[k*T//10:(k+1)*T//10]; b = pit_on_out[k*T//10:(k+1)*T//10]
            r, _ = pearson_voiced(a, b, min_n=3)
            seg_rs.append(r)

        flag = 'FAIL' if label in failing else 'PASS'
        print(f'\n=== {label} [{flag}] ===')
        print(f'  durations: src={d_src:.2f}s pit={d_pit:.2f}s out={d_out:.2f}s  (pit/src ratio={d_pit/d_src:.2f})')
        print(f'  voiced rate: out={vr_out:.2f} pit(stretched)={vr_pit:.2f} src={vr_src:.2f}')
        print(f'  mean Hz:     out={m_out:.0f}  pit={m_pit:.0f}  src={m_src:.0f}')
        print(f'  pearson(out,pit)={r_pit:.3f} n={n_pit}  |  pearson(out,src)={r_src:.3f} n={n_src}')
        print(f'  shift-optimized pearson(out,pit)={best_r:.3f} at lag={best_lag} frames ({best_lag*10}ms)')
        print(f'  per-segment pearson: {["{:.2f}".format(r) if r is not None else "NA" for r in seg_rs]}')

        print(f'  src: {os.path.basename(src)}')
        print(f'  tim: {os.path.basename(tim)}')
        print(f'  pit: {os.path.basename(pit)}')


if __name__ == '__main__':
    main()
