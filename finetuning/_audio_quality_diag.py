"""Diagnose silence/electric-noise artifacts in a wav.

Metrics:
  - Dropout rate: fraction of 100ms windows where RMS < -40 dBFS (muted)
    and the wav is not supposed to be silent (compare to source level)
  - HF energy spikes: 100ms windows where spectral centroid > threshold
  - Zero-crossing rate std: instability indicator (clicks/electric)
  - Frame-to-frame mel cosine: boundary discontinuity detector
"""
import os, sys, argparse, numpy as np, soundfile as sf, librosa


def load_mono(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(-1)
    return a, sr


def rms_db(x):
    r = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    return 20 * np.log10(r + 1e-12)


def analyze(path, label):
    a, sr = load_mono(path)
    print(f'\n=== {label}  ({path})  dur={len(a)/sr:.2f}s  sr={sr} ===')

    # 100ms windowed RMS (hop 50ms)
    win = int(sr * 0.1); hop = int(sr * 0.05)
    n_wins = (len(a) - win) // hop + 1
    rms_seq = []
    for i in range(n_wins):
        seg = a[i * hop: i * hop + win]
        rms_seq.append(rms_db(seg))
    rms_seq = np.array(rms_seq)

    # Dropout detection: windows with RMS < -40 dB
    drop_thresh = -40
    n_drop = int((rms_seq < drop_thresh).sum())
    # Consecutive drop runs (real silences, not just quiet passages)
    drop_mask = rms_seq < drop_thresh
    runs = []
    run_start = None
    for i, d in enumerate(drop_mask):
        if d and run_start is None:
            run_start = i
        elif not d and run_start is not None:
            runs.append((run_start, i - 1, i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(drop_mask) - 1, len(drop_mask) - run_start))
    # Only runs >= 200ms (4 windows at 50ms hop) count as "perceived dropout"
    perceptible_drops = [r for r in runs if r[2] >= 4]
    total_drop_ms = sum(r[2] * 50 for r in perceptible_drops)
    print(f'  RMS: overall={rms_db(a):.1f}dB  window range=[{rms_seq.min():.1f}, {rms_seq.max():.1f}]dB')
    print(f'  Muted windows (<{drop_thresh}dB): {n_drop}/{len(rms_seq)} ({n_drop*100/len(rms_seq):.1f}%)')
    print(f'  Perceptible dropouts (>=200ms continuous silence): {len(perceptible_drops)} runs, total {total_drop_ms/1000:.2f}s')
    if perceptible_drops[:3]:
        for r in perceptible_drops[:3]:
            print(f'    drop at {r[0]*50/1000:.2f}-{(r[1]+1)*50/1000:.2f}s (dur {r[2]*50/1000:.2f}s)')

    # ZCR std (clicks / electric noise surge)
    zcr = librosa.feature.zero_crossing_rate(a, frame_length=win, hop_length=hop)[0]
    print(f'  ZCR: mean={zcr.mean():.3f}  std={zcr.std():.3f}  max={zcr.max():.3f}')

    # Spectral centroid range (high freq energy spike indicator)
    sc = librosa.feature.spectral_centroid(y=a, sr=sr, n_fft=2048, hop_length=hop)[0]
    print(f'  Spec centroid: mean={sc.mean():.0f}Hz  std={sc.std():.0f}  p95={np.percentile(sc, 95):.0f}Hz  max={sc.max():.0f}Hz')

    # Boundary discontinuity: frame-to-frame mel log-cosine
    mel = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    logmel = np.log(mel + 1e-6)
    # Cosine between adjacent frames
    norms = np.linalg.norm(logmel, axis=0) + 1e-9
    dots = (logmel[:, :-1] * logmel[:, 1:]).sum(axis=0)
    cos_adj = dots / (norms[:-1] * norms[1:])
    print(f'  Adjacent-frame mel cos: mean={cos_adj.mean():.4f}  min={cos_adj.min():.4f}')
    # Dips below 0.9 = possible boundary discontinuity
    n_discont = int((cos_adj < 0.9).sum())
    print(f'  Discontinuities (cos<0.9): {n_discont} frames ({n_discont*100/len(cos_adj):.2f}%)')

    return rms_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wavs', nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    args = ap.parse_args()
    labels = args.labels if args.labels else [os.path.basename(w) for w in args.wavs]
    for w, l in zip(args.wavs, labels):
        if not os.path.exists(w):
            print(f'SKIP {l}: {w} not found'); continue
        analyze(w, l)


if __name__ == '__main__':
    main()
