"""F0 tracking + electric noise diagnosis for mic realtime output.

Compute F0 pearson between output and three candidate references:
  - pit cycled (matches sim's get_block_f0 convention)
  - pit stretched nearest (non-stream convention)
  - source/mic F0 (traditional SVC would keep this)

Plus electric-noise scan: spectral centroid spikes, F0 pitch jumps.
"""
import os, sys, argparse, numpy as np, soundfile as sf, librosa, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.svc.f0_extractor import extract_f0


def load_and_f0(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(-1)
    a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
    f0 = extract_f0(a16, 16000, device='cuda:0').cpu().numpy()
    return f0, sr, a


def nearest_resize(arr, target_len):
    sl = len(arr)
    if sl == target_len:
        return arr.copy()
    idx = np.clip(np.round(np.linspace(0, sl - 1, target_len)).astype(int), 0, sl - 1)
    return arr[idx]


def cycle_resize(pit, target_len):
    if len(pit) >= target_len:
        return pit[:target_len].copy()
    out = np.empty(target_len, dtype=np.float32)
    pos = 0
    while pos < target_len:
        take = min(len(pit), target_len - pos)
        out[pos:pos + take] = pit[:take]
        pos += take
    return out


def pearson_v(a, b):
    v = (a > 50) & (b > 50)
    if v.sum() < 10:
        return None, 0, 0
    r = float(np.corrcoef(np.log2(a[v]), np.log2(b[v]))[0, 1])
    diff_st = float(np.log2(a[v] / b[v]).mean() * 12)
    return r, int(v.sum()), diff_st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--pit', required=True)
    ap.add_argument('--window_sec', type=float, default=30.0)
    args = ap.parse_args()

    print('Extracting F0...')
    f0_out, out_sr, out_wav = load_and_f0(args.out)
    f0_src, src_sr, _ = load_and_f0(args.src)
    f0_pit, pit_sr, _ = load_and_f0(args.pit)

    print(f'out: {len(f0_out)/100:.2f}s voiced={int((f0_out>50).sum())}')
    print(f'src: {len(f0_src)/100:.2f}s voiced={int((f0_src>50).sum())}')
    print(f'pit: {len(f0_pit)/100:.2f}s voiced={int((f0_pit>50).sum())}')

    L = len(f0_out)

    # Reproduce sim's voicing-fill: linear interp unvoiced (<=50Hz) to neighbors
    voiced_mask = f0_pit > 50
    all_idx = np.arange(len(f0_pit))
    if voiced_mask.any():
        voiced_idx = all_idx[voiced_mask]
        f0_pit_filled = np.interp(all_idx, voiced_idx, f0_pit[voiced_idx]).astype(np.float32)
    else:
        f0_pit_filled = np.full_like(f0_pit, 220.0)

    f0_pit_cyc = cycle_resize(f0_pit, L)           # raw pit cycled
    f0_pit_cyc_fill = cycle_resize(f0_pit_filled, L)  # voicing-filled pit cycled (what sim actually feeds)
    f0_pit_str = nearest_resize(f0_pit, L)         # pit stretched (non-stream convention)
    f0_pit_str_fill = nearest_resize(f0_pit_filled, L)
    f0_src_al = nearest_resize(f0_src, L)

    print('\n=== Global F0 pearson (log-Hz, both-voiced) ===')
    for name, ref in [('pit cycled raw', f0_pit_cyc),
                       ('pit cycled + voicing_fill (SIM ACTUAL)', f0_pit_cyc_fill),
                       ('pit stretched raw', f0_pit_str),
                       ('pit stretched + voicing_fill', f0_pit_str_fill),
                       ('src/mic (traditional SVC)', f0_src_al)]:
        r, n, dst = pearson_v(f0_out, ref)
        if r is None:
            print(f'  {name:32s}: too few voiced ({n})')
        else:
            print(f'  {name:32s}: pearson={r:+.3f}  overlap={n}  mean_diff={dst:+.2f}st')

    # Per-window
    print(f'\n=== Per-{args.window_sec:.0f}s window pearson ===')
    fpw = int(args.window_sec * 100)
    n_wins = L // fpw
    print(f'{"w":>3} {"t(s)":>7} {"pit_cyc":>10} {"pit_str":>10} {"src":>10} {"voiced":>7}')
    for w in range(n_wins):
        s = w * fpw; e = (w + 1) * fpw
        o = f0_out[s:e]
        cells = []
        for ref in (f0_pit_cyc, f0_pit_str, f0_src_al):
            r, n, _ = pearson_v(o, ref[s:e])
            cells.append((r, n))
        v_cnt = cells[0][1]
        parts = [f'{c[0]:+6.3f}' if c[0] is not None else '   NA  ' for c in cells]
        print(f'{w:>3} {w*args.window_sec:>6.0f}  {parts[0]:>10} {parts[1]:>10} {parts[2]:>10} {v_cnt:>7}')

    # Electric-noise scan
    print('\n=== Electric-noise scan ===')
    hop = 441
    sc = librosa.feature.spectral_centroid(y=out_wav, sr=out_sr, n_fft=2048, hop_length=hop)[0]
    rms = librosa.feature.rms(y=out_wav, frame_length=2048, hop_length=hop)[0]
    high = (sc > 6000) & (rms > 0.005)
    print(f'High-centroid voiced frames (sc>6kHz rms>0.005): '
          f'{int(high.sum())}/{len(sc)} = {high.mean()*100:.2f}%')
    print(f'Spec centroid p90/p95/p99/max: '
          f'{np.percentile(sc, 90):.0f}/{np.percentile(sc, 95):.0f}/'
          f'{np.percentile(sc, 99):.0f}/{sc.max():.0f} Hz')

    voiced_out = f0_out > 50
    jumps = []
    for i in range(1, len(f0_out)):
        if voiced_out[i-1] and voiced_out[i]:
            r = f0_out[i] / f0_out[i-1]
            if r > 1.5 or r < 1/1.5:
                jumps.append((i, f0_out[i-1], f0_out[i], r))
    print(f'F0 pitch-jumps (>1.5x frame-to-frame, both voiced): {len(jumps)}')
    for i, fa, fb, r in jumps[:5]:
        print(f'  frame {i} t={i/100:.2f}s: {fa:.0f}->{fb:.0f}Hz ratio={r:.2f}')

    # Periodicity check — does sim cycle boundary create systematic F0 jumps?
    pit_dur_sec = len(f0_pit) / 100.0
    print(f'\nPit duration: {pit_dur_sec:.2f}s. If sim cycles pit every {pit_dur_sec:.2f}s, expect F0 jumps near t=N×{pit_dur_sec:.2f}s.')
    near_cycle = 0
    for i, _, _, _ in jumps:
        t = i / 100
        t_mod = t % pit_dur_sec
        if t_mod < 0.1 or t_mod > pit_dur_sec - 0.1:
            near_cycle += 1
    print(f'F0 jumps within 100ms of pit-cycle boundary: {near_cycle}/{len(jumps)}')


if __name__ == '__main__':
    main()
