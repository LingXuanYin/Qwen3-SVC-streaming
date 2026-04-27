"""Precompute pitch-shifted wavs + resampled 16k for F0-aug fine-tune.

For each source wav, generate 7 pitch-shifted copies (shift_st in {-6,-4,-2,0,2,4,6}).
Save:
  - a_16k.npy            (original wav at 16kHz, shared across all shifts)
  - b_shift{N}_44k.npy   (pitch-shifted wav at 44.1kHz, for mel target)
  - b_shift{N}_16k.npy   (pitch-shifted wav at 16kHz, for F0 + speaker)

Total per source wav: ~1.2 MB a_16k + 7 × (0.44 MB b_44k + 0.32 MB b_16k) ≈ 6.5 MB.
For 2065 English GTS wavs → ~13 GB disk. Runs in parallel, expect 20–40 min.

Usage:
  python finetuning/_preprocess_f0aug.py --wav_glob 'L:/DATASET/GTSinger_repo/English/*/*/*/Control_Group/*.wav' --workers 8 --out L:/DATASET/svc_f0aug
"""
import os, glob, argparse, numpy as np, librosa, time
from multiprocessing import Pool


SR = 44100
SHIFTS_DEFAULT = (-6, -4, -2, 0, 2, 4, 6)
DUR_MIN = 1.0
DUR_MAX = 30.0


def process_one(args_tuple):
    wav_path, idx, out_root, shifts = args_tuple
    try:
        wav, _ = librosa.load(wav_path, sr=SR)
        if len(wav) < SR * DUR_MIN or len(wav) > SR * DUR_MAX:
            return ('skip_dur', idx)
        out_dir = os.path.join(out_root, f'{idx:06d}')
        if os.path.exists(os.path.join(out_dir, 'a_16k.npy')):
            # already processed
            return ('dup', idx)
        os.makedirs(out_dir, exist_ok=True)

        a_16k = librosa.resample(wav, orig_sr=SR, target_sr=16000).astype(np.float32)
        np.save(os.path.join(out_dir, 'a_16k.npy'), a_16k)

        for shift_st in shifts:
            if shift_st == 0:
                b_44k = wav.astype(np.float32)
            else:
                # res_type='fft' is much faster than soxr_hq and valid for non-integer rates
                b_44k = librosa.effects.pitch_shift(
                    wav, sr=SR, n_steps=shift_st, res_type='fft'
                ).astype(np.float32)
            b_16k = librosa.resample(b_44k, orig_sr=SR, target_sr=16000).astype(np.float32)
            np.save(os.path.join(out_dir, f'b_shift{shift_st:+d}_44k.npy'), b_44k)
            np.save(os.path.join(out_dir, f'b_shift{shift_st:+d}_16k.npy'), b_16k)

        # Write a small manifest entry
        with open(os.path.join(out_dir, 'meta.txt'), 'w') as f:
            f.write(f'src={wav_path}\ndur={len(wav)/SR:.2f}s\nshifts={",".join(str(s) for s in shifts)}\n')
        return ('ok', idx)
    except Exception as e:
        return ('err', f'{idx}:{e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav_glob', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_samples', type=int, default=0, help='0 = all')
    ap.add_argument('--shifts', type=str, default='-6,-4,-2,0,2,4,6')
    args = ap.parse_args()

    shifts = tuple(int(s) for s in args.shifts.split(','))
    wavs = sorted(glob.glob(args.wav_glob))
    if args.max_samples:
        wavs = wavs[:args.max_samples]
    os.makedirs(args.out, exist_ok=True)
    print(f'Found {len(wavs)} wavs, shifts={shifts}, out={args.out}, workers={args.workers}', flush=True)

    tasks = [(w, i, args.out, shifts) for i, w in enumerate(wavs)]
    t0 = time.time()
    ok = dup = skip = err = 0
    with Pool(args.workers) as p:
        for i, (status, payload) in enumerate(p.imap_unordered(process_one, tasks, chunksize=4)):
            if status == 'ok':
                ok += 1
            elif status == 'dup':
                dup += 1
            elif status == 'skip_dur':
                skip += 1
            else:
                err += 1
                if err < 5:
                    print(f'ERR: {payload}', flush=True)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                print(f'[{i + 1}/{len(tasks)}] ok={ok} dup={dup} skip={skip} err={err} '
                      f'rate={rate:.1f}/s eta={eta/60:.1f}m', flush=True)
    elapsed = time.time() - t0
    print(f'\nFinal: ok={ok} dup={dup} skip={skip} err={err} in {elapsed/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
