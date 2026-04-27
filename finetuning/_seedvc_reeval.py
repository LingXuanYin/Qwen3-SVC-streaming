"""Re-evaluate existing Seed-VC outputs with voiced-aware F0 metric.

Previous eval used np.interp (linear) to resize pit F0 → artifact Hz values at
voiced/unvoiced boundaries → artificially lowered pearson.

Correct metric: nearest-neighbor resize of pit F0 to output-F0 length (preserves
voiced/unvoiced structure), then pearson on log F0 over frames voiced in BOTH.
This matches what length_regulator does internally to the F0 embedding.
"""
import os, sys, json, glob, random, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.svc.f0_extractor import extract_f0


def nearest_resize(arr, target_len):
    src_len = len(arr)
    if src_len == target_len:
        return arr.copy()
    idx = np.clip(np.round(np.linspace(0, src_len - 1, target_len)).astype(int), 0, src_len - 1)
    return arr[idx]


def load_mono(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(-1)
    return a, sr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default='output/seedvc_accept_v2')
    ap.add_argument('--n_combos', type=int, default=15)
    args = ap.parse_args()

    device = 'cuda:0'

    # Reproduce the exact same combos (same seed)
    ALTOS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    TENORS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
    SPEECH_F = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))
    SPEECH_M = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0197/*.wav'))
    random.seed(42)
    combos = []
    for i in range(args.n_combos):
        src = random.choice(ALTOS + TENORS)
        tim = random.choice(SPEECH_M + SPEECH_F)
        pit = random.choice(TENORS if src in ALTOS else ALTOS)
        combos.append((src, tim, pit, f'F3A_{i}'))

    pass_f0 = 0
    rows = []
    for src, tim, pit, label in combos:
        out_wav_dir = os.path.join(args.output_dir, label)
        wavs = glob.glob(os.path.join(out_wav_dir, '*.wav'))
        if not wavs:
            print(f'  {label}: no output wav')
            continue
        out_path = wavs[0]

        src_a, src_sr = load_mono(src)
        pit_a, pit_sr = load_mono(pit)
        out_a, out_sr = load_mono(out_path)

        src_16k = librosa.resample(src_a.astype(np.float32), orig_sr=src_sr, target_sr=16000)
        pit_16k = librosa.resample(pit_a.astype(np.float32), orig_sr=pit_sr, target_sr=16000)
        out_16k = librosa.resample(out_a.astype(np.float32), orig_sr=out_sr, target_sr=16000)

        f0_pit = extract_f0(pit_16k, 16000, device=device).cpu().numpy()
        f0_out = extract_f0(out_16k, 16000, device=device).cpu().numpy()

        # Nearest-neighbor resize pit F0 to output F0 length (matches internal F.interpolate 'nearest').
        f0_pit_on_out = nearest_resize(f0_pit, len(f0_out))

        both = (f0_out > 50) & (f0_pit_on_out > 50)
        pearson = diff = None; f0_ok = False
        if both.sum() >= 5:
            diff = np.log2(f0_out[both] / f0_pit_on_out[both]).mean() * 12
            pearson = float(np.corrcoef(np.log2(f0_out[both]), np.log2(f0_pit_on_out[both]))[0, 1])
            f0_ok = pearson > 0.8 and abs(diff) < 1.0
        if f0_ok:
            pass_f0 += 1

        # Also compute a relaxed voiced-overlap coverage metric
        voiced_out = (f0_out > 50).mean()
        voiced_pit = (f0_pit_on_out > 50).mean()
        overlap = both.mean()

        print(f'  {label}: F0p={pearson if pearson is not None else float("nan"):.3f} diff={diff if diff is not None else 0:+.2f}st '
              f'v_out={voiced_out:.2f} v_pit={voiced_pit:.2f} overlap={overlap:.2f} '
              f'[{"P" if f0_ok else "F"}]')
        rows.append(dict(label=label, pearson=pearson, diff=float(diff) if diff is not None else None,
                         voiced_out=float(voiced_out), voiced_pit=float(voiced_pit), overlap=float(overlap)))

    n = len(combos)
    print(f'\n=== RE-EVAL with NEAREST-NEIGHBOR resize (voiced-aware) ===')
    print(f'F0 pearson>0.8 & |diff|<1st: {pass_f0}/{n} ({pass_f0*100//n}%)')

    with open(os.path.join(args.output_dir, 'reeval_report.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
