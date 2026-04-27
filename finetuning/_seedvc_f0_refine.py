"""Test-time F0 refinement via WORLD analysis-resynthesis.

Applied AFTER Seed-VC inference. Replaces output audio's F0 contour with the
user-provided pit_ref F0 contour (resized to output length), preserving the
output's spectral envelope (content + speaker) via pyworld's (F0, sp, ap)
decomposition.

Pipeline per output wav:
  1. Extract pit_ref F0 via RMVPE (matches Seed-VC's internal F0 extractor)
  2. pyworld.dio + stonemask → output F0_world, time axis t
  3. Align pit F0 to the same time axis (resize by nearest)
  4. Preserve pyworld-extracted voiced/unvoiced pattern from output
     (so we don't introduce unvoiced glitches), but replace voiced F0
     values with pit_ref's contour at those times
  5. pyworld.cheaptrick → spectral envelope (sp) from output
  6. pyworld.d4c → aperiodicity (ap) from output
  7. pyworld.synthesize(replaced_F0, sp, ap) → refined output

This is classical DSP — no training, no model change.
"""
import os, sys, argparse, glob, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import soundfile as sf
import pyworld as pw

from qwen_tts.svc.f0_extractor import extract_f0


def refine_f0(out_wav, out_sr, pit_wav, pit_sr, device='cuda:0',
              voicing_merge='output'):
    """Replace out_wav's F0 contour with pit_wav's, keep out's timbre/content.

    voicing_merge: 'output' = use output's voicing mask (safer, preserves unvoiced
                              fricatives/gaps in output)
                   'pit'    = use pit's voicing mask (stricter F0 following)
    """
    # Convert to float64 for pyworld
    out = out_wav.astype(np.float64)
    sr = out_sr
    # WORLD analysis
    f0_world, t = pw.dio(out, sr, f0_floor=50.0, f0_ceil=1100.0, frame_period=5.0)
    f0_world = pw.stonemask(out, f0_world, t, sr)  # (T_w,)
    sp = pw.cheaptrick(out, f0_world, t, sr)       # (T_w, n_spec)
    ap = pw.d4c(out, f0_world, t, sr)              # (T_w, n_spec)

    # Extract target F0 from pit using RMVPE (same as Seed-VC)
    pit16 = librosa.resample(pit_wav.astype(np.float32), orig_sr=pit_sr, target_sr=16000)
    f0_pit = extract_f0(pit16, 16000, device=device).cpu().numpy()  # 100 Hz frame rate

    # Resize pit F0 to WORLD time axis (nearest — preserves voiced/unvoiced binary structure)
    T_w = len(f0_world)
    if len(f0_pit) != T_w:
        idx = np.clip(np.round(np.linspace(0, len(f0_pit) - 1, T_w)).astype(int),
                      0, len(f0_pit) - 1)
        f0_pit_on_world = f0_pit[idx]
    else:
        f0_pit_on_world = f0_pit

    # Merge voicing
    voiced_out = f0_world > 1
    voiced_pit = f0_pit_on_world > 50
    if voicing_merge == 'output':
        # keep output's voicing pattern; where output is voiced, use pit F0 (if pit is also voiced) else keep output F0
        f0_new = f0_world.copy()
        both = voiced_out & voiced_pit
        f0_new[both] = f0_pit_on_world[both]
    elif voicing_merge == 'pit':
        # strictly follow pit voicing
        f0_new = np.where(voiced_pit, f0_pit_on_world, 0.0)
    else:
        raise ValueError(voicing_merge)

    # Resynthesize
    y_new = pw.synthesize(f0_new.astype(np.float64), sp, ap, sr, frame_period=5.0)
    return y_new.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='Directory containing F3A_*/*.wav (zero-shot Seed-VC outputs)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--voicing_merge', choices=['output', 'pit'], default='output')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Replay combos (same seed) to find which pit goes with which label
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

    for src, tim, pit, label in combos:
        in_wavs = glob.glob(os.path.join(args.input_dir, label, '*.wav'))
        if not in_wavs:
            print(f'  {label}: no input wav'); continue
        out_wav, out_sr = sf.read(in_wavs[0], dtype='float32')
        if out_wav.ndim > 1:
            out_wav = out_wav.mean(-1)
        pit_wav, pit_sr = sf.read(pit, dtype='float32')
        if pit_wav.ndim > 1:
            pit_wav = pit_wav.mean(-1)

        refined = refine_f0(out_wav, out_sr, pit_wav, pit_sr,
                             voicing_merge=args.voicing_merge)
        # Keep same filename so acceptance's glob finds it
        dst_dir = os.path.join(args.output_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        sf.write(os.path.join(dst_dir, os.path.basename(in_wavs[0])), refined, out_sr)
        print(f'  {label}: refined (in_dur={len(out_wav)/out_sr:.2f}s out_dur={len(refined)/out_sr:.2f}s)')


if __name__ == '__main__':
    main()
