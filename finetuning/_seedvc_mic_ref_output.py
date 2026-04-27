"""Generate reference non-streaming output on same 30s source used by mic sim.
Used to A/B compare mic realtime pipeline against full pipeline (no realtime).
"""
import os, sys, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
os.environ.setdefault('HF_HUB_CACHE', './external/seed-vc/checkpoints/hf_cache')

import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
from pydub import AudioSegment as _AS
_AS.converter = ffmpeg_path

import numpy as np, soundfile as sf, librosa, torch
from seed_vc_wrapper import SeedVCWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--duration_sec', type=float, default=30.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--diffusion_steps', type=int, default=30)
    args = ap.parse_args()

    tim = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[0]
    pit = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[0]

    # Trim source to match mic sim duration for fair comparison
    sr = 44100
    wav, _ = librosa.load(args.source, sr=sr)
    n_samples = int(args.duration_sec * sr)
    wav = wav[:n_samples]
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    trimmed_src = args.out.replace('.wav', '_src_trimmed.wav')
    sf.write(trimmed_src, wav, sr)

    print('Loading wrapper...')
    wrapper = SeedVCWrapper()
    gen = wrapper.convert_voice(
        source=trimmed_src, target=tim, pitch_ref=pit,
        diffusion_steps=args.diffusion_steps,
        length_adjust=1.0, inference_cfg_rate=0.7,
        f0_condition=True, auto_f0_adjust=False, pitch_shift=0,
        stream_output=False,
    )
    audio = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        audio = e.value
    if isinstance(audio, tuple):
        sr_out, audio = audio
    else:
        sr_out = 44100
    if audio.ndim > 1:
        audio = audio.mean(-1) if audio.shape[1] > audio.shape[0] else audio.mean(0)
    sf.write(args.out, audio.astype(np.float32), sr_out)
    print(f'Saved {args.out}: {len(audio)/sr_out:.2f}s sr={sr_out}')


if __name__ == '__main__':
    main()
