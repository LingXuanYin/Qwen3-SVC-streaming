"""Baseline for CFM stochasticity: run non-streaming TWICE on same inputs,
measure how much two independent samples of the same conditioning diverge.

If stream-vs-non-stream F0 pearson ~= nonstream-vs-nonstream F0 pearson, then
the streaming path is NOT introducing extra divergence beyond the model's
inherent sampling noise — i.e. "consistent within CFM stochasticity envelope".
"""
import os, sys, numpy as np, soundfile as sf, librosa, torch, glob, random

# ffmpeg for pydub (wrapper internally uses it for mp3 encode in streaming — not used here)
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
from seed_vc_wrapper import SeedVCWrapper
from qwen_tts.svc.f0_extractor import extract_f0


def mel_logmel(x, sr):
    if sr != 16000:
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    mel = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
    return np.log(mel + 1e-6)


def cosine_framewise(a, b):
    L = min(a.shape[1], b.shape[1])
    a = a[:, :L]; b = b[:, :L]
    dot = (a * b).sum(axis=0)
    na = np.linalg.norm(a, axis=0) + 1e-8
    nb = np.linalg.norm(b, axis=0) + 1e-8
    return float((dot / (na * nb)).mean())


def run_nonstream(wrapper, src, tim, pit, steps):
    gen = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                diffusion_steps=steps, length_adjust=1.0,
                                inference_cfg_rate=0.7, f0_condition=True,
                                auto_f0_adjust=False, pitch_shift=0, stream_output=False)
    audio = None; sr = 44100
    try:
        while True:
            next(gen)
    except StopIteration as e:
        audio = e.value
    if isinstance(audio, tuple):
        sr, audio = audio
    if audio.ndim > 1:
        audio = audio.mean(-1) if audio.shape[1] > audio.shape[0] else audio.mean(0)
    return audio.astype(np.float32), sr


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=5)
    ap.add_argument('--diffusion_steps', type=int, default=30)
    args = ap.parse_args()

    os.makedirs('output/seedvc_stream_baseline', exist_ok=True)

    # Same 15-combo set
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

    print('Loading SeedVCWrapper...')
    wrapper = SeedVCWrapper()

    f0_scores = []; mel_scores = []
    for i in range(args.n_samples):
        src, tim, pit = combos[i]
        print(f'\n--- Sample {i} ---  src={os.path.basename(src)} pit={os.path.basename(pit)}')

        a, sr_a = run_nonstream(wrapper, src, tim, pit, args.diffusion_steps)
        b, sr_b = run_nonstream(wrapper, src, tim, pit, args.diffusion_steps)

        # F0 pearson between two independent non-streaming samples
        a16 = librosa.resample(a, orig_sr=sr_a, target_sr=16000)
        b16 = librosa.resample(b, orig_sr=sr_b, target_sr=16000)
        f0_a = extract_f0(a16, 16000, device='cuda:0').cpu().numpy()
        f0_b = extract_f0(b16, 16000, device='cuda:0').cpu().numpy()
        L = min(len(f0_a), len(f0_b))
        v = (f0_a[:L] > 50) & (f0_b[:L] > 50)
        f0_p = None
        if v.sum() >= 5:
            f0_p = float(np.corrcoef(np.log2(f0_a[:L][v]), np.log2(f0_b[:L][v]))[0, 1])

        m_a = mel_logmel(a, sr_a); m_b = mel_logmel(b, sr_b)
        mel_c = cosine_framewise(m_a, m_b)

        f0_scores.append(f0_p)
        mel_scores.append(mel_c)
        print(f'  f0_nonstream_vs_nonstream={f0_p if f0_p is not None else float("nan"):.3f}  mel_cos={mel_c:.3f}')

        sf.write(f'output/seedvc_stream_baseline/ns1_{i}.wav', a, sr_a)
        sf.write(f'output/seedvc_stream_baseline/ns2_{i}.wav', b, sr_b)

    print(f'\n=== CFM stochasticity baseline (non-stream vs non-stream, n={args.n_samples}) ===')
    print(f'F0 pearson:  mean={np.mean([x for x in f0_scores if x is not None]):.3f}  min={min(x for x in f0_scores if x is not None):.3f}  max={max(x for x in f0_scores if x is not None):.3f}')
    print(f'Mel cos:     mean={np.mean(mel_scores):.3f}  min={min(mel_scores):.3f}  max={max(mel_scores):.3f}')


if __name__ == '__main__':
    main()
