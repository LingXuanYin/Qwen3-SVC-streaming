"""Test Seed-VC streaming vs non-streaming PERCEPTUAL consistency.

RULES.md criterion 5: "流式/非流式一致：两种模式输出一致".

CFM samples fresh noise on each call (no manual seed), so stream vs non-stream
can't be bit-exact. The correct interpretation: "output is FUNCTIONALLY the same
audio" — same content, same speaker, same F0 contour, same length.

Metrics:
  - Length ratio (>= 0.99)
  - Content: ASR CER between stream and non-stream outputs (< 0.2)
  - Speaker: cos(spk_stream, spk_non) (> 0.95)
  - F0: pearson(f0_stream, f0_non) over both-voiced frames (> 0.85)
  - Mel similarity: mean cos-sim per frame on log-mel (> 0.8)
"""
import os, sys, numpy as np, soundfile as sf, librosa, torch

# Fix ffmpeg for pydub
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
from pydub import AudioSegment as _AS
_AS.converter = ffmpeg_path

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
    # a, b: (n_mels, T)
    L = min(a.shape[1], b.shape[1])
    a = a[:, :L]; b = b[:, :L]
    dot = (a * b).sum(axis=0)
    na = np.linalg.norm(a, axis=0) + 1e-8
    nb = np.linalg.norm(b, axis=0) + 1e-8
    return float((dot / (na * nb)).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=3)
    ap.add_argument('--diffusion_steps', type=int, default=30)
    args = ap.parse_args()

    os.makedirs('output/seedvc_stream', exist_ok=True)

    # ASR (CPU to not contend with VC)
    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')
    def transcribe_arr(a, sr):
        if sr != 16000:
            a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(a, beam_size=1, language='en')
        return ''.join(s.text for s in segs).strip().lower()

    def cer(a, b):
        import difflib
        if not a:
            return 1.0 if b else 0.0
        return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()

    # Speaker encoder (Qwen3-TTS for metric consistency)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                                         torch_dtype=torch.bfloat16, device_map='cuda:0')
    qmodel = qwen.model

    def get_spk(a, sr):
        a24 = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        with torch.no_grad():
            return qmodel.speaker_encoder(mel[:, :400].to(device='cuda:0', dtype=torch.bfloat16)).float()

    print('Loading SeedVCWrapper...')
    wrapper = SeedVCWrapper()

    import glob, random
    # Same 15-combo FULL_3audio set as acceptance (seed=42). Use first n_samples.
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

    n_len = n_cer = n_spk = n_f0 = n_mel = 0
    for i in range(args.n_samples):
        src, tim, pit = combos[i]
        print(f'\n--- Sample {i} ---  src={os.path.basename(src)} tim={os.path.basename(tim)} pit={os.path.basename(pit)}')

        # Non-streaming: convert_voice is a generator (contains yield), return value via StopIteration.
        gen_n = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                       diffusion_steps=args.diffusion_steps,
                                       length_adjust=1.0, inference_cfg_rate=0.7,
                                       f0_condition=True, auto_f0_adjust=False, pitch_shift=0,
                                       stream_output=False)
        non_audio = None; non_sr = 44100
        try:
            while True:
                next(gen_n)
        except StopIteration as e:
            non_audio = e.value
        if isinstance(non_audio, tuple):
            non_sr, non_audio = non_audio

        gen_s = wrapper.convert_voice(source=src, target=tim, pitch_ref=pit,
                                       diffusion_steps=args.diffusion_steps,
                                       length_adjust=1.0, inference_cfg_rate=0.7,
                                       f0_condition=True, auto_f0_adjust=False, pitch_shift=0,
                                       stream_output=True)
        stream_audio = None; stream_sr = 44100
        for item in gen_s:
            if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
                full = item[1]
                if isinstance(full, tuple):
                    stream_sr, stream_audio = full
                else:
                    stream_audio = full
                break

        if non_audio is None or stream_audio is None:
            print(f'  missing output: non={non_audio is not None} stream={stream_audio is not None}')
            continue

        def norm(x):
            if x.ndim > 1:
                x = x.mean(-1) if x.shape[1] > x.shape[0] else x.mean(0)
            return x.astype(np.float32)

        a = norm(non_audio); b = norm(stream_audio)
        len_ratio = min(len(a), len(b)) / max(len(a), len(b))
        len_ok = len_ratio >= 0.99

        sf.write(f'output/seedvc_stream/non_{i}.wav', a, non_sr)
        sf.write(f'output/seedvc_stream/stream_{i}.wav', b, stream_sr)

        # Content
        txt_non = transcribe_arr(a, non_sr)
        txt_str = transcribe_arr(b, stream_sr)
        c = cer(txt_non, txt_str)
        cer_ok = c < 0.2

        # Speaker
        spk_non = get_spk(a, non_sr); spk_str = get_spk(b, stream_sr)
        cos = torch.nn.functional.cosine_similarity(spk_non, spk_str).item()
        spk_ok = cos > 0.95

        # F0
        a16 = librosa.resample(a, orig_sr=non_sr, target_sr=16000)
        b16 = librosa.resample(b, orig_sr=stream_sr, target_sr=16000)
        f0_non = extract_f0(a16, 16000, device='cuda:0').cpu().numpy()
        f0_str = extract_f0(b16, 16000, device='cuda:0').cpu().numpy()
        L = min(len(f0_non), len(f0_str))
        v = (f0_non[:L] > 50) & (f0_str[:L] > 50)
        f0_p = None; f0_ok = False
        if v.sum() >= 5:
            f0_p = float(np.corrcoef(np.log2(f0_non[:L][v]), np.log2(f0_str[:L][v]))[0, 1])
            f0_ok = f0_p > 0.85

        # Mel framewise cos
        m_n = mel_logmel(a, non_sr); m_s = mel_logmel(b, stream_sr)
        mel_cos = cosine_framewise(m_n, m_s)
        mel_ok = mel_cos > 0.8

        if len_ok: n_len += 1
        if cer_ok: n_cer += 1
        if spk_ok: n_spk += 1
        if f0_ok: n_f0 += 1
        if mel_ok: n_mel += 1

        print(f'  len={len_ratio:.3f} cer={c:.2f} spk={cos:.3f} '
              f'f0p={f0_p if f0_p is not None else float("nan"):.3f} mel={mel_cos:.3f}  '
              f'[L={"P" if len_ok else "F"} C={"P" if cer_ok else "F"} '
              f'S={"P" if spk_ok else "F"} F={"P" if f0_ok else "F"} M={"P" if mel_ok else "F"}]')

    n = args.n_samples
    print(f'\n=== STREAMING CONSISTENCY ({n} samples) ===')
    print(f'Length ratio >= 0.99: {n_len}/{n}')
    print(f'Content CER < 0.2:    {n_cer}/{n}')
    print(f'Speaker cos > 0.95:   {n_spk}/{n}')
    print(f'F0 pearson > 0.85:    {n_f0}/{n}')
    print(f'Mel framewise cos>0.8: {n_mel}/{n}')
    all_ok = (n_len == n and n_cer == n and n_spk == n and n_f0 == n and n_mel == n)
    print(f'\nAll PASS: {all_ok}')


if __name__ == '__main__':
    main()
