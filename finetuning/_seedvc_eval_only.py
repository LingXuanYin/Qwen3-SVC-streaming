"""Full 15-combo evaluation on existing output wavs (no inference).

Computes F0 (nearest-neighbor), Speaker (cos), Content (CER) on wavs in a given
directory structure: <output_dir>/F3A_<i>/<any>.wav

Used to evaluate test-time F0 refinement output without re-running inference.
"""
import os, sys, argparse, glob, json, random, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.svc.f0_extractor import extract_f0
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def nearest_resize(arr, target_len):
    sl = len(arr)
    if sl == target_len:
        return arr.copy()
    idx = np.clip(np.round(np.linspace(0, sl - 1, target_len)).astype(int), 0, sl - 1)
    return arr[idx]


def load_mono(path):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(-1)
    return a, sr


def cer(ref, hyp):
    import difflib
    if not ref:
        return 1.0 if hyp else 0.0
    return 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()


def load_lyrics(wav_path):
    p = wav_path.replace('.wav', '.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = [w['word'] for w in data if w.get('word') not in ('<SP>', '<AP>', None)]
        return ' '.join(words).strip().lower()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--n_combos', type=int, default=15)
    args = ap.parse_args()

    device = 'cuda:0'
    print('Loading Qwen3-TTS speaker_encoder...', flush=True)
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                                         torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model

    def get_spk(a, sr):
        a24 = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        with torch.no_grad():
            return model.speaker_encoder(mel[:, :400].to(device=device, dtype=torch.bfloat16)).float()

    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')

    def transcribe(path, lang='en'):
        a, sr = sf.read(path, dtype='float32')
        if a.ndim > 1:
            a = a.mean(-1)
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(a, beam_size=1, language=lang)
        return ''.join(s.text for s in segs).strip().lower()

    # Replay combos
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

    pass_f0 = pass_spk = pass_cer = pass_len = pass_joint = 0
    for src, tim, pit, label in combos:
        in_wavs = glob.glob(os.path.join(args.output_dir, label, '*.wav'))
        if not in_wavs:
            print(f'  {label}: no output wav'); continue
        out_path = in_wavs[0]

        src_a, src_sr = load_mono(src)
        tim_a, tim_sr = load_mono(tim)
        pit_a, pit_sr = load_mono(pit)
        out_a, out_sr = load_mono(out_path)

        # Length
        ratio = (len(out_a) / out_sr) / (len(src_a) / src_sr)
        length_ok = 0.8 <= ratio <= 1.2
        if length_ok:
            pass_len += 1

        # F0 (nearest-neighbor, voiced-aware)
        pit16 = librosa.resample(pit_a, orig_sr=pit_sr, target_sr=16000)
        out16 = librosa.resample(out_a, orig_sr=out_sr, target_sr=16000)
        f0_pit = extract_f0(pit16, 16000, device=device).cpu().numpy()
        f0_out = extract_f0(out16, 16000, device=device).cpu().numpy()
        f0_pit_on_out = nearest_resize(f0_pit, len(f0_out))
        both = (f0_out > 50) & (f0_pit_on_out > 50)
        pearson = diff = None
        f0_ok = False
        if both.sum() >= 5:
            diff = float(np.log2(f0_out[both] / f0_pit_on_out[both]).mean() * 12)
            pearson = float(np.corrcoef(np.log2(f0_out[both]), np.log2(f0_pit_on_out[both]))[0, 1])
            f0_ok = pearson > 0.8 and abs(diff) < 1.0
        if f0_ok:
            pass_f0 += 1

        # Speaker
        spk_out = get_spk(out_a, out_sr)
        spk_tim = get_spk(tim_a, tim_sr)
        spk_src = get_spk(src_a, src_sr)
        cos_tim = torch.nn.functional.cosine_similarity(spk_out, spk_tim).item()
        cos_src = torch.nn.functional.cosine_similarity(spk_out, spk_src).item()
        spk_ok = cos_tim > 0.7 and cos_tim > cos_src
        if spk_ok:
            pass_spk += 1

        # Content
        lyric = load_lyrics(src)
        text_out = transcribe(out_path, 'en')
        text_src = transcribe(src, 'en')
        c = cer(lyric, text_out) if lyric else cer(text_src, text_out)
        cnt_ok = c < 0.5
        if cnt_ok:
            pass_cer += 1

        if f0_ok and spk_ok and cnt_ok:
            pass_joint += 1

        print(f'  {label}: len={ratio:.2f} F0p={pearson if pearson is not None else float("nan"):.3f} '
              f'diff={diff if diff is not None else 0:+.1f}st cosT={cos_tim:.3f} cosS={cos_src:.3f} '
              f'cer={c:.2f} [F0={"P" if f0_ok else "F"} Spk={"P" if spk_ok else "F"} '
              f'Cnt={"P" if cnt_ok else "F"}]')

    n = len(combos)
    print(f'\n=== EVAL ({n} combos) ===')
    print(f'1. F0 pearson>0.8 & |diff|<1st:  {pass_f0}/{n} ({pass_f0*100//n}%)')
    print(f'2. Speaker cos>0.7 & >src:       {pass_spk}/{n} ({pass_spk*100//n}%)')
    print(f'3. Content CER<50%:              {pass_cer}/{n} ({pass_cer*100//n}%)')
    print(f'4. Joint (F0+Spk+Cnt):           {pass_joint}/{n} ({pass_joint*100//n}%)')
    print(f'6. Length 0.8x~1.2x:             {pass_len}/{n} ({pass_len*100//n}%)')


if __name__ == '__main__':
    main()
