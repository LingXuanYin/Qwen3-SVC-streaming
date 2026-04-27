"""Seed-VC pure NATIVE baseline (no pitch_ref patch).

Use source as both content and F0 source. Target = timbre_ref. Evaluate:
  - F0 tracking: output F0 vs source F0 (should be near-perfect since F0 is preserved)
  - Speaker: cos(out, timbre_ref) > 0.7 and > cos(out, source)
  - Content: ASR CER vs source lyrics
  - Length: 0.8x~1.2x

This establishes Seed-VC's true zero-shot capability before any custom changes.
"""
import os, sys, json, glob, random, subprocess, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.svc.f0_extractor import extract_f0
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def load_lyrics(wav_path):
    p = wav_path.replace('.wav', '.json')
    if not os.path.exists(p): return None
    try:
        with open(p, 'r', encoding='utf-8') as f: data = json.load(f)
        words = [w['word'] for w in data if w.get('word') not in ('<SP>', '<AP>', None)]
        return ' '.join(words).strip().lower()
    except: return None


def cer(ref, hyp):
    import difflib
    if not ref: return 1.0 if hyp else 0.0
    return 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_combos', type=int, default=15)
    ap.add_argument('--diffusion_steps', type=int, default=30)
    ap.add_argument('--output_dir', default='output/seedvc_native')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda:0'
    print('Loading Qwen3-TTS speaker_encoder (metric only)...', flush=True)
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model

    def get_spk(a, sr):
        a24 = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1,2)
        with torch.no_grad():
            return model.speaker_encoder(mel[:,:400].to(device=device, dtype=torch.bfloat16)).float()

    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')
    def transcribe(path, lang):
        a, sr = sf.read(path, dtype='float32')
        if a.ndim > 1: a = a.mean(-1)
        if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(a, beam_size=1, language=lang)
        return ''.join(s.text for s in segs).strip().lower()

    ALTOS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    TENORS = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))
    SPEECH_F = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))
    SPEECH_M = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0197/*.wav'))
    random.seed(42)
    combos = []
    for i in range(args.n_combos):
        src = random.choice(ALTOS + TENORS)
        tim = random.choice(SPEECH_M + SPEECH_F)
        combos.append((src, tim, f'N_{i}'))  # no pitch_ref in native mode

    pass_f0 = pass_spk = pass_cer = pass_len = pass_joint = 0
    for src, tim, label in combos:
        out_wav_dir = os.path.join(args.output_dir, label)
        os.makedirs(out_wav_dir, exist_ok=True)
        r = subprocess.run([
            sys.executable, 'external/seed-vc/inference.py',
            '--source', src, '--target', tim,
            '--output', out_wav_dir,
            '--diffusion-steps', str(args.diffusion_steps),
            '--f0-condition', 'True', '--fp16', 'True',
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')
        wavs = glob.glob(os.path.join(out_wav_dir, '*.wav'))
        if not wavs:
            print(f'  {label}: FAIL: {r.stderr[-200:]}')
            continue
        out_path = wavs[0]
        src_a, src_sr = sf.read(src, dtype='float32');   src_a = src_a.mean(-1) if src_a.ndim > 1 else src_a
        tim_a, tim_sr = sf.read(tim, dtype='float32');   tim_a = tim_a.mean(-1) if tim_a.ndim > 1 else tim_a
        out_a, out_sr = sf.read(out_path, dtype='float32'); out_a = out_a.mean(-1) if out_a.ndim > 1 else out_a

        ratio = (len(out_a)/out_sr) / (len(src_a)/src_sr)
        length_ok = 0.8 <= ratio <= 1.2
        if length_ok: pass_len += 1

        # F0 tracking — output F0 should match source F0 (native mode: F0 comes from source)
        src_16k = librosa.resample(src_a.astype(np.float32), orig_sr=src_sr, target_sr=16000)
        out_16k = librosa.resample(out_a.astype(np.float32), orig_sr=out_sr, target_sr=16000)
        f0_src = extract_f0(src_16k, 16000, device=device).cpu().numpy()
        f0_out = extract_f0(out_16k, 16000, device=device).cpu().numpy()
        # Align lengths (Seed-VC preserves length up to hop effects)
        L = min(len(f0_src), len(f0_out))
        f0_src_a = f0_src[:L]; f0_out_a = f0_out[:L]
        both = (f0_src_a > 50) & (f0_out_a > 50)
        pearson = diff = None; f0_ok = False
        if both.sum() >= 5:
            diff = np.log2(f0_out_a[both] / f0_src_a[both]).mean() * 12
            pearson = np.corrcoef(np.log2(f0_out_a[both]), np.log2(f0_src_a[both]))[0, 1]
            f0_ok = pearson > 0.8 and abs(diff) < 1.0
        if f0_ok: pass_f0 += 1

        # Speaker
        spk_out = get_spk(out_a, out_sr)
        spk_tim = get_spk(tim_a, tim_sr)
        spk_src = get_spk(src_a, src_sr)
        cos_tim = torch.nn.functional.cosine_similarity(spk_out, spk_tim).item()
        cos_src = torch.nn.functional.cosine_similarity(spk_out, spk_src).item()
        spk_ok = cos_tim > 0.7 and cos_tim > cos_src
        if spk_ok: pass_spk += 1

        # Content
        lyric = load_lyrics(src)
        text_out = transcribe(out_path, 'en')
        text_src = transcribe(src, 'en')
        c_out = cer(lyric, text_out) if lyric else cer(text_src, text_out)
        cnt_ok = c_out < 0.5
        if cnt_ok: pass_cer += 1
        if f0_ok and spk_ok and cnt_ok: pass_joint += 1

        print(f'  {label}: len={ratio:.2f} F0p={pearson:.3f} diff={diff:+.1f}st '
              f'cosT={cos_tim:.3f} cosS={cos_src:.3f} cer={c_out:.2f} '
              f'[F0={"P" if f0_ok else "F"} Spk={"P" if spk_ok else "F"} Cnt={"P" if cnt_ok else "F"}]')

    n = len(combos)
    print(f'\n=== SEED-VC NATIVE (no pitch_ref patch, F0 from source) {n} combos ===')
    print(f'F0 (out F0 ≈ src F0): {pass_f0}/{n} ({pass_f0*100//n}%)')
    print(f'Speaker:              {pass_spk}/{n} ({pass_spk*100//n}%)')
    print(f'Content CER<50%:      {pass_cer}/{n} ({pass_cer*100//n}%)')
    print(f'Joint:                {pass_joint}/{n} ({pass_joint*100//n}%)')
    print(f'Length:               {pass_len}/{n} ({pass_len*100//n}%)')


if __name__ == '__main__':
    main()
