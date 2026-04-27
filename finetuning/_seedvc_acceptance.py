"""Seed-VC 15-combo FULL_3audio acceptance (zero-shot, separate pitch_ref).

Metrics per RULES.md:
  1. F0 pearson > 0.8 AND |diff| < 1st (output F0 vs pitch_ref F0)
  2. cos(out, timbre_ref) > 0.7 AND > cos(out, source)  (speaker)
  3. ASR CER < 50% vs GT lyrics (content, relaxed threshold)
  4. Joint F0+Spk+Cnt
  5. (streaming deferred)
  6. length ratio 0.8~1.2x
"""
import os, sys, json, glob, random, subprocess, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
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
    ap.add_argument('--output_dir', default='output/seedvc_accept')
    ap.add_argument('--checkpoint', default=None, help='Optional fine-tuned ckpt path')
    ap.add_argument('--config', default=None, help='Optional config path matching the checkpoint')
    ap.add_argument('--cfg_rate', type=float, default=0.7)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Speaker encoder (reuse Qwen3-TTS's for consistency with previous evals)
    device = 'cuda:0'
    print('Loading Qwen3-TTS speaker_encoder (for speaker metric only)...', flush=True)
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

    # Build combos — English singing (lyrics GT) + speech timbre + tenor pitch
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
    all_rows = []
    for src, tim, pit, label in combos:
        out_wav_dir = os.path.join(args.output_dir, label)
        os.makedirs(out_wav_dir, exist_ok=True)
        # Call Seed-VC inference
        cmd = [
            sys.executable,
            'external/seed-vc/inference.py',
            '--source', src,
            '--target', tim,
            '--pitch-ref', pit,
            '--output', out_wav_dir,
            '--diffusion-steps', str(args.diffusion_steps),
            '--f0-condition', 'True',
            '--fp16', 'True',
            '--inference-cfg-rate', str(args.cfg_rate),
        ]
        if args.checkpoint:
            cmd += ['--checkpoint', args.checkpoint]
        if args.config:
            cmd += ['--config', args.config]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        # Find output wav
        wavs = glob.glob(os.path.join(out_wav_dir, '*.wav'))
        if not wavs:
            print(f'  {label}: Seed-VC FAILED. stderr tail:\n{r.stderr[-400:]}')
            continue
        out_path = wavs[0]

        # Load audios
        src_a, src_sr = sf.read(src, dtype='float32');   src_a = src_a.mean(-1) if src_a.ndim > 1 else src_a
        tim_a, tim_sr = sf.read(tim, dtype='float32');   tim_a = tim_a.mean(-1) if tim_a.ndim > 1 else tim_a
        pit_a, pit_sr = sf.read(pit, dtype='float32');   pit_a = pit_a.mean(-1) if pit_a.ndim > 1 else pit_a
        out_a, out_sr = sf.read(out_path, dtype='float32'); out_a = out_a.mean(-1) if out_a.ndim > 1 else out_a

        # 6. Length
        ratio = (len(out_a)/out_sr) / (len(src_a)/src_sr)
        length_ok = 0.8 <= ratio <= 1.2
        if length_ok: pass_len += 1

        # 1. F0 tracking — correctly compare output F0 against what Seed-VC was actually fed.
        # Seed-VC's F0_alt input (after our pitch_ref patch) = linear-resize(F0(pit), len(F0(source))).
        # So output F0 (which is on the source time-axis) should correlate with the *resized* pit F0,
        # not the raw pit F0 at a different time scale. Replicate the resize exactly.
        # Use 16kHz resample (matches Seed-VC's internal F0 extraction on 16k audio).
        src_16k = librosa.resample(src_a.astype(np.float32), orig_sr=src_sr, target_sr=16000)
        pit_16k = librosa.resample(pit_a.astype(np.float32), orig_sr=pit_sr, target_sr=16000)
        out_16k = librosa.resample(out_a.astype(np.float32), orig_sr=out_sr, target_sr=16000)
        f0_src = extract_f0(src_16k, 16000, device=device).cpu().numpy()  # source F0 for length reference
        f0_pit_raw = extract_f0(pit_16k, 16000, device=device).cpu().numpy()
        f0_out_raw = extract_f0(out_16k, 16000, device=device).cpu().numpy()

        target_len = len(f0_src)  # Seed-VC's F0_alt length (matches source F0 length)
        # Resize pit F0 to target_len (same as our seed-vc patch)
        if len(f0_pit_raw) != target_len:
            f0_pit_resized = np.interp(
                np.linspace(0, len(f0_pit_raw) - 1, target_len),
                np.arange(len(f0_pit_raw)),
                f0_pit_raw,
            ).astype(np.float32)
        else:
            f0_pit_resized = f0_pit_raw
        # Output F0 may differ slightly in length from target_len (Seed-VC output length = input length);
        # trim / interp to match
        if len(f0_out_raw) != target_len:
            f0_out_aligned = np.interp(
                np.linspace(0, len(f0_out_raw) - 1, target_len),
                np.arange(len(f0_out_raw)),
                f0_out_raw,
            ).astype(np.float32)
        else:
            f0_out_aligned = f0_out_raw

        both = (f0_out_aligned > 50) & (f0_pit_resized > 50)
        pearson = diff = None; f0_ok = False
        if both.sum() >= 5:
            diff = np.log2(f0_out_aligned[both] / f0_pit_resized[both]).mean() * 12
            pearson = np.corrcoef(np.log2(f0_out_aligned[both]), np.log2(f0_pit_resized[both]))[0, 1]
            f0_ok = pearson > 0.8 and abs(diff) < 1.0
        if f0_ok: pass_f0 += 1

        # 2. Speaker
        spk_out = get_spk(out_a, out_sr)
        spk_tim = get_spk(tim_a, tim_sr)
        spk_src = get_spk(src_a, src_sr)
        cos_tim = torch.nn.functional.cosine_similarity(spk_out, spk_tim).item()
        cos_src = torch.nn.functional.cosine_similarity(spk_out, spk_src).item()
        spk_ok = cos_tim > 0.7 and cos_tim > cos_src
        if spk_ok: pass_spk += 1

        # 3. Content: ASR CER
        lyric = load_lyrics(src)
        text_out = transcribe(out_path, 'en')
        text_src = transcribe(src, 'en')
        c_out = cer(lyric, text_out) if lyric else cer(text_src, text_out)
        c_src_base = cer(lyric, text_src) if lyric else 0
        cnt_ok = c_out < 0.5
        if cnt_ok: pass_cer += 1

        if f0_ok and spk_ok and cnt_ok: pass_joint += 1

        print(f'  {label}: len={ratio:.2f} F0p={pearson if pearson is not None else "NA":.3f} diff={diff if diff is not None else 0:+.1f}st '
              f'cosT={cos_tim:.3f} cosS={cos_src:.3f} cer={c_out:.2f} '
              f'[F0={"P" if f0_ok else "F"} Spk={"P" if spk_ok else "F"} Cnt={"P" if cnt_ok else "F"}]')
        all_rows.append(dict(label=label, ratio=ratio, pearson=pearson, diff=diff,
                             cos_tim=cos_tim, cos_src=cos_src, cer=c_out, src_cer=c_src_base,
                             text_out=text_out, text_src=text_src, lyric=lyric))

    n = len(combos)
    print(f'\n=== SEED-VC ZERO-SHOT ACCEPTANCE ({n} combos) ===')
    print(f'1. F0 pearson>0.8 & |diff|<1st:  {pass_f0}/{n} ({pass_f0*100//n}%)')
    print(f'2. Speaker cos>0.7 & >src:       {pass_spk}/{n} ({pass_spk*100//n}%)')
    print(f'3. Content CER<50%:              {pass_cer}/{n} ({pass_cer*100//n}%)')
    print(f'4. Joint (F0+Spk+Cnt):           {pass_joint}/{n} ({pass_joint*100//n}%)')
    print(f'6. Length 0.8x~1.2x:             {pass_len}/{n} ({pass_len*100//n}%)')

    with open(os.path.join(args.output_dir, 'acceptance_report.json'), 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
