"""Feasibility check before deciding the next-iteration path.

Goals:
A) Does Whisper reliably transcribe source singing? (baseline CER using GTSinger JSON lyrics)
B) What is the HuBERT content cosine between source and current-model SVC output?
C) What does Qwen3-TTS voice_clone sound like on singing src? (informational)

We don't train anything here — pure diagnosis.
"""
import os, sys, json, glob, numpy as np, torch, librosa, soundfile as sf
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def load_gtsinger_lyrics(wav_path):
    """Read the per-clip JSON next to the wav and return concatenated words."""
    json_path = wav_path.replace('.wav', '.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = [w['word'] for w in data if w.get('word') not in ('<SP>', '<AP>', None)]
    return ' '.join(words).strip()


def char_error_rate(ref, hyp):
    import difflib
    if not ref: return 1.0 if hyp else 0.0
    sm = difflib.SequenceMatcher(None, ref, hyp)
    return 1.0 - sm.ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='output/svc_final_v2/checkpoint-5000')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_samples', type=int, default=6)
    args = ap.parse_args()
    device = args.device

    # Load ASR
    from faster_whisper import WhisperModel
    print('Loading Whisper medium...', flush=True)
    asr = WhisperModel('medium', device='cpu', compute_type='int8')
    def transcribe(wav, sr, lang):
        if sr != 16000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(wav.astype(np.float32), beam_size=1, language=lang)
        return ''.join(s.text for s in segs).strip().lower()

    # Load model + our mapper (for check C)
    print('Loading Qwen3-TTS + mapper...', flush=True)
    m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = m.model
    V = model.talker.config.vocab_size
    D = model.talker.config.hidden_size
    hubert = HubertContentEncoder(device=device, dtype=torch.float16)
    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    f0_proj.load_state_dict(torch.load(f'{args.ckpt}/f0_projector.pt', map_location=device, weights_only=True))
    sd = torch.load(f'{args.ckpt}/mapper.pt', map_location=device, weights_only=True)
    n_spk = 0
    for k, v in sd.items():
        if k.startswith('adv_spk_predictor') and k.endswith('weight') and v.ndim == 2:
            n_spk = v.shape[0]
    mapper = SVCMapperHubert(content_dim=768, cond_dim=D, hidden_size=1024, num_layers=4,
                             num_heads=8, vocab_size=V, num_codebooks=16, num_speakers=n_spk
                             ).to(device=device, dtype=torch.float32)
    mapper.load_state_dict(sd); mapper.eval(); f0_proj.eval()

    def get_spk(a, sr):
        a24 = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1,2)
        with torch.no_grad():
            return model.speaker_encoder(mel[:,:400].to(device=device, dtype=torch.bfloat16)).float()

    # Collect test samples — use English GTSinger "all is found" (have JSON with lyrics)
    src_paths = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/Control/all is found/Control_Group/*.wav'))
    if not src_paths:
        # fallback search
        src_paths = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))
    src_paths = src_paths[:args.n_samples]
    tim_paths = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[:args.n_samples]
    pit_paths = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Tenor-1/*/Always Remember Us This Way/Control_Group/*.wav'))[:args.n_samples]

    print(f'\n{"="*70}\nCHECK A: ASR baseline on source singing (GTSinger lyrics as GT)\n{"="*70}')
    cer_src_list = []
    refs = []
    for src in src_paths:
        ref = load_gtsinger_lyrics(src)
        if ref is None:
            print(f'  {os.path.basename(src)}: no JSON'); continue
        src_a, src_sr = sf.read(src, dtype='float32')
        if src_a.ndim > 1: src_a = src_a.mean(-1)
        hyp_src = transcribe(src_a, src_sr, lang='en')
        cer = char_error_rate(ref.lower(), hyp_src.lower())
        cer_src_list.append(cer)
        refs.append((src, ref, hyp_src, src_a, src_sr, cer))
        print(f'  {os.path.basename(src)}:')
        print(f'    GT:  {ref.lower()[:80]}')
        print(f'    ASR: {hyp_src[:80]}')
        print(f'    CER: {cer:.3f}')
    mean_src_cer = float(np.mean(cer_src_list)) if cer_src_list else 1.0
    print(f'\n=> Source singing ASR mean CER: {mean_src_cer:.3f}')
    print('   (if CER >= 0.5 on clean source, ASR itself is unreliable for singing — need HuBERT cosine instead)')

    print(f'\n{"="*70}\nCHECK C: HuBERT content cosine source vs mapper output\n{"="*70}')
    cos_list = []
    cer_out_list = []
    for (src, ref, hyp_src, src_a, src_sr, src_cer), tim, pit in zip(refs, tim_paths, pit_paths):
        tim_a, tim_sr = sf.read(tim, dtype='float32')
        if tim_a.ndim > 1: tim_a = tim_a.mean(-1)
        pit_a, pit_sr = sf.read(pit, dtype='float32')
        if pit_a.ndim > 1: pit_a = pit_a.mean(-1)

        with torch.inference_mode():
            src_codes = model.speech_tokenizer.encode(src_a, sr=src_sr).audio_codes[0]
        T = src_codes.shape[0]
        content_src = hubert.encode(src_a, src_sr, target_frames=T)  # (T, 768)
        content_src_in = content_src.unsqueeze(0).to(device)
        spk = get_spk(tim_a, tim_sr)
        pit_f0 = align_f0_to_codec(extract_f0(pit_a, pit_sr, device=device), T).to(device)
        f0e = f0_proj(pit_f0.unsqueeze(0))

        with torch.no_grad():
            pred = mapper.predict(content_src_in, f0e, spk, temperature=0)
            wav_out, fs = model.speech_tokenizer.decode([{'audio_codes': pred[0]}])

        # Extract HuBERT content from output
        content_out = hubert.encode(wav_out[0], fs, target_frames=T)  # (T, 768)
        cos_per_frame = torch.nn.functional.cosine_similarity(content_src, content_out, dim=-1)
        mean_cos = cos_per_frame.mean().item()
        cos_list.append(mean_cos)

        # ASR on output
        hyp_out = transcribe(wav_out[0], fs, lang='en')
        cer_out = char_error_rate(ref.lower(), hyp_out.lower())
        cer_out_list.append(cer_out)

        print(f'  {os.path.basename(src)}:')
        print(f'    output ASR: {hyp_out[:80]}')
        print(f'    content cos(src, out): {mean_cos:.3f}')
        print(f'    output CER (vs GT):  {cer_out:.3f}  (source CER was {src_cer:.3f})')

    mean_cos = float(np.mean(cos_list)) if cos_list else 0.0
    mean_out_cer = float(np.mean(cer_out_list)) if cer_out_list else 1.0
    print(f'\n=> Mean content cosine(src, output): {mean_cos:.3f}')
    print(f'=> Mean output CER vs GT: {mean_out_cer:.3f}')
    print(f'=> Source baseline CER:  {mean_src_cer:.3f}')
    print(f'=> Delta CER (output - source): {mean_out_cer - mean_src_cer:+.3f}')

    print(f'\n{"="*70}\nDIAGNOSIS\n{"="*70}')
    print(f'If content cos > 0.7 BUT output CER >> source CER:')
    print(f'  → content preserved in HuBERT space but ASR confused (likely ok delivery)')
    print(f'If content cos < 0.5:')
    print(f'  → mapper output really lost content. Need stronger recipe (LoRA on TTS, etc).')
    print(f'If source CER > 0.5:')
    print(f'  → Whisper not reliable for singing; use content cos as primary metric.')


if __name__ == '__main__':
    main()
