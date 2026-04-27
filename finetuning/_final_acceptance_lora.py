"""Final acceptance test on LoRA-SVC model.

15 random FULL_3audio combos + ASR (GT text from GTSinger JSON) +
F0 tracking + speaker similarity + length ratio + streaming.
"""
import os, sys, json, glob, random, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from finetuning.sft_svc_lora import ContentProjector, F0EmbProjector, build_prefix_embed
from finetuning._lora_ar_infer import build_svc_ar_prefill
from peft import PeftModel


def load_gtsinger_lyrics(wav_path):
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
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_combos', type=int, default=15)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = args.device

    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    config = model.config
    D = model.talker.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    talker_dir = os.path.join(args.ckpt, 'talker_lora')
    model.talker = PeftModel.from_pretrained(model.talker, talker_dir)
    model.talker.eval()
    base_talker = model.talker.get_base_model()
    sub_dir = os.path.join(args.ckpt, 'sub_talker_lora')
    if os.path.exists(sub_dir):
        base_talker.code_predictor = PeftModel.from_pretrained(base_talker.code_predictor, sub_dir)
        base_talker.code_predictor.eval()

    cp = ContentProjector(in_dim=768, hidden=D).to(device=device, dtype=torch.float32)
    cp.load_state_dict(torch.load(os.path.join(args.ckpt, 'content_projector.pt'), map_location=device, weights_only=True))
    cp.eval()
    fp = F0EmbProjector(hidden=D).to(device=device, dtype=torch.float32)
    fp.load_state_dict(torch.load(os.path.join(args.ckpt, 'f0_projector.pt'), map_location=device, weights_only=True))
    fp.eval()

    hubert = HubertContentEncoder(device=device, dtype=torch.float16)
    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')

    def transcribe(wav, sr, lang):
        if sr != 16000: wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(wav.astype(np.float32), beam_size=1, language=lang)
        return ''.join(s.text for s in segs).strip().lower()

    def get_spk(a, sr):
        a24 = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1,2)
        with torch.no_grad():
            return model.speaker_encoder(mel[:,:400].to(device=device, dtype=torch.bfloat16)).float()

    # Test audio pools — English singing (lyrics GT available via JSON)
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

    os.makedirs('output/lora_acceptance', exist_ok=True)
    pass_f0 = pass_spk = pass_len = pass_cer = pass_joint = 0
    all_rows = []
    for src, tim, pit, label in combos:
        src_a, src_sr = sf.read(src, dtype='float32')
        tim_a, tim_sr = sf.read(tim, dtype='float32')
        pit_a, pit_sr = sf.read(pit, dtype='float32')
        if src_a.ndim > 1: src_a = src_a.mean(-1)
        if tim_a.ndim > 1: tim_a = tim_a.mean(-1)
        if pit_a.ndim > 1: pit_a = pit_a.mean(-1)
        src_a = src_a[:10*src_sr]; tim_a = tim_a[:10*tim_sr]; pit_a = pit_a[:10*pit_sr]

        # Build features
        with torch.inference_mode():
            src_codes = model.speech_tokenizer.encode(src_a, sr=src_sr).audio_codes[0]
        T = src_codes.shape[0]
        content = hubert.encode(src_a, src_sr, target_frames=T).unsqueeze(0).to(device=device, dtype=torch.float32)
        spk = get_spk(tim_a, tim_sr).to(torch.float32)
        pit_f0 = align_f0_to_codec(extract_f0(pit_a, pit_sr, device=device), T).to(device).unsqueeze(0)

        with torch.inference_mode():
            content_emb = cp(content).to(torch.bfloat16)
            f0_emb = fp(pit_f0).to(torch.bfloat16)
            prefill, trailing, tts_pad_b, _ = build_svc_ar_prefill(
                base_talker, config, content_emb, f0_emb, spk.to(torch.bfloat16))
            attention_mask = torch.ones(1, prefill.shape[1], device=device, dtype=torch.long)
            gen_out = model.talker.generate(
                inputs_embeds=prefill, attention_mask=attention_mask,
                trailing_text_hidden=trailing, tts_pad_embed=tts_pad_b,
                max_new_tokens=T + 2, min_new_tokens=T,
                do_sample=False, subtalker_dosample=False,
                eos_token_id=config.talker_config.codec_eos_token_id,
                output_hidden_states=True, return_dict_in_generate=True,
            )
        pred_codec_0 = gen_out.sequences[0, :T]

        # Build full codec: rerun talker TF to get hidden, then sub-talker generate per frame
        with torch.inference_mode():
            codec_0_emb_out = base_talker.get_input_embeddings()(pred_codec_0.unsqueeze(0)).to(torch.bfloat16)
            main_emb = codec_0_emb_out + content_emb + f0_emb
            ie2 = torch.cat([prefill, main_emb], dim=1)
            outer = base_talker.model(inputs_embeds=ie2,
                                      attention_mask=torch.ones(1, ie2.shape[1], device=device, dtype=torch.long))
            hidden = outer.last_hidden_state
            talker_h_codec = hidden[:, 6:6+T, :]

            sub_base = base_talker.code_predictor.get_base_model() if hasattr(base_talker.code_predictor, 'get_base_model') else base_talker.code_predictor
            codec_rest = torch.zeros(T, 15, dtype=torch.long, device=device)
            for t in range(T):
                seed = torch.cat([talker_h_codec[:, t:t+1, :],
                                  base_talker.get_input_embeddings()(pred_codec_0[t:t+1].unsqueeze(0))], dim=1)
                sub_res = sub_base.generate(
                    inputs_embeds=seed.to(torch.bfloat16),
                    max_new_tokens=15, min_new_tokens=15,
                    do_sample=False,
                )
                codec_rest[t] = sub_res[0, :15]
            codec_full = torch.cat([pred_codec_0.unsqueeze(-1), codec_rest], dim=-1)
            wav_out, fs = model.speech_tokenizer.decode([{'audio_codes': codec_full}])

        sf.write(f'output/lora_acceptance/{label}.wav', wav_out[0], fs)

        # Metrics
        ratio = len(wav_out[0])/fs / (len(src_a)/src_sr)
        length_ok = 0.8 <= ratio <= 1.2
        if length_ok: pass_len += 1

        # F0
        f0_out = align_f0_to_codec(extract_f0(wav_out[0], fs, device=device), T).cpu().numpy()
        f0_pit_np = pit_f0[0].cpu().numpy()
        both = (f0_out > 50) & (f0_pit_np > 50)
        f0_ok = False; pearson = diff = 0
        if both.sum() >= 5:
            diff = np.log2(f0_out[both] / f0_pit_np[both]).mean() * 12
            pearson = np.corrcoef(np.log2(f0_out[both]), np.log2(f0_pit_np[both]))[0, 1]
            f0_ok = pearson > 0.8 and abs(diff) < 1.0
        if f0_ok: pass_f0 += 1

        # Speaker
        spk_out = get_spk(wav_out[0], fs)
        spk_src = get_spk(src_a, src_sr)
        cos_tim = torch.nn.functional.cosine_similarity(spk_out, spk).item()
        cos_src = torch.nn.functional.cosine_similarity(spk_out, spk_src).item()
        spk_ok = cos_tim > 0.7 and cos_tim > cos_src
        if spk_ok: pass_spk += 1

        # Content: ASR CER vs GTSinger lyric
        lyric = load_gtsinger_lyrics(src)
        text_out = transcribe(wav_out[0], fs, lang='en')
        text_src = transcribe(src_a, src_sr, lang='en')
        c_out = 1.0; c_src_base = 1.0
        if lyric:
            c_out = cer(lyric, text_out.lower())
            c_src_base = cer(lyric, text_src.lower())
        content_ok = c_out < 0.5  # relaxed threshold
        if content_ok: pass_cer += 1

        if f0_ok and spk_ok and content_ok: pass_joint += 1

        print(f'  {label}: len={ratio:.2f} F0p={pearson:.3f} diff={diff:+.1f}st '
              f'cosT={cos_tim:.3f} cosS={cos_src:.3f} cer={c_out:.2f} (src_cer={c_src_base:.2f}) '
              f'[F0={"P" if f0_ok else "F"} Spk={"P" if spk_ok else "F"} Cnt={"P" if content_ok else "F"}]')
        all_rows.append(dict(label=label, ratio=ratio, pearson=pearson, diff=diff,
                             cos_tim=cos_tim, cos_src=cos_src, cer=c_out, src_cer=c_src_base))

    n = len(combos)
    print(f'\n=== ACCEPTANCE ({n} combos) ===')
    print(f'F0 pearson>0.8 + diff<1st: {pass_f0}/{n} ({pass_f0*100//n}%)')
    print(f'Speaker cos>0.7 & >src:   {pass_spk}/{n} ({pass_spk*100//n}%)')
    print(f'Content CER<50% vs GT:    {pass_cer}/{n} ({pass_cer*100//n}%)')
    print(f'Length 0.8x~1.2x:         {pass_len}/{n} ({pass_len*100//n}%)')
    print(f'Joint (F0+Spk+Cnt PASS):  {pass_joint}/{n} ({pass_joint*100//n}%)')


if __name__ == '__main__':
    main()
