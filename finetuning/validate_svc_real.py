# coding=utf-8
"""Real SVC validation: 3 independent audio inputs (source, timbre_ref, pitch_ref).

Metrics per RULES.md:
- F0 control: per-frame diff with pitch_ref F0, pearson correlation
- Speaker control: speaker embedding cosine with timbre_ref
- Content preservation: manual listen / ASR
- Length ratio: 0.8x~1.2x
"""
import argparse, os, sys, torch, soundfile as sf, librosa, numpy as np, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.content_encoder import HubertContentEncoder
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True, help="Source audio for content")
    p.add_argument("--timbre_ref", required=True, help="Timbre reference")
    p.add_argument("--pitch_ref", required=True, help="Pitch reference")
    p.add_argument("--output", required=True)
    p.add_argument("--mapper_layers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch_dtype=torch.bfloat16, device_map=device)
    model = m.model
    V = model.talker.config.vocab_size
    D = model.talker.config.hidden_size

    hubert = HubertContentEncoder(device=device, dtype=torch.float16)
    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    f0_proj.load_state_dict(torch.load(f"{args.ckpt}/f0_projector.pt", map_location=device, weights_only=True))
    sd = torch.load(f"{args.ckpt}/mapper.pt", map_location=device, weights_only=True)
    # Detect num_speakers from checkpoint (adv_spk_predictor is not needed at inference but must match shape)
    n_spk = 0
    for k, v in sd.items():
        if k.startswith("adv_spk_predictor") and k.endswith("weight") and v.ndim == 2:
            n_spk = v.shape[0]
    mapper = SVCMapperHubert(content_dim=768, cond_dim=D, hidden_size=1024, num_layers=args.mapper_layers,
                             num_heads=8, vocab_size=V, num_codebooks=16, num_speakers=n_spk).to(device=device, dtype=torch.float32)
    mapper.load_state_dict(sd)
    mapper.eval(); f0_proj.eval()

    def load(path):
        a, sr = sf.read(path, dtype='float32')
        if a.ndim > 1: a = a.mean(-1)
        if len(a) > sr * 10: a = a[:sr*10]
        return a, sr

    def get_spk(a, sr):
        a24 = librosa.resample(a, orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(a24).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1,2)
        with torch.no_grad():
            return model.speaker_encoder(mel[:,:400].to(device=device, dtype=torch.bfloat16)).float()

    # Load 3 audios
    src_a, src_sr = load(args.source)
    tim_a, tim_sr = load(args.timbre_ref)
    pit_a, pit_sr = load(args.pitch_ref)

    # Content from SOURCE
    with torch.inference_mode():
        src_codes = model.speech_tokenizer.encode(src_a, sr=src_sr).audio_codes[0]
    T = src_codes.shape[0]
    content = hubert.encode(src_a, src_sr, target_frames=T).unsqueeze(0).to(device)

    # Speaker from TIMBRE_REF
    spk = get_spk(tim_a, tim_sr)

    # F0 from PITCH_REF, aligned to source length T
    pit_f0 = align_f0_to_codec(extract_f0(pit_a, pit_sr, device=device), T).to(device)

    # Generate
    f0e = f0_proj(pit_f0.unsqueeze(0))
    with torch.no_grad():
        pred = mapper.predict(content, f0e, spk, temperature=0)
        wav_out, fs = model.speech_tokenizer.decode([{"audio_codes": pred[0]}])

    os.makedirs(args.output, exist_ok=True)
    sf.write(f"{args.output}/source.wav", src_a, src_sr)
    sf.write(f"{args.output}/timbre_ref.wav", tim_a, tim_sr)
    sf.write(f"{args.output}/pitch_ref.wav", pit_a, pit_sr)
    sf.write(f"{args.output}/svc_output.wav", wav_out[0], fs)

    # === Quantitative metrics ===
    print("\n=== SVC validation metrics ===")

    # 1. Length ratio
    src_dur = len(src_a)/src_sr
    out_dur = wav_out[0].shape[0]/fs
    ratio = out_dur / src_dur
    print(f"[Length] source={src_dur:.2f}s, output={out_dur:.2f}s, ratio={ratio:.2f}x  {'PASS' if 0.8<=ratio<=1.2 else 'FAIL'}")

    # 2. F0 tracking (output F0 should follow pitch_ref F0)
    f0_out_raw = extract_f0(wav_out[0], fs, device=device)
    f0_out = align_f0_to_codec(f0_out_raw, T).cpu().numpy()
    f0_pit = pit_f0.cpu().numpy()
    both = (f0_out > 50) & (f0_pit > 50)
    if both.sum() >= 5:
        diff_st = np.log2(f0_out[both] / f0_pit[both]) * 12
        mean_diff = diff_st.mean()
        pearson = np.corrcoef(np.log2(f0_out[both]), np.log2(f0_pit[both]))[0, 1]
        pass_f0 = abs(mean_diff) < 1.0 and pearson > 0.8
        print(f"[F0] mean_diff={mean_diff:+.2f}st (ideal 0, ±1st), pearson={pearson:.3f} (>0.8)  {'PASS' if pass_f0 else 'FAIL'}")
    else:
        print(f"[F0] too few voiced frames")

    # 3. Speaker similarity
    spk_tim = get_spk(tim_a, tim_sr)
    spk_out = get_spk(wav_out[0], fs)
    spk_src = get_spk(src_a, src_sr)
    cos_out_tim = torch.nn.functional.cosine_similarity(spk_out, spk_tim).item()
    cos_out_src = torch.nn.functional.cosine_similarity(spk_out, spk_src).item()
    pass_spk = cos_out_tim > 0.7 and cos_out_tim > cos_out_src
    print(f"[Speaker] cos(output, timbre)={cos_out_tim:.3f}, cos(output, source)={cos_out_src:.3f}  {'PASS' if pass_spk else 'FAIL'}")

    print(f"\nAudio files: {args.output}/")


if __name__ == "__main__":
    main()
