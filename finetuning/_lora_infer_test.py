"""Inference test: load LoRA ckpt, run teacher-forcing codec_0 prediction on one train sample,
decode to wav, check content via ASR + HuBERT cosine.

This tests whether high codec_0 acc (98%) actually produces intelligible audio.
Note: we're predicting codec_0 only; codec_1-15 are taken from ground-truth target
(since sub-talker LoRA isn't trained yet). This isolates the talker LoRA effect.
"""
import os, sys, glob, json, torch, numpy as np, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from finetuning.sft_svc_lora import ContentProjector, F0EmbProjector, build_prefix_embed
from peft import PeftModel


def main():
    ckpt = 'output/svc_lora_overfit/checkpoint-500'
    device = 'cuda:0'

    print('Loading base + LoRA...')
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    config = model.config
    D = model.talker.config.hidden_size

    # Freeze base and attach LoRA
    for p in model.parameters(): p.requires_grad = False
    model.talker = PeftModel.from_pretrained(model.talker, ckpt)
    model.talker.eval()

    content_proj = ContentProjector(in_dim=768, hidden=D).to(device=device, dtype=torch.float32)
    content_proj.load_state_dict(torch.load(os.path.join(ckpt, 'content_projector.pt'), map_location=device, weights_only=True))
    content_proj.eval()
    f0_proj = F0EmbProjector(hidden=D).to(device=device, dtype=torch.float32)
    f0_proj.load_state_dict(torch.load(os.path.join(ckpt, 'f0_projector.pt'), map_location=device, weights_only=True))
    f0_proj.eval()

    base_talker = model.talker.get_base_model()

    # Load one train sample (mini30)
    with open('L:/DATASET/svc_mini_sing30_new/manifest.json') as f:
        man = json.load(f)
    sample = torch.load(man[0]['path'], weights_only=True)
    content = sample['content'].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T, 768)
    target_codes = sample['target_codes'].unsqueeze(0).to(device)  # (1, T, 16)
    f0 = sample['f0'].unsqueeze(0).to(device)  # (1, T)
    spk_embed = sample['spk_embed'].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, D)

    T = content.shape[1]
    B = 1
    print(f'Sample: T={T}, audio_path={sample.get("audio_path","?")[-60:]}')

    # Same teacher-forcing forward as training, get codec_0 predictions
    with torch.inference_mode():
        content_emb = content_proj(content).to(torch.bfloat16)
        f0_emb = f0_proj(f0).to(torch.bfloat16)
        prefix_emb = build_prefix_embed(base_talker, config, spk_embed.to(torch.bfloat16), B)
        codec_0 = target_codes[:, :, 0]
        codec_0_emb = base_talker.get_input_embeddings()(codec_0).to(torch.bfloat16)
        main_emb = codec_0_emb + content_emb + f0_emb
        inputs_embeds = torch.cat([prefix_emb, main_emb], dim=1)
        attention_mask = torch.ones(B, 6 + T, device=device, dtype=torch.long)

        out = model.talker(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = out.logits  # (1, 6+T, V)
        pred_codec_0 = logits[:, 5:5+T, :].argmax(-1)  # predict codec_0[0..T-1]

    # Accuracy
    acc = (pred_codec_0 == codec_0).float().mean().item()
    print(f'Teacher-forcing codec_0 acc: {acc*100:.1f}%')

    # Reconstruct full codec: use predicted codec_0 + ground-truth codec_1-15 (baseline)
    codec_full_gt = target_codes[0]  # (T, 16)
    codec_full_pred_0 = target_codes[0].clone()
    codec_full_pred_0[:, 0] = pred_codec_0[0]

    # Decode both
    with torch.inference_mode():
        wav_gt, fs = model.speech_tokenizer.decode([{'audio_codes': codec_full_gt}])
        wav_pred, _ = model.speech_tokenizer.decode([{'audio_codes': codec_full_pred_0}])

    os.makedirs('output/svc_lora_overfit/_infer', exist_ok=True)
    sf.write('output/svc_lora_overfit/_infer/gt.wav', wav_gt[0], fs)
    sf.write('output/svc_lora_overfit/_infer/pred.wav', wav_pred[0], fs)
    print(f'Wav GT:   len={len(wav_gt[0])/fs:.2f}s')
    print(f'Wav PRED: len={len(wav_pred[0])/fs:.2f}s')

    # ASR both
    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')
    def transcribe(wav, sr):
        if sr != 16000: wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(wav.astype(np.float32), beam_size=1, language='zh')
        return ''.join(s.text for s in segs).strip()

    print(f'GT ASR:   {transcribe(wav_gt[0], fs)}')
    print(f'PRED ASR: {transcribe(wav_pred[0], fs)}')

    # HuBERT cosine
    from qwen_tts.svc.content_encoder import HubertContentEncoder
    hubert = HubertContentEncoder(device=device, dtype=torch.float16)
    c_gt = hubert.encode(wav_gt[0], fs, target_frames=T)
    c_pred = hubert.encode(wav_pred[0], fs, target_frames=T)
    cos = torch.nn.functional.cosine_similarity(c_gt, c_pred, dim=-1).mean().item()
    print(f'HuBERT cos(gt, pred): {cos:.3f}')


if __name__ == '__main__':
    main()
