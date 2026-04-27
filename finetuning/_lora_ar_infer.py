"""Full AR inference with LoRA talker. Compare to teacher-forcing and source audio.

Setup:
  Prefill = prefix(6) + gen_start(codec_bos + content[0] + f0[0])
  Trailing = content[t+1] + f0[t+1] for t in 0..T-2, then tts_eos
  Generate loop uses sub-talker prediction for codec_1-15 (sub-talker is NOT fine-tuned,
  using its pretrained prior).
"""
import os, sys, json, glob, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from finetuning.sft_svc_lora import ContentProjector, F0EmbProjector, build_prefix_embed
from peft import PeftModel


def build_svc_ar_prefill(base_talker, config, content_proj_emb, f0_emb, spk_embed):
    """Prefill = 6-token prefix only. trailing[s] = content[s]+f0[s] for s=0..T-2.

    Training alignment:
      training main pos t (t=0..T-1) = codec_sum[t]+content[t]+f0[t]
      training logit[5+t] (HF shift) predicts labels[6+t] = codec_0[t] (for t=0)
                                  or codec_0[t] (for later t — logits[5] → codec_0[0], logits[6] → codec_0[1], ...)
      Actually HF shift: logits[i] predicts labels[i+1]. So:
        - logits[5] (prefix[5]=codec_bos+spk+tts_bos) → codec_0[0]  (no frame signal)
        - logits[6] (main[0]=codec_sum[0]+content[0]+f0[0]) → codec_0[1]
        - logits[5+t] → codec_0[t] for t=0; logits[6+t] → codec_0[t+1]

    Inference:
      prefill final logit → codec_0[0] (from prefix alone, matches training logits[5])
      gen step 0 input = sum(codec[0]) + trailing[0] = codec_sum[0]+content[0]+f0[0] if trailing[0]=content[0]+f0[0].
      This matches training main[0] → predict codec_0[1].
    """
    B = content_proj_emb.shape[0]
    T = content_proj_emb.shape[1]
    device = next(base_talker.parameters()).device
    dtype = next(base_talker.parameters()).dtype

    prefill = build_prefix_embed(base_talker, config, spk_embed, B)  # (B, 6, D), no gen_start

    tts_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device, dtype=torch.long,
    )
    _, tts_eos, tts_pad = base_talker.text_projection(
        base_talker.get_text_embeddings()(tts_ids)
    ).chunk(3, dim=1)

    # trailing[s] = content[s]+f0[s] for s=0..T-2  (length T-1)
    if T > 1:
        trailing = content_proj_emb[:, :-1] + f0_emb[:, :-1]  # (B, T-1, D)
    else:
        trailing = torch.zeros(B, 0, content_proj_emb.shape[-1], device=device, dtype=dtype)

    return prefill, trailing, tts_pad.expand(B, -1, -1), tts_eos.expand(B, -1, -1)


def main():
    ckpt = 'output/svc_lora_overfit3/checkpoint-500'
    device = 'cuda:0'

    print('Loading base + LoRA...')
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    config = model.config
    D = model.talker.config.hidden_size

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

    with open('L:/DATASET/svc_mini_sing30_new/manifest.json') as f:
        man = json.load(f)

    os.makedirs('output/svc_lora_ar_infer', exist_ok=True)
    from qwen_tts.svc.content_encoder import HubertContentEncoder
    hubert = HubertContentEncoder(device=device, dtype=torch.float16)
    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')

    for idx in [0, 1, 2]:
        sample = torch.load(man[idx]['path'], weights_only=True)
        content = sample['content'].unsqueeze(0).to(device=device, dtype=torch.float32)
        target_codes = sample['target_codes'].unsqueeze(0).to(device)
        f0 = sample['f0'].unsqueeze(0).to(device)
        spk_embed = sample['spk_embed'].unsqueeze(0).to(device=device, dtype=torch.float32)
        T = content.shape[1]

        print(f'\n=== Sample {idx} T={T} ===')

        with torch.inference_mode():
            content_emb = content_proj(content).to(torch.bfloat16)
            f0_emb = f0_proj(f0).to(torch.bfloat16)
            prefill, trailing, tts_pad_b, tts_eos_b = build_svc_ar_prefill(
                base_talker, config, content_emb, f0_emb, spk_embed.to(torch.bfloat16))
            attention_mask = torch.ones(1, prefill.shape[1], device=device, dtype=torch.long)

            # Call the underlying talker's generate (PeftModel unwraps automatically for generate)
            gen_out = model.talker.generate(
                inputs_embeds=prefill,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing,
                tts_pad_embed=tts_pad_b,
                max_new_tokens=T + 2,
                min_new_tokens=T,
                do_sample=False,
                subtalker_dosample=False,
                eos_token_id=config.talker_config.codec_eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # gen_out.sequences: (B, num_generated) for codec_0
            # We need codec_1..15 too — from the generator's trace (but HF generate only returns the primary sequence)
            # Since modeling.forward generate-step returns codec_ids by concatenating codec_0 + sub-talker outputs internally,
            # we need to reconstruct 16-codebook from gen_out. This requires digging into model internals.
            pass

        pred_codec_0_ar = gen_out.sequences[0, :T]  # trim to T (drop trailing eos/pad)
        gt_codec_0 = target_codes[0, :, 0]
        codec_match = (pred_codec_0_ar == gt_codec_0).float().mean().item()
        print(f'  AR codec_0 match rate: {codec_match*100:.1f}%')

        # Decode: use AR predicted codec_0 + GT codec_1-15 (since sub-talker is not fine-tuned)
        # This isolates talker LoRA's effect on content.
        codec_full_ar = target_codes[0].clone()
        codec_full_ar[:, 0] = pred_codec_0_ar
        with torch.inference_mode():
            wav_ar, fs = model.speech_tokenizer.decode([{'audio_codes': codec_full_ar}])
            wav_gt, _ = model.speech_tokenizer.decode([{'audio_codes': target_codes[0]}])

        sf.write(f'output/svc_lora_ar_infer/ar_{idx}.wav', wav_ar[0], fs)
        sf.write(f'output/svc_lora_ar_infer/gt_{idx}.wav', wav_gt[0], fs)

        # ASR + HuBERT cosine
        def to16(w, sr):
            return librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else w.astype(np.float32)
        segs_gt, _ = asr.transcribe(to16(wav_gt[0], fs), beam_size=1, language='zh')
        segs_ar, _ = asr.transcribe(to16(wav_ar[0], fs), beam_size=1, language='zh')
        text_gt = ''.join(s.text for s in segs_gt).strip()
        text_ar = ''.join(s.text for s in segs_ar).strip()
        print(f'  GT  ASR: {text_gt}')
        print(f'  AR  ASR: {text_ar}')

        c_gt = hubert.encode(wav_gt[0], fs, target_frames=T)
        c_ar = hubert.encode(wav_ar[0], fs, target_frames=T)
        cos = torch.nn.functional.cosine_similarity(c_gt, c_ar, dim=-1).mean().item()
        print(f'  HuBERT cos(gt, ar): {cos:.3f}')


if __name__ == '__main__':
    main()
