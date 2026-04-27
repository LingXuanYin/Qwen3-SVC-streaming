"""Diagnose intermediate LoRA checkpoint with FULL AR inference (no GT cheating).

Key question: does sub-talker (predicting codec_1-15 using its pretrained prior +
new talker_hidden distribution) produce intelligible audio?

If ASR on output matches source ASR → architecture is sound, just train more.
If ASR is garbage → sub-talker cannot cope with LoRA-modified talker_hidden,
                    need to increase sub_weight or rethink.
"""
import os, sys, json, glob, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from finetuning.sft_svc_lora import ContentProjector, F0EmbProjector, build_prefix_embed
from finetuning._lora_ar_infer import build_svc_ar_prefill
from peft import PeftModel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_samples', type=int, default=5)
    args = ap.parse_args()
    device = args.device

    print(f'Loading checkpoint: {args.ckpt}')
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    config = model.config
    D = model.talker.config.hidden_size
    for p in model.parameters(): p.requires_grad = False

    # Load talker LoRA
    talker_lora_dir = os.path.join(args.ckpt, 'talker_lora')
    if not os.path.exists(talker_lora_dir):
        talker_lora_dir = args.ckpt  # fallback
    model.talker = PeftModel.from_pretrained(model.talker, talker_lora_dir)
    model.talker.eval()

    # Load sub-talker LoRA (if present)
    sub_lora_dir = os.path.join(args.ckpt, 'sub_talker_lora')
    base_talker = model.talker.get_base_model()
    if os.path.exists(sub_lora_dir):
        print('Loading sub-talker LoRA')
        base_talker.code_predictor = PeftModel.from_pretrained(base_talker.code_predictor, sub_lora_dir)
        base_talker.code_predictor.eval()

    content_proj = ContentProjector(in_dim=768, hidden=D).to(device=device, dtype=torch.float32)
    content_proj.load_state_dict(torch.load(os.path.join(args.ckpt, 'content_projector.pt'), map_location=device, weights_only=True))
    content_proj.eval()
    f0_proj = F0EmbProjector(hidden=D).to(device=device, dtype=torch.float32)
    f0_proj.load_state_dict(torch.load(os.path.join(args.ckpt, 'f0_projector.pt'), map_location=device, weights_only=True))
    f0_proj.eval()

    from faster_whisper import WhisperModel
    asr = WhisperModel('medium', device='cpu', compute_type='int8')
    from qwen_tts.svc.content_encoder import HubertContentEncoder
    hubert = HubertContentEncoder(device=device, dtype=torch.float16)

    def transcribe(wav, sr, lang):
        if sr != 16000: wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = asr.transcribe(wav.astype(np.float32), beam_size=1, language=lang)
        return ''.join(s.text for s in segs).strip()

    # Pick samples from sing_parallel (real singing)
    with open('L:/DATASET/svc_sing_parallel/manifest.json') as f:
        man = json.load(f)
    test_man = [x for x in man[:200] if x['T'] <= 80 and x.get('shift') == 0][:args.n_samples]
    print(f'Testing {len(test_man)} samples (shift=0 self-recon)')

    os.makedirs('output/diagnose_ar', exist_ok=True)
    for idx, item in enumerate(test_man):
        sample = torch.load(item['path'], weights_only=True)
        content = sample['content'].unsqueeze(0).to(device=device, dtype=torch.float32)
        target_codes = sample['target_codes'].unsqueeze(0).to(device)
        f0 = sample['f0'].unsqueeze(0).to(device)
        spk_embed = sample['spk_embed'].unsqueeze(0).to(device=device, dtype=torch.float32)
        T = content.shape[1]
        audio_path = sample.get('audio_path', '?')
        print(f'\n=== [{idx}] T={T} src={audio_path[-50:]} ===')

        with torch.inference_mode():
            content_emb = content_proj(content).to(torch.bfloat16)
            f0_emb_ = f0_proj(f0).to(torch.bfloat16)
            prefill, trailing, tts_pad_b, _ = build_svc_ar_prefill(
                base_talker, config, content_emb, f0_emb_, spk_embed.to(torch.bfloat16))
            attention_mask = torch.ones(1, prefill.shape[1], device=device, dtype=torch.long)

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

            # Extract codec_0..15 from generate: HF model.generate returns sequences = codec_0 ids only.
            # codec_1..15 are predicted internally by sub-talker but not exposed. We need to re-run
            # with a custom loop — OR run generate in "teacher-forcing-like" style extracting per-step.
            # For a quick diagnosis, use the model's internal pipeline:
            # Actually the simplest: run generate, then for each generated codec_0[t], manually call
            # sub-talker with (talker_hidden[t], codec_0[t], ...AR) to get codec_1..15.
            # But we don't have talker_hidden per step from gen_out.
            # Use gen_out.hidden_states? Per HF it's the hidden states at each gen step.
            pass

        # gen_out.hidden_states is a tuple of per-step hiddens; last-layer last-pos gives codec hidden.
        pred_codec_0 = gen_out.sequences[0, :T]
        # Build codec_full using predicted codec_0 + predicted codec_1-15 via sub-talker
        # Extract talker hidden at each gen step (last layer, last position)
        # gen_out.hidden_states is tuple of length = num_gen_steps; each element is tuple of per-layer tensors
        # shape: gen_out.hidden_states[step][-1] shape = (B, 1, D) for gen steps (1 for AR gen)
        codec_rest_pred = torch.zeros(T, 15, dtype=torch.long, device=device)
        with torch.inference_mode():
            # We'd need to re-run sub-talker. Since sub-talker is already called inside generate per step,
            # getting codec_1-15 out requires hook or custom loop. Simpler: re-run sub-talker from predicted codec_0 + talker_h.
            # But we don't have talker_h. Approximate: run talker one more time in teacher-forcing with pred_codec_0
            # to get the full hidden trajectory.
            codec_0_emb = base_talker.get_input_embeddings()(pred_codec_0.unsqueeze(0)).to(torch.bfloat16)
            # For codec_1..15 sum, we need them but don't have them yet. Use zeros as placeholder for first pass.
            # Or just use predicted codec_0 with content+f0 and iterate:
            main_emb = codec_0_emb + content_emb + f0_emb_
            inputs_embeds2 = torch.cat([prefill, main_emb], dim=1)
            outer2 = base_talker.model(inputs_embeds=inputs_embeds2,
                                       attention_mask=torch.ones(1, inputs_embeds2.shape[1], device=device, dtype=torch.long))
            hidden2 = outer2.last_hidden_state  # (1, 6+T, D)
            talker_h_codec = hidden2[:, 6:6+T, :]  # (1, T, D)

            sub_base = base_talker.code_predictor.get_base_model() if hasattr(base_talker.code_predictor, 'get_base_model') else base_talker.code_predictor
            code_pred_embs = sub_base.get_input_embeddings()

            # Generate codec_1..15 per frame via sub-talker AR
            for t in range(T):
                codec_so_far = [pred_codec_0[t].item()]
                # Build 16-pos input step by step (teacher-free AR within each frame)
                sub_inputs_step = [talker_h_codec[:, t:t+1, :],
                                   base_talker.get_input_embeddings()(pred_codec_0[t:t+1].unsqueeze(0)).to(torch.bfloat16)]
                for i in range(1, 15):  # predict codec_i for i=1..14 iteratively
                    sub_in = torch.cat(sub_inputs_step, dim=1).to(torch.bfloat16)
                    # Pad to 16 positions (fills unused with zeros so forward works? Or just run sub_base with current length)
                    # Actually forward_finetune expects full 16-token sequence. Use shortcut: full seq with GT-like padding
                    # Better: use sub-talker's own generate() method
                    break  # Abort this custom loop; use sub_base.generate()
                # Use sub-talker generate starting from talker_hidden
                hidden_flat = torch.cat([talker_h_codec[:, t:t+1, :],
                                         base_talker.get_input_embeddings()(pred_codec_0[t:t+1].unsqueeze(0))], dim=1)
                sub_result = sub_base.generate(
                    inputs_embeds=hidden_flat.to(torch.bfloat16),
                    max_new_tokens=config.talker_config.num_code_groups - 1,  # 15
                    min_new_tokens=config.talker_config.num_code_groups - 1,
                    do_sample=False,
                )
                codec_rest_pred[t] = sub_result[0, :15]

        # Stack: pred full codec (T, 16)
        codec_full = torch.cat([pred_codec_0.unsqueeze(-1), codec_rest_pred], dim=-1)

        with torch.inference_mode():
            wav_pred, fs = model.speech_tokenizer.decode([{'audio_codes': codec_full}])
            wav_gt, _ = model.speech_tokenizer.decode([{'audio_codes': target_codes[0]}])

        sf.write(f'output/diagnose_ar/pred_{idx}.wav', wav_pred[0], fs)
        sf.write(f'output/diagnose_ar/gt_{idx}.wav', wav_gt[0], fs)

        text_gt = transcribe(wav_gt[0], fs, lang='zh')
        text_pred = transcribe(wav_pred[0], fs, lang='zh')
        c_gt = hubert.encode(wav_gt[0], fs, target_frames=T)
        c_pred = hubert.encode(wav_pred[0], fs, target_frames=T)
        cos = torch.nn.functional.cosine_similarity(c_gt, c_pred, dim=-1).mean().item()

        codec_0_match = (pred_codec_0 == target_codes[0, :, 0]).float().mean().item()
        codec_rest_match = (codec_rest_pred == target_codes[0, :, 1:16]).float().mean().item()
        print(f'  GT   ASR: {text_gt}')
        print(f'  PRED ASR: {text_pred}')
        print(f'  codec_0 match: {codec_0_match*100:.1f}% | codec_1-15 match: {codec_rest_match*100:.1f}% | HuBERT cos: {cos:.3f}')


if __name__ == '__main__':
    main()
