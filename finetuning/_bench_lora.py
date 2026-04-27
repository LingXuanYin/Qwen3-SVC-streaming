"""Benchmark LoRA training throughput across bs to pick the best one."""
import os, sys, time, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from sft_svc_lora import ContentProjector, F0EmbProjector, build_prefix_embed, SVCDataset


def bench(bs, steps=10, train_sub=True):
    device = 'cuda:0'
    qwen = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    config = model.config
    talker = model.talker
    D = talker.config.hidden_size
    V = talker.config.vocab_size

    for p in model.parameters(): p.requires_grad = False
    model.talker = get_peft_model(talker, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj'], task_type='CAUSAL_LM'))

    if train_sub:
        base_talker = model.talker.get_base_model()
        base_talker.code_predictor = get_peft_model(base_talker.code_predictor, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj'], task_type='CAUSAL_LM'))

    content_proj = ContentProjector(in_dim=768, hidden=D).to(device=device, dtype=torch.float32)
    f0_proj = F0EmbProjector(hidden=D).to(device=device, dtype=torch.float32)
    trainable = list(content_proj.parameters()) + list(f0_proj.parameters()) + \
                [p for p in model.talker.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=1e-4, weight_decay=0.01)

    ds = SVCDataset('L:/DATASET/svc_sing_parallel', 'sing_synth_gts_manifest.json', max_T=90)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=SVCDataset.collate_fn,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    it = iter(loader)
    base_talker = model.talker.get_base_model()

    def one_step():
        nonlocal it
        try: batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        content = batch['content'].to(device, non_blocking=True).to(torch.float32)
        target_codes = batch['target_codes'].to(device, non_blocking=True)
        f0 = batch['f0'].to(device, non_blocking=True)
        spk_embed = batch['spk_embed'].to(device, non_blocking=True).to(torch.float32)
        mask = batch['mask'].to(device, non_blocking=True)
        lengths = batch['lengths'].to(device, non_blocking=True)
        B, T, _ = content.shape

        content_emb = content_proj(content).to(torch.bfloat16)
        f0_emb = f0_proj(f0).to(torch.bfloat16)
        prefix_emb = build_prefix_embed(base_talker, config, spk_embed.to(torch.bfloat16), B)
        codec_0 = target_codes[:, :, 0]
        codec_0_emb = base_talker.get_input_embeddings()(codec_0).to(torch.bfloat16)
        code_pred_embs = (base_talker.code_predictor.get_base_model() if hasattr(base_talker.code_predictor, 'get_base_model') else base_talker.code_predictor).get_input_embeddings()
        codec_rest_sum = 0
        for i in range(1, target_codes.shape[-1]):
            codec_rest_sum = codec_rest_sum + code_pred_embs[i-1](target_codes[:, :, i])
        main_emb = codec_0_emb + codec_rest_sum.to(torch.bfloat16) + content_emb + f0_emb
        inputs_embeds = torch.cat([prefix_emb, main_emb], dim=1)
        labels = torch.cat([torch.full((B, 6), -100, dtype=torch.long, device=device), codec_0.clone()], dim=1)
        for i in range(B): labels[i, 6+lengths[i]:] = -100
        attn = torch.cat([torch.ones(B, 6, device=device, dtype=torch.long), mask.to(torch.long)], dim=1)

        outer = base_talker.model(inputs_embeds=inputs_embeds, attention_mask=attn)
        hidden = outer.last_hidden_state
        logits_t = base_talker.codec_head(hidden)
        shifted_logits = logits_t[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss_t = torch.nn.functional.cross_entropy(shifted_logits.reshape(-1, V), shifted_labels.reshape(-1), ignore_index=-100)

        loss_s = 0.0
        if train_sub:
            talker_h_codec = hidden[:, 6:6+T, :]
            Bf = B * T
            tk_h = talker_h_codec.reshape(Bf, 1, D)
            c0 = base_talker.get_input_embeddings()(codec_0).reshape(Bf, 1, D)
            sub_base = base_talker.code_predictor.get_base_model() if hasattr(base_talker.code_predictor, 'get_base_model') else base_talker.code_predictor
            cpe = sub_base.get_input_embeddings()
            c1_14 = [cpe[i-1](target_codes[:, :, i]).reshape(Bf, 1, D) for i in range(1, 15)]
            sub_inp = torch.cat([tk_h, c0] + c1_14, dim=1).to(torch.bfloat16)
            sub_lbl = target_codes[:, :, 1:16].reshape(Bf, 15).clone()
            sub_lbl[~mask.reshape(Bf)] = -100
            sub_out = sub_base.forward_finetune(inputs_embeds=sub_inp)
            V_s = sub_out.logits.shape[-1]
            loss_s = torch.nn.functional.cross_entropy(sub_out.logits.reshape(-1, V_s), sub_lbl.reshape(-1), ignore_index=-100)
        loss = loss_t + 0.15 * loss_s
        opt.zero_grad(); loss.backward(); opt.step()

    # Warmup
    for _ in range(3): one_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps): one_step()
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f'bs={bs} train_sub={train_sub}: {el/steps*1000:.0f} ms/step ({steps/el:.2f} steps/s, samples/s={steps*bs/el:.1f}) peak_mem={mem:.1f}GB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--steps', type=int, default=10); ap.add_argument('--no_sub', action='store_true')
    args = ap.parse_args()
    bench(args.bs, args.steps, not args.no_sub)
