# coding=utf-8
"""SVC training via LoRA on Qwen3-TTS AR talker (RULES.md compliant).

Architecture:
  prefix(6) + [codec_0_emb(t) + content_proj(t) + f0_emb(t)]_t=0..T-1

  Prefix (from build_svc_prefill, unchanged):
    text track:  [pad, pad, pad, pad, pad, tts_bos]
    codec track: [nothink, think_bos, think_eos, spk_emb, codec_pad, codec_bos]

  At each main position t:
    input = (codec_0[t] talker embed) + content_projected[t] + f0_projected[t]
  Teacher-forcing predict codec_0[t+1] via CausalLM shift.

Sub-talker training is in a separate pass using the talker's last hidden states
plus codec_0/1..14 teacher-forcing to predict codec_1..15.

Only LoRA weights + projectors are trainable. Base talker/sub-talker frozen.
"""
import argparse, json, logging, math, os, sys
import numpy as np
import torch
import soundfile as sf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVCDataset(Dataset):
    def __init__(self, preprocessed_dir, manifest_name='manifest.json', max_T=90):
        with open(os.path.join(preprocessed_dir, manifest_name)) as f:
            self.manifest = [x for x in json.load(f) if x['T'] <= max_T]
        logger.info(f'SVCDataset: {len(self.manifest)} samples (max_T={max_T})')

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        return torch.load(self.manifest[idx]['path'], weights_only=True)

    @staticmethod
    def collate_fn(batch):
        content = pad_sequence([b['content'] for b in batch], batch_first=True)
        target_codes = pad_sequence([b['target_codes'] for b in batch], batch_first=True)
        f0 = pad_sequence([b['f0'] for b in batch], batch_first=True)
        spk_embed = torch.stack([b['spk_embed'] for b in batch])
        lengths = torch.tensor([b['content'].shape[0] for b in batch], dtype=torch.long)
        max_T = int(lengths.max().item())
        mask = torch.arange(max_T)[None, :] < lengths[:, None]
        return dict(content=content, target_codes=target_codes, f0=f0,
                    spk_embed=spk_embed, mask=mask, lengths=lengths)


class ContentProjector(torch.nn.Module):
    """HuBERT content (768d) → talker hidden (2048d). 2-layer MLP."""
    def __init__(self, in_dim=768, hidden=2048):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)


class F0EmbProjector(torch.nn.Module):
    """F0 (Hz, float) → talker hidden (2048d). Log-scale + learned unvoiced embed."""
    def __init__(self, hidden=2048):
        super().__init__()
        self.proj = torch.nn.Linear(1, hidden)
        self.unvoiced = torch.nn.Parameter(torch.zeros(hidden))
        torch.nn.init.normal_(self.unvoiced, std=0.02)

    def forward(self, f0):
        voiced = f0 > 0
        log_f0 = torch.log1p(f0.float()).unsqueeze(-1).to(self.proj.weight.dtype)
        emb = self.proj(log_f0)
        emb = torch.where(voiced.unsqueeze(-1), emb, self.unvoiced.to(emb.dtype))
        return emb


def build_prefix_embed(base_talker, config, spk_embed, B):
    """Return prefix_combined (B, 6, D). spk_embed: (B, D)."""
    device = next(base_talker.parameters()).device
    dtype = next(base_talker.parameters()).dtype

    tts_ids = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device, dtype=torch.long,
    )
    tts_bos, tts_eos, tts_pad = base_talker.text_projection(
        base_talker.get_text_embeddings()(tts_ids)
    ).chunk(3, dim=1)  # each (1, 1, D)

    codec_prefix_ids = torch.tensor([[
        config.talker_config.codec_nothink_id,
        config.talker_config.codec_think_bos_id,
        config.talker_config.codec_think_eos_id,
    ]], device=device, dtype=torch.long)
    codec_prefix_embed = base_talker.get_input_embeddings()(codec_prefix_ids)  # (1, 3, D)
    codec_pad_bos_ids = torch.tensor([[
        config.talker_config.codec_pad_id,
        config.talker_config.codec_bos_id,
    ]], device=device, dtype=torch.long)
    codec_pad_bos = base_talker.get_input_embeddings()(codec_pad_bos_ids)  # (1, 2, D)

    spk = spk_embed.view(B, 1, -1).to(device=device, dtype=dtype)  # (B, 1, D)

    # Text track: [pad, pad, pad, pad, pad, tts_bos]  (5 pads + 1 bos = 6)
    prefix_text = torch.cat([tts_pad.expand(1, 5, -1), tts_bos], dim=1).expand(B, -1, -1)
    # Codec track: [nothink, think_bos, think_eos, spk, codec_pad, codec_bos] (3+1+2=6)
    prefix_codec = torch.cat([
        codec_prefix_embed.expand(B, -1, -1),
        spk,
        codec_pad_bos.expand(B, -1, -1),
    ], dim=1)
    return prefix_text + prefix_codec  # (B, 6, D)


def train():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--manifest_name', default='manifest.json')
    p.add_argument('--output_dir', default='output/svc_lora')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--min_lr', type=float, default=5e-6)
    p.add_argument('--max_steps', type=int, default=2000)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=500)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lora_rank', type=int, default=32)
    p.add_argument('--lora_alpha', type=int, default=64)
    p.add_argument('--sub_lora_rank', type=int, default=16)
    p.add_argument('--sub_lora_alpha', type=int, default=32)
    p.add_argument('--sub_weight', type=float, default=0.15, help='Weight for sub-talker loss')
    p.add_argument('--train_sub', action='store_true', default=True, help='Train sub-talker LoRA too')
    p.add_argument('--freeze_talker', action='store_true', help='Freeze talker LoRA; train only sub-talker (Phase 2)')
    p.add_argument('--resume_ckpt', type=str, default=None, help='Load talker_lora + sub_lora + projectors from this ckpt dir')
    p.add_argument('--max_T', type=int, default=90)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tb'))

    logger.info('Loading Qwen3-TTS base model...')
    qwen = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)
    model = qwen.model
    talker = model.talker  # Qwen3TTSTalkerForConditionalGeneration
    config = model.config
    D = talker.config.hidden_size
    V = talker.config.vocab_size
    logger.info(f'Talker hidden={D}, vocab={V}')

    # Freeze base
    for p_ in model.parameters():
        p_.requires_grad = False

    from peft import PeftModel
    talker_resume_dir = os.path.join(args.resume_ckpt, 'talker_lora') if args.resume_ckpt else None
    sub_resume_dir = os.path.join(args.resume_ckpt, 'sub_talker_lora') if args.resume_ckpt else None

    # Attach LoRA to talker — either resume from ckpt or new init
    if talker_resume_dir and os.path.exists(talker_resume_dir):
        logger.info(f'Loading talker LoRA from {talker_resume_dir}')
        model.talker = PeftModel.from_pretrained(talker, talker_resume_dir, is_trainable=True)
    else:
        talker_lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            task_type='CAUSAL_LM',
        )
        model.talker = get_peft_model(talker, talker_lora_cfg)
    model.talker.print_trainable_parameters()

    # Attach LoRA to sub-talker
    if args.train_sub:
        base_talker_for_sub = model.talker.get_base_model()
        if sub_resume_dir and os.path.exists(sub_resume_dir):
            logger.info(f'Loading sub-talker LoRA from {sub_resume_dir}')
            base_talker_for_sub.code_predictor = PeftModel.from_pretrained(
                base_talker_for_sub.code_predictor, sub_resume_dir, is_trainable=True)
        else:
            sub_lora_cfg = LoraConfig(
                r=args.sub_lora_rank, lora_alpha=args.sub_lora_alpha, lora_dropout=0.05,
                target_modules=['q_proj', 'k_proj', 'v_proj'],
                task_type='CAUSAL_LM',
            )
            base_talker_for_sub.code_predictor = get_peft_model(base_talker_for_sub.code_predictor, sub_lora_cfg)
        base_talker_for_sub.code_predictor.print_trainable_parameters()

    # Trainable projectors
    device = args.device
    content_projector = ContentProjector(in_dim=768, hidden=D).to(device=device, dtype=torch.float32)
    f0_projector = F0EmbProjector(hidden=D).to(device=device, dtype=torch.float32)
    trainable = list(content_projector.parameters()) + list(f0_projector.parameters()) + \
                [p_ for p_ in model.talker.parameters() if p_.requires_grad]
    # code_predictor is under model.talker.base_model.model after peft wrap of talker, and then we wrapped it too.
    # Its LoRA params' `.requires_grad` is True automatically via PeftModel. They're already included via model.talker.parameters().
    n_trainable = sum(p_.numel() for p_ in trainable)
    logger.info(f'Total trainable params (LoRA + projectors): {n_trainable:,}')

    # Resume projectors (talker and sub-talker LoRA already resumed above via PeftModel.from_pretrained)
    if args.resume_ckpt:
        content_pth = os.path.join(args.resume_ckpt, 'content_projector.pt')
        if os.path.exists(content_pth):
            content_projector.load_state_dict(torch.load(content_pth, map_location=device, weights_only=True))
            logger.info(f'Loaded content_projector from {content_pth}')
        f0_pth = os.path.join(args.resume_ckpt, 'f0_projector.pt')
        if os.path.exists(f0_pth):
            f0_projector.load_state_dict(torch.load(f0_pth, map_location=device, weights_only=True))
            logger.info(f'Loaded f0_projector from {f0_pth}')

    # Freeze talker LoRA if requested (Phase 2: sub-talker only)
    if args.freeze_talker:
        for n, p_ in model.talker.named_parameters():
            if 'lora' in n and 'code_predictor' not in n:
                p_.requires_grad = False
        logger.info('Phase 2: talker LoRA frozen, training sub-talker LoRA + projectors only')
        # Re-collect trainable
        trainable = list(content_projector.parameters()) + list(f0_projector.parameters()) + \
                    [p_ for p_ in model.talker.parameters() if p_.requires_grad]
        n_trainable = sum(p_.numel() for p_ in trainable)
        logger.info(f'Phase-2 trainable params: {n_trainable:,}')
        opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)

    dataset = SVCDataset(args.preprocessed_dir, args.manifest_name, max_T=args.max_T)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=SVCDataset.collate_fn, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=(args.num_workers > 0))

    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.talker.train()
    content_projector.train(); f0_projector.train()

    base_talker = model.talker.get_base_model() if hasattr(model.talker, 'get_base_model') else model.talker
    logger.info(f'Base talker type: {type(base_talker).__name__}')

    global_step = 0
    data_iter = iter(loader)

    while global_step < args.max_steps:
        try: batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)

        content = batch['content'].to(device, non_blocking=True).to(torch.float32)  # (B, T, 768)
        target_codes = batch['target_codes'].to(device, non_blocking=True)           # (B, T, 16)
        f0 = batch['f0'].to(device, non_blocking=True)                                # (B, T)
        spk_embed = batch['spk_embed'].to(device, non_blocking=True).to(torch.float32)  # (B, D)
        mask = batch['mask'].to(device, non_blocking=True)                            # (B, T)
        lengths = batch['lengths'].to(device, non_blocking=True)

        B, T, _ = content.shape

        # Project to talker hidden (bf16)
        content_emb = content_projector(content).to(torch.bfloat16)  # (B, T, D)
        f0_emb = f0_projector(f0).to(torch.bfloat16)                  # (B, T, D)

        # Build prefix (6 tokens, bf16)
        prefix_emb = build_prefix_embed(base_talker, config, spk_embed.to(torch.bfloat16), B)  # (B, 6, D)

        # Codec teacher forcing: sum all 16 codebook embeds per frame (matches inference).
        # code_predictor's codec_embedding is nn.Embedding(V, 2048) using talker_config.hidden_size
        # (modeling line 1165), so all embeds are in the same 2048 dim and can be summed.
        codec_0 = target_codes[:, :, 0]  # (B, T)
        codec_0_emb = base_talker.get_input_embeddings()(codec_0).to(torch.bfloat16)
        code_pred_embs = base_talker.code_predictor.get_input_embeddings()  # ModuleList of 15 Embedding(V, 2048)
        codec_rest_sum = 0
        for i in range(1, target_codes.shape[-1]):  # i = 1..15
            emb_i = code_pred_embs[i - 1](target_codes[:, :, i])  # (B, T, 2048)
            codec_rest_sum = codec_rest_sum + emb_i
        codec_sum_emb = (codec_0_emb + codec_rest_sum.to(torch.bfloat16))
        main_emb = codec_sum_emb + content_emb + f0_emb  # (B, T, D)

        inputs_embeds = torch.cat([prefix_emb, main_emb], dim=1)  # (B, 6+T, D)

        # Labels: [-100]*6 for prefix, then codec_0 (HF shifts logits[i] to predict labels[i+1])
        labels_prefix = torch.full((B, 6), -100, dtype=torch.long, device=device)
        # codec_0 labels with -100 on padding positions (beyond actual length)
        labels_codec = codec_0.clone()
        for i in range(B):
            labels_codec[i, lengths[i]:] = -100
        labels = torch.cat([labels_prefix, labels_codec], dim=1)  # (B, 6+T)

        # Attention mask: 1 for prefix, and mask for main
        attn_prefix = torch.ones(B, 6, device=device, dtype=torch.long)
        attn_main = mask.to(torch.long)
        attention_mask = torch.cat([attn_prefix, attn_main], dim=1)  # (B, 6+T)

        # Call inner decoder directly (LoRA on self_attn still applies) to get last_hidden_state.
        # Qwen3TTSTalkerOutputWithPast doesn't expose last_hidden_state, so we compute manually.
        outer = base_talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden = outer.last_hidden_state  # (B, 6+T, D)
        logits_talker = base_talker.codec_head(hidden)  # (B, 6+T, V)
        # Shift-by-1 causal LM loss (standard HF convention)
        shifted_logits = logits_talker[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss_talker = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, V), shifted_labels.reshape(-1), ignore_index=-100
        )

        loss_sub = torch.tensor(0.0, device=device)
        sub_out = None
        if args.train_sub:
            talker_h_at_codec = hidden[:, 6:6+T, :]  # (B, T, D)
            B_flat = B * T
            talker_h_flat = talker_h_at_codec.reshape(B_flat, 1, D)
            codec_0_emb_flat = base_talker.get_input_embeddings()(codec_0).reshape(B_flat, 1, D)
            # code_predictor may be PeftModel-wrapped; get its base to access its input embeddings ModuleList
            sub_base = base_talker.code_predictor.get_base_model() \
                if hasattr(base_talker.code_predictor, 'get_base_model') else base_talker.code_predictor
            code_pred_embs = sub_base.get_input_embeddings()  # ModuleList of 15 Embedding(V, D)
            codec_1_14_embs = []
            for i in range(1, 15):
                e = code_pred_embs[i - 1](target_codes[:, :, i])
                codec_1_14_embs.append(e.reshape(B_flat, 1, D))
            sub_inputs = torch.cat([talker_h_flat, codec_0_emb_flat] + codec_1_14_embs, dim=1)
            sub_inputs = sub_inputs.to(torch.bfloat16)
            sub_labels = target_codes[:, :, 1:16].reshape(B_flat, 15)
            valid_frame = mask.reshape(B_flat)
            sub_labels_masked = sub_labels.clone()
            sub_labels_masked[~valid_frame] = -100

            sub_out = sub_base.forward_finetune(inputs_embeds=sub_inputs)  # no labels → just logits
            # sub_out.logits: (B*T, 15, V). Each position i predicts codec_{i+1}[t].
            # No shift needed (not standard AR — it's MTP with 15 separate heads).
            V_sub = sub_out.logits.shape[-1]
            loss_sub = torch.nn.functional.cross_entropy(
                sub_out.logits.reshape(-1, V_sub),
                sub_labels_masked.reshape(-1),
                ignore_index=-100,
            )

        loss = loss_talker + args.sub_weight * loss_sub

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        # lr schedule
        if global_step < args.warmup_steps:
            lr = args.lr * (global_step + 1) / max(args.warmup_steps, 1)
        else:
            progress = (global_step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups: pg['lr'] = lr
        opt.step()

        if (global_step + 1) % args.log_every == 0:
            with torch.no_grad():
                # codec_0 accuracy: logits[5..5+T-1] predict labels[6..5+T]=codec_0[0..T-1]
                pred = logits_talker[:, 5:5+T, :].argmax(-1)  # (B, T)
                gt = labels_codec
                valid = mask
                acc_0 = ((pred == gt) & valid).float().sum() / valid.float().sum()
                acc_sub = 0.0
                if args.train_sub and sub_out is not None and sub_out.logits.numel() > 0:
                    sub_pred = sub_out.logits.argmax(-1)
                    sub_gt = sub_labels
                    sub_valid = valid_frame.unsqueeze(-1).expand_as(sub_gt)
                    acc_sub = (((sub_pred == sub_gt) & sub_valid).float().sum() / sub_valid.float().sum()).item()
            logger.info(f'Step {global_step+1}: total={loss.item():.3f} talker={loss_talker.item():.3f} '
                        f'sub={loss_sub.item():.3f} acc0={acc_0*100:.1f}% acc_sub={acc_sub*100:.1f}% lr={lr:.2e}')
            writer.add_scalar('loss/talker', loss_talker.item(), global_step + 1)
            writer.add_scalar('loss/sub', loss_sub.item(), global_step + 1)
            writer.add_scalar('loss/total', loss.item(), global_step + 1)
            writer.add_scalar('acc/codec_0', acc_0.item(), global_step + 1)
            writer.add_scalar('acc/sub', acc_sub, global_step + 1)
            writer.add_scalar('lr', lr, global_step + 1)

        if (global_step + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f'checkpoint-{global_step+1}')
            save_all(save_dir, model, content_projector, f0_projector, args.train_sub)
            logger.info(f'Saved checkpoint at step {global_step+1}')

        global_step += 1

    save_dir = os.path.join(args.output_dir, f'checkpoint-{args.max_steps}')
    save_all(save_dir, model, content_projector, f0_projector, args.train_sub)
    logger.info('Training done.')


def save_all(save_dir, model, content_projector, f0_projector, train_sub):
    os.makedirs(save_dir, exist_ok=True)
    model.talker.save_pretrained(os.path.join(save_dir, 'talker_lora'))
    if train_sub:
        base_talker = model.talker.get_base_model()
        base_talker.code_predictor.save_pretrained(os.path.join(save_dir, 'sub_talker_lora'))
    torch.save(content_projector.state_dict(), os.path.join(save_dir, 'content_projector.pt'))
    torch.save(f0_projector.state_dict(), os.path.join(save_dir, 'f0_projector.pt'))


if __name__ == '__main__':
    train()
