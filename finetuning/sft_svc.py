# coding=utf-8
"""SVC LoRA fine-tuning script for Qwen3-TTS → SVC conversion.

Usage:
    python finetuning/sft_svc.py \
        --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --train_jsonl data/svc_train.jsonl \
        --output_dir output/svc_lora
"""

import argparse
import json
import logging
import os

import torch
from peft import PeftModel
from svc_dataset import SVCDataset
from svc_dataset_cached import SVCCachedDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_adapter import apply_svc_lora, save_svc_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SVC LoRA fine-tuning")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="output/svc_lora")
    parser.add_argument("--train_jsonl", type=str, default=None, help="Raw JSONL (will preprocess on-the-fly)")
    parser.add_argument("--preprocessed_dir", type=str, default=None, help="Preprocessed .pt dir (fast, preferred)")
    parser.add_argument("--manifest_name", type=str, default="manifest.json", help="Manifest filename in preprocessed_dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--sub_talker_lora_rank", type=int, default=16)
    parser.add_argument("--sub_talker_loss_weight", type=float, default=0.3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    return parser.parse_args()


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    return base_lr


def train():
    args = parse_args()
    device = "cuda:0"

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, "runs"))

    # Load base model
    logger.info(f"Loading base model from {args.init_model_path}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = qwen3tts.model

    # SVC config
    svc_config = Qwen3TTSSVCConfig(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        sub_talker_lora_rank=args.sub_talker_lora_rank,
        sub_talker_loss_weight=args.sub_talker_loss_weight,
    )

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    f0_projector, content_projector, trainable_count = apply_svc_lora(model, svc_config)
    f0_projector = f0_projector.to(device=device, dtype=torch.bfloat16)
    content_projector = content_projector.to(device=device, dtype=torch.bfloat16)
    logger.info(f"Trainable parameters: {trainable_count:,}")

    # Gradient checkpointing
    if args.gradient_checkpointing:
        base_talker = model.talker.base_model.model if isinstance(model.talker, PeftModel) else model.talker
        if hasattr(base_talker, 'model') and hasattr(base_talker.model, 'gradient_checkpointing_enable'):
            base_talker.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Dataset
    if args.preprocessed_dir:
        logger.info(f"Loading preprocessed data from {args.preprocessed_dir}")
        dataset = SVCCachedDataset(args.preprocessed_dir, manifest_name=args.manifest_name)
        collate_fn = SVCCachedDataset.collate_fn
    elif args.train_jsonl:
        logger.info(f"Loading raw training data from {args.train_jsonl}")
        with open(args.train_jsonl, encoding='utf-8') as f:
            data_list = [json.loads(line) for line in f]
        dataset = SVCDataset(
            data_list=data_list,
            speech_tokenizer=model.speech_tokenizer,
            config=model.config,
            f0_device=device,
        )
        collate_fn = SVCDataset.collate_fn
    else:
        raise ValueError("Must specify --preprocessed_dir or --train_jsonl")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(f0_projector.parameters())
    trainable_params += list(content_projector.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Training loop
    model.train()
    f0_projector.train()
    content_projector.train()
    global_step = 0
    optimizer.zero_grad()
    data_iter = iter(dataloader)

    logger.info(f"Starting training for {args.max_steps} steps")

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        source_codes = batch["source_codes"].to(device)  # (B, T, Q)
        target_codes = batch["target_codes"].to(device)  # (B, T, Q)
        f0 = batch["f0"].to(device)                      # (B, T)
        mask = batch["mask"].to(device)                   # (B, T)
        ref_mels = batch["ref_mels"].to(device=device, dtype=torch.bfloat16)

        B, T, Q = source_codes.shape

        # Extract speaker embedding (frozen)
        with torch.no_grad():
            speaker_embedding = model.speaker_encoder(ref_mels)  # (B, D)

        # F0 → embedding
        f0_embed = f0_projector(f0)  # (B, T, D)

        base_talker = model.talker.base_model.model if isinstance(model.talker, PeftModel) else model.talker
        D = base_talker.config.hidden_size

        # === ICL-based SVC training structure ===
        # Matches build_svc_prefill ICL approach:
        # [prefix(6)] [ICL_ref: tts_pad + source_codec(T+1)] [gen: f0 + target_codec(T) + eos(1)]
        #
        # ICL region: model sees source audio codec as context (text=tts_pad, codec=source)
        # Gen region: model generates target (text=f0, codec=teacher-forced target)

        # Special embeddings
        tts_pad_embed = base_talker.text_projection(base_talker.get_text_embeddings()(
            torch.tensor([[model.config.tts_pad_token_id]], device=device, dtype=torch.long)))
        tts_bos_embed = base_talker.text_projection(base_talker.get_text_embeddings()(
            torch.tensor([[model.config.tts_bos_token_id]], device=device, dtype=torch.long)))
        tts_eos_embed = base_talker.text_projection(base_talker.get_text_embeddings()(
            torch.tensor([[model.config.tts_eos_token_id]], device=device, dtype=torch.long)))
        codec_bos_embed = base_talker.get_input_embeddings()(
            torch.tensor([[model.config.talker_config.codec_bos_id]], device=device, dtype=torch.long))

        # Prefix (6 tokens)
        prefix_codec_ids = torch.tensor([[
            model.config.talker_config.codec_nothink_id,
            model.config.talker_config.codec_think_bos_id,
            model.config.talker_config.codec_think_eos_id,
            model.config.talker_config.codec_pad_id,
            model.config.talker_config.codec_pad_id,
            model.config.talker_config.codec_bos_id,
        ]], device=device, dtype=torch.long).expand(B, -1)
        prefix_codec = base_talker.get_input_embeddings()(prefix_codec_ids)
        prefix_codec[:, 3, :] = speaker_embedding
        prefix_text = torch.cat([tts_pad_embed.expand(B, 5, -1), tts_bos_embed.expand(B, 1, -1)], dim=1)

        # ICL reference: sum all 16 codebook embeddings of source audio
        src_all_embed = base_talker.get_input_embeddings()(source_codes[:, :, 0])
        for ci in range(1, Q):
            ci_emb = base_talker.code_predictor.get_input_embeddings()[ci-1](source_codes[:, :, ci])
            if isinstance(ci_emb, tuple): ci_emb = ci_emb[0]
            src_all_embed = src_all_embed + ci_emb
        # Prepend codec_bos to ICL ref
        icl_codec = torch.cat([codec_bos_embed.expand(B, -1, -1), src_all_embed], dim=1)  # (B, T+1, D)
        icl_text = tts_pad_embed.expand(B, T + 1, -1)

        # F0 embedding (text track for generation region)
        f0_text = f0_embed  # (B, T, D)
        if content_projector is not None:
            f0_text = content_projector(f0_text)

        # Generation region: f0 + target codec (teacher-forced, all 16 codebooks)
        tgt_all_embed = base_talker.get_input_embeddings()(target_codes[:, :, 0])
        for ci in range(1, Q):
            ci_emb = base_talker.code_predictor.get_input_embeddings()[ci-1](target_codes[:, :, ci])
            if isinstance(ci_emb, tuple): ci_emb = ci_emb[0]
            tgt_all_embed = tgt_all_embed + ci_emb

        # Gen start: f0[0] + codec_bos
        gen_start = f0_text[:, :1] + codec_bos_embed.expand(B, -1, -1)
        # Gen body: f0[1:T] + target[0:T-1] (shifted)
        if T > 1:
            gen_body = f0_text[:, 1:T] + tgt_all_embed[:, :T-1]
        # Gen EOS: tts_eos + target[T-1]
        gen_eos = tts_eos_embed.expand(B, -1, -1) + tgt_all_embed[:, T-1:T]

        # === Full sequence: prefix(6) + ICL(T+1) + gen_start(1) + gen_body(T-1) + gen_eos(1) ===
        # Total S = 6 + (T+1) + 1 + (T-1) + 1 = 2T + 8
        S = 6 + (T + 1) + 1 + max(T - 1, 0) + 1
        full_embeds = torch.zeros(B, S, D, device=device, dtype=prefix_codec.dtype)
        p = 0
        full_embeds[:, p:p+6] = prefix_text + prefix_codec; p += 6
        full_embeds[:, p:p+T+1] = icl_text + icl_codec; p += T + 1
        full_embeds[:, p:p+1] = gen_start; p += 1
        if T > 1:
            full_embeds[:, p:p+T-1] = gen_body; p += T - 1
        full_embeds[:, p:p+1] = gen_eos; p += 1

        # Labels: only in generation region (predict target codec_0 + EOS)
        gen_start_pos = 6 + (T + 1)  # position of gen_start
        full_labels = torch.full((B, S), -100, device=device, dtype=torch.long)
        for i in range(B):
            seq_len = mask[i].sum().int().item()
            full_labels[i, gen_start_pos:gen_start_pos+seq_len] = target_codes[i, :seq_len, 0]
            full_labels[i, gen_start_pos+seq_len] = model.config.talker_config.codec_eos_token_id

        # Attention mask
        full_attention = torch.zeros(B, S, device=device, dtype=torch.long)
        for i in range(B):
            seq_len = mask[i].sum().int().item()
            full_attention[i, :gen_start_pos+seq_len+1] = 1

        # Forward: main talker loss
        outputs = model.talker(
            inputs_embeds=full_embeds[:, :-1],
            attention_mask=full_attention[:, :-1],
            labels=full_labels[:, 1:],
            output_hidden_states=True,
        )
        main_loss = outputs.loss

        # Sub-talker loss
        # hidden_states is a tuple of per-layer tensors: ((layer0, layer1, ..., layerN),)
        # or for PeftModel: tuple of tuples. We need the last layer output.
        hs = outputs.hidden_states
        if isinstance(hs, tuple) and isinstance(hs[0], tuple):
            hidden_states = hs[0][-1]  # nested: first element is tuple of layers
        elif isinstance(hs, tuple):
            hidden_states = hs[-1]  # flat tuple: last layer
        else:
            hidden_states = hs

        # hidden_states: (B, seq_len-1, D)
        if global_step == 0:
            logger.info(f"hidden_states shape={hidden_states.shape}, full_embeds: (B={B}, S={S}={full_embeds.shape[1]})")
        # Extract hidden states for generation region only (exclude ICL context and EOS)
        content_mask = torch.zeros(B, S - 1, dtype=torch.bool, device=device)
        for i in range(B):
            seq_len = mask[i].sum().int().item()
            content_mask[i, gen_start_pos:gen_start_pos+seq_len] = True

        talker_hidden = hidden_states[content_mask]  # (N, D)
        # Corresponding target codec ids (all Q codebooks)
        codec_labels = []
        for i in range(B):
            seq_len = mask[i].sum().int().item()
            codec_labels.append(target_codes[i, :seq_len])
        codec_labels = torch.cat(codec_labels, dim=0)  # (N, Q)

        if global_step == 0:
            logger.info(f"talker_hidden: {talker_hidden.shape}, codec_labels: {codec_labels.shape}, content_mask sum: {content_mask.sum()}")

        # Truncate to match (can differ by 1 due to label shift)
        min_n = min(talker_hidden.shape[0], codec_labels.shape[0])
        talker_hidden = talker_hidden[:min_n]
        codec_labels = codec_labels[:min_n]

        if min_n > 0:
            sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(
                codec_labels, talker_hidden
            )
        else:
            sub_loss = torch.tensor(0.0, device=device)

        # Combined loss
        total_loss = main_loss + svc_config.sub_talker_loss_weight * sub_loss

        # Backward
        scaled_loss = total_loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            # LR schedule
            lr = get_lr(global_step // args.gradient_accumulation_steps, args.warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if (global_step + 1) % (args.log_every * args.gradient_accumulation_steps) == 0:
                effective_step = (global_step + 1) // args.gradient_accumulation_steps
                writer.add_scalar("loss/total", total_loss.item(), effective_step)
                writer.add_scalar("loss/main", main_loss.item(), effective_step)
                writer.add_scalar("loss/sub_talker", sub_loss.item(), effective_step)
                writer.add_scalar("lr", lr, effective_step)
                writer.add_scalar("grad_norm", grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, effective_step)
                logger.info(
                    f"Step {effective_step} | loss={total_loss.item():.4f} "
                    f"main={main_loss.item():.4f} sub={sub_loss.item():.4f} lr={lr:.2e}"
                )

        # Save checkpoint
        if (global_step + 1) % (args.save_every * args.gradient_accumulation_steps) == 0:
            effective_step = (global_step + 1) // args.gradient_accumulation_steps
            logger.info(f"Saving checkpoint at step {effective_step}")
            save_svc_checkpoint(
                model=model,
                f0_projector=f0_projector,
                svc_config=svc_config,
                output_dir=args.output_dir,
                step=effective_step,
                content_projector=content_projector,
            )

        global_step += 1

    # Final save
    logger.info("Training complete. Saving final checkpoint.")
    save_svc_checkpoint(
        model=model,
        f0_projector=f0_projector,
        svc_config=svc_config,
        output_dir=args.output_dir,
        step=args.max_steps // args.gradient_accumulation_steps,
    )
    writer.close()


if __name__ == "__main__":
    train()
