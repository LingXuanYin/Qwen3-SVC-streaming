# coding=utf-8
"""SVC Mapper training + in-process validation.

Non-autoregressive frame-level codec_0 prediction.
Validates by decoding predicted codec tokens and saving audio.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from svc_dataset_cached import SVCCachedDataset

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSVCConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_mapper import SVCMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--manifest_name", default="manifest.json")
    p.add_argument("--output_dir", default="output/svc_mapper")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--mapper_layers", type=int, default=4)
    p.add_argument("--mapper_heads", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--preload", action="store_true", help="Preload all data into RAM for max GPU util")
    p.add_argument("--validate_samples", type=int, default=3, help="Validate N samples after training")
    return p.parse_args()


def train():
    args = parse_args()
    device = "cuda:0"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model (frozen - only used for embeddings and sub-talker)
    logger.info("Loading base model...")
    qwen = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=device)
    model = qwen.model
    talker = model.talker
    D = talker.config.hidden_size
    V = talker.config.vocab_size

    # Freeze entire base model
    for p in model.parameters():
        p.requires_grad = False

    # Trainable modules
    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    mapper = SVCMapper(hidden_size=D, num_layers=args.mapper_layers, num_heads=args.mapper_heads, vocab_size=V).to(device=device, dtype=torch.float32)

    trainable_params = list(f0_proj.parameters()) + list(mapper.parameters())
    n_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable params: {n_params:,} (mapper + f0_proj)")

    # Dataset
    dataset = SVCCachedDataset(args.preprocessed_dir, args.manifest_name, preload=args.preload)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=SVCCachedDataset.collate_fn, num_workers=args.num_workers, pin_memory=True)

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    mapper.train(); f0_proj.train()

    global_step = 0
    data_iter = iter(dataloader)

    logger.info(f"Training {args.max_steps} steps, bs={args.batch_size}")

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sc = batch["source_codes"].to(device)
        tc = batch["target_codes"].to(device)
        f0 = batch["f0"].to(device)
        mask = batch["mask"].to(device)
        rm = batch["ref_mels"].to(device=device, dtype=torch.float16)
        B, T, Q = sc.shape

        # Speaker embedding (frozen, cast to float32)
        with torch.no_grad():
            spk = model.speaker_encoder(rm).float()  # (B, D)

        # Source codec embeddings (frozen, cast to float32)
        with torch.no_grad():
            src_embed = talker.get_input_embeddings()(sc[:, :, 0]).float()
            for ci in range(1, Q):
                src_embed = src_embed + talker.code_predictor.get_input_embeddings()[ci-1](sc[:, :, ci]).float()

        # F0 embedding (trainable)
        f0_embed = f0_proj(f0)

        # Mapper forward - predicts all 16 codebooks
        all_logits = mapper(src_embed, f0_embed, spk, padding_mask=mask)  # list of (B, T, V)

        # Cross-entropy loss on ALL codebooks
        total_loss = 0
        for cb_idx, logits in enumerate(all_logits):
            cb_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V),
                tc[:, :, cb_idx].reshape(-1),
                reduction='none'
            )
            total_loss = total_loss + (cb_loss.view(B, T) * mask.float()).sum() / mask.float().sum()
        loss = total_loss / len(all_logits)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

        # LR warmup
        lr = args.lr * min(1.0, (global_step + 1) / max(args.warmup_steps, 1))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()

        if (global_step + 1) % args.log_every == 0:
            # Accuracy (average over all codebooks)
            with torch.no_grad():
                total_acc = 0
                for cb_idx, logits_cb in enumerate(all_logits):
                    preds_cb = logits_cb.argmax(dim=-1)
                    total_acc += ((preds_cb == tc[:, :, cb_idx]) * mask).sum().float() / mask.sum()
                acc = total_acc / len(all_logits)
            logger.info(f"Step {global_step+1}: loss={loss.item():.4f} acc={acc.item()*100:.1f}% lr={lr:.2e}")

        if (global_step + 1) % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step+1}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(mapper.state_dict(), os.path.join(save_dir, "mapper.pt"))
            torch.save(f0_proj.state_dict(), os.path.join(save_dir, "f0_projector.pt"))
            logger.info(f"Saved checkpoint at step {global_step+1}")

        global_step += 1

    # Final save
    save_dir = os.path.join(args.output_dir, f"checkpoint-{args.max_steps}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(mapper.state_dict(), os.path.join(save_dir, "mapper.pt"))
    torch.save(f0_proj.state_dict(), os.path.join(save_dir, "f0_projector.pt"))

    # === IN-PROCESS VALIDATION ===
    logger.info("=== Validation (in-process, no save/load) ===")
    mapper.eval(); f0_proj.eval()
    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    # Load first N samples
    for idx in range(min(args.validate_samples, len(dataset))):
        feat = torch.load(dataset.manifest[idx]["path"], weights_only=True)
        codes = feat["source_codes"].to(device)  # (T, Q)
        tgt = feat["target_codes"].to(device)
        f0_val = feat["f0"].to(device)
        mel = feat["ref_mel"][:400].unsqueeze(0).to(device=device, dtype=torch.float16)
        T_val = codes.shape[0]

        with torch.no_grad():
            spk_val = model.speaker_encoder(mel).float()
            src_e = talker.get_input_embeddings()(codes[:, 0:1].unsqueeze(0))[:, :, 0].float()
            for ci in range(1, Q):
                src_e = src_e + talker.code_predictor.get_input_embeddings()[ci-1](codes[:, ci:ci+1].unsqueeze(0))[:, :, 0].float()
            f0_e = f0_proj(f0_val.unsqueeze(0))
            pred_all = mapper.predict(src_e, f0_e, spk_val)  # (1, T, 16)

        # Accuracy (all codebooks)
        acc_val = (pred_all[0, :T_val] == tgt.to(device)).float().mean().item()

        # Decode predicted (all codebooks from mapper)
        with torch.no_grad():
            wavs_pred, fs = model.speech_tokenizer.decode([{"audio_codes": pred_all[0, :T_val]}])
            wavs_gt, _ = model.speech_tokenizer.decode([{"audio_codes": tgt}])

        sf.write(os.path.join(val_dir, f"sample{idx}_predicted.wav"), wavs_pred[0], fs)
        sf.write(os.path.join(val_dir, f"sample{idx}_groundtruth.wav"), wavs_gt[0], fs)
        logger.info(f"Sample {idx}: T={T_val} acc={acc_val*100:.1f}% pred_dur={wavs_pred[0].shape[0]/fs:.2f}s")

    logger.info(f"Validation audio saved to {val_dir}")


if __name__ == "__main__":
    train()
