# coding=utf-8
"""In-process grid search for SVC Mapper. No subprocess."""
import gc
import json
import os
import sys
from itertools import product

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

GRID = {
    "lr": [1e-4, 3e-4, 5e-4],
    "mapper_layers": [2, 4],
}
BATCH_SIZE = 128
MAX_STEPS = 200
WARMUP = 30
OUTPUT_BASE = "L:/DATASET/svc_output/mapper_grid"


def run_one(model, talker, dataset, collate_fn, lr, layers, device):
    from qwen_tts.svc.f0_projector import F0Projector
    from qwen_tts.svc.svc_mapper import SVCMapper
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    D = talker.config.hidden_size
    V = talker.config.vocab_size
    Q = 16

    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    mapper = SVCMapper(hidden_size=D, num_layers=layers, num_heads=8, vocab_size=V).to(device=device, dtype=torch.float32)

    params = list(f0_proj.parameters()) + list(mapper.parameters())
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    mapper.train(); f0_proj.train()

    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    data_iter = iter(dl)

    last_loss = last_acc = 0
    for step in range(MAX_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        sc = batch["source_codes"].to(device)
        tc = batch["target_codes"].to(device)
        f0 = batch["f0"].to(device)
        mask = batch["mask"].to(device)
        rm = batch["ref_mels"].to(device=device, dtype=torch.float16)
        B, T, _ = sc.shape

        with torch.no_grad():
            spk = model.speaker_encoder(rm).float()
            src_e = talker.get_input_embeddings()(sc[:, :, 0]).float()
            for ci in range(1, Q):
                src_e = src_e + talker.code_predictor.get_input_embeddings()[ci - 1](sc[:, :, ci]).float()

        f0e = f0_proj(f0)
        logits = mapper(src_e, f0e, spk, mask)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, V), tc[:, :, 0].reshape(-1), reduction="none")
        loss = (loss.view(B, T) * mask.float()).sum() / mask.float().sum()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        cur_lr = lr * min(1.0, (step + 1) / max(WARMUP, 1))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        opt.step()

        last_loss = loss.item()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            last_acc = ((preds == tc[:, :, 0]) * mask).sum().float().item() / mask.sum().item() * 100

    # Cleanup
    del mapper, f0_proj, opt, params
    gc.collect()
    torch.cuda.empty_cache()

    return last_loss, last_acc


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    device = "cuda:0"

    # Load model once
    print("Loading model...", flush=True)
    from svc_dataset_cached import SVCCachedDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch_dtype=torch.float16, device_map=device)
    model = m.model
    talker = model.talker
    for p in model.parameters():
        p.requires_grad = False

    ds = SVCCachedDataset("L:/DATASET/svc_preprocessed_v2", "manifest_p80.json")
    collate_fn = SVCCachedDataset.collate_fn

    configs = list(product(GRID["lr"], GRID["mapper_layers"]))
    print(f"Grid: {len(configs)} configs, bs={BATCH_SIZE}, steps={MAX_STEPS}", flush=True)

    results = []
    for i, (lr, layers) in enumerate(configs):
        name = f"lr{lr:.0e}_L{layers}"
        print(f"[{i+1}/{len(configs)}] {name}...", end=" ", flush=True)
        try:
            loss, acc = run_one(model, talker, ds, collate_fn, lr, layers, device)
            print(f"loss={loss:.4f} acc={acc:.1f}%", flush=True)
            results.append({"lr": lr, "layers": layers, "loss": loss, "acc": acc})
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            results.append({"lr": lr, "layers": layers, "loss": 999, "acc": 0})
            torch.cuda.empty_cache()

    print("\n" + "=" * 60, flush=True)
    print(f"{'lr':>8} {'layers':>6} | {'loss':>8} {'acc':>8}", flush=True)
    print("-" * 40, flush=True)
    for r in sorted(results, key=lambda x: x["loss"]):
        print(f"{r['lr']:>8.0e} {r['layers']:>6} | {r['loss']:>8.4f} {r['acc']:>7.1f}%", flush=True)

    best = sorted(results, key=lambda x: x["loss"])[0]
    print(f"\nBest: lr={best['lr']:.0e} layers={best['layers']} loss={best['loss']:.4f} acc={best['acc']:.1f}%", flush=True)

    with open(os.path.join(OUTPUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
