# coding=utf-8
"""Grid search for SVC HuBERT mapper hyperparameters on PG199."""
import gc, json, os, sys, random
from itertools import product
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sft_svc_hubert import HubertSVCDataset

GRID = {
    "lr": [1e-4, 3e-4, 5e-4],
    "mapper_layers": [2, 4],
    "adv_weight": [0.05, 0.1],
}
BATCH_SIZE = 384  # safe margin under 512 max
MAX_STEPS = 300
WARMUP = 30
OUTPUT_BASE = "L:/DATASET/svc_output/hp_hubert"


def run_one(model, dataset, lr, layers, adv_weight, device):
    from qwen_tts.svc.f0_projector import F0Projector
    from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert

    talker = model.talker
    V = talker.config.vocab_size
    D = talker.config.hidden_size

    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    mapper = SVCMapperHubert(content_dim=768, cond_dim=D, hidden_size=1024,
                              num_layers=layers, num_heads=8, vocab_size=V, num_codebooks=16).to(device=device, dtype=torch.float32)

    params = list(f0_proj.parameters()) + list(mapper.parameters())
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    mapper.train(); f0_proj.train()

    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=HubertSVCDataset.collate_fn, num_workers=0)
    data_iter = iter(dl)

    last_codec_loss = last_acc = 0
    for step in range(MAX_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl); batch = next(data_iter)

        content = batch["content"].to(device)
        tc = batch["target_codes"].to(device)
        f0 = batch["f0"].to(device)
        mask = batch["mask"].to(device)
        rm = batch["ref_mels"].to(device=device, dtype=torch.bfloat16)
        B, T, _ = tc.shape

        with torch.no_grad():
            spk = model.speaker_encoder(rm).float()
        f0e = f0_proj(f0)

        adv_lambda = min(1.0, (step + 1) / WARMUP)
        codec_logits, adv_logits = mapper(content, f0e, spk, padding_mask=mask, adv_lambda=adv_lambda)

        codec_loss = 0
        for ci, lg in enumerate(codec_logits):
            cb = torch.nn.functional.cross_entropy(lg.view(-1, V), tc[:,:,ci].reshape(-1), reduction='none')
            codec_loss = codec_loss + (cb.view(B, T) * mask.float()).sum() / mask.float().sum()
        codec_loss = codec_loss / 16

        f0_bins = mapper.f0_to_bin(f0, n_bins=mapper.f0_num_bins)
        adv_loss = torch.nn.functional.cross_entropy(
            adv_logits.view(-1, mapper.f0_num_bins), f0_bins.view(-1), reduction='none'
        )
        adv_loss = (adv_loss.view(B, T) * mask.float()).sum() / mask.float().sum()

        loss = codec_loss + adv_weight * adv_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        cur_lr = lr * min(1.0, (step + 1) / WARMUP)
        for pg in opt.param_groups: pg["lr"] = cur_lr
        opt.step()

        last_codec_loss = codec_loss.item()
        with torch.no_grad():
            total_acc = 0
            for ci, lg in enumerate(codec_logits):
                total_acc += ((lg.argmax(-1) == tc[:,:,ci]) * mask).sum().float() / mask.sum()
            last_acc = (total_acc / 16).item() * 100

    del mapper, f0_proj, opt, params
    gc.collect(); torch.cuda.empty_cache()
    return last_codec_loss, last_acc


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    device = "cuda:0"
    print("Loading model...", flush=True)
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch_dtype=torch.bfloat16, device_map=device)
    model = m.model
    for p in model.parameters(): p.requires_grad = False

    ds = HubertSVCDataset("L:/DATASET/svc_hubert_v3")
    configs = list(product(GRID["lr"], GRID["mapper_layers"], GRID["adv_weight"]))
    print(f"Grid: {len(configs)} configs, bs={BATCH_SIZE}, steps={MAX_STEPS}", flush=True)

    results = []
    for i, (lr, layers, adv_w) in enumerate(configs):
        name = f"lr{lr:.0e}_L{layers}_adv{adv_w}"
        print(f"[{i+1}/{len(configs)}] {name}...", end=" ", flush=True)
        try:
            loss, acc = run_one(model, ds, lr, layers, adv_w, device)
            print(f"codec_loss={loss:.4f} acc={acc:.1f}%", flush=True)
            results.append({"lr": lr, "layers": layers, "adv_weight": adv_w, "loss": loss, "acc": acc})
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            results.append({"lr": lr, "layers": layers, "adv_weight": adv_w, "loss": 999, "acc": 0})
            torch.cuda.empty_cache()

    print("\n" + "=" * 70, flush=True)
    print(f"{'lr':>8} {'L':>3} {'adv':>5} | {'loss':>8} {'acc':>7}", flush=True)
    print("-" * 50, flush=True)
    for r in sorted(results, key=lambda x: x["loss"]):
        print(f"{r['lr']:>8.0e} {r['layers']:>3} {r['adv_weight']:>5} | {r['loss']:>8.4f} {r['acc']:>6.1f}%", flush=True)

    best = sorted(results, key=lambda x: x["loss"])[0]
    print(f"\nBest: lr={best['lr']:.0e} layers={best['layers']} adv={best['adv_weight']} loss={best['loss']:.4f}", flush=True)

    with open(os.path.join(OUTPUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
