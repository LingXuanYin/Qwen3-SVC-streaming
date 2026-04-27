# coding=utf-8
"""Grid search for SVC training hyperparameters."""

import itertools
import json
import os
import subprocess
import sys

GRID = {
    "lr": [3e-4, 5e-4, 8e-4],
    "lora_rank": [16, 32],
    "sub_talker_loss_weight": [0.1, 0.3],
}

FIXED = {
    "init_model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "preprocessed_dir": "L:/DATASET/svc_preprocessed_v2",
    "manifest_name": "manifest_p80.json",
    "batch_size": 32,
    "max_steps": 300,
    "gradient_accumulation_steps": 1,
    "save_every": 9999,
    "log_every": 50,
    "warmup_steps": 30,
    "lora_alpha_ratio": 2,  # alpha = rank * ratio
    "sub_talker_lora_rank_ratio": 0.5,  # sub_rank = rank * ratio
    "num_workers": 4,
}

OUTPUT_BASE = "L:/DATASET/svc_output/grid_search"


def run_one(config, run_name):
    output_dir = os.path.join(OUTPUT_BASE, run_name)
    log_path = os.path.join(OUTPUT_BASE, f"{run_name}.log")
    os.makedirs(output_dir, exist_ok=True)

    rank = config["lora_rank"]
    cmd = [
        sys.executable, "finetuning/sft_svc.py",
        "--init_model_path", FIXED["init_model_path"],
        "--preprocessed_dir", FIXED["preprocessed_dir"],
        "--manifest_name", FIXED["manifest_name"],
        "--output_dir", output_dir,
        "--batch_size", str(FIXED["batch_size"]),
        "--lr", str(config["lr"]),
        "--max_steps", str(FIXED["max_steps"]),
        "--gradient_accumulation_steps", str(FIXED["gradient_accumulation_steps"]),
        "--save_every", str(FIXED["save_every"]),
        "--log_every", str(FIXED["log_every"]),
        "--warmup_steps", str(FIXED["warmup_steps"]),
        "--lora_rank", str(rank),
        "--lora_alpha", str(int(rank * FIXED["lora_alpha_ratio"])),
        "--sub_talker_lora_rank", str(max(8, int(rank * FIXED["sub_talker_lora_rank_ratio"]))),
        "--sub_talker_loss_weight", str(config["sub_talker_loss_weight"]),
        "--num_workers", str(FIXED["num_workers"]),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "J:/Qwen3-SVC-streaming"

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    # Force GPU memory cleanup between runs
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return proc.returncode, log_path


def extract_final_loss(log_path):
    last_step = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Step" in line and "loss=" in line:
                last_step = line.strip()
    if last_step:
        # Parse: "Step N | loss=X main=Y sub=Z lr=W"
        metrics = {}
        import re
        for key in ["loss", "main", "sub"]:
            m = re.search(rf"{key}=([0-9.e+-]+)", last_step)
            if m:
                metrics[key] = float(m.group(1))
        return metrics if metrics else None
    return None


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(itertools.product(*values))

    print(f"Grid search: {len(combos)} configurations")
    print(f"Params: {keys}")
    print()

    results = []
    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        run_name = f"lr{config['lr']}_r{config['lora_rank']}_w{config['sub_talker_loss_weight']}"
        print(f"[{i+1}/{len(combos)}] {run_name}...", end=" ", flush=True)

        rc, log_path = run_one(config, run_name)
        metrics = extract_final_loss(log_path)

        if rc == 0 and metrics:
            print(f"loss={metrics.get('loss', '?'):.4f} main={metrics.get('main', '?'):.4f} sub={metrics.get('sub', '?'):.4f}")
            results.append({**config, **metrics, "status": "ok"})
        else:
            print(f"FAILED (rc={rc})")
            results.append({**config, "status": "failed"})

    # Summary
    print("\n" + "=" * 80)
    print(f"{'lr':>8} {'rank':>5} {'w_sub':>5} | {'loss':>8} {'main':>8} {'sub':>8} | status")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("loss", 999)):
        print(f"{r['lr']:>8.0e} {r['lora_rank']:>5} {r['sub_talker_loss_weight']:>5.1f} | "
              f"{r.get('loss', 0):>8.4f} {r.get('main', 0):>8.4f} {r.get('sub', 0):>8.4f} | {r['status']}")

    # Save results
    with open(os.path.join(OUTPUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_BASE}/results.json")


if __name__ == "__main__":
    main()
