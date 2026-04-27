# coding=utf-8
"""Upgrade existing preprocessed .pt files to offline spk_embed + f0_bins.

Converts legacy feat dict (content, target_codes, f0, ref_mel, ...) into
new layout (content, target_codes, f0, f0_bins, spk_embed, ...) so the
training loop no longer has to run speaker_encoder or f0_to_bin each step.

Run once per dataset directory.
"""
import argparse, json, os, torch, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Preprocessed dir containing manifest.json + NNNNNN.pt")
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32, help="Batch size for speaker_encoder forward")
    p.add_argument("--io_workers", type=int, default=8)
    args = p.parse_args()

    with open(os.path.join(args.dir, "manifest.json")) as f:
        manifest = json.load(f)
    print(f"Upgrading {len(manifest)} samples in {args.dir}", flush=True)

    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map=args.device)
    spk_enc = m.model.speaker_encoder

    io_pool = ThreadPoolExecutor(max_workers=args.io_workers)

    def _load(path): return path, torch.load(path, weights_only=True)
    def _save(path_feat):
        path, feat = path_feat
        torch.save(feat, path)

    paths = [item["path"] for item in manifest]
    converted = skipped = 0

    # Double-buffer: while GPU processes batch N, IO loads batch N+1 and writes batch N-1
    for start in range(0, len(paths), args.batch):
        chunk = paths[start:start + args.batch]
        loaded = list(io_pool.map(_load, chunk))
        batch_feats, batch_paths = [], []
        mels = []
        for path, feat in loaded:
            if "spk_embed" in feat and "f0_bins" in feat:
                skipped += 1
                continue
            if "ref_mel" not in feat or "f0" not in feat:
                skipped += 1
                continue
            batch_feats.append(feat)
            batch_paths.append(path)
            mels.append(feat["ref_mel"][:400])

        if not mels:
            if start % (args.batch * 10) == 0:
                print(f"  {start}/{len(paths)} converted={converted} skipped={skipped}", flush=True)
            continue

        # Pad to (B, max_T_mel, 128) for speaker_encoder
        max_T = max(mel.shape[0] for mel in mels)
        mel_dim = mels[0].shape[-1]
        padded = torch.zeros(len(mels), max_T, mel_dim, dtype=torch.float16, device=args.device)
        for i, mel in enumerate(mels):
            padded[i, :mel.shape[0]] = mel.to(device=args.device, dtype=torch.float16)

        with torch.inference_mode():
            spk = spk_enc(padded).float().cpu()  # (B, D)

        to_save = []
        for i, feat in enumerate(batch_feats):
            feat["spk_embed"] = spk[i].clone()
            feat["f0_bins"] = SVCMapperHubert.f0_to_bin(feat["f0"], n_bins=360)
            feat.pop("ref_mel", None)
            to_save.append((batch_paths[i], feat))
            converted += 1

        list(io_pool.map(_save, to_save))

        if start % (args.batch * 10) == 0:
            print(f"  {start}/{len(paths)} converted={converted} skipped={skipped}", flush=True)

    io_pool.shutdown()
    print(f"Done: converted={converted} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
