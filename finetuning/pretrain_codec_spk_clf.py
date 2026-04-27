# coding=utf-8
"""Pretrain codec-level speaker classifier on real target_codec.

The classifier will then be used (frozen) to supervise cross-speaker SVC training:
given a mapper's output with swapped spk_embed, the classifier's speaker prediction
should match the swapped speaker.
"""
import argparse, json, logging, math, os, re, sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts.svc.codec_speaker_classifier import CodecSpeakerClassifier
from sft_svc_hubert import _parse_speaker

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


class CodecSpkDataset(Dataset):
    def __init__(self, preprocessed_dir, manifest_name, speaker_map, preload=False):
        with open(os.path.join(preprocessed_dir, manifest_name)) as f:
            self.manifest = json.load(f)
        self.speaker_map = speaker_map
        self._cache = None
        if preload:
            self._cache = [self._load(item["path"]) for item in self.manifest]
        logger.info(f"Loaded {len(self.manifest)} samples, {len(speaker_map)} speakers")

    def _load(self, path):
        feat = torch.load(path, weights_only=True)
        return {
            "target_codes": feat["target_codes"],
            "speaker_id": torch.tensor(self.speaker_map.get(_parse_speaker(feat["audio_path"]), 0), dtype=torch.long),
        }

    def __len__(self): return len(self.manifest)

    def __getitem__(self, idx):
        if self._cache is not None: return self._cache[idx]
        return self._load(self.manifest[idx]["path"])

    @staticmethod
    def collate(batch):
        tokens = pad_sequence([b["target_codes"] for b in batch], batch_first=True)
        lengths = torch.tensor([b["target_codes"].shape[0] for b in batch], dtype=torch.long)
        max_T = int(lengths.max().item())
        mask = torch.arange(max_T)[None, :] < lengths[:, None]
        spk = torch.stack([b["speaker_id"] for b in batch])
        return {"tokens": tokens, "mask": mask, "speaker_id": spk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed_dir", required=True)
    ap.add_argument("--manifest_name", default="manifest.json")
    ap.add_argument("--speaker_map", required=True)
    ap.add_argument("--output", required=True, help="Path to save classifier state_dict")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    with open(args.speaker_map, "r", encoding="utf-8") as f:
        speaker_map = json.load(f)
    n_spk = len(speaker_map)

    device = "cuda:0"
    clf = CodecSpeakerClassifier(vocab_size=3072, num_codebooks=16,
                                 hidden=args.hidden, num_layers=args.layers,
                                 num_speakers=n_spk).to(device)

    dataset = CodecSpkDataset(args.preprocessed_dir, args.manifest_name, speaker_map, preload=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=CodecSpkDataset.collate,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))

    opt = AdamW(clf.parameters(), lr=args.lr, weight_decay=0.01)

    clf.train()
    it = iter(loader)
    step = 0
    while step < args.max_steps:
        try: batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        spk_id = batch["speaker_id"].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = clf.forward_tokens(tokens, mask)
            loss = torch.nn.functional.cross_entropy(logits, spk_id)

        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups: pg["lr"] = lr

        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == spk_id).float().mean().item()
            logger.info(f"Step {step+1}: loss={loss.item():.4f} acc={acc*100:.1f}% lr={lr:.2e}")
        step += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "state_dict": clf.state_dict(),
        "num_speakers": n_spk,
        "hidden": args.hidden,
        "layers": args.layers,
    }, args.output)
    logger.info(f"Saved classifier to {args.output}")


if __name__ == "__main__":
    main()
