# coding=utf-8
"""SVC training dataset using precomputed features (no GPU in dataloader)."""

import json
import logging
import os

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SVCCachedDataset(Dataset):
    """Loads precomputed .pt feature files. Zero GPU work in __getitem__."""

    def __init__(self, preprocessed_dir: str, manifest_name: str = "manifest.json", preload: bool = False):
        manifest_path = os.path.join(preprocessed_dir, manifest_name)
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self._cache = None
        if preload:
            logger.info(f"SVCCachedDataset: preloading {len(self.manifest)} samples into RAM...")
            self._cache = []
            for i, item in enumerate(self.manifest):
                self._cache.append(torch.load(item["path"], weights_only=True))
                if (i + 1) % 5000 == 0:
                    logger.info(f"  preloaded {i+1}/{len(self.manifest)}")
            logger.info(f"SVCCachedDataset: preload complete")
        else:
            logger.info(f"SVCCachedDataset: {len(self.manifest)} samples from {manifest_path}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        path = self.manifest[idx]["path"]
        return torch.load(path, weights_only=True)

    @staticmethod
    def collate_fn(batch):
        B = len(batch)
        max_T = max(b["source_codes"].shape[0] for b in batch)
        Q = batch[0]["source_codes"].shape[1]
        MAX_MEL_FRAMES = 400

        source_codes = torch.zeros(B, max_T, Q, dtype=torch.long)
        target_codes = torch.zeros(B, max_T, Q, dtype=torch.long)
        f0 = torch.zeros(B, max_T)
        mask = torch.zeros(B, max_T, dtype=torch.bool)

        for i, b in enumerate(batch):
            T = b["source_codes"].shape[0]
            source_codes[i, :T] = b["source_codes"]
            target_codes[i, :T] = b["target_codes"]
            f0[i, :T] = b["f0"]
            mask[i, :T] = True

        mel_dim = batch[0]["ref_mel"].shape[-1]
        max_mel_T = min(max(b["ref_mel"].shape[0] for b in batch), MAX_MEL_FRAMES)
        ref_mels = torch.zeros(B, max_mel_T, mel_dim)
        for i, b in enumerate(batch):
            mel_T = min(b["ref_mel"].shape[0], MAX_MEL_FRAMES)
            ref_mels[i, :mel_T] = b["ref_mel"][:mel_T]

        return {
            "source_codes": source_codes,
            "target_codes": target_codes,
            "f0": f0,
            "mask": mask,
            "ref_mels": ref_mels,
        }
