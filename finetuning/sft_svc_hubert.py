# coding=utf-8
"""SVC training with HuBERT content encoder.

Self-reconstruction training: content from HuBERT (pitch/speaker stripped)
+ F0 + speaker → codec tokens. At inference, swap F0/speaker to control.
"""
import argparse, json, logging, math, os, re, sys
import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.svc.codec_speaker_classifier import CodecSpeakerClassifier
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_speaker(audio_path: str) -> str:
    p = audio_path.replace("\\", "/")
    m = re.search(r"GTSinger_repo/([^/]+)/([^/]+)/", p)
    if m:
        return f"GTS-{m.group(1)}-{m.group(2)}"
    m = re.search(r"vc_training/train/wav/([^/]+)/", p)
    if m:
        return f"VCT-{m.group(1)}"
    return "UNKNOWN-" + os.path.basename(os.path.dirname(p))


def _speaker_from_feat(feat, speaker_map):
    # Prefer explicit ref_speaker (synthesized parallel data); fall back to path parsing
    if "ref_speaker" in feat:
        name = f"VCT-{feat['ref_speaker']}"
    else:
        name = _parse_speaker(feat.get("audio_path", ""))
    return speaker_map.get(name, 0)


class HubertSVCDataset(Dataset):
    def __init__(self, preprocessed_dir, manifest_name="manifest.json", preload=False,
                 speaker_map=None):
        with open(os.path.join(preprocessed_dir, manifest_name)) as f:
            self.manifest = json.load(f)
        self.speaker_map = speaker_map  # dict {speaker_name: int_id} or None
        self._cache = None
        if preload:
            self._cache = [self._load_and_tag(item["path"]) for item in self.manifest]
        logger.info(f"HubertSVCDataset: {len(self.manifest)} samples"
                    + (f" ({len(speaker_map)} speakers)" if speaker_map else ""))

    def _load_and_tag(self, path):
        feat = torch.load(path, weights_only=True)
        if self.speaker_map is not None:
            feat["speaker_id"] = torch.tensor(_speaker_from_feat(feat, self.speaker_map), dtype=torch.long)
        return feat

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_and_tag(self.manifest[idx]["path"])

    @staticmethod
    def collate_fn(batch):
        content = pad_sequence([b["content"] for b in batch], batch_first=True)
        target_codes = pad_sequence([b["target_codes"] for b in batch], batch_first=True)
        f0 = pad_sequence([b["f0"] for b in batch], batch_first=True)
        f0_bins = pad_sequence([b["f0_bins"] for b in batch], batch_first=True)
        spk_embed = torch.stack([b["spk_embed"] for b in batch])
        lengths = torch.tensor([b["content"].shape[0] for b in batch], dtype=torch.long)
        max_T = int(lengths.max().item())
        mask = torch.arange(max_T)[None, :] < lengths[:, None]
        out = {"content": content, "target_codes": target_codes, "f0": f0,
               "f0_bins": f0_bins, "mask": mask, "spk_embed": spk_embed}
        if "speaker_id" in batch[0]:
            out["speaker_id"] = torch.stack([b["speaker_id"] for b in batch])
        return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--manifest_name", default="manifest.json")
    p.add_argument("--output_dir", default="output/svc_hubert")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5, help="cosine decay end lr")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--mapper_layers", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--validate_samples", type=int, default=3)
    p.add_argument("--adv_weight", type=float, default=1.0, help="Adversarial F0 removal weight")
    p.add_argument("--adv_spk_weight", type=float, default=0.0, help="Adversarial speaker removal weight")
    p.add_argument("--speaker_map", type=str, default=None, help="Path to speaker_map.json (required if adv_spk_weight>0 or cross_spk_weight>0)")
    p.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint dir to continue from (loads mapper.pt + f0_projector.pt)")
    p.add_argument("--cross_spk_weight", type=float, default=0.0, help="Cross-speaker cycle-consistency weight (requires --codec_spk_clf)")
    p.add_argument("--cross_spk_warmup", type=int, default=2000, help="Steps before cross_spk loss activates (let codec reconstruction stabilize first)")
    p.add_argument("--cross_spk_ramp", type=int, default=1000, help="Steps to ramp cross_spk_weight from 0 to target after warmup")
    p.add_argument("--codec_spk_clf", type=str, default=None, help="Path to pretrained codec speaker classifier .pt")
    return p.parse_args()


def train():
    args = parse_args()
    device = "cuda:0"
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base model...")
    qwen = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=device)
    model = qwen.model
    D_cond = model.talker.config.hidden_size  # 2048
    V = model.talker.config.vocab_size

    for p in model.parameters():
        p.requires_grad = False

    speaker_map = None
    num_speakers = 0
    if args.adv_spk_weight > 0 or args.cross_spk_weight > 0:
        assert args.speaker_map, "--speaker_map is required when adv_spk_weight > 0 or cross_spk_weight > 0"
        with open(args.speaker_map, "r", encoding="utf-8") as f:
            speaker_map = json.load(f)
        num_speakers = len(speaker_map)
        logger.info(f"Speaker map: {num_speakers} speakers, adv_spk={args.adv_spk_weight}, cross_spk={args.cross_spk_weight}")

    codec_spk_clf = None
    if args.cross_spk_weight > 0:
        assert args.codec_spk_clf, "--codec_spk_clf checkpoint required when cross_spk_weight > 0"
        clf_ckpt = torch.load(args.codec_spk_clf, map_location=device, weights_only=True)
        assert clf_ckpt["num_speakers"] == num_speakers, "classifier/speaker_map size mismatch"
        codec_spk_clf = CodecSpeakerClassifier(vocab_size=V, num_codebooks=16,
                                               hidden=clf_ckpt["hidden"], num_layers=clf_ckpt["layers"],
                                               num_speakers=num_speakers).to(device=device)
        codec_spk_clf.load_state_dict(clf_ckpt["state_dict"])
        codec_spk_clf.eval()  # frozen supervisor; we backprop through it but do not update its weights
        for p in codec_spk_clf.parameters():
            p.requires_grad = False
        logger.info(f"Loaded codec speaker classifier ({clf_ckpt['hidden']}x{clf_ckpt['layers']}) from {args.codec_spk_clf}")

    f0_proj = F0Projector(D_cond).to(device=device, dtype=torch.float32)
    mapper = SVCMapperHubert(
        content_dim=768, cond_dim=D_cond,
        hidden_size=1024, num_layers=args.mapper_layers, num_heads=8,
        vocab_size=V, num_codebooks=16,
        num_speakers=num_speakers,
    ).to(device=device, dtype=torch.float32)

    if args.resume_from:
        mapper_sd = torch.load(f"{args.resume_from}/mapper.pt", map_location=device, weights_only=True)
        missing = mapper.load_state_dict(mapper_sd, strict=False)
        logger.info(f"Loaded mapper from {args.resume_from} (missing keys: {len(missing.missing_keys)}, unexpected: {len(missing.unexpected_keys)})")
        f0_proj.load_state_dict(torch.load(f"{args.resume_from}/f0_projector.pt", map_location=device, weights_only=True))
        logger.info(f"Loaded f0_projector from {args.resume_from}")

    trainable = list(f0_proj.parameters()) + list(mapper.parameters())
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    tb_dir = os.path.join(args.output_dir, "tb")
    writer = SummaryWriter(tb_dir)
    logger.info(f"TensorBoard: {tb_dir}")

    # Windows uses spawn: preload + workers duplicates RAM per worker. Use preload XOR workers.
    if args.num_workers > 0 and args.preload:
        logger.warning("Disabling preload because num_workers>0 (preload would duplicate RAM across spawn'd workers).")
        preload = False
    else:
        preload = args.preload
    dataset = HubertSVCDataset(args.preprocessed_dir, args.manifest_name, preload=preload,
                               speaker_map=speaker_map)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=HubertSVCDataset.collate_fn,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))

    opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    mapper.train(); f0_proj.train()

    global_step = 0
    data_iter = iter(dataloader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        content = batch["content"].to(device, non_blocking=True)
        tc = batch["target_codes"].to(device, non_blocking=True)
        f0 = batch["f0"].to(device, non_blocking=True)
        f0_bins = batch["f0_bins"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        spk = batch["spk_embed"].to(device, non_blocking=True)
        spk_id = batch["speaker_id"].to(device, non_blocking=True) if "speaker_id" in batch else None
        B, T, _ = tc.shape

        # Warm up adversarial lambdas from 0 to 1 over warmup_steps
        adv_lambda = min(1.0, global_step / max(args.warmup_steps, 1))
        adv_spk_lambda = adv_lambda

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            f0e = f0_proj(f0)
            codec_logits, adv_f0_logits, adv_spk_logits = mapper(
                content, f0e, spk, padding_mask=mask,
                adv_lambda=adv_lambda, adv_spk_lambda=adv_spk_lambda
            )

            # Codec reconstruction loss (over 16 codebooks)
            codec_loss = 0
            for cb_idx, logits in enumerate(codec_logits):
                cb_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), tc[:,:,cb_idx].reshape(-1), reduction='none'
                )
                codec_loss += (cb_loss.view(B, T) * mask.float()).sum() / mask.float().sum()
            codec_loss = codec_loss / len(codec_logits)

            # Adversarial F0 loss: predictor tries to recover F0 from reversed content
            adv_loss = torch.nn.functional.cross_entropy(
                adv_f0_logits.view(-1, mapper.f0_num_bins), f0_bins.view(-1), reduction='none'
            )
            adv_loss = (adv_loss.view(B, T) * mask.float()).sum() / mask.float().sum()

            # Adversarial speaker loss: predictor tries to recover speaker from pooled reversed content
            adv_spk_loss = torch.tensor(0.0, device=device)
            if adv_spk_logits is not None and spk_id is not None:
                adv_spk_loss = torch.nn.functional.cross_entropy(adv_spk_logits, spk_id)

            # Cross-speaker cycle-consistency: permute spk within batch, classifier should identify permuted speaker
            # Only activated after codec reconstruction has stabilized (cross_spk_warmup) to avoid destroying content.
            cross_spk_loss = torch.tensor(0.0, device=device)
            cross_spk_ramp = 0.0
            if codec_spk_clf is not None and spk_id is not None and global_step >= args.cross_spk_warmup:
                ramp_progress = min(1.0, (global_step - args.cross_spk_warmup) / max(args.cross_spk_ramp, 1))
                cross_spk_ramp = args.cross_spk_weight * ramp_progress
                if cross_spk_ramp > 0:
                    perm = torch.randperm(B, device=device)
                    spk_shuffled = spk[perm]
                    spk_id_shuffled = spk_id[perm]
                    codec_logits_cross, _, _ = mapper(content, f0e, spk_shuffled, padding_mask=mask,
                                                      adv_lambda=0.0, adv_spk_lambda=0.0)
                    clf_logits = codec_spk_clf.forward_logits(codec_logits_cross, mask)
                    cross_spk_loss = torch.nn.functional.cross_entropy(clf_logits, spk_id_shuffled)

            loss = (codec_loss + args.adv_weight * adv_loss
                    + args.adv_spk_weight * adv_spk_loss
                    + cross_spk_ramp * cross_spk_loss)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if global_step < args.warmup_steps:
            lr = args.lr * (global_step + 1) / max(args.warmup_steps, 1)
        else:
            progress = (global_step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()

        if (global_step + 1) % args.log_every == 0:
            with torch.no_grad():
                total_acc = 0
                for ci, lg in enumerate(codec_logits):
                    total_acc += ((lg.argmax(-1) == tc[:,:,ci]) * mask).sum().float() / mask.sum()
                acc = total_acc / len(codec_logits)
                adv_acc = ((adv_f0_logits.argmax(-1) == f0_bins) * mask).sum().float() / mask.sum()
                adv_spk_acc = 0.0
                if adv_spk_logits is not None and spk_id is not None:
                    adv_spk_acc = (adv_spk_logits.argmax(-1) == spk_id).float().mean().item()
            logger.info(f"Step {global_step+1}: codec={codec_loss.item():.3f} "
                        f"advF0={adv_loss.item():.3f} advSpk={adv_spk_loss.item():.3f} "
                        f"crossSpk={cross_spk_loss.item():.3f} "
                        f"cAcc={acc.item()*100:.1f}% aF0={adv_acc.item()*100:.1f}% "
                        f"aSpk={adv_spk_acc*100:.1f}% lr={lr:.2e}")
            step = global_step + 1
            writer.add_scalar("loss/codec", codec_loss.item(), step)
            writer.add_scalar("loss/adv_f0", adv_loss.item(), step)
            writer.add_scalar("loss/adv_spk", adv_spk_loss.item(), step)
            writer.add_scalar("loss/cross_spk", cross_spk_loss.item(), step)
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("acc/codec", acc.item(), step)
            writer.add_scalar("acc/adv_f0", adv_acc.item(), step)
            writer.add_scalar("acc/adv_spk", adv_spk_acc, step)
            writer.add_scalar("lr", lr, step)

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

    # In-process validation
    logger.info("=== Validation ===")
    mapper.eval(); f0_proj.eval()
    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    pearson_scores, diff_scores = [], []
    for idx in range(min(args.validate_samples, len(dataset))):
        feat = torch.load(dataset.manifest[idx]["path"], weights_only=True)
        content = feat["content"].unsqueeze(0).to(device)
        tgt = feat["target_codes"].to(device)
        f0_val = feat["f0"].unsqueeze(0).to(device)
        spk = feat["spk_embed"].unsqueeze(0).to(device)
        T_val = content.shape[1]

        with torch.no_grad():
            f0e = f0_proj(f0_val)
            pred = mapper.predict(content, f0e, spk, temperature=0)

        acc = (pred[0, :T_val] == tgt.to(device)).float().mean().item()
        with torch.no_grad():
            wav_pred, fs = model.speech_tokenizer.decode([{"audio_codes": pred[0, :T_val]}])
            wav_gt, _ = model.speech_tokenizer.decode([{"audio_codes": tgt}])
        sf.write(os.path.join(val_dir, f"sample{idx}_predicted.wav"), wav_pred[0], fs)
        sf.write(os.path.join(val_dir, f"sample{idx}_groundtruth.wav"), wav_gt[0], fs)

        # F0 pearson — output F0 should track the conditioning f0_val contour
        f0_out = align_f0_to_codec(extract_f0(wav_pred[0], fs, device=device), T_val).cpu().numpy()
        f0_in = f0_val.squeeze(0).cpu().numpy()
        both = (f0_out > 50) & (f0_in > 50)
        if both.sum() >= 5:
            diff_st = (np.log2(f0_out[both] / f0_in[both]) * 12).mean()
            pearson = np.corrcoef(np.log2(f0_out[both]), np.log2(f0_in[both]))[0, 1]
            pearson_scores.append(pearson)
            diff_scores.append(diff_st)
            logger.info(f"Sample {idx}: T={T_val} acc={acc*100:.1f}% F0_diff={diff_st:+.2f}st pearson={pearson:.3f}")
        else:
            logger.info(f"Sample {idx}: T={T_val} acc={acc*100:.1f}% F0: too few voiced frames")

    if pearson_scores:
        mean_pearson = float(np.mean(pearson_scores))
        mean_diff = float(np.mean(np.abs(diff_scores)))
        pass_f0 = mean_pearson > 0.8 and mean_diff < 1.0
        logger.info(f"=== F0 tracking: mean pearson={mean_pearson:.3f} (>0.8), mean |diff|={mean_diff:.2f}st (<1.0) "
                    f"{'PASS' if pass_f0 else 'FAIL'} ===")


if __name__ == "__main__":
    train()
