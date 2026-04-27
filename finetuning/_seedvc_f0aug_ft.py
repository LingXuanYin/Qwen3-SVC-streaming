"""Seed-VC F0-aug self-reconstruction fine-tune (v3: full precompute).

Reads precomputed features from `_preprocess_f0aug.py` + `_preprocess_f0aug_feats.py`.
Training loop only runs length_regulator + CFM forward/backward — zero CPU-bound
ops, zero CPU↔GPU transfers besides the minimal npy→device copy.

Per sample dir provides:
  - a_whisper.npy         : (T_w, D)      Whisper encoder output, shift-invariant
  - b_shift{s}_mel.npy    : (n_mels, T)   mel target (shift-specific)
  - b_shift{s}_f0.npy     : (T,)          RMVPE F0 (shift-specific)
  - b_shift{s}_spk.npy    : (192,)        campplus speaker embedding (shift-specific)
"""
import os, sys, argparse, glob, random, time
sys.path.insert(0, 'external/seed-vc')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_HUB_CACHE', './external/seed-vc/checkpoints/hf_cache')

import numpy as np
import torch
import torch.utils.data
import yaml

from modules.commons import recursive_munch, build_model, load_checkpoint
from optimizers import build_optimizer
from hf_utils import load_custom_model_from_hf


class F0AugFeatsDataset(torch.utils.data.Dataset):
    """Returns (a_whisper, b_mel, b_f0, b_spk) all precomputed tensors."""
    def __init__(self, preproc_root, shifts=(-6, -4, -2, 0, 2, 4, 6)):
        self.shifts = list(shifts)
        self.dirs = []
        for d in sorted(glob.glob(os.path.join(preproc_root, '*'))):
            if not os.path.isdir(d):
                continue
            if not os.path.exists(os.path.join(d, 'a_whisper.npy')):
                continue
            # require at least one shift with full feature set
            for s in self.shifts:
                mel_p = os.path.join(d, f'b_shift{s:+d}_mel.npy')
                f0_p  = os.path.join(d, f'b_shift{s:+d}_f0.npy')
                spk_p = os.path.join(d, f'b_shift{s:+d}_spk.npy')
                if os.path.exists(mel_p) and os.path.exists(f0_p) and os.path.exists(spk_p):
                    self.dirs.append(d)
                    break
        assert len(self.dirs) > 0, f'No precomputed samples with features in {preproc_root}'

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        d = self.dirs[idx % len(self.dirs)]
        avail = [s for s in self.shifts
                 if os.path.exists(os.path.join(d, f'b_shift{s:+d}_mel.npy'))
                 and os.path.exists(os.path.join(d, f'b_shift{s:+d}_f0.npy'))
                 and os.path.exists(os.path.join(d, f'b_shift{s:+d}_spk.npy'))]
        s = random.choice(avail)
        a_w = np.load(os.path.join(d, 'a_whisper.npy'))  # (T_w, D) fp32
        b_mel = np.load(os.path.join(d, f'b_shift{s:+d}_mel.npy'))  # (n_mels, T) fp16
        b_f0  = np.load(os.path.join(d, f'b_shift{s:+d}_f0.npy'))   # (T,) fp32
        b_spk = np.load(os.path.join(d, f'b_shift{s:+d}_spk.npy'))  # (192,) fp32
        return (torch.from_numpy(a_w).float(),
                torch.from_numpy(b_mel.astype(np.float32)),
                torch.from_numpy(b_f0).float(),
                torch.from_numpy(b_spk).float())


def collate_feats(batch):
    B = len(batch)
    # sort by mel length desc
    batch = sorted(batch, key=lambda x: x[1].size(1), reverse=True)
    a_ws, b_mels, b_f0s, b_spks = zip(*batch)

    n_mels = b_mels[0].size(0)
    d_wh = a_ws[0].size(1)
    max_Tw = max(x.size(0) for x in a_ws)
    max_Tm = max(x.size(1) for x in b_mels)
    max_Tf = max(x.size(0) for x in b_f0s)

    a_w_pad = torch.zeros(B, max_Tw, d_wh)
    mel_pad = torch.full((B, n_mels, max_Tm), -10.0)
    f0_pad  = torch.zeros(B, max_Tf)
    spk_pad = torch.stack(list(b_spks))  # (B, 192)
    a_w_lens = torch.zeros(B, dtype=torch.long)
    mel_lens = torch.zeros(B, dtype=torch.long)
    f0_lens  = torch.zeros(B, dtype=torch.long)

    for i in range(B):
        a_w_pad[i, :a_ws[i].size(0)] = a_ws[i]
        mel_pad[i, :, :b_mels[i].size(1)] = b_mels[i]
        f0_pad[i, :b_f0s[i].size(0)] = b_f0s[i]
        a_w_lens[i] = a_ws[i].size(0)
        mel_lens[i] = b_mels[i].size(1)
        f0_lens[i]  = b_f0s[i].size(0)
    return a_w_pad, mel_pad, f0_pad, spk_pad, a_w_lens, mel_lens, f0_lens


class F0AugTrainer:
    def __init__(self, config_path, pretrained_ckpt_path, run_name, preproc_root,
                 batch_size=4, num_workers=4, max_steps=1500, save_interval=500,
                 device='cuda:0', lr=1e-5, seed=42, shifts=(-6, -4, -2, 0, 2, 4, 6)):
        self.device = torch.device(device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        config = yaml.safe_load(open(config_path))
        self.log_dir = os.path.join('runs', run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        import shutil
        shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        self.max_steps = max_steps
        self.save_interval = save_interval
        self.log_interval = 5

        dataset = F0AugFeatsDataset(preproc_root, shifts=shifts)
        print(f'Dataset: {len(dataset)} samples, shifts={shifts}', flush=True)
        dl_kwargs = dict(
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_feats, drop_last=True, pin_memory=True,
        )
        if num_workers > 0:
            dl_kwargs['persistent_workers'] = True
            dl_kwargs['prefetch_factor'] = 4
        self.train_dataloader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

        # Only length_regulator + cfm receive gradient. No frozen encoders loaded in trainer.
        self.f0_condition = config['model_params']['DiT'].get('f0_condition', False)
        assert self.f0_condition
        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, stage='DiT')
        _ = [self.model[key].to(self.device) for key in self.model]
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
        self.optimizer = build_optimizer({k: self.model[k] for k in self.model}, lr=lr)

        if pretrained_ckpt_path is None:
            ckpt = load_custom_model_from_hf('Plachta/Seed-VC', config['pretrained_model'], None)
        else:
            ckpt = pretrained_ckpt_path
        self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
            self.model, self.optimizer, ckpt, load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        print(f'Loaded pretrained {ckpt}', flush=True)
        self.iters = 0

    def train_one_step(self, batch):
        a_w, b_mel, b_f0, b_spk, a_w_lens, mel_lens, f0_lens = [x.to(self.device, non_blocking=True) for x in batch]
        B = a_w.size(0)

        target = b_mel
        target_lengths = mel_lens

        cond, _, _, commit_loss, cb_loss = self.model.length_regulator(
            a_w, ylens=target_lengths, f0=b_f0,
        )
        if commit_loss is None:
            commit_loss = 0; cb_loss = 0

        prompt_len = torch.zeros(B, dtype=torch.long, device=self.device)
        common_min = min(target.size(2), cond.size(1))
        target = target[:, :, :common_min]
        cond = cond[:, :common_min]
        target_lengths = torch.clamp(target_lengths, max=common_min)

        y = b_spk  # (B, 192) precomputed

        loss, _ = self.model.cfm(target, target_lengths, prompt_len, cond, y)
        loss_total = loss + commit_loss * 0.05 + cb_loss * 0.15

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        self.optimizer.step('cfm')
        self.optimizer.step('length_regulator')
        self.optimizer.scheduler(key='cfm')
        self.optimizer.scheduler(key='length_regulator')
        return loss.detach().item()

    def save_ckpt(self, suffix=''):
        state = {'net': {k: self.model[k].state_dict() for k in self.model},
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.optimizer.scheduler_state_dict(),
                 'iters': self.iters, 'epoch': 0}
        p = os.path.join(self.log_dir, f'ft_model{suffix}.pth')
        torch.save(state, p)
        print(f'Saved {p}', flush=True)

    def fit(self):
        _ = [self.model[k].train() for k in self.model]
        ema = None; losses = []; t0 = time.time()
        while self.iters < self.max_steps:
            for batch in self.train_dataloader:
                loss = self.train_one_step(batch)
                ema = loss if ema is None else ema * 0.95 + loss * 0.05
                losses.append(loss)
                if self.iters % self.log_interval == 0:
                    rate = (self.iters + 1) / (time.time() - t0 + 1e-9)
                    print(f'step {self.iters:5d}  loss={loss:.4f}  ema={ema:.4f}  {rate:.2f} it/s', flush=True)
                if self.iters > 0 and self.iters % self.save_interval == 0:
                    self.save_ckpt(f'_step{self.iters}')
                self.iters += 1
                if self.iters >= self.max_steps:
                    break
        self.save_ckpt('')
        return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='external/seed-vc/configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
    ap.add_argument('--pretrained_ckpt', default=None)
    ap.add_argument('--run_name', default='f0aug_v3')
    ap.add_argument('--preproc_root', default='L:/DATASET/svc_f0aug')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_steps', type=int, default=1500)
    ap.add_argument('--save_interval', type=int, default=500)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--shifts', type=str, default='-6,-4,-2,0,2,4,6')
    args = ap.parse_args()

    shifts = tuple(int(s) for s in args.shifts.split(','))
    tr = F0AugTrainer(
        config_path=args.config, pretrained_ckpt_path=args.pretrained_ckpt,
        run_name=args.run_name, preproc_root=args.preproc_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_steps=args.max_steps, save_interval=args.save_interval, lr=args.lr,
        shifts=shifts,
    )
    losses = tr.fit()
    if len(losses) >= 20:
        print(f'\nFirst-10 mean loss: {np.mean(losses[:10]):.4f}')
        print(f'Last-10 mean loss:  {np.mean(losses[-10:]):.4f}')


if __name__ == '__main__':
    main()
