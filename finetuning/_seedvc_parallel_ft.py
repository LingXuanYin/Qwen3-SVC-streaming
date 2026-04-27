"""Seed-VC parallel fine-tune: (A content, B F0+mel+speaker) → B mel.

Purpose: close the (phoneme, pitch) co-occurrence distribution gap that zero-shot
Seed-VC has in pitch_ref mode. Training data = GTSinger cross-singer parallel
pairs (same song, different singer) from find_parallel_pairs().

Differences vs official train.py:
  - Dataset returns paired (A_wav, B_wav, B_mel) instead of single utterance
  - train_one_step: content from A, F0 from B, target mel from B, speaker y from B
  - No timbre perturbation on A (already cross-singer); no prompt bootstrapping
    (simpler — prompt region reuses B content anyway)

Usage:
  python finetuning/_seedvc_parallel_ft.py \
      --max_steps 200 --batch_size 2 --run_name parallel_gts_overfit --overfit 20

Notes:
  - Uses Seed-VC's official singing config (44k + f0 + whisper_base)
  - Logs CFM loss every 10 steps, saves ckpt every `save_interval`
  - Works on cuda:0 only (matches existing project convention)
"""
import os, sys, argparse, glob, json, random, time
sys.path.insert(0, 'external/seed-vc')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_HUB_CACHE', './external/seed-vc/checkpoints/hf_cache')

import numpy as np
import torch
import torch.utils.data
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import yaml
import soundfile as sf

from modules.commons import recursive_munch, build_model, load_checkpoint
from modules.audio import mel_spectrogram
from optimizers import build_optimizer
from hf_utils import load_custom_model_from_hf

# Parallel pair discovery — reuse existing helper
sys.path.insert(0, 'finetuning')
from preprocess_gts_parallel import find_parallel_pairs


DUR_MIN = 1.0
DUR_MAX = 30.0


class ParallelDataset(torch.utils.data.Dataset):
    """Returns (A_wav, B_wav, B_mel, lengths) for CFM supervised training.

    A: content reference (source-like)
    B: target (F0 + mel + speaker all from B)
    """
    def __init__(self, pairs, spect_params, sr, both_directions=True):
        self.sr = sr
        self.mel_kwargs = {
            'n_fft': spect_params['n_fft'],
            'win_size': spect_params.get('win_length', spect_params.get('win_size', 1024)),
            'hop_size': spect_params.get('hop_length', spect_params.get('hop_size', 256)),
            'num_mels': spect_params.get('n_mels', spect_params.get('num_mels', 80)),
            'sampling_rate': sr,
            'fmin': spect_params['fmin'],
            'fmax': None if spect_params['fmax'] == 'None' else spect_params['fmax'],
            'center': False,
        }
        # Expand to directed tuples
        directed = []
        for a, b, sa, sb in pairs:
            directed.append((a, b))
            if both_directions:
                directed.append((b, a))
        self.pairs = directed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path = self.pairs[idx % len(self.pairs)]
        try:
            a_wav, a_sr = librosa.load(a_path, sr=self.sr)
            b_wav, b_sr = librosa.load(b_path, sr=self.sr)
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))
        if (len(a_wav) < self.sr * DUR_MIN or len(a_wav) > self.sr * DUR_MAX
            or len(b_wav) < self.sr * DUR_MIN or len(b_wav) > self.sr * DUR_MAX):
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))
        a_t = torch.from_numpy(a_wav).float().unsqueeze(0)
        b_t = torch.from_numpy(b_wav).float().unsqueeze(0)
        b_mel = mel_spectrogram(b_t, **self.mel_kwargs).squeeze(0)
        return a_t.squeeze(0), b_t.squeeze(0), b_mel


def collate_parallel(batch):
    B = len(batch)
    # Sort by B_mel length descending (reduces padding in CFM)
    batch = sorted(batch, key=lambda x: x[2].shape[1], reverse=True)
    a_waves = [b[0] for b in batch]
    b_waves = [b[1] for b in batch]
    b_mels  = [b[2] for b in batch]

    max_a_len = max(w.size(0) for w in a_waves)
    max_b_len = max(w.size(0) for w in b_waves)
    max_m_len = max(m.size(1) for m in b_mels)
    nmels = b_mels[0].size(0)

    a_pad = torch.zeros(B, max_a_len)
    b_pad = torch.zeros(B, max_b_len)
    m_pad = torch.full((B, nmels, max_m_len), -10.0)
    a_lens = torch.zeros(B, dtype=torch.long)
    b_lens = torch.zeros(B, dtype=torch.long)
    m_lens = torch.zeros(B, dtype=torch.long)

    for i in range(B):
        a_pad[i, :a_waves[i].size(0)] = a_waves[i]
        b_pad[i, :b_waves[i].size(0)] = b_waves[i]
        m_pad[i, :, :b_mels[i].size(1)] = b_mels[i]
        a_lens[i] = a_waves[i].size(0)
        b_lens[i] = b_waves[i].size(0)
        m_lens[i] = b_mels[i].size(1)
    return a_pad, b_pad, m_pad, a_lens, b_lens, m_lens


class ParallelTrainer:
    def __init__(self, config_path, pretrained_ckpt_path, run_name,
                 batch_size=2, num_workers=0, max_steps=200, save_interval=200,
                 overfit=0, device='cuda:0', lr=1e-5, seed=42):
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

        self.sr = config['preprocess_params']['sr']
        preprocess_params = config['preprocess_params']

        # Data
        pairs = find_parallel_pairs()
        random.shuffle(pairs)
        if overfit > 0:
            pairs = pairs[:overfit]
            print(f'[OVERFIT] using only {len(pairs)} pairs')
        dataset = ParallelDataset(pairs, preprocess_params['spect_params'], self.sr, both_directions=True)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_parallel, drop_last=True,
        )
        print(f'Dataset: {len(dataset)} directed tuples, batch_size={batch_size}, '
              f'iters_per_epoch={len(self.train_dataloader)}', flush=True)

        self.f0_condition = config['model_params']['DiT'].get('f0_condition', False)
        assert self.f0_condition, 'Parallel FT requires f0_condition=True (singing config)'

        # Build semantic/f0/SV/vocoder (reuse code from official train.py via helpers)
        self._build_sv_model(config)
        self._build_semantic_fn(config)
        self._build_f0_fn()

        # Build DiT model
        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, stage='DiT')
        _ = [self.model[key].to(self.device) for key in self.model]
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        self.optimizer = build_optimizer({key: self.model[key] for key in self.model}, lr=lr)

        # Load pretrained
        if pretrained_ckpt_path is None:
            ckpt = load_custom_model_from_hf('Plachta/Seed-VC', config['pretrained_model'], None)
        else:
            ckpt = pretrained_ckpt_path
        self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
            self.model, self.optimizer, ckpt, load_only_params=True, ignore_modules=[], is_distributed=False,
        )
        print(f'Loaded pretrained {ckpt}', flush=True)
        self.iters = 0  # start from 0 for fine-tune

    def _build_sv_model(self, config):
        from modules.campplus.DTDNN import CAMPPlus
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        p = load_custom_model_from_hf('funasr/campplus', 'campplus_cn_common.bin', config_filename=None)
        self.campplus_model.load_state_dict(torch.load(p, map_location='cpu'))
        self.campplus_model.eval().to(self.device)

    def _build_semantic_fn(self, config):
        speech_tokenizer = config['model_params']['speech_tokenizer'].get('type', 'whisper')
        assert speech_tokenizer == 'whisper', f'Expected whisper, got {speech_tokenizer}'
        from transformers import AutoFeatureExtractor, WhisperModel
        wname = config['model_params']['speech_tokenizer']['name']
        self.whisper_model = WhisperModel.from_pretrained(wname, torch_dtype=torch.float16).to(self.device)
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(wname)
        self.whisper_model.eval()
        del self.whisper_model.decoder
        def semantic_fn(waves_16k):
            # Do NOT pass padding=True — default is 'max_length' which pads to 3000 (Whisper requirement).
            inp = self.whisper_feature_extractor(
                [w.cpu().numpy() for w in waves_16k], return_tensors='pt',
                return_attention_mask=True, sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inp.input_features, attention_mask=inp.attention_mask,
            ).to(self.device)
            with torch.no_grad():
                out = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None, output_attentions=False, output_hidden_states=False,
                    return_dict=True,
                )
            S = out.last_hidden_state.to(torch.float32)
            S = S[:, :waves_16k.size(-1) // 320 + 1]
            return S
        self.semantic_fn = semantic_fn

    def _build_f0_fn(self):
        from modules.rmvpe import RMVPE
        p = load_custom_model_from_hf('lj1995/VoiceConversionWebUI', 'rmvpe.pt', None)
        self.rmvpe = RMVPE(p, is_half=False, device=self.device)

    def train_one_step(self, batch):
        a_waves, b_waves, b_mels, a_wlens, b_wlens, b_mlens = [x.to(self.device) for x in batch]
        B = a_waves.size(0)

        target = b_mels
        target_lengths = b_mlens

        # Resample both to 16k for semantic + F0
        a_waves_16k = torchaudio.functional.resample(a_waves, self.sr, 16000)
        b_waves_16k = torchaudio.functional.resample(b_waves, self.sr, 16000)
        a_wlens_16k = (a_wlens.float() * 16000 / self.sr).long()
        b_wlens_16k = (b_wlens.float() * 16000 / self.sr).long()

        # Content = Whisper(A_16k) — in A's time axis (will be nearest-stretched to target_lengths)
        S_a = self.semantic_fn(a_waves_16k)

        # F0 = RMVPE(B_16k) — in B's time axis, matches target mel timing
        F0_b = self.rmvpe.infer_from_audio_batch(b_waves_16k)

        # length_regulator pulls S_a → target_lengths (B time axis) and does same for F0
        cond, _, codes, commit_loss, cb_loss = self.model.length_regulator(
            S_a, ylens=target_lengths, f0=F0_b,
        )
        if commit_loss is None:
            commit_loss = 0
            cb_loss = 0

        # No prompt bootstrapping in parallel mode — cond=alt everywhere
        prompt_len = torch.zeros(B, dtype=torch.long, device=self.device)

        common_min_len = min(target.size(2), cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)

        # Speaker y from B (target)
        feat_list = []
        for bib in range(B):
            feat = kaldi.fbank(
                b_waves_16k[bib:bib + 1, :b_wlens_16k[bib]],
                num_mel_bins=80, dither=0, sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.campplus_model(feat.unsqueeze(0))
                y_list.append(y)
        y = torch.cat(y_list, dim=0)

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
        ema = None
        losses = []
        t0 = time.time()
        while self.iters < self.max_steps:
            for batch in self.train_dataloader:
                loss = self.train_one_step(batch)
                ema = loss if ema is None else ema * 0.95 + loss * 0.05
                losses.append(loss)
                if self.iters % self.log_interval == 0:
                    rate = (self.iters + 1) / (time.time() - t0 + 1e-9)
                    print(f'step {self.iters:5d}  loss={loss:.4f}  ema={ema:.4f}  {rate:.2f} it/s',
                          flush=True)
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
    ap.add_argument('--run_name', default='parallel_gts_v1')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--save_interval', type=int, default=200)
    ap.add_argument('--overfit', type=int, default=0, help='if >0, use only first N pairs')
    ap.add_argument('--lr', type=float, default=1e-5)
    args = ap.parse_args()

    tr = ParallelTrainer(
        config_path=args.config, pretrained_ckpt_path=args.pretrained_ckpt,
        run_name=args.run_name, batch_size=args.batch_size,
        num_workers=args.num_workers, max_steps=args.max_steps,
        save_interval=args.save_interval, overfit=args.overfit, lr=args.lr,
    )
    losses = tr.fit()
    # Print first/last mean for quick health check
    if len(losses) >= 20:
        print(f'\nFirst-10 mean loss: {np.mean(losses[:10]):.4f}')
        print(f'Last-10 mean loss:  {np.mean(losses[-10:]):.4f}')


if __name__ == '__main__':
    main()
