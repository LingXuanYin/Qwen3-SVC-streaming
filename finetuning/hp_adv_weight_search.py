# coding=utf-8
"""Search adv_weight: must balance codec_loss and F0 disentanglement.
Eval metric: per-frame F0 tracking (not just codec acc).
"""
import gc, os, sys, json
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sft_svc_hubert import HubertSVCDataset

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.svc.f0_projector import F0Projector
from qwen_tts.svc.svc_mapper_hubert import SVCMapperHubert
from qwen_tts.svc.f0_extractor import extract_f0, align_f0_to_codec, pitch_shift


def train_and_eval(model, ds, adv_weight, device, steps=1000, bs=20, lr=5e-4):
    talker = model.talker
    V = talker.config.vocab_size; D = talker.config.hidden_size

    f0_proj = F0Projector(D).to(device=device, dtype=torch.float32)
    mapper = SVCMapperHubert(content_dim=768, cond_dim=D, hidden_size=1024, num_layers=4, num_heads=8, vocab_size=V, num_codebooks=16).to(device=device, dtype=torch.float32)
    params = list(f0_proj.parameters()) + list(mapper.parameters())
    opt = AdamW(params, lr=lr, weight_decay=0.01)
    mapper.train(); f0_proj.train()
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=HubertSVCDataset.collate_fn, num_workers=0)
    data_iter = iter(dl)

    for step in range(steps):
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(dl); batch = next(data_iter)
        content = batch["content"].to(device); tc = batch["target_codes"].to(device)
        f0 = batch["f0"].to(device); mask = batch["mask"].to(device)
        rm = batch["ref_mels"].to(device=device, dtype=torch.bfloat16)
        B, T, _ = tc.shape
        with torch.no_grad(): spk = model.speaker_encoder(rm).float()
        f0e = f0_proj(f0)
        adv_lambda = min(1.0, (step+1) / 100)
        codec_logits, adv_logits = mapper(content, f0e, spk, padding_mask=mask, adv_lambda=adv_lambda)
        codec_loss = 0
        for ci, lg in enumerate(codec_logits):
            cb = torch.nn.functional.cross_entropy(lg.view(-1, V), tc[:,:,ci].reshape(-1), reduction='none')
            codec_loss = codec_loss + (cb.view(B,T)*mask.float()).sum()/mask.float().sum()
        codec_loss = codec_loss / 16
        f0_bins = mapper.f0_to_bin(f0, n_bins=mapper.f0_num_bins)
        adv_loss = torch.nn.functional.cross_entropy(adv_logits.view(-1, mapper.f0_num_bins), f0_bins.view(-1), reduction='none')
        adv_loss = (adv_loss.view(B,T)*mask.float()).sum()/mask.float().sum()
        loss = codec_loss + adv_weight * adv_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        cur_lr = lr * min(1.0, (step+1)/100)
        for pg in opt.param_groups: pg["lr"] = cur_lr
        opt.step()

    # Evaluate: F0 tracking on a shift=0 sample
    mapper.eval(); f0_proj.eval()
    shift0 = next((x for x in ds.manifest if x.get('shift') == 0), ds.manifest[0])
    feat = torch.load(shift0['path'], weights_only=True)
    content = feat['content'].unsqueeze(0).to(device)
    f0_orig = feat['f0'].to(device)
    T = content.shape[1]
    mel = feat['ref_mel'][:400].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad(): spk = model.speaker_encoder(mel).float()

    tracking = {}
    for shift_test in [0, 6, -6, 12]:
        f0_test = pitch_shift(f0_orig, float(shift_test)).to(device) if shift_test != 0 else f0_orig
        f0e = f0_proj(f0_test.unsqueeze(0))
        with torch.no_grad():
            pred = mapper.predict(content, f0e, spk, temperature=0)
            w, fs = model.speech_tokenizer.decode([{'audio_codes': pred[0]}])
        f0_out = align_f0_to_codec(extract_f0(w[0], fs, device=device), T).cpu().numpy()
        inp = f0_test.cpu().numpy()
        both = (inp > 50) & (f0_out > 50)
        if both.sum() < 5: tracking[shift_test] = 999; continue
        diff = (np.log2(f0_out[both]/inp[both]) * 12).mean()
        tracking[shift_test] = diff

    # Codec acc on training set
    mapper.eval()
    total_acc = 0; n_batches = 0
    with torch.no_grad():
        for batch in dl:
            content = batch["content"].to(device); tc = batch["target_codes"].to(device)
            f0 = batch["f0"].to(device); mask = batch["mask"].to(device)
            rm = batch["ref_mels"].to(device=device, dtype=torch.bfloat16)
            B, T, _ = tc.shape
            spk = model.speaker_encoder(rm).float()
            f0e = f0_proj(f0)
            codec_logits, _ = mapper(content, f0e, spk, padding_mask=mask, adv_lambda=0.0)
            for ci, lg in enumerate(codec_logits):
                total_acc += ((lg.argmax(-1) == tc[:,:,ci]) * mask).sum().float() / mask.sum()
            n_batches += 1
            break  # one batch
        acc = (total_acc / 16 / n_batches).item() * 100

    del mapper, f0_proj, opt, params
    gc.collect(); torch.cuda.empty_cache()
    return acc, tracking


def main():
    device = "cuda:0"
    print("Loading model...")
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", torch_dtype=torch.bfloat16, device_map=device)
    model = m.model
    for p in model.parameters(): p.requires_grad = False

    ds = HubertSVCDataset("L:/DATASET/svc_hubert_v4", "manifest_mini.json")

    print(f"\n{'adv_w':>6} | {'codec_acc':>10} | {'F0 tracking per-frame diff (ideal 0)':>40}")
    print("-" * 80)
    for adv_w in [0.2, 0.3, 0.5, 0.7]:
        acc, track = train_and_eval(model, ds, adv_w, device, steps=1500)
        t_str = "  ".join([f"{s:+d}st:{v:+.2f}" for s, v in track.items()])
        print(f"{adv_w:>6.2f} | {acc:>9.1f}% | {t_str}", flush=True)


if __name__ == "__main__":
    main()
