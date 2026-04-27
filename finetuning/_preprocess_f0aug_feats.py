"""Stage 2 preprocessing: compute and cache frozen-encoder features.

Input: L:/DATASET/svc_f0aug/<idx>/{a_16k.npy, b_shift{s}_16k.npy, b_shift{s}_44k.npy}
Output (per sample dir):
  - a_whisper.npy             : Whisper encoder output for a_16k, shape (T_w, D)
  - b_shift{s}_mel.npy        : mel_spectrogram(b_shift{s}_44k)    (shift-specific target)
  - b_shift{s}_f0.npy         : RMVPE(b_shift{s}_16k)              (shift-specific)
  - b_shift{s}_spk.npy        : campplus(b_shift{s}_16k)           (shift-specific)

This removes all heavy ops from the training loop — trainer only does
length_regulator + CFM forward/backward + optimizer step.

Storage estimate per sample (10s typical):
  a_whisper  ~500 KB
  b_mel      ~220 KB × 7 = 1.5 MB
  b_f0       ~4 KB × 7   = 28 KB
  b_spk      ~0.8 KB × 7 = 6 KB
Total ~2 MB × 2065 = ~4 GB on top of stage-1 33 GB.
"""
import os, sys, argparse, glob, time
sys.path.insert(0, 'external/seed-vc')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('HF_HUB_CACHE', './external/seed-vc/checkpoints/hf_cache')

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml

from modules.audio import mel_spectrogram
from hf_utils import load_custom_model_from_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preproc_root', default='L:/DATASET/svc_f0aug')
    ap.add_argument('--config', default='external/seed-vc/configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shifts', type=str, default='-6,-4,-2,0,2,4,6')
    ap.add_argument('--batch_mel', type=int, default=8, help='batch size for GPU ops')
    args = ap.parse_args()
    shifts = [int(s) for s in args.shifts.split(',')]

    device = torch.device(args.device)
    config = yaml.safe_load(open(args.config))
    spect = config['preprocess_params']['spect_params']
    mel_kwargs = {
        'n_fft': spect['n_fft'],
        'win_size': spect.get('win_length', 1024),
        'hop_size': spect.get('hop_length', 256),
        'num_mels': spect.get('n_mels', 80),
        'sampling_rate': config['preprocess_params']['sr'],
        'fmin': spect['fmin'],
        'fmax': None if spect['fmax'] == 'None' else spect['fmax'],
        'center': False,
    }

    # Load encoders
    print('Loading Whisper...', flush=True)
    from transformers import AutoFeatureExtractor, WhisperModel
    wname = config['model_params']['speech_tokenizer']['name']
    whisper = WhisperModel.from_pretrained(wname, torch_dtype=torch.float16).to(device).eval()
    del whisper.decoder
    wfe = AutoFeatureExtractor.from_pretrained(wname)

    print('Loading RMVPE...', flush=True)
    from modules.rmvpe import RMVPE
    rmvpe_path = load_custom_model_from_hf('lj1995/VoiceConversionWebUI', 'rmvpe.pt', None)
    rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)

    print('Loading CAMPPlus...', flush=True)
    from modules.campplus.DTDNN import CAMPPlus
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    cp_path = load_custom_model_from_hf('funasr/campplus', 'campplus_cn_common.bin', config_filename=None)
    campplus.load_state_dict(torch.load(cp_path, map_location='cpu'))
    campplus.eval().to(device)

    dirs = sorted(glob.glob(os.path.join(args.preproc_root, '*')))
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'a_16k.npy'))]
    print(f'Found {len(dirs)} sample dirs', flush=True)

    # Process one dir at a time (simpler, still fast since GPU ops are batched within a sample).
    # For whisper, we do single-sample forward (tests showed batch 8 memory pressure; keep simple).
    t0 = time.time()
    ok = skip = err = 0
    for i, d in enumerate(dirs):
        try:
            # Whisper encoder on a_16k
            a16_path = os.path.join(d, 'a_16k.npy')
            a_whisper_path = os.path.join(d, 'a_whisper.npy')
            if not os.path.exists(a_whisper_path):
                a16 = np.load(a16_path)
                a_t = torch.from_numpy(a16).float().unsqueeze(0)  # (1, T)
                inp = wfe([a_t[0].numpy()], return_tensors='pt',
                          return_attention_mask=True, sampling_rate=16000)
                features = inp.input_features.to(device, dtype=torch.float16)
                with torch.no_grad():
                    out = whisper.encoder(features, head_mask=None,
                                          output_attentions=False, output_hidden_states=False,
                                          return_dict=True)
                S = out.last_hidden_state.to(torch.float32)
                # Trim to real length (same as train.py convention)
                S = S[:, :a_t.size(-1) // 320 + 1]
                np.save(a_whisper_path, S.cpu().numpy()[0])  # shape (T_w, D)

            # Per-shift features
            for s in shifts:
                b44_p = os.path.join(d, f'b_shift{s:+d}_44k.npy')
                b16_p = os.path.join(d, f'b_shift{s:+d}_16k.npy')
                if not os.path.exists(b44_p):
                    continue
                mel_p = os.path.join(d, f'b_shift{s:+d}_mel.npy')
                f0_p  = os.path.join(d, f'b_shift{s:+d}_f0.npy')
                spk_p = os.path.join(d, f'b_shift{s:+d}_spk.npy')
                if all(os.path.exists(p) for p in (mel_p, f0_p, spk_p)):
                    continue

                b44 = torch.from_numpy(np.load(b44_p)).float().unsqueeze(0).to(device)
                b16 = torch.from_numpy(np.load(b16_p)).float().unsqueeze(0).to(device)

                if not os.path.exists(mel_p):
                    mel = mel_spectrogram(b44, **mel_kwargs).squeeze(0).cpu().numpy()  # (n_mels, T)
                    np.save(mel_p, mel.astype(np.float16))

                if not os.path.exists(f0_p):
                    F0 = rmvpe.infer_from_audio_batch(b16).squeeze(0).cpu().numpy()
                    np.save(f0_p, F0.astype(np.float32))

                if not os.path.exists(spk_p):
                    feat = kaldi.fbank(b16, num_mel_bins=80, dither=0, sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)
                    with torch.no_grad():
                        spk = campplus(feat.unsqueeze(0)).squeeze(0).cpu().numpy()  # (192,)
                    np.save(spk_p, spk.astype(np.float32))

            ok += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dirs) - i - 1) / rate
                print(f'[{i + 1}/{len(dirs)}] ok={ok} skip={skip} err={err} '
                      f'rate={rate:.2f}/s eta={eta/60:.1f}m', flush=True)
        except Exception as e:
            err += 1
            if err < 5:
                print(f'ERR at {d}: {e}', flush=True)

    elapsed = time.time() - t0
    print(f'\nFinal: ok={ok} skip={skip} err={err} in {elapsed/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
