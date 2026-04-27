"""Mic-to-mic realtime VC simulator with F0/pitch_ref support.

Based on seed-vc's real-time-gui.py custom_infer pattern, extended to:
  - Load singing F0 44k model (not the default speech tiny)
  - Precompute F0 from pitch_ref, align by mic-time when serving each block
  - Simulate mic feed by reading a file at block_time pace and measuring end-to-end
    latency (wall-clock from block ready → output ready)

What we measure:
  T_E2E per block = time.perf_counter at "block arrives" (file read + enqueue) →
                     time.perf_counter when custom_infer returns corresponding output
  Steady-state RTF = T_E2E / block_time (< 1 → can keep up in realtime)
  Underrun count = blocks where T_E2E > block_time (stream starts to fall behind)

Mic input comes from file; we simulate real-time by pacing with time.sleep between
blocks so the loop timing matches wall-clock. Output is accumulated to check
correctness (SOLA crossfade seamlessness).

Not aiming for true audio playback — this is a benchmark, not a GUI.
"""
import os, sys, argparse, glob, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'external/seed-vc')
os.environ.setdefault('HF_HUB_CACHE', './external/seed-vc/checkpoints/hf_cache')

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
import soundfile as sf
import yaml

from modules.commons import recursive_munch, build_model, load_checkpoint
from hf_utils import load_custom_model_from_hf


def load_singing_f0_model(device, config_path, checkpoint_path=None):
    """Variant of real-time-gui.load_models specialized for singing F0 model.

    Loads:
      - DiT f0_44k checkpoint + config (Plachta/Seed-VC default if paths None)
      - BigVGAN v2 44kHz vocoder
      - Whisper-base semantic encoder (singing config default)
      - CAMPPlus speaker encoder
      - RMVPE F0 extractor
    """
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    model_params.dit_type = 'DiT'
    model = build_model(model_params, stage='DiT')
    hop_length = config['preprocess_params']['spect_params']['hop_length']
    sr = config['preprocess_params']['sr']

    if checkpoint_path is None:
        checkpoint_path = load_custom_model_from_hf(
            'Plachta/Seed-VC', config['pretrained_model'], None,
        )
    model, _, _, _ = load_checkpoint(
        model, None, checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for k in model:
        model[k].eval().to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Vocoder (bigvgan)
    from modules.bigvgan import bigvgan
    vocoder = bigvgan.BigVGAN.from_pretrained(model_params.vocoder.name, use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)

    # Whisper
    from transformers import AutoFeatureExtractor, WhisperModel
    wname = model_params.speech_tokenizer.name
    whisper = WhisperModel.from_pretrained(wname, torch_dtype=torch.float16).to(device).eval()
    del whisper.decoder
    wfe = AutoFeatureExtractor.from_pretrained(wname)

    def semantic_fn(waves_16k):
        inp = wfe([waves_16k.squeeze(0).cpu().numpy()],
                  return_tensors='pt', return_attention_mask=True)
        feats = whisper._mask_input_features(
            inp.input_features, attention_mask=inp.attention_mask,
        ).to(device)
        with torch.no_grad():
            out = whisper.encoder(feats.to(whisper.encoder.dtype),
                                  head_mask=None, output_attentions=False,
                                  output_hidden_states=False, return_dict=True)
        S = out.last_hidden_state.to(torch.float32)
        S = S[:, :waves_16k.size(-1) // 320 + 1]
        return S

    # CAMPPlus
    from modules.campplus.DTDNN import CAMPPlus
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    cp = load_custom_model_from_hf('funasr/campplus', 'campplus_cn_common.bin', config_filename=None)
    campplus.load_state_dict(torch.load(cp, map_location='cpu'))
    campplus.eval().to(device)

    # RMVPE
    from modules.rmvpe import RMVPE
    rmvpe_path = load_custom_model_from_hf('lj1995/VoiceConversionWebUI', 'rmvpe.pt', None)
    rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)

    from modules.audio import mel_spectrogram
    mel_args = {
        'n_fft': config['preprocess_params']['spect_params']['n_fft'],
        'win_size': config['preprocess_params']['spect_params']['win_length'],
        'hop_size': hop_length,
        'num_mels': config['preprocess_params']['spect_params']['n_mels'],
        'sampling_rate': sr,
        'fmin': config['preprocess_params']['spect_params'].get('fmin', 0),
        'fmax': None if config['preprocess_params']['spect_params'].get('fmax', 'None') == 'None' else 8000,
        'center': False,
    }

    return {
        'model': model, 'semantic_fn': semantic_fn, 'vocoder': vocoder,
        'campplus': campplus, 'rmvpe': rmvpe,
        'sr': sr, 'hop_length': hop_length, 'mel_args': mel_args,
    }


class RealtimeSession:
    """One realtime session: fixed timbre_ref + pitch_ref, streaming mic input."""

    def __init__(self, model_set, timbre_ref_path, pitch_ref_path, device,
                 block_time=0.25, extra_time_ce=2.5, extra_time=0.5,
                 extra_time_right=0.02, crossfade_time=0.04,
                 max_prompt_length=3.0, diffusion_steps=10, cfg_rate=0.7,
                 fp16=True, use_source_voicing=False):
        self.use_source_voicing = use_source_voicing
        self.ms = model_set
        self.device = device
        self.sr = model_set['sr']
        self.hop_length = model_set['hop_length']
        self.block_time = block_time
        self.extra_time_ce = extra_time_ce
        self.extra_time = extra_time  # DiT extra (<=ce)
        self.extra_time_right = extra_time_right
        self.crossfade_time = crossfade_time
        self.max_prompt_length = max_prompt_length
        self.diffusion_steps = diffusion_steps
        self.cfg_rate = cfg_rate
        self.fp16 = fp16

        self.zc = self.sr // 50  # 50 Hz frame base
        self.block_frame = int(round(block_time * self.sr / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = int(round(crossfade_time * self.sr / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(extra_time_ce * self.sr / self.zc)) * self.zc
        self.extra_frame_right = int(round(extra_time_right * self.sr / self.zc)) * self.zc

        # Sliding input buffer
        total_frames = (self.extra_frame + self.crossfade_frame +
                        self.sola_search_frame + self.block_frame + self.extra_frame_right)
        self.input_wav = torch.zeros(total_frames, device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(320 * total_frames // self.zc, device=device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)

        # Fade windows
        self.fade_in_window = (torch.sin(0.5 * np.pi *
            torch.linspace(0.0, 1.0, self.sola_buffer_frame, device=device)) ** 2)
        self.fade_out_window = 1.0 - self.fade_in_window

        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc

        # Precompute prompt from timbre_ref
        self._setup_timbre_prompt(timbre_ref_path)
        # Precompute pit F0 from pitch_ref (100 Hz frame rate)
        self._setup_pitch_f0(pitch_ref_path)

    def _setup_timbre_prompt(self, timbre_ref_path):
        """Precompute prompt_condition, mel2, style2 from timbre_ref (first max_prompt_length sec)."""
        wav, _ = librosa.load(timbre_ref_path, sr=self.sr)
        wav = wav[:int(self.sr * self.max_prompt_length)]
        wav_t = torch.from_numpy(wav).float().to(self.device)
        ori_16k = torchaudio.functional.resample(wav_t, self.sr, 16000)
        # F0_ori for prompt
        F0_ori = self.ms['rmvpe'].infer_from_audio(ori_16k, thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
        self.F0_ori = F0_ori
        # Content embedding for prompt
        S_ori = self.ms['semantic_fn'](ori_16k.unsqueeze(0))
        feat2 = kaldi.fbank(ori_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        self.style2 = self.ms['campplus'](feat2.unsqueeze(0))
        from modules.audio import mel_spectrogram
        self.mel2 = mel_spectrogram(wav_t.unsqueeze(0), **self.ms['mel_args'])
        tgt2_len = torch.LongTensor([self.mel2.size(2)]).to(self.device)
        # IMPORTANT: pass f0=None for prompt, so prompt doesn't bias output F0 toward
        # timbre's natural range. Only the per-block cond carries F0 (from pitch_ref).
        # In non-stream convert_voice the source-side cond is long (covers full source),
        # which dominates over prompt's F0 contribution. In streaming each per-block cond
        # is ~1s, short enough that prompt F0 dominates — observed as output F0 pulled
        # toward timbre_ref's F0 range rather than tracking pitch_ref.
        self.prompt_condition = self.ms['model'].length_regulator(
            S_ori, ylens=tgt2_len, n_quantizers=3, f0=None,
        )[0]

    def set_expected_source_duration(self, dur_sec):
        """If set, pre-stretch pit F0 linearly to cover this many seconds
        (matches non-stream architecture that gives best pit-tracking). When not
        set (None), fallback to cycle mode (treats pit as a repeating template).
        """
        self.expected_src_dur = float(dur_sec) if dur_sec else None
        if self.expected_src_dur is not None:
            target_frames = int(round(self.expected_src_dur * 100))
            # Linear-resize the already-filled pit_f0 to target_frames (nearest, like length_regulator)
            orig = self.pit_f0_np
            src_idx = np.linspace(0, len(orig) - 1, target_frames).astype(np.float32)
            # Use nearest-neighbor (preserve pit values; no new interp across voiced frames)
            nn = np.clip(np.round(src_idx).astype(int), 0, len(orig) - 1)
            self.pit_f0_stretched = orig[nn]
            print(f'  pit F0 pre-stretched from {len(orig)/100:.2f}s to {self.expected_src_dur:.2f}s ({target_frames} frames)')
        else:
            self.pit_f0_stretched = None

    def _setup_pitch_f0(self, pitch_ref_path):
        """Extract full F0 from pitch_ref at 100 Hz frame rate.

        Fill unvoiced (<=50Hz) frames via linear interpolation between voiced
        neighbors. Rationale: model's F0 condition interprets 0 as "unvoiced" and
        refuses to generate voiced output even if the source content is voiced.
        Cycling pit means its unvoiced gaps recur every pit_duration; if source
        is voiced at those moments, model drops out. Continuous voiced F0
        contour avoids this — the voicing decision should come from source
        content (Whisper features already encode phoneme voicing), not from pit.
        """
        wav, _ = librosa.load(pitch_ref_path, sr=16000)
        wav_t = torch.from_numpy(wav).float().to(self.device)
        f0 = self.ms['rmvpe'].infer_from_audio(wav_t, thred=0.03).astype(np.float32)
        # Linear interp across unvoiced gaps
        voiced_idx = np.where(f0 > 50)[0]
        if len(voiced_idx) >= 2:
            all_idx = np.arange(len(f0))
            f0_filled = np.interp(all_idx, voiced_idx, f0[voiced_idx]).astype(np.float32)
        else:
            f0_filled = np.full_like(f0, 220.0)  # fallback if pit has no voicing
        self.pit_f0 = torch.from_numpy(f0_filled).to(self.device)
        self.pit_f0_np = f0_filled
        self.pit_duration_sec = len(f0_filled) / 100.0
        voicing_rate = float((f0 > 50).mean())
        print(f'  pit F0: orig voicing={voicing_rate:.2%}, unvoiced frames filled via linear interp')
        self.pit_f0_stretched = None  # set via set_expected_source_duration()
        self.expected_src_dur = None

    def get_block_f0(self, mic_time_start_sec, mic_time_end_sec):
        """Return F0 slice for given mic-time window.

        Two modes:
          - stretch mode: pit F0 pre-stretched to expected_src_dur. mic_time linearly
            maps to stretched pit position. Matches non-stream architecture.
          - cycle mode (fallback): pit F0 repeats every pit_duration, sampled at
            absolute mic_time.
        """
        n_frames_needed = int(round((mic_time_end_sec - mic_time_start_sec) * 100))
        if self.pit_f0_stretched is not None:
            pit_arr = self.pit_f0_stretched
            T = len(pit_arr)
            start = int(round(mic_time_start_sec * 100))
            # Clip into [0, T), clamp at end (don't overflow)
            idxs = np.arange(n_frames_needed) + start
            idxs = np.clip(idxs, 0, T - 1)
            out = pit_arr[idxs]
        else:
            pit_arr = self.pit_f0_np
            T = len(pit_arr)
            if T == 0:
                return torch.zeros(n_frames_needed, device=self.device).unsqueeze(0)
            start_idx = int(round(mic_time_start_sec * 100)) % T
            out = np.empty(n_frames_needed, dtype=np.float32)
            pos = start_idx; i = 0
            while i < n_frames_needed:
                remain = n_frames_needed - i
                avail = T - pos
                take = min(remain, avail)
                out[i:i + take] = pit_arr[pos:pos + take]
                i += take
                pos = (pos + take) % T
        return torch.from_numpy(out.astype(np.float32)).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def process_block(self, new_block_44k, mic_time_end_sec, diag=None):
        """Process one block of new audio (length = self.block_frame), return output block.
        mic_time_end_sec = the absolute mic time at the end of this new block.
        """
        assert len(new_block_44k) == self.block_frame
        # Slide input_wav
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(new_block_44k).to(self.device)
        # Slide input_wav_res (16k)
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        # Resample last 2*zc + block_frame at 44k, take 16k corresponding segment
        src_end = self.input_wav.shape[0]
        src_start = src_end - self.block_frame - 2 * self.zc
        seg_44k = self.input_wav[src_start:src_end].cpu().numpy()
        seg_16k = librosa.resample(seg_44k, orig_sr=self.sr, target_sr=16000)[320:]
        self.input_wav_res[-self.block_frame_16k:] = torch.from_numpy(seg_16k).to(self.device)[: self.block_frame_16k]

        # Build S_alt (content) from full rolling 16k window
        S_alt = self.ms['semantic_fn'](self.input_wav_res.unsqueeze(0))
        # Apply ce_dit_difference: skip leftmost (extra_time_ce - extra_time) seconds of frames
        ce_dit_diff_sec = self.extra_time_ce - self.extra_time
        ce_dit_frames = int(ce_dit_diff_sec * 50)
        S_alt = S_alt[:, ce_dit_frames:]

        # F0 for the window covered by S_alt post-trim.
        # input_wav_res covers [mic_time_end - (extra_time_ce + block + extra_right), mic_time_end + extra_right]
        # at 16kHz. Full whisper frames ~= window_sec * 50. After ce_dit trim we drop
        # `ce_dit_diff_sec` worth of frames from front. So S_alt post-trim covers:
        #   [mic_time_end - (extra_time + block + extra_right), mic_time_end + extra_right]
        # which is extra_time + block + extra_right = extra_time + block_time + extra_time_right seconds
        t_start = mic_time_end_sec - self.block_time - self.extra_time - self.extra_time_right
        t_end   = mic_time_end_sec + self.extra_time_right
        F0_alt = self.get_block_f0(t_start, t_end)  # (1, T_f0) — pit values, all voiced (filled)

        # Mask pit F0 with source's voicing pattern. Source F0 extracted from the same
        # rolling 16k window. This makes output inherit source's voiced/unvoiced timing
        # (so user silences stay silent) while pitch values follow pit_ref.
        if self.use_source_voicing:
            # Extract F0 from the tail portion of input_wav_res matching F0_alt's time range.
            # F0_alt covers (extra_time + block_time + 2*extra_time_right) seconds, starting
            # from (mic_time_end - block - extra_time - extra_right).
            win_sec = self.block_time + self.extra_time + 2 * self.extra_time_right
            win_samples_16k = int(round(win_sec * 16000))
            src_seg_16k = self.input_wav_res[-win_samples_16k:]
            with torch.no_grad():
                src_f0_np = self.ms['rmvpe'].infer_from_audio(src_seg_16k, thred=0.03)
            src_voicing = src_f0_np > 50  # (T_src_f0,) bool
            # Resize voicing mask to F0_alt length (both nominally 100Hz, small off-by-one)
            T_f0 = F0_alt.size(1)
            if len(src_voicing) != T_f0:
                idx = np.clip(np.round(np.linspace(0, len(src_voicing) - 1, T_f0)).astype(int),
                              0, len(src_voicing) - 1)
                src_voicing = src_voicing[idx]
            mask = torch.from_numpy(src_voicing.astype(np.float32)).to(self.device).unsqueeze(0)
            F0_alt = F0_alt * mask

        target_len_frames = int((self.skip_head + self.return_length + self.skip_tail - ce_dit_frames) / 50 * self.sr / self.hop_length)
        target_lengths = torch.LongTensor([target_len_frames]).to(self.device)

        cond = self.ms['model'].length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=F0_alt,
        )[0]
        cat_cond = torch.cat([self.prompt_condition, cond], dim=1)
        # Pin CFM init noise to same seed every block: makes per-block output depend
        # only on condition (content + F0). Without this, each block samples fresh
        # z ~ N(0,1), making long-term F0 trajectory drift randomly even when condition
        # is identical — observed as systematic negative pearson with pit.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_mel = self.ms['model'].cfm.inference(
                cat_cond,
                torch.LongTensor([cat_cond.size(1)]).to(self.device),
                self.mel2, self.style2, None,
                n_timesteps=self.diffusion_steps,
                inference_cfg_rate=self.cfg_rate,
            )
            vc_mel = vc_mel[:, :, self.mel2.size(-1):]
            vc_wave = self.ms['vocoder'](vc_mel).squeeze()
        vc_wave = vc_wave.float()  # cast back from fp16 for SOLA conv1d

        # Extract return segment
        output_len = self.return_length * self.sr // 50
        tail_len = self.skip_tail * self.sr // 50
        if tail_len == 0:
            infer_wav = vc_wave[-output_len:]
        else:
            infer_wav = vc_wave[-output_len - tail_len: -tail_len]

        # Fixed-offset crossfade (SOLA search unreliable with per-block random CFM noise —
        # search peaks on noise correlations rather than real phase alignment, resulting
        # in ~random ms-level shifts per block that degrade F0 coherence and create
        # metallic artifacts at boundaries). Use offset=0 and rely on fade windows.
        sola_offset = 0
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:self.sola_buffer_frame] = (
            infer_wav[:self.sola_buffer_frame] * self.fade_in_window
            + self.sola_buffer * self.fade_out_window
        )
        # Save new sola_buffer for next call
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]
        # Return only the block portion
        out_blk = infer_wav[:self.block_frame].cpu().numpy()
        if diag is not None:
            diag.append({
                'mic_t': mic_time_end_sec,
                'out_rms_db': 20 * float(np.log10(np.sqrt(np.mean(out_blk ** 2)) + 1e-9)),
                'in_rms_db': 20 * float(np.log10(np.sqrt(np.mean(new_block_44k ** 2)) + 1e-9)),
                'f0_voiced_frames': int((F0_alt[0] > 50).sum().item()),
                'f0_total_frames': int(F0_alt.size(1)),
                'sola_offset': sola_offset,
                'mel_energy': float(vc_mel.abs().mean().item()),
            })
        return out_blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='output/longform_10m/src_long.wav',
                    help='Source wav (simulated as mic input, read at real-time pace)')
    ap.add_argument('--duration_sec', type=float, default=60.0,
                    help='How many seconds to simulate (cap, useful for quick tests)')
    ap.add_argument('--block_time', type=float, default=0.25)
    ap.add_argument('--extra_time_ce', type=float, default=2.5)
    ap.add_argument('--extra_time', type=float, default=0.5)
    ap.add_argument('--extra_time_right', type=float, default=0.02)
    ap.add_argument('--diffusion_steps', type=int, default=10,
                    help='Lower is faster; realtime wants <= 10')
    ap.add_argument('--cfg_rate', type=float, default=0.7)
    ap.add_argument('--config', default='external/seed-vc/configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')
    ap.add_argument('--sleep_to_realtime', action='store_true',
                    help='Sleep to match real-time pace (else run as fast as possible)')
    ap.add_argument('--pit_stretch_to', type=float, default=0.0,
                    help='If >0: pre-stretch pit F0 to this duration (seconds). Matches '
                         'non-stream architecture that tracks pit contour linearly instead '
                         'of cycling. Typically set to duration_sec.')
    ap.add_argument('--source_voicing', action='store_true',
                    help='Use source mic F0 voicing mask (voiced when user speaks/sings) + '
                         'pit F0 values for voiced frames. Keeps user silences silent.')
    ap.add_argument('--out', default='output/mic_realtime_out.wav')
    args = ap.parse_args()

    device = torch.device('cuda:0')
    tim = sorted(glob.glob('L:/DATASET/vc_training/train/wav/SSB0005/*.wav'))[0]
    pit = sorted(glob.glob('L:/DATASET/GTSinger_repo/English/EN-Alto-1/*/all is found/Control_Group/*.wav'))[0]
    print(f'src={args.source}\ntim={tim}\npit={pit}', flush=True)

    print('Loading singing F0 model set...', flush=True)
    model_set = load_singing_f0_model(device, args.config)

    session = RealtimeSession(
        model_set, tim, pit, device,
        block_time=args.block_time, extra_time_ce=args.extra_time_ce,
        extra_time=args.extra_time, extra_time_right=args.extra_time_right,
        diffusion_steps=args.diffusion_steps, cfg_rate=args.cfg_rate,
        use_source_voicing=args.source_voicing,
    )
    print(f'block_frame={session.block_frame} ({args.block_time}s) '
          f'sr={session.sr} extra_ce={args.extra_time_ce}s extra_right={args.extra_time_right}s')
    if args.pit_stretch_to > 0:
        session.set_expected_source_duration(args.pit_stretch_to)

    # Load source file for mic simulation
    src_wav, src_sr = librosa.load(args.source, sr=session.sr)
    total_samples = min(int(args.duration_sec * session.sr), len(src_wav))
    n_blocks = total_samples // session.block_frame
    print(f'Simulating {n_blocks} blocks of {args.block_time}s each '
          f'({n_blocks * args.block_time:.1f}s mic audio)', flush=True)

    # Buffer-fill warmup: feed enough source blocks to fill the rolling context.
    # After extra_time_ce seconds of feed, input_wav is fully populated with real
    # audio (not zeros), so whisper encoder sees proper context from block 0.
    # Also serves as CUDA warmup. We throw away these outputs.
    n_warmup_blocks = int(np.ceil(session.extra_time_ce / args.block_time)) + 2
    n_warmup_blocks = min(n_warmup_blocks, len(src_wav) // session.block_frame)
    print(f'\n[warmup {n_warmup_blocks} blocks — filling buffer + CUDA warmup]')
    warmup_offset_samples = 0
    for i in range(n_warmup_blocks):
        blk = src_wav[i * session.block_frame:(i + 1) * session.block_frame]
        if len(blk) < session.block_frame:
            break
        # NOTE: during warmup we advance mic_time but keep it separate from measurement
        session.process_block(blk, (i + 1) * args.block_time)
    warmup_offset_samples = n_warmup_blocks * session.block_frame
    warmup_offset_time = n_warmup_blocks * args.block_time
    # Do NOT reset buffers — keep the warm state.

    latencies = []
    output_chunks = []
    diag_log = []
    t_start = time.perf_counter()
    mic_time = 0.0
    # Start reading from where warmup left off to avoid double-feeding the same samples
    for i in range(n_blocks):
        src_idx = warmup_offset_samples + i * session.block_frame
        blk = src_wav[src_idx: src_idx + session.block_frame]
        if len(blk) < session.block_frame:
            break
        mic_time_end = warmup_offset_time + (i + 1) * args.block_time
        if args.sleep_to_realtime:
            # Pace ingestion to real-time: block i is available at time i*block_time
            target_wall = t_start + i * args.block_time
            now = time.perf_counter()
            if now < target_wall:
                time.sleep(target_wall - now)
        t_block_ready = time.perf_counter()
        out = session.process_block(blk, mic_time_end, diag=diag_log)
        t_block_done = time.perf_counter()
        lat = t_block_done - t_block_ready
        latencies.append(lat)
        output_chunks.append(out)
        mic_time = mic_time_end
        if i < 5 or i % 20 == 0 or i == n_blocks - 1:
            rtf = lat / args.block_time
            print(f'  block {i:4d}: mic_t={mic_time_end:6.2f}s  latency={lat*1000:.0f}ms  '
                  f'chunkRTF={rtf:.3f}', flush=True)

    total_wall = time.perf_counter() - t_start
    lats = np.array(latencies)
    print(f'\n=== Summary: {len(lats)} blocks, block_time={args.block_time}s, diff_steps={args.diffusion_steps} ===')
    print(f'End-to-end per-block latency: mean={lats.mean()*1000:.0f}ms '
          f'median={np.median(lats)*1000:.0f}ms p95={np.percentile(lats, 95)*1000:.0f}ms '
          f'max={lats.max()*1000:.0f}ms min={lats.min()*1000:.0f}ms')
    print(f'Chunk RTF = latency / block_time: mean={lats.mean()/args.block_time:.3f} p95={np.percentile(lats, 95)/args.block_time:.3f}')
    n_underrun = int((lats > args.block_time).sum())
    print(f'Blocks over block_time (underrun candidates): {n_underrun}/{len(lats)} ({n_underrun*100/len(lats):.1f}%)')
    print(f'User-perceived E2E latency ≈ block_time + mean(process) + extra_time_right')
    e2e = args.block_time + lats.mean() + args.extra_time_right
    print(f'  = {args.block_time*1000:.0f} + {lats.mean()*1000:.0f} + {args.extra_time_right*1000:.0f} = {e2e*1000:.0f}ms')

    # Save output
    if output_chunks:
        full = np.concatenate(output_chunks)
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        sf.write(args.out, full.astype(np.float32), session.sr)
        print(f'Saved {args.out}: {len(full)/session.sr:.2f}s')

    # Save diag log and find weak blocks
    if diag_log:
        import json as _json
        diag_path = args.out.replace('.wav', '_diag.json')
        with open(diag_path, 'w') as f:
            _json.dump(diag_log, f, indent=2)
        # Summary of weak blocks
        weak = [d for d in diag_log if d['out_rms_db'] < -35]
        print(f'\n=== Block-level diag (saved {diag_path}) ===')
        print(f'Weak blocks (output RMS < -35dB): {len(weak)}/{len(diag_log)} ({len(weak)*100/len(diag_log):.1f}%)')
        if weak:
            import numpy as _np
            in_rms_weak = [d['in_rms_db'] for d in weak]
            voicing_weak = [d['f0_voiced_frames']/max(d['f0_total_frames'],1) for d in weak]
            print(f'  Weak blocks: input RMS mean={_np.mean(in_rms_weak):.1f}dB  pit F0 voicing mean={_np.mean(voicing_weak):.2f}')
            # Compare to healthy blocks
            strong = [d for d in diag_log if d['out_rms_db'] >= -20]
            if strong:
                in_rms_strong = [d['in_rms_db'] for d in strong]
                voicing_strong = [d['f0_voiced_frames']/max(d['f0_total_frames'],1) for d in strong]
                print(f'  Strong blocks (out>-20dB): input RMS mean={_np.mean(in_rms_strong):.1f}dB  pit F0 voicing mean={_np.mean(voicing_strong):.2f}')


if __name__ == '__main__':
    main()
