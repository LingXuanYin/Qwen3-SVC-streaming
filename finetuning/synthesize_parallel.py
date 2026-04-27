# coding=utf-8
"""Synthesize cross-speaker parallel data via Qwen3-TTS voice cloning.

For each source utterance (text_A, original audio_A), use Qwen3-TTS zero-shot
voice clone to generate audio_AB = TTS(text_A, speaker_ref=audio_B) where B is
a DIFFERENT speaker. This produces genuine cross-speaker parallel tuples:
(audio_A, audio_AB) where audio_A and audio_AB have SAME content but DIFFERENT
speaker. Training then learns the speaker-swap behaviour.

Output layout:
  svc_synth_parallel/
    manifest.jsonl           # one line per synthesized sample
      {text, source_path, speaker_ref_path, synth_path, src_speaker, ref_speaker}
    synth/
      NNNNNN.wav             # generated audio (24kHz mono float32)
"""
import argparse, json, os, random, sys, time
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def load_aishell_texts(content_path, max_len=60):
    """Parse AISHELL content.txt: 'SSB00050001.wav\\t<chinese text with pinyin>'."""
    samples = []
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            wav_name = parts[0]
            # The second column mixes Chinese + pinyin; keep only Chinese characters
            pieces = parts[1].split()
            chinese_only = ''.join(p for p in pieces if any('一' <= c <= '鿿' for c in p))
            if not chinese_only or len(chinese_only) > max_len or len(chinese_only) < 5:
                continue
            speaker = wav_name[:7]  # e.g. SSB0005
            samples.append({
                'wav_name': wav_name,
                'text': chinese_only,
                'speaker': speaker,
            })
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aishell_content', default='L:/DATASET/vc_training/train/content.txt')
    ap.add_argument('--aishell_wav_root', default='L:/DATASET/vc_training/train/wav')
    ap.add_argument('--output_dir', default='L:/DATASET/svc_synth_parallel')
    ap.add_argument('--model_path', default='Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
    ap.add_argument('--num_texts', type=int, default=600, help='How many distinct texts to synthesize')
    ap.add_argument('--refs_per_text', type=int, default=6, help='How many speaker refs per text')
    ap.add_argument('--num_ref_speakers', type=int, default=40, help='Distinct AISHELL speakers to use as refs')
    ap.add_argument('--batch_size', type=int, default=4, help='TTS batch size (higher = faster but more VRAM)')
    ap.add_argument('--text_start', type=int, default=0, help='Text index range start (inclusive)')
    ap.add_argument('--text_end', type=int, default=-1, help='Text index range end (exclusive); -1 means num_texts')
    ap.add_argument('--shard_id', type=str, default='', help='Shard tag used in manifest filename and wav filenames')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_chunks', type=int, default=0, help='Stop after this many generations (0=no limit)')
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.join(args.output_dir, 'synth'), exist_ok=True)
    tag = f'_{args.shard_id}' if args.shard_id else ''
    manifest_path = os.path.join(args.output_dir, f'manifest{tag}.jsonl')

    # Resume: how many already done?
    done = 0
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            done = sum(1 for _ in f)
        print(f"Resuming: {done} samples already in manifest", flush=True)

    print(f"Loading AISHELL transcripts...", flush=True)
    all_samples = load_aishell_texts(args.aishell_content)
    # Group by speaker
    by_speaker = {}
    for s in all_samples:
        by_speaker.setdefault(s['speaker'], []).append(s)
    speakers = sorted(by_speaker.keys())
    print(f"AISHELL: {len(all_samples)} utterances, {len(speakers)} unique speakers", flush=True)

    # Pick ref speakers (balanced sampling: one long utterance from each)
    ref_speakers = random.sample(speakers, min(args.num_ref_speakers, len(speakers)))
    ref_audios = {}
    for sp in ref_speakers:
        # Pick the utterance with median text length (likely decent quality)
        utts = by_speaker[sp]
        utts_sorted = sorted(utts, key=lambda x: len(x['text']))
        pick = utts_sorted[len(utts_sorted) // 2]
        ref_audios[sp] = os.path.join(args.aishell_wav_root, sp, pick['wav_name']).replace('\\', '/')
    print(f"Selected {len(ref_audios)} reference speakers", flush=True)

    # Pick source texts (different from ref_speakers to ensure cross-speaker parallel)
    text_pool = [s for s in all_samples if s['speaker'] not in set(ref_speakers)]
    random.shuffle(text_pool)
    all_texts = text_pool[:args.num_texts]
    end_idx = args.num_texts if args.text_end < 0 else args.text_end
    source_texts = all_texts[args.text_start:end_idx]
    print(f"Shard {args.shard_id or '(full)'}: texts [{args.text_start}:{end_idx}], {len(source_texts)} entries, "
          f"{len(set(s['speaker'] for s in source_texts))} src speakers", flush=True)

    print(f"Loading Qwen3-TTS on {args.device}...", flush=True)
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    m = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=dtype, device_map=args.device)

    # Open manifest in append mode
    f_man = open(manifest_path, 'a', encoding='utf-8')
    total = 0
    t0 = time.time()

    # Build a flat queue of (text, ref_sp, src_wav_path, ref_path, src_speaker) pending jobs
    queue = []
    for src in source_texts:
        candidates = [sp for sp in ref_speakers if sp != src['speaker']]
        chosen_refs = random.sample(candidates, min(args.refs_per_text, len(candidates)))
        src_wav_path = os.path.join(args.aishell_wav_root, src['speaker'], src['wav_name']).replace('\\', '/')
        for ref_sp in chosen_refs:
            queue.append({
                'text': src['text'],
                'src_speaker': src['speaker'],
                'ref_speaker': ref_sp,
                'src_wav_path': src_wav_path,
                'ref_path': ref_audios[ref_sp],
            })
    print(f'Total queue: {len(queue)} generations, batch_size={args.batch_size}', flush=True)

    # Filenames embed the shard tag so two processes don't clash
    pending = []
    for j, job in enumerate(queue):
        out_path = os.path.join(args.output_dir, 'synth', f'{args.shard_id}_{j:06d}.wav' if args.shard_id
                                 else f'{done + len(pending):06d}.wav').replace('\\', '/')
        if os.path.exists(out_path):
            total += 1
            continue
        job['out_path'] = out_path
        pending.append(job)

    # Batched generation
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        texts = [b['text'] for b in batch]
        refs = [b['ref_path'] for b in batch]
        try:
            with torch.inference_mode():
                wavs, fs = m.generate_voice_clone(
                    text=texts,
                    ref_audio=refs,
                    language=['chinese'] * len(batch),
                    x_vector_only_mode=[True] * len(batch),
                    non_streaming_mode=True,
                )
        except Exception as e:
            print(f"  FAIL batch_start={batch_start}: {e}", flush=True)
            continue

        for b, wav in zip(batch, wavs):
            if wav.ndim > 1: wav = wav[0]
            max_len = int(fs * 10)
            if wav.shape[0] > max_len: wav = wav[:max_len]
            sf.write(b['out_path'], wav.astype(np.float32), fs)
            entry = {
                'idx': done + total,
                'text': b['text'],
                'source_path': b['src_wav_path'],
                'speaker_ref_path': b['ref_path'],
                'synth_path': b['out_path'],
                'src_speaker': b['src_speaker'],
                'ref_speaker': b['ref_speaker'],
                'sample_rate': int(fs),
                'duration': float(wav.shape[0] / fs),
            }
            f_man.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f_man.flush()
            total += 1
            if args.max_chunks and total >= args.max_chunks:
                break

        if batch_start // args.batch_size % 10 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1)
            eta = (len(pending) - batch_start) / max(rate, 0.1)
            print(f"  batch {batch_start}/{len(pending)} | generated={total} "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min)", flush=True)
        if args.max_chunks and total >= args.max_chunks:
            break

    f_man.close()
    print(f"Done: {total} samples generated", flush=True)


if __name__ == '__main__':
    main()
