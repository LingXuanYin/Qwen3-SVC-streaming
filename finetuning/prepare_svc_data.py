# coding=utf-8
"""Prepare SVC training data from audio directories.

Supports:
1. Self-reconstruction mode: source == target (same audio)
2. Cross-speaker mode: random source/target from different speakers
3. Pitch augmentation: generates additional entries with pitch-shifted references

Usage:
    python finetuning/prepare_svc_data.py \
        --audio_dirs L:/DATASET/vc_training/train/wav L:/DATASET/GTSinger_Chinese/Chinese \
        --output_jsonl L:/DATASET/svc_train.jsonl \
        --pitch_augment \
        --max_duration 30.0 \
        --min_duration 1.0
"""

import argparse
import json
import os
import random

import librosa
import numpy as np
import soundfile as sf


def find_wav_files(root_dir: str, max_duration: float = 30.0, min_duration: float = 1.0):
    """Recursively find wav files grouped by speaker directory."""
    speakers = {}
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.wav'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                duration = librosa.get_duration(path=fpath)
                file_sr = librosa.get_samplerate(fpath)
                if duration < min_duration or duration > max_duration:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            # Use parent directory as speaker ID
            rel = os.path.relpath(dirpath, root_dir)
            parts = rel.replace('\\', '/').split('/')
            # For AISHELL-3: wav/SSB0005/file.wav → speaker = SSB0005
            # For GTSinger: ZH-Alto-1/Breathy/song/Group/file.wav → speaker = ZH-Alto-1
            speaker = parts[0] if parts[0] != 'wav' else (parts[1] if len(parts) > 1 else parts[0])

            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append({
                'path': fpath.replace('\\', '/'),
                'duration': duration,
                'sr': file_sr,
            })

    print(f"Found {sum(len(v) for v in speakers.values())} files from {len(speakers)} speakers (skipped {skipped})")
    return speakers


def generate_self_reconstruction(speakers: dict) -> list:
    """Generate self-reconstruction pairs (source == target)."""
    entries = []
    for speaker, files in speakers.items():
        for f in files:
            entries.append({
                'source_audio': f['path'],
                'target_audio': f['path'],
                # pitch_audio omitted → defaults to target
            })
    return entries


def generate_cross_speaker(speakers: dict, pairs_per_speaker: int = 50) -> list:
    """Generate cross-speaker pairs for timbre conversion training."""
    entries = []
    speaker_list = list(speakers.keys())
    if len(speaker_list) < 2:
        return entries

    for spk in speaker_list:
        other_spks = [s for s in speaker_list if s != spk]
        for _ in range(min(pairs_per_speaker, len(speakers[spk]))):
            source_file = random.choice(speakers[spk])
            target_spk = random.choice(other_spks)
            target_file = random.choice(speakers[target_spk])
            entries.append({
                'source_audio': source_file['path'],
                'target_audio': target_file['path'],
                'pitch_audio': target_file['path'],
            })
    return entries


def generate_pitch_augmented(speakers: dict, shifts=(-5, -3, -1, 1, 3, 5, 7, 12),
                              samples_per_shift: int = 500) -> list:
    """Generate entries with pitch shift metadata for training augmentation.

    NOTE: Pitch shift is applied during training via F0 manipulation,
    not by physically shifting the audio files.
    """
    entries = []
    all_files = [(spk, f) for spk, files in speakers.items() for f in files]

    for shift in shifts:
        sampled = random.sample(all_files, min(samples_per_shift, len(all_files)))
        for spk, f in sampled:
            entries.append({
                'source_audio': f['path'],
                'target_audio': f['path'],
                'pitch_shift': shift,
            })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Prepare SVC training JSONL")
    parser.add_argument('--audio_dirs', nargs='+', required=True, help='Directories containing wav files')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Output JSONL path')
    parser.add_argument('--max_duration', type=float, default=30.0)
    parser.add_argument('--min_duration', type=float, default=1.0)
    parser.add_argument('--pitch_augment', action='store_true', help='Add pitch-augmented entries')
    parser.add_argument('--cross_speaker', action='store_true', help='Add cross-speaker pairs')
    parser.add_argument('--cross_speaker_pairs', type=int, default=50, help='Cross-speaker pairs per speaker')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mini', action='store_true', help='Create mini dataset for overfitting test')
    parser.add_argument('--mini_size', type=int, default=20, help='Mini dataset size')
    args = parser.parse_args()

    random.seed(args.seed)

    # Scan all audio directories
    all_speakers = {}
    for audio_dir in args.audio_dirs:
        print(f"Scanning {audio_dir}...")
        speakers = find_wav_files(audio_dir, args.max_duration, args.min_duration)
        for spk, files in speakers.items():
            if spk in all_speakers:
                all_speakers[spk].extend(files)
            else:
                all_speakers[spk] = files

    print(f"\nTotal: {sum(len(v) for v in all_speakers.values())} files, {len(all_speakers)} speakers")

    if args.mini:
        # Create a tiny dataset for overfitting test
        entries = []
        all_files = [(spk, f) for spk, files in all_speakers.items() for f in files]
        sampled = random.sample(all_files, min(args.mini_size, len(all_files)))
        for spk, f in sampled:
            entries.append({
                'source_audio': f['path'],
                'target_audio': f['path'],
            })
        print(f"Mini dataset: {len(entries)} entries")
    else:
        # Self-reconstruction (main training signal)
        entries = generate_self_reconstruction(all_speakers)
        print(f"Self-reconstruction: {len(entries)} entries")

        # Cross-speaker pairs
        if args.cross_speaker:
            cross = generate_cross_speaker(all_speakers, args.cross_speaker_pairs)
            entries.extend(cross)
            print(f"Cross-speaker: {len(cross)} entries")

        # Pitch augmentation
        if args.pitch_augment:
            pitch = generate_pitch_augmented(all_speakers)
            entries.extend(pitch)
            print(f"Pitch augmented: {len(pitch)} entries")

    # Shuffle and save
    random.shuffle(entries)
    os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(entries)} entries to {args.output_jsonl}")


if __name__ == '__main__':
    main()
