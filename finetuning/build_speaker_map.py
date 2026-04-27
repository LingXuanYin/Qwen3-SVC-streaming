# coding=utf-8
"""Build speaker_id mapping from audio_paths across preprocessed datasets."""
import argparse, json, os, re, torch


def parse_speaker(audio_path: str) -> str:
    p = audio_path.replace("\\", "/")
    m = re.search(r"GTSinger_repo/([^/]+)/([^/]+)/", p)
    if m:
        return f"GTS-{m.group(1)}-{m.group(2)}"
    m = re.search(r"vc_training/train/wav/([^/]+)/", p)
    if m:
        return f"VCT-{m.group(1)}"
    return "UNKNOWN-" + os.path.basename(os.path.dirname(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Preprocessed dirs to scan")
    ap.add_argument("--output", required=True, help="Path to save speaker_map.json")
    args = ap.parse_args()

    all_speakers = set()
    for d in args.dirs:
        with open(f"{d}/manifest.json") as f:
            man = json.load(f)
        for item in man:
            feat = torch.load(item["path"], weights_only=True)
            all_speakers.add(parse_speaker(feat["audio_path"]))
        print(f"{d}: total unique so far={len(all_speakers)}")

    # Deterministic ordering: sort speakers
    speaker_map = {sp: i for i, sp in enumerate(sorted(all_speakers))}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output} with {len(speaker_map)} speakers")


if __name__ == "__main__":
    main()
