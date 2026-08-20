from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from discql.audio_analysis import analyze_track

# Mirrors config.py's default (CACHE_DIR / "essentia_models"), without going
# through load_config() - that also handles the Discogs token prompt/config
# file, neither of which this standalone tool needs.
DEFAULT_MODELS_DIR = Path(
    os.environ.get(
        "ESSENTIA_MODELS_DIR",
        str(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "discql" / "essentia_models"),
    )
).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline (BPM/key via Essentia, mood/genre via "
        "Discogs-EffNet) on a single local audio file and print the results."
    )
    parser.add_argument("audio_path", type=Path, help="Path to a local audio file (any ffmpeg-readable format)")
    parser.add_argument(
        "--waveform-out",
        type=Path,
        default=None,
        help="Where to write the waveform PNG (default: <audio file>.waveform.png next to the input)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory for cached Essentia model files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a formatted report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.audio_path.is_file():
        sys.exit(f"No such file: {args.audio_path}")

    waveform_out = args.waveform_out or args.audio_path.with_suffix(".waveform.png")
    result = analyze_track(args.audio_path, waveform_out, args.models_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "bpm": result.bpm,
                    "musical_key": result.musical_key,
                    "musical_key_scale": result.musical_key_scale,
                    "mood_scores": result.mood_scores,
                    "mood_summary": result.mood_summary,
                    "genre": result.genre,
                    "style": result.style,
                    "waveform_path": str(result.waveform_path),
                },
                indent=2,
            )
        )
        return

    binary_scores = {k: v for k, v in result.mood_scores.items() if ":" not in k}
    # Multi-class classifiers (currently just genre_discogs400) key their
    # scores as "<classifier>:<class>" - group by classifier so any newly added
    # one shows up here without further changes to this script.
    multiclass_groups: dict[str, dict[str, float]] = {}
    for key, score in result.mood_scores.items():
        if ":" not in key:
            continue
        classifier, cls = key.split(":", 1)
        multiclass_groups.setdefault(classifier, {})[cls] = score

    print(f"File:          {args.audio_path}")
    print(f"BPM:           {result.bpm:.1f}" if result.bpm is not None else "BPM:           -")
    print(f"Key:           {result.musical_key} {result.musical_key_scale}" if result.musical_key else "Key:           -")
    print(f"Mood summary:  {result.mood_summary}")
    print(f"Genre:         {result.genre or '-'}")
    print(f"Style:         {result.style or '-'}")
    print()
    print("Mood scores:")
    for name, score in sorted(binary_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {score:.3f}")
    for classifier, scores in multiclass_groups.items():
        print()
        # genre_discogs400 has 400 classes - only the top 10 are shown here to
        # keep the report readable; --json includes every raw score.
        top_scores = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
        print(f"{classifier} scores (top {len(top_scores)} of {len(scores)}):")
        for name, score in top_scores:
            print(f"  {name:<30} {score:.3f}")
    print()
    print(f"Waveform:      {result.waveform_path}")


if __name__ == "__main__":
    main()
