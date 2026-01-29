#!/usr/bin/env python3
"""Audio processing script for video clips.

Processes audio in video files through a pipeline:
1. Denoise (highpass/lowpass filters)
2. Reduce reverb/echo
3. Equalize for better voice
4. Compress
5. Normalize to target loudness
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_audio_filter_chain():
    """Build FFmpeg audio filter chain for the processing pipeline."""
    filters = []

    # 1. Denoise - highpass to remove rumble, lowpass to remove hiss
    filters.append("highpass=f=80")
    filters.append("lowpass=f=12000")

    # 2. Reduce reverb/echo (less aggressive for more natural sound)
    filters.append("afftdn=nf=-20:tn=1")

    # 3. Equalize for better voice - boost lower range for male voice
    # Boost 100-300Hz range by 3dB
    filters.append("equalizer=f=150:width_type=h:width=100:g=3")

    # 4. Compress (gentler attack for more natural sound)
    filters.append(
        "acompressor=threshold=-20dB:ratio=4:attack=25:release=100:makeup=3dB"
    )

    # 5. Normalize to target loudness
    filters.append("loudnorm=I=-12:TP=-1.5:LRA=11")

    return ",".join(filters)


def process_video(input_path, output_path=None):
    """Process audio in video file.

    Args:
        input_path: Path to input video file.
        output_path: Path to output video file. If None, creates
                     input_processed.ext file.

    Returns:
        True if successful, False otherwise.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        msg = f"Error: Input file '{input_path}' does not exist"
        print(msg, file=sys.stderr)
        return False

    if output_path is None:
        output_dir = input_path.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / input_path.name
    else:
        output_path = Path(output_path)

    audio_filter = build_audio_filter_chain()

    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-c:v",
        "copy",
        "-af",
        audio_filter,
        "-y",
        str(output_path),
    ]

    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Processing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        if e.stderr:
            print(f"FFmpeg stderr: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        msg = "Error: ffmpeg not found. Please install FFmpeg."
        print(msg, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process audio in video files for better voice quality."
    )
    parser.add_argument("input", nargs="+", help="Input video file path(s)")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for processed files "
        "(default: media/[release]/processed/)",
        default=None,
    )

    args = parser.parse_args()

    success_count = 0
    total_count = len(args.input)

    for input_file in args.input:
        input_path = Path(input_file)

        if not input_path.exists():
            print(
                f"Warning: Input file '{input_path}' does not exist, skipping",
                file=sys.stderr,
            )
            continue

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / input_path.name
        else:
            output_path = None

        if process_video(input_path, output_path):
            success_count += 1

    print(f"\nProcessed {success_count}/{total_count} files successfully")
    sys.exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()
