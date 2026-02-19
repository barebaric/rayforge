#!/usr/bin/env python3
"""
Script to remove silent pauses from a video file.

This script analyzes the audio track of a video, detects silent portions,
and removes them by cutting out those segments and joining the remaining parts.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def detect_silence(
    input_file: str,
    silence_threshold: float = -50,
    min_silence_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Detect silent portions in a video file.

    Args:
        input_file: Path to the input video file.
        silence_threshold: Noise threshold in dB (negative value).
        min_silence_duration: Minimum silence duration in seconds.

    Returns:
        List of (start, end) tuples representing silent segments.
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-af",
        (
            f"silencedetect=noise={silence_threshold}dB:"
            f"duration={min_silence_duration}"
        ),
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    stderr = result.stderr
    silent_segments = []

    lines = stderr.split("\n")
    silence_start = None

    for line in lines:
        if "silence_start" in line:
            match = re.search(r"silence_start[:\s]+([\d.]+)", line)
            if match:
                silence_start = float(match.group(1))
        elif "silence_end" in line and silence_start is not None:
            match = re.search(r"silence_end[:\s]+([\d.]+)", line)
            if match:
                silence_end = float(match.group(1))
                silent_segments.append((silence_start, silence_end))
                silence_start = None

    return silent_segments


def merge_adjacent_silence(
    segments: list[tuple[float, float]],
    gap: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Merge silence segments that are close to each other.

    Args:
        segments: List of (start, end) tuples.
        gap: Maximum gap between segments to merge.

    Returns:
        List of merged (start, end) tuples.
    """
    if not segments:
        return []

    merged = [list(segments[0])]

    for start, end in segments[1:]:
        last_end = merged[-1][1]
        if start - last_end <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def get_video_duration(input_file: str) -> float:
    """
    Get the duration of a video file.

    Args:
        input_file: Path to the input video file.

    Returns:
        Duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        input_file,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    if "format" not in data or "duration" not in data["format"]:
        raise RuntimeError("Could not determine video duration")
    return float(data["format"]["duration"])


def has_audio_stream(input_file: str) -> bool:
    """
    Check if a video file has an audio stream.

    Args:
        input_file: Path to the input video file.

    Returns:
        True if audio stream exists, False otherwise.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        input_file,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return bool(result.stdout.strip())


def extract_segment(
    input_file: str,
    output_file: str,
    start: float,
    end: float,
) -> None:
    """
    Extract a segment from a video file.

    Args:
        input_file: Path to the input video file.
        output_file: Path to the output segment file.
        start: Start time in seconds.
        end: End time in seconds.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss",
        str(start),
        "-i",
        input_file,
        "-t",
        str(duration),
        "-c",
        "copy",
        "-y",
        output_file,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(f"Segment extraction failed: {result.stderr}")


def create_concat_file(
    segment_files: list[str],
    output_file: str,
) -> None:
    """
    Create a concat file for FFmpeg concat demuxer.

    Args:
        segment_files: List of segment file paths.
        output_file: Path to the concat file.
    """
    with open(output_file, "w") as f:
        for seg_file in segment_files:
            f.write(f"file '{seg_file}'\n")


def concat_segments(
    concat_file: str,
    output_file: str,
) -> None:
    """
    Concatenate segments using FFmpeg concat demuxer.

    Args:
        concat_file: Path to the concat file.
        output_file: Path to the output video file.
    """
    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c",
        "copy",
        "-y",
        output_file,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(f"Concatenation failed: {result.stderr}")


def get_default_output_file(input_file: str) -> str:
    """
    Generate default output filename from input file.

    Args:
        input_file: Path to the input file.

    Returns:
        Default output file path.
    """
    path = Path(input_file)
    return str(path.with_name(f"{path.stem}.nosilence{path.suffix}"))


def remove_silence(
    input_file: str,
    output_file: str,
    silence_threshold: float = -50,
    min_silence_duration: float = 0.5,
    merge_gap: float = 0.1,
) -> None:
    """
    Remove silent portions from a video file.

    Args:
        input_file: Path to the input video file.
        output_file: Path to the output video file.
        silence_threshold: Noise threshold in dB (negative value).
        min_silence_duration: Minimum silence duration to remove in seconds.
        merge_gap: Gap between silent segments to merge.
    """
    if not has_audio_stream(input_file):
        print("No audio stream found. Copying file as-is.")
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", output_file],
            check=True,
        )
        return

    print(f"Analyzing audio in {input_file}...")

    silent_segments = detect_silence(
        input_file, silence_threshold, min_silence_duration
    )

    if not silent_segments:
        print("No silence detected. Copying file as-is.")
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", output_file],
            check=True,
        )
        return

    silent_segments = merge_adjacent_silence(silent_segments, merge_gap)

    total_silence = sum(end - start for start, end in silent_segments)
    print(f"Found {len(silent_segments)} silent segment(s)")
    print(f"Total silence to remove: {total_silence:.2f} seconds")

    video_duration = get_video_duration(input_file)

    # Create keep segments (inverse of silent segments)
    keep_segments = []
    prev_end = 0

    for start, end in silent_segments:
        if start > prev_end:
            keep_segments.append((prev_end, start))
        prev_end = end

    if prev_end < video_duration:
        keep_segments.append((prev_end, video_duration))

    print(f"Extracting {len(keep_segments)} non-silent segments...")

    # Use temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []

        for i, (start, end) in enumerate(keep_segments):
            seg_file = os.path.join(temp_dir, f"segment_{i:04d}.mkv")
            print(
                f"  Extracting segment {i + 1}/{len(keep_segments)} "
                f"({start:.2f}s - {end:.2f}s)..."
            )
            extract_segment(input_file, seg_file, start, end)
            segment_files.append(seg_file)

        print("Concatenating segments...")

        concat_file = os.path.join(temp_dir, "concat.txt")
        create_concat_file(segment_files, concat_file)
        concat_segments(concat_file, output_file)

    print(f"Done! Output saved to {output_file}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remove silent pauses from a video file."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video file paths (can specify multiple)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output video file path (default: input.nosilence.ext). "
            "Ignored when processing multiple files."
        ),
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=-50,
        help="Silence threshold in dB (default: -50)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=0.5,
        help="Minimum silence duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gap",
        type=float,
        default=0.1,
        help="Gap between silences to merge in seconds (default: 0.1)",
    )

    args = parser.parse_args()

    # Validate inputs
    for input_file in args.inputs:
        if not os.path.exists(input_file):
            print(
                f"Error: Input file '{input_file}' not found",
                file=sys.stderr,
            )
            return 1

    # Determine output files
    if len(args.inputs) > 1:
        if args.output:
            print(
                "Warning: -o/--output ignored when processing multiple files",
                file=sys.stderr,
            )
        output_files = [get_default_output_file(f) for f in args.inputs]
    else:
        output_files = [
            args.output
            if args.output
            else get_default_output_file(args.inputs[0])
        ]

    # Process each file
    success = True
    for input_file, output_file in zip(args.inputs, output_files):
        print(f"\n{'=' * 60}")
        print(f"Processing: {input_file}")
        print(f"Output: {output_file}")
        print(f"{'=' * 60}\n")

        try:
            remove_silence(
                input_file,
                output_file,
                args.threshold,
                args.duration,
                args.gap,
            )
        except Exception as e:
            print(f"Error processing {input_file}: {e}", file=sys.stderr)
            traceback.print_exc()
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
