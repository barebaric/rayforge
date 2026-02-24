#!/usr/bin/env python3
"""Audio processing script for video clips.

Processes audio in video files through a pipeline:
1. Optional: Remove silence
2. Denoise (highpass/lowpass filters)
3. Reduce reverb/echo
4. Equalize for better voice
5. Compress
6. Normalize to target loudness
7. Optional: Time compress (speed up) without pitch artifacts
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TEMPO = 1.05


def build_audio_filter_chain(tempo: float = 1.0) -> str:
    """Build FFmpeg audio filter chain for the processing pipeline.

    Args:
        tempo: Tempo multiplier (1.0 = normal, 1.05 = 5% faster).
               Values > 2.0 or < 0.5 require chaining atempo filters.
    """
    filters = []

    if tempo != 1.0:
        filters.append(build_atempo_filter(tempo))

    filters.append("highpass=f=80")
    filters.append("afftdn=nf=-25:tn=1")
    filters.append(
        "acompressor=threshold=-20dB:ratio=4:attack=15:release=100:makeup=3dB"
    )
    filters.append("equalizer=f=150:t=q:w=3:g=4")
    filters.append("loudnorm=I=-18:TP=-1.5:LRA=20")

    return ",".join(filters)


def build_atempo_filter(tempo: float) -> str:
    """Build atempo filter chain for time compression/expansion.

    FFmpeg's atempo filter only accepts values between 0.5 and 2.0.
    For values outside this range, we chain multiple atempo filters.

    Args:
        tempo: Desired tempo multiplier.

    Returns:
        FFmpeg atempo filter string.
    """
    if 0.5 <= tempo <= 2.0:
        return f"atempo={tempo}"

    atempo_filters = []
    remaining = tempo

    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining /= 0.5

    if remaining != 1.0:
        atempo_filters.append(f"atempo={remaining}")

    return ",".join(atempo_filters)


def build_video_filter(tempo: float) -> str:
    """Build setpts filter for video time compression.

    Args:
        tempo: Tempo multiplier (1.0 = normal, 1.05 = 5% faster).

    Returns:
        FFmpeg setpts filter string.
    """
    return f"setpts={1 / tempo}*PTS"


def detect_silence(
    input_file: str,
    silence_threshold: float = -50,
    min_silence_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Detect silent portions in a video file.

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
    """Merge silence segments that are close to each other.

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
    """Get the duration of a video file.

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
    """Check if a video file has an audio stream.

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
    """Extract a segment from a video file.

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
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
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
    """Create a concat file for FFmpeg concat demuxer.

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
    """Concatenate segments using FFmpeg concat demuxer.

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


def remove_silence(
    input_file: str,
    output_file: str,
    silence_threshold: float = -50,
    min_silence_duration: float = 0.5,
    merge_gap: float = 0.1,
) -> bool:
    """Remove silent portions from a video file.

    Args:
        input_file: Path to the input video file.
        output_file: Path to the output video file.
        silence_threshold: Noise threshold in dB (negative value).
        min_silence_duration: Minimum silence duration to remove in seconds.
        merge_gap: Gap between silent segments to merge.

    Returns:
        True if successful, False if no silence removed (file copied).
    """
    if not has_audio_stream(input_file):
        print("  No audio stream found. Copying file as-is.")
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", "-y", output_file],
            check=True,
            capture_output=True,
        )
        return False

    print("  Analyzing audio for silence...")
    silent_segments = detect_silence(
        input_file, silence_threshold, min_silence_duration
    )

    if not silent_segments:
        print("  No silence detected. Copying file as-is.")
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", "-y", output_file],
            check=True,
            capture_output=True,
        )
        return False

    silent_segments = merge_adjacent_silence(silent_segments, merge_gap)

    video_duration = get_video_duration(input_file)

    margin = 0.1
    silent_segments = [
        (start, end)
        for start, end in silent_segments
        if start > margin and end < video_duration - margin
    ]

    if not silent_segments:
        print("  No mid-video silence detected. Copying file as-is.")
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-c", "copy", "-y", output_file],
            check=True,
            capture_output=True,
        )
        return False

    total_silence = sum(end - start for start, end in silent_segments)
    print(f"  Found {len(silent_segments)} silent segment(s)")
    print(f"  Total silence to remove: {total_silence:.2f} seconds")

    margin = 0.005
    keep_segments = []
    prev_end = 0

    for start, end in silent_segments:
        if start > prev_end + margin:
            keep_segments.append((prev_end, start - margin))
        prev_end = end

    if prev_end < video_duration - margin:
        keep_segments.append((prev_end, video_duration))

    print(f"  Extracting {len(keep_segments)} non-silent segments...")

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []

        for i, (start, end) in enumerate(keep_segments):
            if end - start < 0.01:
                continue
            seg_file = os.path.join(temp_dir, f"segment_{i:04d}.mkv")
            print(f"    Segment {i + 1}/{len(keep_segments)}")
            extract_segment(input_file, seg_file, start, end)
            segment_files.append(seg_file)

        if not segment_files:
            print("  No non-silent segments found. Copying file as-is.")
            subprocess.run(
                ["ffmpeg", "-i", input_file, "-c", "copy", "-y", output_file],
                check=True,
                capture_output=True,
            )
            return False

        print("  Concatenating segments...")
        concat_file = os.path.join(temp_dir, "concat.txt")
        create_concat_file(segment_files, concat_file)
        concat_segments(concat_file, output_file)

    return True


def process_video(
    input_path,
    output_path=None,
    tempo: float = 1.0,
    remove_silence_flag: bool = False,
    silence_threshold: float = -50,
    min_silence_duration: float = 0.5,
    silence_merge_gap: float = 0.1,
):
    """Process audio in video file.

    Args:
        input_path: Path to input video file.
        output_path: Path to output video file. If None, creates
                     input_processed.ext file.
        tempo: Tempo multiplier for time compression (1.05 = 5% faster).
        remove_silence_flag: Whether to remove silence before processing.
        silence_threshold: Silence detection threshold in dB.
        min_silence_duration: Minimum silence duration to remove.
        silence_merge_gap: Gap between silences to merge.

    Returns:
        True if successful, False otherwise.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        msg = f"Error: Input file '{input_path}' does not exist"
        print(msg, file=sys.stderr)
        return False

    if output_path is None:
        raise ValueError(
            "Output path must be specified. Use -o/--output-dir option."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleanup_files = []
    current_input = str(input_path)

    try:
        if remove_silence_flag:
            print("Removing silence...")
            temp_silence = output_path.with_suffix(".temp_silence.mkv")
            cleanup_files.append(temp_silence)

            remove_silence(
                current_input,
                str(temp_silence),
                silence_threshold,
                min_silence_duration,
                silence_merge_gap,
            )
            current_input = str(temp_silence)

        audio_filter = build_audio_filter_chain(tempo)

        cmd = [
            "ffmpeg",
            "-i",
            current_input,
        ]

        if tempo != 1.0:
            video_filter = build_video_filter(tempo)
            cmd.extend(["-vf", video_filter])
            cmd.extend(["-c:v", "libx264", "-preset", "fast"])
        else:
            cmd.extend(["-c:v", "copy"])

        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-af",
                audio_filter,
                "-y",
                str(output_path),
            ]
        )

        speed_info = f" (tempo: {tempo}x)" if tempo != 1.0 else ""
        silence_info = " + silence removal" if remove_silence_flag else ""
        print(f"Processing{silence_info}{speed_info}: {input_path}")
        print(f"Output: {output_path}")

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
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False
    finally:
        for f in cleanup_files:
            if f.exists():
                f.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Process audio in video files for better voice quality."
    )
    parser.add_argument("input", nargs="+", help="Input video file path(s)")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for processed files",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO,
        help=f"Tempo multiplier for time compression "
        f"(default: {DEFAULT_TEMPO}, i.e. 5%% faster). "
        f"Use 1.0 for no speed change.",
    )
    parser.add_argument(
        "--no-remove-silence",
        action="store_true",
        help="Disable automatic silence removal.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-50,
        help="Silence threshold in dB (default: -50).",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.5,
        help="Minimum silence duration to remove in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--silence-gap",
        type=float,
        default=0.1,
        help="Gap between silences to merge in seconds (default: 0.1).",
    )

    args = parser.parse_args()

    remove_silence_flag = not args.no_remove_silence
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = output_dir / input_path.name

        if process_video(
            input_path,
            output_path,
            tempo=args.tempo,
            remove_silence_flag=remove_silence_flag,
            silence_threshold=args.silence_threshold,
            min_silence_duration=args.silence_duration,
            silence_merge_gap=args.silence_gap,
        ):
            success_count += 1

    print(f"\nProcessed {success_count}/{total_count} files successfully")
    sys.exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()
