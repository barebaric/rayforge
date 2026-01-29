#!/usr/bin/env python3
"""Generate Blender setup for video project assembly.

This script directly creates a Blender .blend file with clips, audio tracks,
and text overlays from a CHANGELOG.md.
"""

import argparse
import sys
from pathlib import Path

import bpy


def parse_changelog(changelog_path):
    """Parse CHANGELOG.md to extract version headers and content.

    Args:
        changelog_path: Path to CHANGELOG.md file.

    Returns:
        List of tuples: (version, content_lines).
    """
    changelog_path = Path(changelog_path)

    if not changelog_path.exists():
        msg = f"Error: CHANGELOG.md not found at '{changelog_path}'"
        print(msg, file=sys.stderr)
        return []

    sections = []
    current_version = None
    current_content = []

    with open(changelog_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## ["):
                if current_version is not None:
                    sections.append((current_version, current_content))
                current_version = line[4:].split("]")[0]
                current_content = []
            elif current_version is not None:
                current_content.append(line)

    if current_version is not None:
        sections.append((current_version, current_content))

    return sections


def find_media_files(media_dir, extensions=(".mp4", ".mkv", ".mov", ".avi")):
    """Find video files in media directory.

    Args:
        media_dir: Path to media directory.
        extensions: Tuple of video file extensions to include.

    Returns:
        Sorted list of video file paths.
    """
    media_dir = Path(media_dir)

    if not media_dir.exists():
        msg = f"Error: Media directory '{media_dir}' does not exist"
        print(msg, file=sys.stderr)
        return []

    video_files = []
    for ext in extensions:
        video_files.extend(media_dir.glob(f"*{ext}"))

    return sorted(video_files)


def find_previous_blend_file(release_dir):
    """Find the previous release's blend file.

    Args:
        release_dir: Path to current release directory.

    Returns:
        Path to previous blend file or None.
    """
    media_dir = Path(__file__).parent.parent / "media"
    if not media_dir.exists():
        return None

    release_dirs = sorted(
        [d for d in media_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )

    for release in release_dirs:
        if release == release_dir:
            continue
        blend_file = release / "draft_edit.blend"
        if blend_file.exists():
            return blend_file
    return None


def find_thumbnail(release_dir):
    """Find the thumbnail for the current release.

    Args:
        release_dir: Path to current release directory.

    Returns:
        Path to thumbnail file or None.
    """
    release_dir = Path(release_dir)
    drafts_dir = release_dir / "drafts"
    if drafts_dir.exists():
        for i in range(1, 6):
            thumbnail = drafts_dir / f"thumbnail{i}.png"
            if thumbnail.exists():
                return thumbnail
    return None


def load_template(template_path):
    """Load template blend file.

    Args:
        template_path: Path to template .blend file.

    Returns:
        True if template was loaded, False otherwise.
    """
    if template_path and Path(template_path).exists():
        bpy.ops.wm.open_mainfile(filepath=str(template_path))
        return True
    return False


def clear_sequencer():
    """Clear all strips from the video sequencer."""
    if bpy.context.scene.sequence_editor is None:
        bpy.context.scene.sequence_editor_create()

    seq_editor = bpy.context.scene.sequence_editor
    strips = list(seq_editor.strips)
    for strip in strips:
        seq_editor.strips.remove(strip)


def setup_scene_settings(res_x, res_y, fps_val):
    """Configure render settings.

    Args:
        res_x: Resolution width in pixels.
        res_y: Resolution height in pixels.
        fps_val: Frames per second.
    """
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.context.scene.render.fps = fps_val


def add_thumbnail(thumbnail_path):
    """Add thumbnail as first frame.

    Args:
        thumbnail_path: Path to thumbnail image.

    Returns:
        Frame number to start video clips from.
    """
    if not thumbnail_path or not Path(thumbnail_path).exists():
        return 1

    seq_editor = bpy.context.scene.sequence_editor
    seq_editor.strips.new_image(
        name="thumbnail",
        filepath=str(thumbnail_path),
        channel=1,
        frame_start=1,
    )
    strip = seq_editor.strips["thumbnail"]
    strip.frame_final_duration = 1
    return 2


def add_video_clips(video_files, start_frame):
    """Add video clips to the timeline.

    Args:
        video_files: List of video file paths.
        start_frame: Frame number to start adding clips.

    Returns:
        Last frame number after all clips.
    """
    current_frame = start_frame
    track_video = 1
    seq_editor = bpy.context.scene.sequence_editor

    for i, filepath in enumerate(video_files):
        full_path = str(filepath)
        strip_name = f"video_{i}"

        seq_editor.strips.new_movie(
            name=strip_name,
            filepath=full_path,
            channel=track_video,
            frame_start=current_frame,
        )

        strip = seq_editor.strips[strip_name]
        current_frame += strip.frame_final_duration

    return current_frame - 1


def add_text_overlays(sections, start_frame, duration=180):
    """Add text strips from changelog sections.

    Args:
        sections: List of (version, content) tuples.
        start_frame: Frame number to start adding text overlays.
        duration: Duration of each text overlay in frames.
    """
    if not sections:
        return

    track_text = 3
    current_frame = start_frame
    seq_editor = bpy.context.scene.sequence_editor

    for i, (version, _) in enumerate(sections):
        title_text = f"Release {version}"
        strip_name = f"text_{version}"

        seq_editor.strips.new_effect(
            name=strip_name,
            type="TEXT",
            channel=track_text,
            frame_start=current_frame,
            length=duration,
        )

        strip = seq_editor.strips[strip_name]
        strip.text = title_text
        strip.location = (0.5, 0.7)
        strip.font_size = 150
        strip.color = (1.0, 1.0, 1.0, 1.0)

        current_frame += duration


def generate_blender_file(
    media_dir,
    output_blend,
    changelog_path=None,
    resolution_x=1920,
    resolution_y=1080,
    fps=30,
    template_path=None,
    thumbnail_path=None,
):
    """Generate Blender .blend file directly.

    Args:
        media_dir: Path to media directory containing video files.
        output_blend: Path for output .blend file.
        changelog_path: Path to CHANGELOG.md for text overlays.
        resolution_x: Video width in pixels.
        resolution_y: Video height in pixels.
        fps: Frames per second.
        template_path: Path to template .blend file.
        thumbnail_path: Path to thumbnail image.
    """
    media_dir = Path(media_dir).resolve()
    output_blend = Path(output_blend).resolve()
    video_files = find_media_files(media_dir)

    if not video_files:
        msg = f"Warning: No video files found in '{media_dir}'"
        print(msg, file=sys.stderr)

    changelog_sections = []
    if changelog_path:
        changelog_sections = parse_changelog(changelog_path)

    loaded = load_template(template_path)
    if not loaded:
        clear_sequencer()
    else:
        if bpy.context.scene.sequence_editor is None:
            bpy.context.scene.sequence_editor_create()

    setup_scene_settings(resolution_x, resolution_y, fps)

    start_frame = add_thumbnail(thumbnail_path)

    end_frame = add_video_clips(video_files, start_frame)

    if changelog_sections:
        add_text_overlays(changelog_sections, end_frame + 30)

    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    print(f"Blender project saved to: {output_blend}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Blender .blend file for video project."
    )

    # Filter out Blender's arguments from sys.argv
    # When run with Blender, sys.argv contains Blender's arguments first
    # We need to find where our script's arguments start
    argv = sys.argv
    if "--" in argv:
        # Arguments after -- are for our script
        argv = argv[argv.index("--") + 1 :]
    elif len(argv) > 1 and argv[0].endswith("generate_blender_setup.py"):
        # Script is first argument, rest are our arguments
        argv = argv[1:]
    else:
        # Try to find our script in the path
        for i, arg in enumerate(argv):
            if "generate_blender_setup.py" in arg:
                argv = argv[i + 1 :]
                break

    # Get the script's directory to resolve relative paths
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent.parent
    parser.add_argument(
        "media_dir",
        help="Directory containing video files",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .blend file path",
        default="draft_edit.blend",
    )
    parser.add_argument(
        "-c",
        "--changelog",
        help="Path to CHANGELOG.md for text overlays",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        help="Resolution as WIDTHxHEIGHT (default: 1920x1080)",
        default="1920x1080",
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Frames per second (default: 30)",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-t",
        "--template",
        help="Path to template .blend file",
        default=None,
    )
    parser.add_argument(
        "--thumbnail",
        help="Path to thumbnail image",
        default=None,
    )
    parser.add_argument(
        "--auto-find",
        action="store_true",
        help="Automatically find template and thumbnail from media dir",
    )

    args = parser.parse_args(argv)

    resolution_parts = args.resolution.lower().split("x")
    if len(resolution_parts) != 2:
        print(
            "Error: Resolution must be in WIDTHxHEIGHT format",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        res_x = int(resolution_parts[0])
        res_y = int(resolution_parts[1])
    except ValueError:
        print("Error: Resolution values must be integers", file=sys.stderr)
        sys.exit(1)

    # Resolve paths relative to project directory
    media_dir = Path(args.media_dir)
    if not media_dir.is_absolute():
        media_dir = project_dir / media_dir

    output_blend = Path(args.output)
    if not output_blend.is_absolute():
        output_blend = project_dir / output_blend

    changelog_path = args.changelog
    if changelog_path is not None:
        changelog_path = Path(changelog_path)
        if not changelog_path.is_absolute():
            changelog_path = project_dir / changelog_path
        if not changelog_path.exists():
            changelog_path = None
    else:
        changelog_path = project_dir / "CHANGELOG.md"
        if not changelog_path.exists():
            changelog_path = None

    template_path = args.template
    thumbnail_path = args.thumbnail

    if args.auto_find:
        if not template_path:
            template_path = find_previous_blend_file(media_dir)
            if template_path:
                print(f"Found template: {template_path}")
        if not thumbnail_path:
            thumbnail_path = find_thumbnail(media_dir)
            if thumbnail_path:
                print(f"Found thumbnail: {thumbnail_path}")

    generate_blender_file(
        media_dir=media_dir,
        output_blend=output_blend,
        changelog_path=changelog_path,
        resolution_x=res_x,
        resolution_y=res_y,
        fps=args.fps,
        template_path=template_path,
        thumbnail_path=thumbnail_path,
    )


if __name__ == "__main__":
    main()
