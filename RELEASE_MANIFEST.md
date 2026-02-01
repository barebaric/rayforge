# Release Agent Manifest

## Purpose

The Release Agent automates the preparation of video content for software releases.
It bridges the gap between changelog data and Blender-based video editing by
generating scripts and assets for the release workflow.

## Workflow Overview

When instructed to "Prepare a release", the agent executes the following pipeline:

### Phase 1: Context Analysis

1. **Read CHANGELOG.md** to extract version information and release notes
2. **Read git log** to identify commit history and changes
3. **Identify media assets** in `media/[release]/` directory to learn about the style and language of the content on each platform (reddit, github, patreon)
4. **Fetch Patreon supporters** using `pixi run list-supporters` to get the current list of supporters
5. Research the web using the webReader tool, for any recent mentions of Rayforge, to identify criticism or praise. See if you can incorporate this into the
   content in phase 2 - not by directly adressing it, but to understand what users care about.

### Phase 2: Content Drafting

Generate the following files in `media/[release]/drafts/`:

- `youtube.txt` - Text for the YouTube release video description
- `reddit_post.md` - Formatted post for Reddit, following the tone of the previous Reddit posts. Include credits to paying Patreon supporters.
- `patreon_post.md` - Formatted post for Patreon, following the tone of the previous Patreon posts
- `blog_post.md` - Formatted post for the website blog. Don't store this in the drafts, store it in `website/content/blog/posts/`.
  Give credits to paying Patreon supporters.
- Update the changelog in the appstream file (`data/org.rayforge.rayforge.metainfo.xml`)
- Generate five release thumbnails using the MCP tool. Something like "make a YouTube thumbnail for Rayforge 1.1 with ... [something creative]".
  Put the thumbnails into media/[release]/thumbs/
- Depending on the changes, check that the user documentation on the website is up to date. Check the docs by reading the application code.
  Update the documentation accordingly, but keep it user-centric - this is not intended as developer documentation.
- Depending on the changes listed in the changelog, re-create relevant screenshots for the docs using the `scripts/media/take_screenshot.py` tool.

In the text files, use a maximum line length of 100 chars.
Ensure you use proper links for discord and patreon and the homepage and github, not placeholders.

### Phase 3: User manually creates clips with audio

Clips will be stored in `media/[release]/raw/*.mp4`. The user may also use other formats such as mkv or avi, or plain audio files such as wav or mp3 or aac.

### Phase 4: Audio Preprocessing (FFmpeg)

Generate and execute audio processing commands using the existing script:

```bash
pixi run process-audio media/[release]/raw/*.mp4
```

Output: `media/[release]/processed/` with processed video files

Remove silent pauses from the processed videos:

```bash
pixi run remove-silence media/[release]/processed/*.mp4
```

Output: `media/[release]/processed/` with .nosilence files

### Phase 5: Blender Project Generation

Generate the Blender project:

```bash
pixi run generate-blender-setup \
    media/[release]/processed/ \
    -o media/[release]/draft_edit.blend \
    -c CHANGELOG.md \
    -r 1920x1080 \
    -f 30 \
    --auto-find
```

The script:

- Uses the blender video from the previous release as a template.
- Adds the thumbnail as the first frame.
- Adds video and audio clips to the timeline

Output: `media/[release]/draft_edit.blend`
