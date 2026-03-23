#!/usr/bin/env python3
"""
Normalize SVG icons for GTK compatibility.

This script normalizes all SVG icons in rayforge/resources/icons/ to have:
- viewBox="0 0 24 24"
- Path coordinates transformed to fit within 0-24 range
- Fill color set to #000000
- Transform attributes updated (translate values scaled)
- Coordinate attributes (x, y, cx, cy, etc.) transformed
- Dimension attributes (width, height, r, etc.) scaled

Preserves original XML structure, namespaces, and formatting.
"""

import re
import sys
from pathlib import Path


def parse_viewbox(viewbox_str):
    """Parse viewBox attribute into (min_x, min_y, width, height)."""
    parts = viewbox_str.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Invalid viewBox: {viewbox_str}")
    return tuple(float(p) for p in parts)


def tokenize_path(path_data):
    """
    Tokenize SVG path data into a list of commands and numbers.
    Handles negative numbers without spaces (e.g., "784-120" -> 784, -120).
    """
    tokens = []
    i = 0
    data = path_data.strip()

    while i < len(data):
        c = data[i]

        if c.isalpha():
            tokens.append(c)
            i += 1
        elif c in "-0123456789.":
            j = i
            if data[j] == "-":
                j += 1
            while j < len(data) and data[j].isdigit():
                j += 1
            if j < len(data) and data[j] == ".":
                j += 1
                while j < len(data) and data[j].isdigit():
                    j += 1
            if j < len(data) and data[j] in "eE":
                j += 1
                if j < len(data) and data[j] in "+-":
                    j += 1
                while j < len(data) and data[j].isdigit():
                    j += 1

            num_str = data[i:j]
            if num_str and num_str != "-":
                tokens.append(float(num_str))
            i = j
        elif c in ", \t\n\r":
            i += 1
        else:
            i += 1

    return tokens


def transform_path(path_data, min_x, min_y, width, height, target_size=32):
    """
    Transform path data from original viewBox to target 0-32 viewBox.
    Returns the transformed path data string.
    """
    tokens = tokenize_path(path_data)
    scale = target_size / max(width, height)
    offset_x = (target_size - width * scale) / 2 - min_x * scale
    offset_y = (target_size - height * scale) / 2 - min_y * scale

    def fmt(n):
        if n == int(n):
            return str(int(n))
        s = f"{n:.4f}".rstrip("0").rstrip(".")
        return s

    def tx(x):
        return x * scale + offset_x

    def ty(y):
        return y * scale + offset_y

    result = []
    i = 0
    at_start = True

    while i < len(tokens):
        token = tokens[i]

        if isinstance(token, str):
            cmd = token
            i += 1

            if cmd.lower() == "z":
                result.append(cmd)
                at_start = False
            elif cmd.lower() in ("h", "v"):
                args = []
                while i < len(tokens) and isinstance(tokens[i], float):
                    if cmd == "H":
                        args.append(fmt(tx(tokens[i])))
                    elif cmd == "V":
                        args.append(fmt(ty(tokens[i])))
                    elif cmd == "h":
                        args.append(fmt(tokens[i] * scale))
                    else:
                        args.append(fmt(tokens[i] * scale))
                    i += 1
                result.append(cmd + " " + " ".join(args))
                at_start = False
            elif cmd.lower() == "a":
                while i + 6 < len(tokens) and all(
                    isinstance(tokens[i + j], float) for j in range(7)
                ):
                    rx = tokens[i] * scale
                    ry = tokens[i + 1] * scale
                    rot = int(tokens[i + 2])
                    large_arc = int(tokens[i + 3])
                    sweep = int(tokens[i + 4])
                    if cmd.isupper():
                        x = tx(tokens[i + 5])
                        y = ty(tokens[i + 6])
                    else:
                        x = tokens[i + 5] * scale
                        y = tokens[i + 6] * scale
                    result.append(
                        f"{cmd} {fmt(rx)} {fmt(ry)} {rot} {large_arc} "
                        f"{sweep} {fmt(x)} {fmt(y)}"
                    )
                    i += 7
                at_start = False
            elif cmd.lower() in ("m", "l", "t", "q", "s", "c"):
                coords = []
                first_pair = True
                while (
                    i + 1 < len(tokens)
                    and isinstance(tokens[i], float)
                    and isinstance(tokens[i + 1], float)
                ):
                    if cmd.isupper():
                        x = tx(tokens[i])
                        y = ty(tokens[i + 1])
                    elif at_start and first_pair and cmd.lower() == "m":
                        x = tx(tokens[i])
                        y = ty(tokens[i + 1])
                    else:
                        x = tokens[i] * scale
                        y = tokens[i + 1] * scale
                    coords.append((fmt(x), fmt(y)))
                    i += 2
                    first_pair = False

                if coords:
                    first_x, first_y = coords[0]
                    result.append(f"{cmd} {first_x} {first_y}")
                    for x, y in coords[1:]:
                        result.append(f"{x} {y}")
                at_start = False
        else:
            i += 1

    return " ".join(result)


def transform_translate(transform_str, scale):
    """
    Transform translate() values in a transform attribute.
    Translate values are relative offsets, so only scale them (no offset).
    Returns the updated transform string.
    """

    def replace_translate(m):
        tx_val = float(m.group(1)) if m.group(1) else 0
        ty_val = float(m.group(2)) if m.group(2) else 0
        new_tx = tx_val * scale
        new_ty = ty_val * scale

        def fmt(n):
            if n == int(n):
                return str(int(n))
            return f"{n:.4f}".rstrip("0").rstrip(".")

        return f"translate({fmt(new_tx)},{fmt(new_ty)})"

    return re.sub(
        r"translate\s*\(\s*([-+]?\d*\.?\d+)\s*(?:,\s*([-+]?\d*\.?\d+))?\s*\)",
        replace_translate,
        transform_str,
    )


def transform_style(style_str, scale):
    """
    Transform dimensional values in style attribute (e.g., stroke-width,
    font-size). Returns the updated style string.
    """

    def replace_dimension(m):
        prop = m.group(1)
        val = float(m.group(2))
        unit = m.group(3) or ""
        new_val = val * scale

        def fmt(n):
            if n == int(n):
                return str(int(n))
            return f"{n:.4f}".rstrip("0").rstrip(".")

        return f"{prop}:{fmt(new_val)}{unit}"

    return re.sub(
        r"(stroke-width|font-size|stroke-dasharray)\s*:\s*"
        r"([-+]?\d*\.?\d+)(px|pt|em|%)?",
        replace_dimension,
        style_str,
    )


def normalize_svg(content, target_size=32):
    """
    Normalize SVG content.

    Returns the normalized SVG content with:
    - viewBox set to "0 0 {target_size} {target_size}"
    - Path data transformed to fit the new viewBox
    - fill color set to #000000
    - Transform attributes updated
    - Coordinate/dimension attributes transformed
    """
    viewbox_match = re.search(r'viewBox\s*=\s*"([^"]+)"', content)
    if not viewbox_match:
        return None, "no viewBox found"

    viewbox_str = viewbox_match.group(1)
    min_x, min_y, width, height = parse_viewbox(viewbox_str)

    if (
        min_x == 0
        and min_y == 0
        and width == target_size
        and height == target_size
    ):
        content = re.sub(
            r'fill\s*=\s*"[^"]*"', 'fill="#000000"', content, count=1
        )
        return content, "already normalized"

    scale = target_size / max(width, height)
    offset_x = (target_size - width * scale) / 2 - min_x * scale
    offset_y = (target_size - height * scale) / 2 - min_y * scale

    def fmt(n):
        if n == int(n):
            return str(int(n))
        return f"{n:.4f}".rstrip("0").rstrip(".")

    def tx(x):
        return x * scale + offset_x

    def ty(y):
        return y * scale + offset_y

    def scale_val(v):
        return v * scale

    def replace_path(match):
        indent = match.group(1)
        path_data = match.group(2)
        rest = match.group(3)
        try:
            new_path = transform_path(
                path_data, min_x, min_y, width, height, target_size
            )
            return f'{indent}d="{new_path}"{rest}'
        except Exception:
            return match.group(0)

    path_pattern = r'(\n\s+)d\s*=\s*"([^"]+)"(\s*(?:/?>|\n))'
    content = re.sub(path_pattern, replace_path, content)

    def replace_transform(match):
        indent = match.group(1)
        transform_val = match.group(2)
        rest = match.group(3)
        try:
            new_transform = transform_translate(transform_val, scale)
            return f'{indent}transform="{new_transform}"{rest}'
        except Exception:
            return match.group(0)

    transform_pattern = r'(\n\s+)transform\s*=\s*"([^"]+)"(\s*(?:/?>|\n|\s))'
    content = re.sub(transform_pattern, replace_transform, content)

    def replace_x(match):
        indent = match.group(1)
        x_val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}x="{fmt(tx(x_val))}"{rest}'

    content = re.sub(
        r'(\n\s+)x\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_x,
        content,
    )

    def replace_y(match):
        indent = match.group(1)
        y_val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}y="{fmt(ty(y_val))}"{rest}'

    content = re.sub(
        r'(\n\s+)y\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_y,
        content,
    )

    def replace_cx(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}cx="{fmt(tx(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)cx\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_cx,
        content,
    )

    def replace_cy(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}cy="{fmt(ty(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)cy\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_cy,
        content,
    )

    def replace_r(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}r="{fmt(scale_val(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)r\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_r,
        content,
    )

    def replace_rx_attr(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}rx="{fmt(scale_val(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)rx\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_rx_attr,
        content,
    )

    def replace_ry_attr(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}ry="{fmt(scale_val(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)ry\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_ry_attr,
        content,
    )

    def replace_stroke_width(match):
        indent = match.group(1)
        val = float(match.group(2))
        rest = match.group(3)
        return f'{indent}stroke-width="{fmt(scale_val(val))}"{rest}'

    content = re.sub(
        r'(\n\s+)stroke-width\s*=\s*"([-+]?\d*\.?\d+)"(\s*(?:/?>|\n|\s))',
        replace_stroke_width,
        content,
    )

    def replace_style(match):
        indent = match.group(1)
        style_val = match.group(2)
        rest = match.group(3)
        try:
            new_style = transform_style(style_val, scale)
            return f'{indent}style="{new_style}"{rest}'
        except Exception:
            return match.group(0)

    style_pattern = r'(\n\s+)style\s*=\s*"([^"]+)"(\s*(?:/?>|\n))'
    content = re.sub(style_pattern, replace_style, content)

    content = re.sub(
        r'viewBox\s*=\s*"[^"]*"',
        f'viewBox="0 0 {target_size} {target_size}"',
        content,
    )

    content = re.sub(r'fill\s*=\s*"[^"]*"', 'fill="#000000"', content, count=1)

    return content, None


def main():
    icons_dir = (
        Path(__file__).parent.parent / "rayforge" / "resources" / "icons"
    )

    if not icons_dir.exists():
        print(f"Icons directory not found: {icons_dir}")
        sys.exit(1)

    svg_files = list(icons_dir.glob("**/*.svg"))
    print(f"Found {len(svg_files)} SVG files to process")

    processed = 0
    skipped = 0

    for svg_file in svg_files:
        print(f"Processing {svg_file.name}...", end="", flush=True)
        try:
            content = svg_file.read_text(encoding="utf-8")
            new_content, error = normalize_svg(content)

            if error:
                if error == "already normalized":
                    print(" (already normalized)")
                else:
                    print(f" SKIP: {error}")
                    skipped += 1
                continue

            if new_content:
                svg_file.write_text(new_content, encoding="utf-8")
                print(" OK")
                processed += 1
        except Exception as e:
            print(f" ERROR: {e}")
            skipped += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
