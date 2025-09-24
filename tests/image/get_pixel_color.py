#!/usr/bin/env python3
import argparse
from PIL import Image
import sys


def get_pixel_color(image_path, x, y):
    """
    Get the RGBA values of a pixel at the specified (x, y) coordinate in a
    PNG image.

    Args:
        image_path (str): Path to the PNG file
        x (int): X coordinate of the pixel
        y (int): Y coordinate of the pixel

    Returns:
        tuple: RGBA values of the pixel or None if coordinates are invalid
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGBA to ensure alpha channel exists and format
            # is consistent
            rgba_img = img.convert("RGBA")
            width, height = rgba_img.size

            # Check if coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                return rgba_img.getpixel((x, y))
            else:
                print(
                    f"Error: Coordinates ({x}, {y}) are out of bounds. "
                    f"Image size is {width}x{height}"
                )
                return None
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Get the color and alpha values of a pixel in a PNG file"
    )
    parser.add_argument("image", help="Path to the PNG image file")
    parser.add_argument("x", type=int, help="X coordinate of the pixel")
    parser.add_argument("y", type=int, help="Y coordinate of the pixel")
    parser.add_argument(
        "--format",
        "-f",
        choices=["rgb", "hex", "all"],
        default="all",
        help="Output format: rgb (RGB values), hex (hex values), all (both)",
    )

    args = parser.parse_args()

    pixel_data = get_pixel_color(args.image, args.x, args.y)

    if pixel_data:
        assert isinstance(pixel_data, tuple), (
            "Pixel data is not a valid tuple."
        )
        r, g, b, a = pixel_data

        if args.format in ["rgb", "all"]:
            print(f"Pixel at ({args.x}, {args.y}):")
            print(f"Red: {r}, Green: {g}, Blue: {b}, Alpha: {a}")

        if args.format in ["hex", "all"]:
            # Minor improvement for clarity on the alpha value's range
            print(f"Hex: #{r:02x}{g:02x}{b:02x}, Alpha: {a} (0-255)")

        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
