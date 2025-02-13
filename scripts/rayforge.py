#!/usr/bin/python3
import sys
sys.path.pop(0)  # Prevent importing myself
import argparse
from rayforge.app import App
from rayforge.config import config_mgr


def main():
    parser = argparse.ArgumentParser(
            description="A GCode generator for laser cutters.")
    parser.add_argument("filename",
                        help="Path to the input SVG or image file.",
                        nargs='?')

    args = parser.parse_args()
    app = App(args)
    app.run(None)
    config_mgr.save()

if __name__ == "__main__":
    main()
