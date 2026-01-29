#!/usr/bin/env python3
# Attempts to import key native modules and run a vtracer call.
import sys
import traceback
import faulthandler

faulthandler.enable()
print(f"=== Python executable: {sys.executable} ===", flush=True)
print(f"=== Python version: {sys.version} ===", flush=True)

for m in ["vtracer", "pyvips", "gi", "cv2", "numpy"]:
    try:
        __import__(m)
        print(f"=== Imported {m} OK ===", flush=True)
    except Exception as e:
        print(
            f"=== Import {m} failed: {type(e).__name__}: {e} ===", flush=True
        )

try:
    import numpy as np
    import cv2

    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[4:12, 4:12] = 255
    success, buf = cv2.imencode(".bmp", arr)
    if success:
        import vtracer

        print(
            "=== Calling vtracer.convert_raw_image_to_svg... ===", flush=True
        )
        # Use positional arguments to avoid ABI crash in pyo3
        # extract_arguments_fastcall
        # Signature: (img_bytes, img_format, colormode, hierarchical, mode, ...)
        out = vtracer.convert_raw_image_to_svg(
            buf.tobytes(),  # img_bytes
            "bmp",  # img_format
            "binary",  # colormode
            "stacked",  # hierarchical
            "polygon",  # mode
        )
        print(
            f"=== vtracer returned length: {len(out) if out else 0} ===",
            flush=True,
        )
    else:
        print("=== cv2.imencode failed ===", flush=True)
except Exception as e:
    print(
        f"=== Error in vtracer call: {type(e).__name__}: {e} ===", flush=True
    )
    traceback.print_exc()
