---
description: "Automatically detect stock material on the laser bed using computer vision. Capture a reference image and let Smart Stock create stock items from your camera feed."
---

# Smart Stock

Smart Stock uses computer vision to detect material placed on your laser bed and
creates matching stock items in your document. By comparing a reference image of
the empty bed against the current camera view, the addon identifies the
outlines of physical stock and generates correctly positioned stock items with
the right shape and size.

## Prerequisites

You need a configured and calibrated camera attached to your machine. The camera
must be set up with perspective correction so that the captured image aligns to
the machine's physical coordinate system. You also need a configured machine so
that the addon knows the work area dimensions.

## Opening the Detection Dialog

Open the dialog from **Tools - Detect Stock from Camera**. The window shows a
live camera preview on the left and detection settings on the right.

## Capturing a Reference Image

Before detecting stock, you need a reference image of the empty laser bed. With
no material on the bed, click the **Capture** button next to **Capture
Reference**. The addon stores this image and compares it against the live camera
feed to find new objects.

Reference images are saved per camera. When you reopen the dialog with the same
camera, the previously captured reference is loaded automatically and detection
runs immediately if material is already on the bed.

## Detecting Stock

Place your material on the laser bed, then click **Detect Stock** at the bottom
of the settings panel. The addon compares the current camera frame against the
reference image and traces the outlines of any new objects. Detected shapes
appear on the preview as magenta outlines with green fill.

The status row at the bottom of the settings panel reports how many items were
found. If no stock is detected, adjust the placement or lighting and try again.

## Detection Settings

**Camera** shows the currently selected camera. Click **Change** to switch to a
different configured camera.

**Sensitivity** controls how much visual change is required to register as
stock. At higher values, smaller or subtler differences between the reference
and the current frame are detected. At lower values, only large changes are
picked up. If the addon misses material that is present, increase sensitivity.
If it detects shadows or reflections as stock, decrease it.

**Smoothing** controls how smooth the detected outlines are. Higher values
produce rounder, simpler contours by filtering out small jagged edges from the
camera image. Lower values preserve more detail from the actual shape of the
material.

## Creating Stock Items

Once the preview shows the detected outlines matching your material, click
**Create Stock Items** in the header bar. The addon adds a stock asset and a
stock item to your document for each detected shape, positioned at the correct
physical coordinates on the canvas. The dialog closes after the items are
created.

## Related Topics

- [Camera Setup](../machine/camera) - Configure and calibrate your camera
- [Stock Handling](../features/stock-handling) - Work with stock items in your document
