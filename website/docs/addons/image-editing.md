---
description:
  "Edit the image of a workpiece without changing the imported file, for example to invert it before
  engraving or tracing."
---

# Image Editing

The Image Editing addon changes how a workpiece's image looks without touching the imported file.
Each edit is stored with the workpiece, so you can undo it, remove it later, or place the same file
several times with different edits.

The canvas, the 3D preview and every operation use the edited image. What you see on the canvas is
what the laser engraves.

## Invert Image

Select one or more workpieces and choose **Tools - Invert Image**. Light areas become dark and dark
areas become light; transparent areas stay transparent, so the background is still not engraved.

![A grayscale photo of mountains and a lake, as imported](/images/image-edit-invert-before.webp)

![The same photo after Invert Image](/images/image-edit-invert-after.webp)

Choose it again to remove the inversion. With a mixed selection, the workpieces that are not
inverted yet get inverted, so the whole selection ends up the same.

Use it for lithophanes and embossing in an Engrave operation, where light areas should be engraved
deep, or to engrave light artwork on a dark background.

## Adjust Image

Select one or more workpieces and choose **Tools - Adjust Image...** to correct a photo before
engraving or tracing it:

- **Brightness** lightens or darkens the image.
- **Contrast** spreads the tones away from mid-gray, or pulls them towards it.
- **Gamma** above 1 lightens the midtones, below 1 darkens them; black and white stay as they are.
- **Sharpen** strengthens edges and fine details (an unsharp mask). **Sharpen Radius** sets the size
  of the details it works on, in millimeters, so the result is the same at any line interval or
  canvas zoom.

The canvas shows the changes while you edit. **Apply** keeps them as a single undo step; closing the
dialog without applying discards them, and **Reset** sets every value back to neutral. Transparent
areas stay transparent.

The dialog opens with the values of the first selected workpiece, and Apply gives every selected
workpiece the same values. Sharpening runs before the tone corrections, and other edits such as
Invert Image keep their place.

## Older Projects

Before this addon, Invert was a setting of the Engrave operation. When you open an older project
that used it, Rayforge moves the setting to the workpieces of that layer, so the output stays the
same.

## When the Addon Is Disabled

A project keeps its image edits when the addon is disabled or not installed. Rayforge tells you that
the project uses features that are not available and shows the images without those edits until the
addon is enabled again.
