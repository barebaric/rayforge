---
description:
  "Fill closed sketch regions with solid color or gradient fills in the Rayforge sketcher."
---

# Fill Areas

The fill tool (`G+F`) fills closed regions of a sketch with a solid area. Fills are useful for
regions that will be engraved as a single piece.

![A filled rectangle](/screenshots/addons-sketcher-tool-fill.webp)

## Creating and Removing Fills

1. Draw one or more closed contours (for example with the [rectangle](rectangle.md) or
   [path](path.md) tools).
2. Pick the fill tool from the pie menu, the **Sketch** menu, or press `G+F`.
3. Click anywhere inside a closed region to fill it.
4. Click a filled region again to remove its fill.

Clicking inside a text box toggles the fill of the text glyphs instead of creating a region fill.

## Fill Color

The fill color for new fills is picked with the **Fill color** button in the sketcher toolbar.
Existing fills keep their color until they are removed and recreated.

Like everything in the sketcher, a fill is tied to its boundary: resize the surrounding geometry and
the fill follows.
