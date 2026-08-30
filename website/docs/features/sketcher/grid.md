---
description:
  "Create a construction grid of rows and columns as drawing scaffolding in the Rayforge sketcher."
---

# Grid

The grid tool (`G+G`) creates a homogeneous grid of construction lines — rows and columns of evenly
spaced guides that serve as drawing scaffolding, for example to lay out a perforation pattern or to
align repeated elements.

![A 4x6 construction grid](/screenshots/addons-sketcher-tool-grid.webp)

1. Select the grid tool from the pie menu, the **Sketch** menu, or with `G+G`.
2. A dialog asks for the number of **rows** and **columns**.
3. Confirm to create the grid at the sketch origin with 10 mm cells.

The grid consists of construction geometry: it is drawn dashed, acts as snap and alignment reference
like any other geometry, and is excluded from the toolpaths when the sketch is manufactured (see
[Construction geometry](index.md#construction-geometry)). Individual lines can be moved or deleted
like any other geometry, and selecting them and toggling construction mode with `G+N` turns the
scaffolding into real geometry.
