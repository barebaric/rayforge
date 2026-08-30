---
description: "Bevel sharp corners with the chamfer tool or round them with the fillet tool in the Rayforge sketcher."
---

# Chamfer and Fillet

The sketcher provides two tools to modify corners where two lines meet:

- **Chamfer** (`C+H`): replaces a sharp corner with a beveled edge.
- **Fillet** (`C+F`): replaces a sharp corner with a rounded edge.

![A chamfered rectangle next to a filleted rectangle](/screenshots/addons-sketcher-tool-chamfer-fillet.webp)

To apply one of them:

1. Select a junction point where exactly two lines meet.
2. Press `C+H` for chamfer or `C+F` for fillet, or pick the tool from
   the pie menu.

The corner is replaced in a single step. The two lines are trimmed back
and the new edge is inserted between them, along with constraints that
keep the trimmed segments collinear with the originals and the corner
symmetric. On a chamfer, the bevel length defaults to a fraction of the
shorter adjoining line; on a fillet, the arc radius is chosen to fit.
Dragging the endpoints of the inserted edge afterwards adjusts its size,
with the constraints keeping the corner intact.
