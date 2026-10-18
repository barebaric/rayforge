---
description:
  "Optimize laser cutting paths to minimize travel distance. Reduce job time and improve cut quality
  with intelligent path ordering in Rayforge."
---

# Path Optimization

Path optimization reorders cutting segments to minimize travel distance. The laser moves efficiently
between cuts instead of jumping randomly across the work area.

## How It Works

Without optimization, paths are cut in the order they appear in your design file. Optimization
analyzes all path segments and rearranges them so the laser travels the shortest total distance
between cuts.

**Before optimization** — the laser jumps back and forth across the material:

![Travel moves zigzag between cuts in drawn order](/images/path-optimization-before.webp)

**After optimization** — the laser moves sequentially from cut to cut:

![Travel moves connect neighboring cuts](/images/path-optimization-after.webp)

## Settings

![The Optimize settings on the post-processing page of a Contour step](/screenshots/step-settings-contour-post-optimize.webp)

### Enable Optimization

Toggle path optimization on or off. Enabled by default for most operations.

### Choose Best Start Point

Starts each closed path at the vertex nearest the laser head instead of where the path was drawn.
This shortens travel moves. Open paths, and paths with lead-ins, overcut or tabs, keep their start
point. Off by default.

### Prefer Corners

Only available when **Choose Best Start Point** is on. Closed paths start at a sharp corner where
the path has one, which makes the seam less visible. Paths without corners, such as circles, fall
back to the nearest vertex. Off by default.

The same closed path entered where it was drawn (1), at the vertex nearest the laser head with
**Choose Best Start Point** (2), and at the nearest sharp corner with **Prefer Corners** (3):

![Entry points of a closed path: as drawn, at the nearest vertex, at the nearest corner](/images/path-optimization-start-point.webp)

## When to Use Optimization

**Enable for:**

- Designs with many separate shapes
- Reducing total job time
- Minimizing wear on motion system
- Complex nested layouts

**Disable for:**

- Designs where cut order matters (e.g., inside features before outside)
- Debugging path issues
- When you need predictable, repeatable execution order

## How It Affects Your Job

**Time savings:** Can reduce job time by 20-50% for designs with many separate cuts.

**Motion efficiency:** Less rapid movement means less wear on belts, motors, and bearings.

**Heat distribution:** Optimized paths may concentrate heat in one area. For heat-sensitive
materials, consider if order matters.

<!-- prettier-ignore-start -->
:::tip
Optimization runs automatically. Just enable it and the software handles the rest.
:::
<!-- prettier-ignore-end -->

---

## Related Topics

- [Contour Cutting](operations/contour) - Primary cutting operation
- [Holding Tabs](holding-tabs) - Keeping parts secure during cutting
