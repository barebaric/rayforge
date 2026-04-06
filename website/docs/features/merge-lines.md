# Merge Lines

When you import a design that contains overlapping paths, the laser may end up
cutting the same line more than once. This wastes time, can cause excessive
charring, and may widen the kerf beyond what you intended.

The **Merge Lines** post-processor detects overlapping and coincident path
segments and merges them into a single pass. The laser follows each unique line
only once.

## When to Use It

This comes up most often when:

- You import an SVG or DXF where shapes share edges (for example, a grid pattern
  or tessellation)
- You combine multiple workpieces whose outlines overlap
- Your design software exports duplicate paths

## When Not to Use It

If overlapping cuts are intentional — for example, making multiple passes over
the same line to cut through thicker material — leave Merge Lines disabled. In
that case, you may want to use the [Multi-Pass](multi-pass) feature instead,
which gives you explicit control over the number of passes.

## Related Pages

- [Path Optimization](path-optimization) - Reducing unnecessary travel moves
- [Multi-Pass](multi-pass) - Intentional multiple passes over the same path
- [Contour Cutting](operations/contour) - The main cutting operation
