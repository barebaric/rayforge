---
description:
  "Cut dashed lines in Rayforge with the Perforation post-processor: set a cut length and a skip
  length for fold lines, tear-off lines and stitch holes."
---

# Perforation

The **Perforation** post-processor turns a continuous cut into a dashed one. The laser fires for the
**Cut Length**, then travels with the laser off for the **Skip Length**, and repeats this along
every contour of the workpiece.

## When to Use It

- Fold lines in card stock and corrugated cardboard
- Tear-off lines (tickets, coupons, packaging)
- Stitch holes in leather
- Dashed lines for decoration

## Settings

Perforation is available in the post-processing settings of **Contour** operations. It is off by
default.

- **Cut Length**: the distance the laser fires before each gap.
- **Skip Length**: the distance travelled with the laser off between two cuts.

The pattern is measured along each contour of the workpiece and starts again at the start of every
contour, beginning with a full cut. A contour shorter than one cut plus one skip is cut in full.

## Tips

- For a fold line that should not cut through, start with a skip length about as long as the cut
  length and lower the power.
- Very short lengths (a few tenths of a millimetre) make the laser switch on and off rapidly, which
  lowers the effective power.
- Perforation works together with [Holding Tabs](holding-tabs), [Lead-In/Out](lead-in-out) and
  [Multi-Pass](multi-pass).

## Related Pages

- [Contour Cutting](operations/contour) - The cutting operation that uses perforation
- [Holding Tabs](holding-tabs) - Gaps placed by hand to keep parts attached
