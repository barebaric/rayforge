---
description: "Add lead-in and lead-out moves to contour cuts for smoother entry and exit points. Improve cut quality at starting positions."
---

# Lead-In / Lead-Out

Lead-in and lead-out moves extend each contour path with short zero-power
segments before the cut starts and after it ends. This gives the laser head
time to reach a steady speed before the actual cutting begins and to slow
down gradually after the cut finishes, which produces cleaner results at the
start and end points of each cut.

## How It Works

When lead-in/out is enabled, Rayforge looks at the tangent direction of each
contour path at its start and end points. It then inserts a short straight
move at zero laser power along that tangent before the first cutting point and
another after the last cutting point. The laser is off during these extra
segments, so no material is removed outside the intended path.

## Settings

### Enable Lead-In/Out

Turns the feature on or off for the operation. When disabled, cutting begins
and ends exactly at the path endpoints with no extra approach or exit moves.

### Automatic Distance

When this option is enabled, Rayforge calculates the lead-in and lead-out
distance automatically based on the cutting speed and the machine's
acceleration setting. The formula uses a safety factor of two to ensure the
laser head has enough room to reach full speed. Whenever you change the cutting
speed or the machine's acceleration is updated, the distance is recalculated.

### Lead-In Distance

The length of the zero-power approach move before the cut starts, in
millimeters. The default is 2 mm. This field is only editable when automatic
distance is turned off.

### Lead-Out Distance

The length of the zero-power exit move after the cut ends, in millimeters.
The default is 2 mm. This field is only editable when automatic distance is
turned off.

## When to Use Lead-In/Out

Lead-in/out is most helpful when you notice burn marks, over-burning, or
inconsistent cut quality at the start and end points of your contours. The
zero-power approach gives the machine time to accelerate to cutting speed so
that the laser reaches the material at full velocity, and the zero-power exit
lets it decelerate smoothly instead of lingering at full power on the last
point.

It is available as a post-processing option on contour, frame outline, and
shrink wrap operations.

---

## Related Topics

- [Contour Cutting](operations/contour) - Primary cutting operation
- [Frame Outline](operations/frame-outline) - Rectangular boundary cutting
- [Shrink Wrap](operations/shrink-wrap) - Efficient boundary cutting
- [Holding Tabs](holding-tabs) - Keeping parts secured during cutting
