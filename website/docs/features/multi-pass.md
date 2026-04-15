---
description: "Multi-pass cutting runs multiple passes over the same path to cut thicker materials. Configure pass count, power ramping, and progressive depth."
---

# Multi-Pass

Multi-pass repeats the cutting or engraving path multiple times, optionally stepping down in Z between passes. This is useful for thick materials or creating deeper engravings.

## How It Works

Each pass traces the same path again. With Z step-down enabled, the laser moves closer to the material between passes, progressively cutting deeper.

## Settings

### Number of Passes

How many times to repeat the entire step (1-100). Each pass follows the same path.

- **1 pass:** Single cut (default)
- **2-3 passes:** Common for medium-thick materials
- **4+ passes:** Very thick or hard materials

### Z Step-Down per Pass

Distance to lower the Z-axis between passes (0-50 mm). Only works if your machine has Z-axis control.

- **0 mm:** All passes at same depth (default)
- **Material thickness ÷ passes:** Progressive depth cutting
- **Small increments (0.1-0.5mm):** Fine control for deep engraving

:::warning Z-Axis Required
Z step-down only works with machines that have motorized Z-axis control. For machines without Z-axis, all passes occur at the same focus height.
:::

## When to Use Multi-Pass

**Cutting thick materials:**

Multiple passes at the same depth often cut cleaner than a single slow pass. The first pass creates a kerf, and subsequent passes follow the same path more efficiently.

**Deep engraving:**

With Z step-down, you can carve deep relief patterns or engravings that would be impossible in a single pass.

**Improved edge quality:**

Multiple faster passes often produce cleaner edges than one slow pass, especially in materials that char easily.

## Tips

- Start with 2-3 passes at your normal cutting speed
- For thick materials, increase passes rather than slowing down
- Enable Z step-down only if your machine supports it
- Test on scrap material to find optimal pass count

---

## Related Topics

- [Contour Cutting](operations/contour) - Primary cutting operation
- [Engrave](operations/engrave) - Engraving operations
