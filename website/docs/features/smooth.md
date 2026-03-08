# Smooth Path

Path smoothing reduces jagged edges and sharp transitions in your cutting paths, resulting in cleaner curves and smoother machine motion.

## How It Works

Smoothing applies a filter to your path geometry that rounds off angular corners and smooths out rough edges. The laser follows a gentler trajectory instead of making abrupt direction changes.

## Settings

### Enable Smoothing

Toggle smoothing on or off for this operation. Smoothing is disabled by default.

### Smoothness

Controls how much the path is smoothed (0-100). Higher values produce rounder curves but may deviate more from the original path.

- **Low (0-30):** Minimal smoothing, preserves sharp details
- **Medium (30-60):** Balanced smoothing for most designs
- **High (60-100):** Aggressive smoothing, best for organic shapes

### Corner Angle Threshold

Angles sharper than this value are preserved as corners rather than smoothed (0-179 degrees). This prevents important sharp features from being rounded.

- **Lower values:** More corners are smoothed, rounder result
- **Higher values:** More corners are preserved, sharper result

## When to Use Smoothing

**Good for:**

- Designs imported from pixel-based sources with stair-stepping
- Reducing mechanical stress on rapid direction changes
- Improving cut quality on curves
- Designs with many small line segments

**Not needed for:**

- Clean vector artwork with smooth bezier curves
- Designs where sharp corners must be preserved exactly
- Technical drawings requiring precise geometry

---

## Related Topics

- [Contour Cutting](operations/contour) - Primary cutting operation
- [Path Optimization](path-optimization) - Reducing travel distance
