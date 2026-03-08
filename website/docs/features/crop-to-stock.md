# Crop to Stock

Crop to Stock limits cutting paths to your material boundary. Any cuts that extend beyond the stock area are trimmed, preventing the laser from cutting outside your material.

## How It Works

The transformer compares your cutting paths against the defined stock boundary. Path segments outside this boundary are removed or clipped to the stock edge.

## Settings

### Enable Crop-to-Stock

Toggle cropping on or off. Disabled by default.

### Offset

Adjust the effective stock boundary before cropping (-100 to +100 mm).

- **Positive values:** Shrink the boundary (cuts more conservatively)
- **Negative values:** Expand the boundary (allows cuts closer to edge)
- **0 mm:** Use exact stock boundary

Use offset when you want a safety margin from the stock edge, or when your material placement isn't perfectly aligned.

## When to Use Crop to Stock

**Partial designs:** Your design is larger than your material, but you want to cut only the portion that fits.

**Safety margin:** Prevent accidental cuts beyond material edges.

**Nested sheets:** Cut only the parts that fit on your current piece of material.

**Test cuts:** Limit a test to a specific area of your material.

## Example

You have a large design but only a small piece of material:

1. Define your stock size to match your material
2. Enable Crop to Stock
3. Set offset to 2mm for safety margin
4. Only the portions within your material boundary will be cut

---

## Related Topics

- [Stock Handling](stock-handling) - Setting up material boundaries
- [Contour Cutting](operations/contour) - Primary cutting operation
