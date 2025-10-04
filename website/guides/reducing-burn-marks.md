# Reducing Burn Marks

Learn how to minimize burn marks and charring on your laser-cut and engraved projects using overscan, air assist, and other optimization techniques.

## Overview

Burn marks occur when heat accumulates at the edges or surface of your material during laser cutting or engraving. This guide covers techniques to reduce or eliminate these marks.

## Common Causes

- Laser dwell time at direction changes
- Insufficient air assist
- Excessive power or slow speed
- Poor material ventilation
- Dirty or misaligned optics

## Techniques

### 1. Use Overscan

Overscan extends raster engraving passes beyond the visible boundary, preventing the laser from dwelling at the edges:

1. **Enable overscan** in raster operation settings
2. **Set overscan distance** (typically 2-5mm)
3. **Test** with different values to find optimal setting

See [Overscan & Kerf](../features/overscan-kerf.md) for more details.

### 2. Optimize Air Assist

Proper air flow removes combustion byproducts and cools the material:

- **Adjust air pressure** to blow smoke away without disturbing material
- **Clean air nozzle** regularly
- **Position nozzle** close to the cut point
- **Use cross-draft** or exhaust fan for additional ventilation

### 3. Adjust Power and Speed

Find the optimal balance between power and speed:

- **Use higher speed, lower power** when possible
- **Run material tests** to find ideal settings
- **Multiple passes** at lower power can reduce burning
- **Consider material thickness** and density

See [Creating Material Test Grid](creating-material-test-grid.md) to find optimal settings.

### 4. Material Preparation

Prepare materials to minimize burning:

- **Mask material surfaces** with transfer tape
- **Remove protective film** if it contributes to burning
- **Use scrap backing** underneath cuts
- **Elevate material** on a honeycomb bed for airflow

### 5. Machine Maintenance

Keep your laser cutter in optimal condition:

- **Clean lenses and mirrors** regularly
- **Check beam alignment** periodically
- **Verify focus accuracy**
- **Inspect air assist system** for blockages

## Testing and Refinement

1. **Start with conservative settings** (higher speed, lower power)
2. **Run test cuts** on scrap material
3. **Adjust one parameter at a time**
4. **Document successful settings** for future reference

## Troubleshooting

**Heavy charring on edges**: Increase speed, reduce power, or improve air assist

**Burn marks at corners**: Enable overscan or reduce acceleration settings

**Uneven burning**: Check for bed flatness issues or material warping

## Related Topics

- [Overscan & Kerf](../features/overscan-kerf.md)
- [Creating Material Test Grid](creating-material-test-grid.md)
- [Achieving Perfect Focus](achieving-perfect-focus.md)
