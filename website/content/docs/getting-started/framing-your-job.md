# Framing Your Job

Learn how to use the framing feature to preview your job boundaries and ensure proper alignment before cutting.

## Overview

Framing allows you to preview the exact boundaries of your laser job by tracing an outline with the laser at low power or with the laser off. This helps verify positioning and prevent costly mistakes.

## When to Use Framing

- **First-time setups**: Verify material placement
- **Precise positioning**: Ensure design fits within material boundaries
- **Multiple jobs**: Confirm alignment before each run
- **Expensive materials**: Double-check before committing to cuts

## How to Frame

### Method 1: Outline Only

Trace the job boundary without turning on the laser:

1. **Load your design** in Rayforge
2. **Position material** on the laser bed
3. **Click the Frame button** in the toolbar
4. **Watch the laser head** trace the boundary rectangle
5. **Verify positioning** and adjust material if needed

### Method 2: Low Power Preview

Some machines support low-power framing with a visible beam:

1. **Enable low-power mode** in machine settings
2. **Set framing power** (typically 1-5%)
3. **Run frame operation**
4. **Observe the outline** traced on material surface

!!! warning "Check Your Machine"
    Not all lasers support low-power framing safely. Consult your machine documentation before using this feature.

## Frame Settings

Configure framing behavior in Settings → Machine:

- **Frame speed**: How fast the laser head moves during framing
- **Frame power**: Laser power during framing (0 for off, low % for visible trace)
- **Pause at corners**: Brief pause at each corner for visibility
- **Repeat count**: Number of times to trace the outline

## Using Frame Results

After framing, you can:

- **Adjust material position** if needed
- **Reframe** to verify new position
- **Proceed with job** once satisfied with placement

## Tips for Effective Framing

- **Mark corners**: Place small pieces of tape at corners for reference
- **Check clearance**: Ensure adequate space around your design
- **Verify orientation**: Confirm material is oriented correctly
- **Account for kerf**: Remember that cuts will be slightly wider than outlines

## Framing with Camera

If your machine has camera support, you can:

1. **Capture camera image** of material placement
2. **Overlay design** on camera view
3. **Adjust position** virtually before framing
4. **Frame to confirm** physical alignment

See [Camera Integration](../features/camera.md) for details.

## Troubleshooting

**Frame doesn't match design**: Check job origin and coordinate system settings

**Laser fires during frame**: Disable frame power or check machine settings

**Frame too fast to see**: Reduce frame speed in settings

**Head doesn't reach corners**: Verify design is within machine work area

## Safety Notes

- **Never leave machine unattended** during framing
- **Verify laser is off** if using zero-power framing
- **Keep hands clear** of the laser head path
- **Watch for obstructions** that could interfere with motion

## Related Topics

- [Camera Integration](../features/camera.md)
- [Machine Profiles](../machine/profiles.md)
- [Quick Start Guide](quick-start.md)
