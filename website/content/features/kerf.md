# Kerf Compensation

Kerf is the material removed by the laser beam during cutting. Kerf compensation adjusts toolpaths to account for this, ensuring cut parts match their designed dimensions.

## What is Kerf?

**Kerf** = the width of material removed by the cutting process.

When a laser cuts material, it vaporizes or melts a path through the material. This removed material creates a gap - the kerf.

**Components of kerf:**
- Laser spot size (beam diameter at focal point)
- Heat-affected zone (material vaporized beyond spot)
- Material interaction (burning, melting expands the cut)

**Example:**
- Laser spot size: 0.2mm
- Material interaction: adds ~0.1mm on each side
- **Total kerf:** ~0.4mm

## The Problem Without Compensation

Without kerf compensation, parts don't match their designed dimensions:

**Outside cuts (cutting a part):**
```
Designed:     Actual cut:

  50mm        49.6mm      Part too small
                          (kerf removed from part)
```

**Inside cuts (cutting a hole):**
```
Designed:     Actual cut:

  50mm        50.4mm      Hole too big
                          (kerf removed into hole)
```

**Why this happens:**

- Laser follows the design path exactly
- Material removal occurs on both sides of the path
- Half the kerf is removed from each side
- Final dimensions are off by the kerf width

## When Kerf Matters

**Critical for:**
- Precision assemblies (tight tolerances < 0.5mm)
- Snap-fit assemblies
- Interlocking designs (boxes, puzzles, gears)
- Mechanical parts with specific dimensions
- Parts that must fit together

**Not critical for:**
- Decorative cutting (tolerance > 1mm acceptable)
- Single-piece projects with no assembly
- Artistic work where exact dimensions aren't important
- Engraving (no kerf in engraving)

**Rule of thumb:** If parts need to fit together, compensate for kerf.

## How Kerf Compensation Works

Kerf compensation **offsets the toolpath** inward or outward to account for material removal:

**For outside cuts (cutting a part):**
- Offset path **outward** by half the kerf width
- Laser cuts slightly outside the design
- After kerf is removed, part matches design size

**For inside cuts (cutting a hole):**
- Offset path **inward** by half the kerf width
- Laser cuts slightly inside the design
- After kerf is removed, hole matches design size

**Example with 0.4mm kerf:**

```
Original design:  50mm square (part)
Kerf measured:    0.4mm total width
Compensation:     Offset outward by 0.2mm (half kerf)
Laser follows:    50.4mm square path
After cutting:    Part measures 50.0mm (perfect!)
```

**Visual representation:**

```
Design path:        [==========]  50mm
Kerf compensation: --[==========]-- offset 0.2mm out
Laser cuts here:    [============]  50.4mm
Material removed:    X          X   0.2mm each side
Final part:         [==========]   50.0mm
```

## Measuring Kerf

Accurate kerf measurement is essential for proper compensation.

**Test procedure:**

1. **Create a test file:**
   - Draw a 50mm x 50mm square (for outside cut test)
   - Draw a 30mm diameter circle (for inside cut test)
   - Export to RayForge

2. **Cut the test:**
   - Use your normal cutting settings (power/speed)
   - Cut completely through material
   - Let material cool completely

3. **Measure with calipers:**

   **Outer square (part):**
   - Measure actual width and height
   - If part measures 49.6mm: kerf removed 0.4mm total (0.2mm per side)
   - Kerf = (Designed - Measured)
   - Example: 50 - 49.6 = 0.4mm kerf

   **Inner circle (hole):**
   - Measure actual diameter
   - If hole measures 30.4mm: kerf added 0.4mm total
   - Kerf = (Measured - Designed)
   - Example: 30.4 - 30 = 0.4mm kerf

4. **Average the results:**
   - Take multiple measurements (at least 3)
   - Average them for accuracy
   - Use this kerf value for compensation

**Measurement tips:**
- Use digital calipers for accuracy (not a ruler)
- Measure at multiple points and average
- Ensure material has cooled (hot material may be slightly larger)
- Clean edges of test pieces before measuring

## Typical Kerf Values

Kerf varies by material, thickness, and cutting settings:

| Material | Thickness | Typical Kerf | Notes |
|----------|-----------|--------------|-------|
| **Cardboard** | 1-3mm | 0.2-0.3mm | Thin, burns easily |
| **Plywood** | 3mm | 0.3-0.4mm | Standard cutting |
| **Plywood** | 6mm | 0.4-0.6mm | Thicker, wider kerf |
| **Acrylic** | 3mm | 0.2-0.3mm | Clean, narrow kerf |
| **Acrylic** | 6mm | 0.3-0.5mm | Thicker material |
| **MDF** | 3mm | 0.3-0.5mm | Dense, wider kerf |
| **Hardwood** | 3mm | 0.3-0.4mm | Varies by species |
| **Leather** | 2-3mm | 0.2-0.3mm | Soft material |

**Factors affecting kerf:**
- **Laser power:** Higher power = wider kerf
- **Cutting speed:** Slower speed = wider kerf
- **Material density:** Denser materials = narrower kerf
- **Focus distance:** Optimal focus = narrower kerf
- **Air assist:** Strong air assist = slightly wider kerf

**Important:** Always measure kerf for your specific material and settings. Don't rely on typical values.

## Kerf Compensation in RayForge

**Current implementation status:**

RayForge may not have a dedicated built-in kerf compensation feature yet. Check your version for:

- Workflow transformers labeled "Offset" or "Kerf"
- Operation settings with offset options
- Manual path offset tools

**If available:**

1. **Measure your kerf** (see above)
2. **Add kerf transformer** to workflow
3. **Enter offset value:** Half the measured kerf
4. **Select direction:**
   - Outward for parts (outside cuts)
   - Inward for holes (inside cuts)
5. **Apply to cutting operations** only (not engraving)

**If not available:** Use manual compensation methods (see below).

## Manual Kerf Compensation

If automated kerf compensation isn't available in RayForge, compensate in your design software before import.

### Inkscape

**Offset path method:**

1. **Select the path** you want to offset
2. **Path > Dynamic Offset** (or Ctrl+J)
3. **Drag the handle** to offset by half your kerf measurement
   - Outward (larger) for parts
   - Inward (smaller) for holes
4. **Click outside** to apply
5. **Path > Object to Path** to finalize the offset
6. **Delete original path**, keep offset path
7. **Export to SVG** for RayForge

**Linked offset (advanced):**
- Dynamic Offset creates a live link
- You can adjust offset later by selecting and dragging handle
- Useful for testing different kerf values

### Adobe Illustrator

**Offset path method:**

1. **Select the path**
2. **Object > Path > Offset Path**
3. **Enter offset value:**
   - Positive value (e.g., +0.2mm) for outward offset (parts)
   - Negative value (e.g., -0.2mm) for inward offset (holes)
4. **Click OK**
5. **Delete original path**, keep offset path
6. **Export to SVG** or other format

**Tips:**
- Preview offset before applying
- Check units (mm vs inches)
- Offset creates new path, original remains (delete original)

### Fusion 360 / CAD Software

**Sketch offset:**

1. **In your sketch:** Select entities to offset
2. **Use offset tool** in sketch toolbar
3. **Enter offset distance:** Half your kerf value
4. **Select direction:** Outward for parts, inward for holes
5. **Apply offset**
6. **Export DXF** with offset paths

**Benefits of CAD offset:**
- Parametric (easy to change later)
- Precise dimensional control
- Can create offset as separate sketch for comparison

### Manual Scaling (Simple but Inaccurate)

**For simple rectangular parts only:**

1. **Calculate scale factor:**
   - For outside cuts: Scale up by kerf
   - For inside cuts: Scale down by kerf
2. **Apply uniform scaling** to design
3. **Export**

**Not recommended because:**
- Only works for simple shapes
- Non-uniform kerf compensation needed for complex shapes
- Corners don't offset correctly

## Inside vs Outside Cuts

**Determining direction:**

**Outside cut (cutting a part):**
- Path defines the part outline
- Material outside the path is waste
- **Offset outward** (path gets larger)
- Examples: cutting a shape from a sheet, making a part

**Inside cut (cutting a hole):**
- Path defines the hole outline
- Material inside the path is removed
- **Offset inward** (path gets smaller)
- Examples: cutting holes in a part, creating negative space

**Complex parts with both:**
- Separate into different layers or paths
- Outside cuts: offset outward
- Inside cuts: offset inward
- Apply different compensation to each

**Example (box with holes):**
```
Box outline (outside): Offset +0.2mm outward
Ventilation holes (inside): Offset -0.2mm inward
```

## Testing Kerf Compensation

**Verification procedure:**

1. **Create a test assembly:**
   - Part A: 50mm x 10mm tab
   - Part B: 50mm x 10mm slot
   - Designed to fit together snugly

2. **Apply kerf compensation:**
   - Tab (outside cut): offset outward by half kerf
   - Slot (inside cut): offset inward by half kerf

3. **Cut both parts**

4. **Test fit:**
   - Tab should fit into slot snugly
   - Not too tight (can't insert)
   - Not too loose (wobbles)

5. **Adjust if needed:**
   - Too tight: reduce compensation (use smaller kerf value)
   - Too loose: increase compensation (use larger kerf value)
   - Re-cut test parts with adjusted values

**Iterative refinement:**
- First test gives you a baseline
- Adjust compensation based on fit
- Re-test until perfect fit achieved
- Document final kerf value for future projects

## Advanced: Variable Kerf

**Why kerf varies:**

Different parts of the same design may have different effective kerf due to:

- **Corner vs straight:** Corners may have wider kerf (laser dwells longer)
- **Power variation:** Different power settings = different kerf
- **Speed variation:** Different speeds = different kerf

**Dealing with variable kerf:**

1. **Test at actual settings:** Measure kerf using the exact power/speed you'll use
2. **Separate by settings:** Different layers for different power/speed combos
3. **Measure corners separately:** Some materials need different compensation at corners
4. **Iterate:** Test, measure, adjust, repeat

**Advanced compensation:**
- Some CAD software allows variable offset along a path
- Useful for compensating corner differences
- Complex, usually not necessary for most work

## Kerf and Multi-Pass Cutting

**Multi-pass affects kerf:**

- First pass: Normal kerf
- Second pass: May widen the kerf slightly (cuts into already-cut edges)
- Result: Multi-pass kerf is often slightly wider

**Measuring multi-pass kerf:**
- Use the same number of passes you'll use in production
- Measure kerf after all passes complete
- Apply that compensation value

## Best Practices

1. **Always measure kerf for your setup:**
   - Your machine, material, and settings are unique
   - Don't rely on typical values or others' measurements
   - Re-measure when changing materials or settings

2. **Test before production:**
   - Cut a test assembly to verify compensation
   - Adjust based on actual fit
   - Document final values for reuse

3. **Use consistent settings:**
   - Kerf changes if power/speed changes
   - Keep cutting settings consistent for repeatable results
   - Re-measure if you change settings

4. **Consider material variation:**
   - Different batches of same material may behave differently
   - Test each new batch if precision is critical
   - Keep sample test cuts for reference

5. **When in doubt, test:**
   - Cut a small test piece before committing to full job
   - Verify dimensions before proceeding
   - Better to waste one test piece than an entire sheet

## Troubleshooting

### Parts Still Wrong Size After Compensation

**Problem:** Applied kerf compensation but parts are still incorrect dimensions.

**Diagnosis:**
1. Measure the error: How much too big/too small?
2. Check if error matches kerf (forgot to apply?) or is different amount
3. Verify offset direction (should be outward for parts, inward for holes)

**Solutions:**
- Re-measure kerf (may have measured incorrectly)
- Check offset direction (may be backwards)
- Verify compensation was actually applied (check paths in design software)
- Re-calculate offset (use half kerf, not full kerf)

### Inconsistent Results

**Problem:** Some parts fit perfectly, others don't, using same compensation.

**Diagnosis:**
- Material thickness variation
- Laser power variation (dirty optics)
- Focus height inconsistent

**Solutions:**
- Use more consistent material
- Clean optics before cutting
- Verify focus height is correct and consistent
- Check belt tension and mechanical components

### Corners Don't Fit Right

**Problem:** Straight edges fit well, but corners are too tight or loose.

**Diagnosis:**
- Laser dwells at corners (more kerf)
- Sharp corners vs rounded corners behave differently

**Solutions:**
- Add slight radius to sharp corners (e.g., 0.5mm fillet)
- Test corner fit specifically
- May need different compensation for corners (advanced)

## Related Pages

- [Overscan](overscan.md) - Related feature for raster quality
- [Contour Cutting](operations/contour.md) - Cutting operations that use kerf compensation
- [Calibrating Your Workspace](../guides/calibrating-your-workspace.md) - Dimensional accuracy
- [Power vs Speed](../concepts/power-vs-speed.md) - How settings affect kerf
