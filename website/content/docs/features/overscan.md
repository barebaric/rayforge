# Overscan

Overscan extends raster engraving lines beyond the actual content area to ensure the laser reaches constant velocity during engraving, eliminating acceleration artifacts.

## The Problem: Acceleration Marks

Without overscan, raster engraving suffers from **acceleration artifacts**:

**What happens:**

```
Laser head motion during a raster line:

Without overscan:
  Accelerating  Constant Speed  Decelerating
  ___/>>>>>>>>>>>\___
     ^           ^
  Too light   Too dark
  (moving fast) (moving slow)
```

**Visual result:**

- **Light edges** where acceleration starts (laser moving too fast for power level)
- **Dark edges** where deceleration occurs (laser dwelling longer)
- **Inconsistent engraving depth/darkness** across each line
- Visible banding or streaking at line edges

## How Overscan Works

Overscan **extends the toolpath** before and after each raster line:

```
With overscan:

            Constant Speed
 Accel    Actual Engraving    Decel
(Laser OFF)                   (Laser OFF)
```

**Process:**

1. **Lead-in:** Laser moves to a position _before_ the line starts
2. **Accelerate:** Laser accelerates to target speed (laser OFF)
3. **Engrave:** Laser turns on and engraves at constant speed
4. **Decelerate:** Laser turns off and decelerates _after_ the line ends

**Result:** The entire engraved area receives consistent power at constant velocity.

**Benefits:**

- Even engraving depth across entire raster line
- No light/dark edges
- Higher quality photo engraving
- Professional-looking results

## Configuring Overscan

Overscan is a **transformer** in the Rayforge workflow pipeline.

**To enable:**

1. **Select the layer** with raster engraving
2. **Open workflow settings** (or operation settings)
3. **Add Overscan transformer** if not already present
4. **Configure distance**

**Settings:**

| Setting           | Description             | Typical Value   |
| ----------------- | ----------------------- | --------------- |
| **Enabled**       | Toggle overscan on/off  | ON (for raster) |
| **Distance (mm)** | How far to extend lines | 2-5 mm          |

## Choosing Overscan Distance

The overscan distance should allow the machine to **fully accelerate** to target speed.

**Formula (approximate):**

```
distance = (speed^2) / (2 * acceleration)
```

**Practical guidelines:**

| Max Speed              | Acceleration | Recommended Overscan |
| ---------------------- | ------------ | -------------------- |
| 3000 mm/min (50 mm/s)  | Low          | 5 mm                 |
| 3000 mm/min (50 mm/s)  | Medium       | 3 mm                 |
| 3000 mm/min (50 mm/s)  | High         | 2 mm                 |
| 6000 mm/min (100 mm/s) | Low          | 10 mm                |
| 6000 mm/min (100 mm/s) | Medium       | 6 mm                 |
| 6000 mm/min (100 mm/s) | High         | 4 mm                 |

**Factors affecting required distance:**

- **Speed:** Higher speed = need more distance to accelerate
- **Acceleration:** Lower acceleration = need more distance
- **Machine mechanics:** Belt-driven vs direct-drive affects acceleration

**Tuning:**

- **Too little:** Acceleration marks still visible at edges
- **Too much:** Wastes time, may hit machine boundaries
- **Start with 3mm** and adjust based on results

## Testing Overscan Settings

**Test procedure:**

1. **Create a test engraving:**

   - Solid filled rectangle (50mm x 20mm)
   - Use your typical engraving settings
   - Enable overscan at 3mm

2. **Engrave the test:**

   - Run the job
   - Allow to complete

3. **Examine the edges:**

   - Look at left and right edges of the rectangle
   - Check for darkness variation at edges
   - Compare edge darkness to center darkness

4. **Adjust:**
   - **If edges are lighter/darker:** Increase overscan
   - **If edges match center:** Overscan is sufficient
   - **If edges are perfect:** Try reducing overscan slightly to save time

**Visual inspection guide:**

```
Too little overscan:     Correct overscan:
|LIGHT|DARK|LIGHT|       |EVEN|EVEN|EVEN|
 edge center edge         edge center edge
```

## When to Use Overscan

**Always use for:**

- Photo engraving (raster)
- Fill patterns
- Any high-detail raster work
- Grayscale image engraving
- Text engraving (raster mode)

**Not needed for:**

- Vector cutting (contour operations)
- Very slow engraving (acceleration is negligible)
- Machines with very high acceleration (rare)
- Single-line operations (no raster scanning)

**Recommended by default:** Enable overscan for all raster operations unless you have a specific reason not to.

## Overscan and Machine Boundaries

**Important:** Overscan extends toolpaths beyond the visible content.

**Example:**

- Content: 100mm x 100mm square
- Overscan: 3mm
- **Actual motion:** 106mm x 100mm (3mm added to each side horizontally)

**Space requirements:**

- Horizontal overscan: Added to left and right of content
- Vertical: No overscan added (raster lines run horizontally)
- Total width: Content width + (2 x overscan distance)

**Ensure:**

- The extended paths fit within machine work area
- Content is positioned to allow for overscan extension
- Use Frame function to verify boundaries before running

**Out of bounds warning:**

If overscan causes paths to exceed machine limits, Rayforge will warn you:

```
Warning: Overscan extends beyond machine boundaries.
Reduce overscan distance or reposition the job.
```

**Solutions:**

- Reduce overscan distance
- Move content further from edges
- Reduce content size
- Change work area settings if machine is actually larger

## Technical Details

**How Rayforge implements overscan:**

1. **Identifies raster sections** in the operation
2. **For each raster line:**
   - Calculates line direction vector
   - Extends start point backward by `distance_mm`
   - Extends end point forward by `distance_mm`
3. **Wraps with power control:**
   - Moves to extended start (laser OFF)
   - Turns laser ON at actual start
   - Engraves content at constant speed
   - Turns laser OFF at actual end
   - Continues to extended end (deceleration zone)

**For variable power (photo engraving):**

- Pads the power data array with zeros at edges
- Ensures smooth power ramp at edges
- No abrupt power changes that could cause artifacts

**G-code representation:**

```gcode
; Without overscan
G0 X10 Y20        ; Move to start
M4 S500           ; Laser on
G1 X50 F3000      ; Engrave line
M5                ; Laser off

; With overscan (3mm)
G0 X7 Y20         ; Move to overscan start (10-3=7)
G1 X10 F3000      ; Accelerate to speed (laser off)
M4 S500           ; Laser on at content start
G1 X50            ; Engrave content
M5                ; Laser off at content end
G1 X53            ; Continue to overscan end (50+3=53)
```

## Bidirectional vs Unidirectional Raster

Overscan affects both scanning directions:

**Bidirectional raster (default):**

- Raster lines alternate: left-to-right, right-to-left
- Overscan extends both directions
- Faster (no time wasted returning to left side)
- May show slight misalignment if machine backlash exists

**Unidirectional raster:**

- All lines go left-to-right only
- Head returns to left side between lines (laser off)
- Slower but more consistent
- Overscan still used for acceleration/deceleration

**Overscan is important for both modes** to ensure constant velocity during actual engraving.

## Advanced: Variable Overscan

Some advanced use cases may need different overscan distances:

**Slow engraving:**

- Lower speeds need less overscan
- Can reduce to 1-2mm to save time/space

**Fast engraving:**

- Higher speeds need more overscan
- May need 5-10mm for very fast raster work

**Per-layer configuration:**

- Set different overscan for different layers
- Fast raster layer: 5mm overscan
- Slow detail layer: 2mm overscan

## Troubleshooting

### Acceleration Marks Still Visible

**Problem:** Edges of engraving are lighter or darker than center despite overscan being enabled.

**Diagnosis:**

- Overscan distance too small for current speed/acceleration
- Overscan not actually enabled
- Machine acceleration settings wrong

**Solutions:**

1. Increase overscan distance (try doubling it)
2. Verify overscan is enabled in layer workflow
3. Reduce engraving speed (less acceleration needed)
4. Check machine acceleration settings in firmware

### Overscan Causes Out-of-Bounds Error

**Problem:** Rayforge reports job exceeds machine boundaries when overscan is enabled.

**Diagnosis:**

- Content positioned too close to edge
- Overscan distance too large
- Machine work area configured incorrectly

**Solutions:**

1. Reduce overscan distance
2. Reposition content further from edges (at least overscan distance + 5mm margin)
3. Reduce content size slightly
4. Verify machine work area dimensions in settings

### Inconsistent Results

**Problem:** Some lines show artifacts, others don't.

**Diagnosis:**

- Mechanical issues (belt tension, binding)
- Variable material (thickness changes)
- Firmware acceleration settings inconsistent

**Solutions:**

1. Check belt tension and mechanical components
2. Use flat, consistent material for testing
3. Verify firmware acceleration settings ($120/$121 in GRBL)
4. Clean linear rails and ensure smooth motion

### Engraving Takes Too Long

**Problem:** Job time increased significantly with overscan.

**Diagnosis:**

- Large overscan distance adds travel time
- Many short raster lines amplify time cost

**Analysis:**

- Each line adds: 2 x overscan distance to travel
- 100 lines with 5mm overscan = 1000mm extra travel

**Solutions:**

1. Reduce overscan to minimum needed (test to find smallest acceptable value)
2. Increase engraving speed if quality allows
3. Combine raster lines where possible
4. Accept the time cost for quality improvement

## Best Practices

1. **Enable by default for raster:**

   - Always use overscan for photo engraving
   - Enable for any raster operations
   - Disable only if you have specific reasons

2. **Start with 3mm:**

   - Good baseline for most machines
   - Test and adjust from there
   - Document optimal value for your machine

3. **Test on your machine:**

   - Every machine is different
   - Run test engraving to find ideal distance
   - Re-test if you change acceleration settings

4. **Account for boundaries:**

   - Position content with overscan in mind
   - Leave margin from work area edges
   - Frame before running to verify

5. **Balance quality vs time:**
   - More overscan = better quality
   - But also longer job time
   - Find the sweet spot for your needs

## Related Pages

- [Raster Engraving](operations/raster.md) - Raster operations that use overscan
- [Multi-Layer Workflow](multi-layer.md) - Organizing layers with different overscan settings
- [Kerf Compensation](kerf.md) - Related feature for cutting accuracy
- [Power vs Speed](../concepts/power-vs-speed.md) - Understanding speed effects on quality
