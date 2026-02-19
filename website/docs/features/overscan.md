# Overscan

Overscan extends raster engraving lines beyond the actual content area to ensure the laser reaches constant velocity during engraving, eliminating acceleration artifacts.

## The Problem: Acceleration Marks

Without overscan, raster engraving suffers from **acceleration artifacts**:

- **Light edges** where acceleration starts (laser moving too fast for power level)
- **Dark edges** where deceleration occurs (laser dwelling longer)
- **Inconsistent engraving depth/darkness** across each line
- Visible banding or streaking at line edges

## How Overscan Works

Overscan **extends the toolpath** before and after each raster line:

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

## When to Use Overscan

**Always use for:**

- Photo engraving (raster)
- Fill patterns
- Any high-detail raster work
- Grayscale image engraving
- Text engraving (raster mode)

**Optional for:**

- Vector cutting (not needed)
- Very slow engraving (acceleration less noticeable)
- Large simple shapes (edges less critical)

**Disable for:**

- Vector operations
- Very small work areas (may exceed boundaries)
- When edge quality is not important

---

## Related Topics

- [Engraving Operations](engrave) - Configure engraving settings
- [Material Test Grid](material-test-grid) - Find optimal power/speed settings
