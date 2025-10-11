# Creating Material Test Grid

Find optimal laser settings for new materials using RayForge's built-in Material Test Grid generator.

## Goal

Quickly determine the best power and speed settings for cutting or engraving a material by running a systematic test that tries multiple combinations automatically.

## Prerequisites

- RayForge installed and connected to laser
- Test material (scrap piece of the material you want to test)
- Basic understanding of laser power and speed concepts
- Machine homed and ready

## Why Use Material Test Grid?

Every material responds differently to laser cutting:

- Different woods burn at different rates
- Acrylic thickness affects required power
- Material batches vary (different moisture, density, etc.)

**Manual testing is tedious:**
- Test at 50% power, 500mm/min - doesn't cut through
- Try 60% power - still not enough
- Try 70% power - works but too much charring
- Try different speeds...
- Hours of trial and error

**Material Test Grid automates this:**
- Test 16-25 power/speed combinations in one job
- Organized grid makes results easy to compare
- Find optimal settings in 5-10 minutes

## Step 1: Prepare Test Material

Set up material for the test grid.

**Material requirements:**

- **Size:** At least 50mm x 50mm (larger is better, e.g., 100mm x 100mm)
- **Condition:** Representative of material you'll actually use
  - Same thickness
  - Same batch/supplier
  - Similar moisture content
- **Placement:** Flat, secured on laser bed

**Secure material:**
- Use tape, magnets, or hold-downs
- Ensure material won't shift during test
- Position within laser work area

## Step 2: Access Material Test Grid Generator

Open the Material Test Grid feature in RayForge.

**Navigate to generator:**

1. Open RayForge
2. Go to **Features > Operations > Material Test Grid**
3. Material Test Grid configuration dialog opens

**Alternative access:**
- Some versions may have it in: **Tools > Material Test Grid**
- Or in operation selection dropdown

## Step 3: Configure Test Parameters

Set up the power and speed ranges to test.

### Grid Size

**Number of cells:**
- **4x4 grid:** 16 test cells (faster, less comprehensive)
- **5x5 grid:** 25 test cells (recommended, good coverage)

**Choose based on:**
- Unknown material: 5x5 for wider range
- Similar to known material: 4x4 to refine settings

### Cell Size

**Default:** 10mm x 10mm per cell

**Adjust if needed:**
- Smaller cells (8mm): Save material, harder to evaluate
- Larger cells (15mm): Easier to see results, uses more material

### Power Range

**What to test:**

Define minimum and maximum power to test:

- **Min Power:** Starting power (e.g., 20%)
- **Max Power:** Ending power (e.g., 80%)

**Grid will test evenly spaced values between min and max.**

**Example (5x5 grid, 20-80% power):**
- Row 1: 20% power
- Row 2: 35% power
- Row 3: 50% power
- Row 4: 65% power
- Row 5: 80% power

**How to choose range:**

| Material Type | Suggested Power Range | Notes |
|---------------|----------------------|-------|
| **Thin paper/cardboard** | 10% - 30% | Very low power needed |
| **3mm wood** | 40% - 80% | Medium power |
| **6mm wood** | 60% - 100% | Higher power for thick material |
| **3mm acrylic** | 70% - 100% | High power for clean cuts |
| **Engraving (any)** | 10% - 40% | Lower power for surface marking |

**Tip:** If unsure, use wider range (e.g., 20% - 100%) to ensure you capture working settings.

### Speed Range

**What to test:**

Define minimum and maximum speed to test:

- **Min Speed:** Slowest speed (e.g., 100 mm/min)
- **Max Speed:** Fastest speed (e.g., 1000 mm/min)

**Grid will test evenly spaced values between min and max.**

**Example (5x5 grid, 100-1000 mm/min):**
- Column 1: 100 mm/min
- Column 2: 325 mm/min
- Column 3: 550 mm/min
- Column 4: 775 mm/min
- Column 5: 1000 mm/min

**How to choose range:**

| Material Type | Suggested Speed Range | Notes |
|---------------|----------------------|-------|
| **Thin paper/cardboard** | 1000 - 3000 mm/min | Fast speeds for thin material |
| **3mm wood** | 200 - 1000 mm/min | Moderate speeds |
| **6mm wood** | 100 - 500 mm/min | Slower for thick material |
| **3mm acrylic** | 100 - 400 mm/min | Very slow for clean cuts |
| **Engraving (any)** | 1000 - 4000 mm/min | Fast for surface engraving |

### Test Type

**Choose operation type:**

- **Cut:** Tests through-cutting (Contour operation)
- **Engrave:** Tests surface engraving (Raster operation)

**Select based on your goal:**
- Testing for cutting projects: Choose "Cut"
- Testing for engraving projects: Choose "Engrave"

### Preset Selection (Optional)

Some versions offer presets to auto-fill parameters:

**Common presets:**
- **Diode Laser - Cut:** Low power range, moderate speeds
- **Diode Laser - Engrave:** Very low power, high speeds
- **CO2 Laser - Cut:** Medium/high power, moderate speeds
- **CO2 Laser - Engrave:** Low power, high speeds

**Use presets to:**
- Get reasonable starting ranges
- Save time configuring
- Ensure you're in the right ballpark

**You can still adjust preset values** to fine-tune ranges.

## Step 4: Review and Generate Grid

Check settings and generate the test grid.

**Summary view:**

RayForge shows:
- Grid dimensions (e.g., 5x5, 50mm x 50mm total)
- Power range (20% to 80%)
- Speed range (100 to 1000 mm/min)
- Number of test cells (25)
- Estimated time (varies by settings)

**Click "Generate" or "Create Grid":**

- Grid workpiece is added to canvas
- Grid appears as a series of small squares
- Each square represents one power/speed combination

## Step 5: Position the Grid

Place the grid on your canvas/material.

**Position grid:**

1. Grid appears on canvas in RayForge
2. Drag to position where your test material is located
3. Use job framing to verify position on actual laser bed

**Framing:**

1. Click **Frame Job**
2. Laser traces outline of grid boundary
3. Verify grid will fit on your test material
4. Adjust position if needed

## Step 6: Run the Test Grid

Execute the material test.

**Safety check:**
- Ventilation running
- Fire extinguisher ready
- Test material secured
- Emergency stop accessible

**Start job:**

1. Click **Start Job** in RayForge
2. Laser begins cutting/engraving grid cells
3. Monitor the test closely (watch for fires on high-power cells)

**Execution order:**

RayForge uses **risk-optimized execution order:**
- Starts with lowest power/speed (safest)
- Gradually increases power
- Reduces fire risk by testing conservative settings first
- If material starts burning at high power, you can stop before completing all cells

**During execution:**
- Watch each cell being cut
- Note which cells cut through (for cut tests) or produce desired darkness (for engrave tests)
- Emergency stop if fire occurs

## Step 7: Inspect Results

Examine the completed test grid to find optimal settings.

**Cutting test results:**

Check each cell:

- **Didn't cut through:** Too little power or too fast speed
- **Cut through cleanly:** Good candidate settings
- **Excessive charring:** Too much power or too slow speed
- **Melted edges:** Too slow or too much power (material dependent)

**Visual grid:**

```
Speed ->  100    325    550    775   1000 mm/min
Power
  |
  v
20%      No cut  No cut No cut No cut No cut
35%      Char   Light  No cut No cut No cut
50%      Char   Good!  Light  No cut No cut
65%      Burn   Char   Good!  Light  No cut
80%      Burn   Burn   Char   Good!  Light
```

**Optimal settings:** Cells marked "Good!" - balance of cutting through without excessive charring.

**Engraving test results:**

Check each cell:

- **No visible mark:** Too little power or too fast
- **Light mark:** Low energy, suitable for subtle engraving
- **Medium mark:** Good contrast, common target
- **Dark/burned mark:** High energy, may be too much

**Choose based on desired darkness:**
- Light engraving: Low power + fast speed
- Medium engraving: Medium power + medium speed
- Dark engraving: High power + slow speed (watch for charring)

## Step 8: Record and Use Settings

Document the optimal settings and apply them to your project.

**Record settings:**

1. Identify the best cell in the grid
2. Note its row (power) and column (speed)
3. Write down exact values:
   - Power: ___ %
   - Speed: ___ mm/min

**Example:**

Best cell: Row 3, Column 2
- Power: 50%
- Speed: 325 mm/min

**Apply to your project:**

1. Create your actual design in RayForge
2. Set operation power to: 50%
3. Set operation speed to: 325 mm/min
4. Run job with confidence

**Tip:** Take a photo of the test grid with your phone and label the settings for future reference.

## Advanced Testing

### Multi-Pass Testing

To test multi-pass cutting (multiple passes over same path):

**Create multiple grids:**

- Grid 1: Test 1 pass at various power/speed
- Grid 2: Test 2 passes at various power/speed
- Grid 3: Test 3 passes at various power/speed

**Compare:**
- Which configuration gives cleanest cut?
- More passes at lower power often produces less charring

### Variable Testing

For more advanced testing, create custom grids:

**Test additional variables:**
- Different pass counts
- Different focus heights
- With/without air assist
- Different layer heights (for depth engraving)

**Method:** Create separate test grids for each variable, keep other settings constant.

### Fine-Tuning

After finding rough optimal settings, narrow the range:

**Example:**

Initial test showed 50% power at 325 mm/min works well.

**Fine-tune test:**
- Power range: 45% to 55% (narrow range around 50%)
- Speed range: 250 to 400 mm/min (narrow range around 325)
- Run new 5x5 grid with tighter spacing
- Find even more precise optimal settings

## Troubleshooting

### All Cells Failed to Cut

**Problem:** No cell in grid cut through material.

**Cause:** Power range too low or speed range too fast.

**Solution:**
- Increase max power (try up to 100%)
- Decrease min speed (try down to 50-100 mm/min)
- Re-run test with adjusted ranges

### All Cells Over-Burned

**Problem:** Every cell burned excessively or caught fire.

**Cause:** Power range too high or speed range too slow.

**Solution:**
- Decrease max power
- Increase min speed
- Test with lower energy settings

### Results Inconsistent

**Problem:** Same settings produce different results in different cells.

**Cause:** Material inconsistency, focus issues, or mechanical problems.

**Solution:**
- Use more uniform material for testing
- Verify laser focus is correct
- Check belt tension and mechanical components
- Ensure material is flat

### Can't Tell Which Cell is Best

**Problem:** Multiple cells look similar, hard to choose.

**Cause:** Test range too narrow or cells too small.

**Solution:**
- Use larger cell size (15mm instead of 10mm)
- Widen test range to see more variation
- Test on material with better visual contrast

## Best Practices

1. **Test on representative material:**
   - Same thickness as production material
   - Same batch/supplier if possible
   - Similar condition (moisture, finish, etc.)

2. **Use appropriate ranges:**
   - Start wide if material is unknown
   - Narrow range to fine-tune known settings
   - Don't be afraid to test high power (monitor closely)

3. **Monitor during test:**
   - Watch for fires (high power cells)
   - Note which cells look promising
   - Stop test if all high-power cells burn

4. **Label and save results:**
   - Write settings on test material with marker
   - Take photo for records
   - Keep test samples for future reference

5. **Re-test when needed:**
   - Different material batches may need different settings
   - Re-test after changing laser tube or optics
   - Test again if results change unexpectedly

## Related Pages

- [Power vs Speed Concepts](../concepts/power-vs-speed.md) - Understanding power/speed relationship
- [Material Test Grid Feature](../features/operations/material-test-grid.md) - Detailed feature reference
- [Understanding Operations](../concepts/understanding-operations.md) - Operation types explained
- [Laser Safety](../concepts/laser-safety.md) - Safe material testing practices
