# Calibrating Your Workspace

Ensure your laser cuts accurately to the dimensions you design.

## Goal

Verify and calibrate your laser's dimensional accuracy so that a 100mm square in Rayforge cuts as a true 100mm square on your material.

## Prerequisites

- Rayforge installed and connected to your laser
- Machine homed and ready
- Test material (cardboard, plywood, or acrylic scrap)
- Ruler or calipers (digital calipers preferred for accuracy)
- Basic understanding of machine settings

## Why Calibration Matters

Laser machines use stepper motors with a configured steps-per-millimeter setting. If this setting is incorrect:

- Parts won't fit together properly
- Dimensions will be consistently off (e.g., 100mm design cuts as 98mm or 102mm)
- Assembly projects will fail

**Calibration ensures:** 1mm in Rayforge = 1mm on your material

## Step 1: Check Current Machine Configuration

First, verify your machine dimensions are configured correctly in Rayforge.

**Open machine settings:**

1. Navigate to **Settings > Machine > Profile**
2. Check **Work Area** dimensions:
   - Width: Should match your laser bed width (e.g., 300mm)
   - Height: Should match your laser bed height (e.g., 400mm)
3. Check **Origin Position**:
   - Top-left, bottom-left, or center (must match your actual machine)
4. Check **Y-axis Direction**:
   - Does Y increase upward or downward? (must match your machine)

**If dimensions are wrong:** Correct them now and save settings.

## Step 2: Create a Test Pattern

Create a simple test pattern with known dimensions.

**Design the test:**

1. Create a new project in Rayforge
2. Add a **square** workpiece:
   - Width: 100mm
   - Height: 100mm
3. Add a second **rectangle** workpiece:
   - Width: 50mm
   - Height: 100mm
4. Position them with some spacing

**Why these dimensions?**

- 100mm is easy to measure
- 50mm tests both axes independently
- Multiple shapes verify consistency

## Step 3: Configure Cut Settings

Set up appropriate cut settings for your test material.

**For cardboard test (fast and cheap):**

- **Operation:** Contour
- **Power:** 15-20% (just enough to cut through)
- **Speed:** 1500-2000 mm/min
- **Passes:** 1

**For plywood test (more accurate):**

- **Operation:** Contour
- **Power:** 60-80%
- **Speed:** 300-500 mm/min
- **Passes:** 1-2

**For acrylic test (most accurate):**

- **Operation:** Contour
- **Power:** 80-100%
- **Speed:** 150-200 mm/min
- **Passes:** 1

## Step 4: Preview and Frame

Before cutting, verify the job will run in the correct location.

**Frame the job:**

1. In Rayforge, click **Frame Job** (or use keyboard shortcut)
2. Laser head will trace the outline of the job boundary
3. Verify:
   - Position is correct on your material
   - Job fits within work area
   - No collisions with clamps or obstacles

**If position is wrong:** Adjust job origin or reposition material.

## Step 5: Cut the Test Pattern

Run the calibration test cut.

**Safety first:**

- Ensure ventilation is running
- Have fire extinguisher ready
- Stay near the machine (never leave unattended)

**Run the job:**

1. Click **Start Job** in Rayforge
2. Monitor the cut closely
3. Wait for completion
4. Allow material to cool before removing

## Step 6: Measure the Results

Carefully measure the cut pieces to determine accuracy.

**What to measure:**

1. **100mm square:**
   - Measure width (X-axis): Should be 100.0mm
   - Measure height (Y-axis): Should be 100.0mm
2. **50mm x 100mm rectangle:**
   - Measure width: Should be 50.0mm
   - Measure height: Should be 100.0mm

**Measurement tips:**

- Measure from cut edge to cut edge (ignore kerf)
- Take 3 measurements and average them
- Use calipers for best accuracy (rulers are less precise)

**Record your results:**

```
Target: 100mm x 100mm square
Actual: ___mm x ___mm

Target: 50mm x 100mm rectangle
Actual: ___mm x ___mm
```

## Step 7: Calculate Calibration Error

Determine how far off your machine is from true dimensions.

**Calculate error percentage:**

```
Error (X-axis) = (Measured Width - Target Width) / Target Width * 100%
Error (Y-axis) = (Measured Height - Target Height) / Target Height * 100%
```

**Example:**

Target: 100mm x 100mm
Actual: 98.5mm x 101.2mm

```
X error = (98.5 - 100) / 100 * 100% = -1.5%
Y error = (101.2 - 100) / 100 * 100% = +1.2%
```

**Interpretation:**

- X-axis is cutting 1.5% too small
- Y-axis is cutting 1.2% too large

## Step 8: Determine if Calibration is Needed

Decide if your accuracy is acceptable or needs correction.

**Accuracy thresholds:**

| Application             | Acceptable Error | Action                     |
| ----------------------- | ---------------- | -------------------------- |
| **Artistic/decorative** | +/- 2mm (2%)     | Probably OK as-is          |
| **General projects**    | +/- 0.5mm (0.5%) | Calibrate if outside this  |
| **Precision/assembly**  | +/- 0.1mm (0.1%) | Calibrate, verify hardware |

**If within tolerance:** You're done! No calibration needed.

**If outside tolerance:** Proceed to Step 9.

## Step 9: Adjust Firmware Settings (GRBL)

Correct dimensional errors by adjusting steps-per-mm in firmware.

### Understanding Steps-Per-Millimeter

Your machine's controller needs to know how many motor steps equal one millimeter of movement.

**Default GRBL settings:**

- $100: X-axis steps/mm (typical: 80-100)
- $101: Y-axis steps/mm (typical: 80-100)

**If parts are too small:** Increase steps/mm
**If parts are too large:** Decrease steps/mm

### Calculate New Settings

**Formula:**

```
New steps/mm = Current steps/mm * (Target dimension / Measured dimension)
```

**Example:**

Current X steps/mm: $100 = 80.00
Target: 100mm
Measured: 98.5mm

```
New X steps/mm = 80.00 * (100 / 98.5) = 81.22
```

Current Y steps/mm: $101 = 80.00
Target: 100mm
Measured: 101.2mm

```
New Y steps/mm = 80.00 * (100 / 101.2) = 79.05
```

### Read Current Settings

**In Rayforge console:**

1. Open **Console** tab
2. Send command: `$$` (two dollar signs)
3. Firmware will respond with all settings
4. Note current values for $100 and $101

**Example output:**

```
$100=80.000
$101=80.000
$102=80.000
...
```

### Update Settings

**Send new values:**

In the Rayforge console:

```
$100=81.22
$101=79.05
```

Wait for "ok" response from firmware.

### Save Settings

**GRBL automatically saves settings** to EEPROM, but verify:

```
$$
```

Check that $100 and $101 now show the new values.

## Step 10: Verify Calibration

Cut another test pattern to confirm accuracy.

**Repeat the test:**

1. Use the same test pattern (100mm square, 50mm rectangle)
2. Cut on fresh material
3. Measure results again
4. Verify dimensions are now accurate

**Expected results:**

```
Target: 100mm x 100mm
Actual: 100.0mm +/- 0.2mm x 100.0mm +/- 0.2mm
```

**If still off:** Repeat calibration process. Large errors may require multiple iterations.

**If now accurate:** Calibration complete!

## Advanced Calibration

### Belt Tension and Mechanical Issues

If calibration doesn't solve dimensional errors, check mechanical components:

**Symptoms of mechanical issues:**

- Errors vary between tests (not consistent)
- One axis much worse than the other
- Dimensions change based on speed

**Check:**

1. **Belt tension:**
   - Should be tight but not over-tensioned
   - Loose belts cause backlash and dimensional errors
2. **Pulley set screws:**
   - Ensure pulleys are tight on motor shafts
   - Loose pulleys slip and cause errors
3. **Linear rails:**
   - Should move smoothly without binding
   - Clean and lubricate if needed
4. **Stepper drivers:**
   - Verify current settings are correct
   - Underpowered motors can skip steps

### Kerf Compensation vs Calibration

**Kerf** is the material removed by the laser beam (typically 0.1-0.3mm).

**Kerf is NOT a calibration issue:**

- Kerf makes parts slightly smaller than designed
- Kerf is compensated in Rayforge settings (see Overscan & Kerf)

**Calibration corrects:**

- Systematic dimensional errors across the entire workspace
- Steps-per-mm configuration errors

**If your 100mm square measures 99.8mm:** This might be kerf, not calibration error.

### Testing at Different Scales

For precision work, test calibration at multiple scales:

**Create test patterns:**

- Small: 20mm x 20mm square
- Medium: 100mm x 100mm square
- Large: 200mm x 200mm square

**Measure all three:**

- Errors should be proportional (e.g., 1% error at all scales)
- If errors vary by scale, there may be mechanical non-linearity

## Troubleshooting

### Dimensions Off by Large Amount (>5%)

**Possible causes:**

- Wrong firmware settings
- Wrong machine profile in Rayforge
- Mechanical issue (loose pulley, belt)

**Solutions:**

- Verify machine work area dimensions in settings
- Check $100/$101 settings in firmware
- Inspect mechanical components

### Dimensions Inconsistent Between Tests

**Possible causes:**

- Loose belts or pulleys
- Stepper motors skipping steps
- Material warping or movement during cut

**Solutions:**

- Tighten belts to proper tension
- Verify stepper driver current settings
- Secure material better, use flat material

### One Axis Accurate, One Axis Wrong

**Possible causes:**

- One axis has incorrect steps/mm
- Mechanical issue on one axis only

**Solutions:**

- Calibrate the problem axis separately
- Inspect that axis's belt, pulley, rails

### Still Inaccurate After Calibration

**Possible causes:**

- Measurement error (using inaccurate ruler)
- Kerf not accounted for (measure outer edge to outer edge)
- Severe mechanical issues

**Solutions:**

- Use digital calipers for accurate measurement
- Re-check calculation of new steps/mm
- Consult machine manufacturer documentation

## Best Practices

1. **Calibrate when:**

   - Setting up a new machine
   - After replacing belts or pulleys
   - If dimensional errors are noticed
   - Periodically (every 6-12 months)

2. **Use good test materials:**

   - Flat, stable materials (not warped)
   - Acrylic or plywood preferred over cardboard
   - Large enough to measure accurately

3. **Measure carefully:**

   - Use calipers, not rulers
   - Take multiple measurements and average
   - Measure edge to edge, not including kerf

4. **Document your settings:**

   - Record $100 and $101 values
   - Note any mechanical adjustments
   - Keep calibration test results for reference

5. **Re-verify periodically:**
   - Test dimensional accuracy on actual projects
   - Re-calibrate if you notice consistent errors

## Related Pages

- [Machine Setup](../machine/device-config.md) - Configure machine dimensions
- [GRBL Settings](../machine/grbl-settings.md) - Firmware configuration reference
- [Kerf](../features/kerf.md) - Kerf compensation techniques
- [Coordinates and Origin](../concepts/coordinates-and-origin.md) - Understanding coordinate systems
