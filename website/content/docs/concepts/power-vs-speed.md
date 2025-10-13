# Power vs Speed

Understanding the relationship between laser power, movement speed, and material interaction is crucial for achieving high-quality cuts and engravings.

## The Fundamental Relationship

**Energy delivered to material = Power x Time**

When the laser moves more slowly, it delivers more energy per unit area. When it moves faster, less energy is delivered.

**Key principle:**

```
More power + slow speed = High energy (deep cuts, dark burns)
Less power + fast speed = Low energy (light marks, shallow cuts)
```

**Balance is key:** Different materials and operations require different combinations.

---

## Understanding Power

### What is Power?

**Laser power** is the intensity of the laser beam, typically measured in watts (W).

**Power control:**

- Rayforge uses the S parameter (0-1000) to control power
- S0 = 0% power (off)
- S500 = 50% power
- S1000 = 100% power (maximum your laser can produce)

**Example:**

```gcode
M4 S250   ; 25% power
M4 S750   ; 75% power
M4 S1000  ; 100% power
```

---

### Absolute Power vs Percentage

**Your laser has a maximum power rating:**

- 5W diode laser
- 40W CO2 laser
- 80W CO2 laser

**When you set 50% power:**

- 5W laser delivers: 2.5W
- 40W laser delivers: 20W
- 80W laser delivers: 40W

**Implication:** Settings that work on one laser won't directly transfer to a different wattage laser. You must adjust for your specific laser power.

---

## Understanding Speed

### What is Speed?

**Feed rate** (or cutting speed) is how fast the laser head moves across the material, measured in mm/min or inches/min.

**Speed ranges:**

- Very slow: 50-300 mm/min (deep cuts, thick materials)
- Moderate: 500-1500 mm/min (general cutting, engraving)
- Fast: 2000-5000 mm/min (light engraving, travel moves)

**Example:**

```gcode
G1 X100 F500   ; Move at 500 mm/min (slow cut)
G1 X200 F3000  ; Move at 3000 mm/min (fast travel)
```

---

### Speed Affects Energy Density

**Slower movement** = More time at each point = More energy delivered

**Example:**

- Laser power: 40W
- Speed 1: 100 mm/min
- Speed 2: 1000 mm/min

At 100 mm/min, the laser spends **10x longer** at each point compared to 1000 mm/min, delivering 10x more energy.

---

## The Power-Speed Matrix

Different combinations of power and speed produce different results:

| Power    | Speed | Result                           | Use Case                           |
| -------- | ----- | -------------------------------- | ---------------------------------- |
| **High** | Slow  | Very deep cuts, charring         | Cutting thick materials            |
| **High** | Fast  | Moderate cuts/engraving          | Fast cutting, efficient production |
| **Low**  | Slow  | Deep engraving, controlled burns | Detailed engraving, dark marks     |
| **Low**  | Fast  | Light engraving, surface marks   | High-speed raster engraving        |

**Visualization:**

```
Power
  ^
  |  Charring/   |  Through-cut
  |  Too deep    |  (ideal)
  |              |
  |------------- +-------------
  |  Good        |  Too fast/
  |  engraving   |  Not cutting
  |              |
  +-----------------------------> Speed
```

---

## Material-Specific Considerations

### Wood

**Cutting:**

- Medium to high power
- Moderate speed
- Multiple passes for thick wood

**Example (3mm plywood, 40W CO2):**

- Power: 70-80%
- Speed: 200-400 mm/min
- Passes: 1-2

**Engraving:**

- Low to medium power
- High speed for raster
- Lower speed for deeper marks

**Example (wood engraving, 40W CO2):**

- Power: 20-30%
- Speed: 2000-3000 mm/min

---

### Acrylic

**Cutting:**

- High power
- Very slow speed
- Produces clean, flame-polished edges

**Example (3mm cast acrylic, 40W CO2):**

- Power: 80-100%
- Speed: 100-200 mm/min
- Passes: 1 (through-cut)

**Engraving:**

- Low power
- Fast speed
- Produces frosted appearance

**Example (acrylic engraving, 40W CO2):**

- Power: 15-25%
- Speed: 2500-4000 mm/min

---

### Cardboard / Paper

**Cutting:**

- Very low power
- Fast speed
- **High fire risk** - monitor constantly

**Example (cardboard, 40W CO2):**

- Power: 10-20%
- Speed: 1000-2000 mm/min

**Engraving:**

- Extremely low power
- Very fast speed

**Example (paper engraving, 40W CO2):**

- Power: 5-10%
- Speed: 3000-5000 mm/min

---

### Leather

**Cutting:**

- Medium power
- Moderate speed

**Example (leather, 40W CO2):**

- Power: 40-60%
- Speed: 300-600 mm/min

**Engraving:**

- Low power
- Fast to moderate speed
- Creates dark, detailed marks

**Example (leather engraving, 40W CO2):**

- Power: 15-25%
- Speed: 2000-3000 mm/min

---

## Multi-Pass Strategies

### When to Use Multiple Passes

**Reasons:**

1. Material too thick for single pass
2. Preventing excessive charring
3. Achieving cleaner cuts
4. Reducing heat buildup

**How it works:**
Instead of one deep cut, make several shallower cuts:

- Pass 1: Cuts 30% through
- Pass 2: Cuts 60% through
- Pass 3: Cuts 90% through
- Pass 4: Cuts fully through

---

### Benefits of Multi-Pass

**Cleaner cuts:**

- Less charring on edges
- Less heat stress on material
- Better edge quality

**Thicker materials:**

- Cut materials beyond single-pass capability
- Safer than maxing out power

**Heat management:**

- Allows material to cool between passes
- Reduces warping and melting

---

### Multi-Pass Settings

**General approach:**

1. **Determine total energy needed** to cut through
2. **Divide into multiple passes** (2-5 passes typical)
3. **Adjust speed or power** to deliver energy gradually

**Example (6mm plywood, 40W CO2):**

**Single pass (may fail or char heavily):**

- Power: 100%
- Speed: 100 mm/min
- Passes: 1
- Result: Excessive charring, incomplete cut

**Multi-pass (better):**

- Power: 80%
- Speed: 200 mm/min
- Passes: 3
- Result: Clean cut, minimal charring

---

### Configuring Multi-Pass in Rayforge

**In operation settings:**

1. Select Contour operation
2. Set **Passes** to desired number (e.g., 3)
3. Adjust power/speed as needed
4. Rayforge will automatically repeat the cut path

**G-code result:**

```gcode
; Pass 1
G0 X10 Y10
M4 S800
G1 X50 Y10 F200
; ...path...
M5

; Pass 2 (same path repeated)
G0 X10 Y10
M4 S800
G1 X50 Y10 F200
; ...path...
M5

; Pass 3
; ...
```

---

## Reading Burn Marks and Adjusting

### Visual Feedback

Material tells you if settings are correct:

**Too much power or too slow:**

- Heavy charring (black edges)
- Excessive smoke
- Melted/deformed edges
- Material warping

**Too little power or too fast:**

- Incomplete cuts
- Light surface marks only
- No visible change

**Just right:**

- Clean cuts all the way through
- Minimal charring
- Smooth edges
- Consistent depth

---

### Adjusting Based on Results

**If cut doesn't go through:**

1. Increase power by 10-20%
2. OR decrease speed by 20-30%
3. OR add another pass

**If too much charring:**

1. Decrease power by 10-20%
2. OR increase speed by 20-30%
3. OR switch to multi-pass with lower power

**If engraving too light:**

1. Increase power by 10-15%
2. OR decrease speed by 15-25%

**If engraving too dark/burned:**

1. Decrease power by 10-15%
2. OR increase speed by 15-25%

---

## Diode Lasers vs CO2 Lasers

### Diode Lasers (Typically 5-20W)

**Characteristics:**

- Lower power output
- 445nm wavelength (blue light)
- Absorbed differently by materials

**Power/speed considerations:**

- Slower speeds needed (lower power)
- Multiple passes often required
- Struggles with thick materials
- Excellent for wood engraving

**Example (wood engraving, 5W diode):**

- Power: 60-80%
- Speed: 500-1500 mm/min

---

### CO2 Lasers (Typically 40-150W)

**Characteristics:**

- Higher power output
- 10,600nm wavelength (infrared)
- Efficiently absorbed by organic materials

**Power/speed considerations:**

- Faster speeds possible
- Clean through-cuts on thicker materials
- Better for cutting acrylic, wood

**Example (wood cutting, 40W CO2):**

- Power: 70-90%
- Speed: 200-500 mm/min

---

## Material Testing Workflow

### Using the Material Test Grid

**Rayforge's Material Test Grid** automates power/speed testing:

1. **Create grid:** Features > Operations > Material Test Grid
2. **Set ranges:**
   - Power: 20% to 80% (or appropriate range)
   - Speed: 500 to 2000 mm/min (or appropriate range)
3. **Run grid** on scrap material
4. **Inspect results:**
   - Find the cell with best cut quality
   - Note the power/speed values
5. **Use those settings** in your actual job

**Example grid result:**

```
        500mm/min  1000mm/min  1500mm/min  2000mm/min
20%     Too light  Too light   Too light   Too light
40%     Good       Too light   Too light   Too light
60%     Too dark   Good        Light       Too light
80%     Charred    Too dark    Good        Light
```

**Conclusion:** Use 60% power at 1500 mm/min for this material.

See [Material Test Grid](../features/operations/material-test-grid.md) for details.

---

## Advanced Topics

### Kerf Compensation

**Kerf** is the width of material removed by the laser beam.

**Power/speed affect kerf:**

- Higher power = Wider kerf
- Slower speed = Wider kerf (more material burned away)

**Adjust kerf compensation** if parts come out wrong size:

- Parts too small: Reduce kerf compensation
- Parts too large: Increase kerf compensation

See [Kerf](../features/kerf.md) for details.

---

### Raster Speed Variation

For raster engraving, Rayforge varies speed to create different gray tones:

**How it works:**

- Light areas: Fast speed, low power
- Dark areas: Slow speed, higher power (or just higher power at constant speed)

**Bidirectional raster:**

- Laser engraves left-to-right, then right-to-left
- Must maintain consistent power at varying speeds
- M4 (laser mode) ensures constant power

---

### Air Assist and Cooling

**Air assist affects results:**

- Reduces charring
- Cools material
- Blows away smoke/debris

**Settings interaction:**

- With air assist: May need slightly higher power
- Without air assist: More charring, may need lower power/faster speed

---

## Common Mistakes

### Mistake 1: Maxing Out Power

**Problem:** Using 100% power for everything.

**Why it's bad:**

- Excessive charring
- Accelerated wear on laser tube
- Less control and precision

**Solution:** Use appropriate power (usually 60-80% for cuts).

---

### Mistake 2: Ignoring Material Variation

**Problem:** Using same settings for all "wood" without testing.

**Why it's bad:**

- Different wood species vary significantly
- Plywood vs solid wood behaves differently
- Moisture content affects results

**Solution:** Test each new material type with a Material Test Grid.

---

### Mistake 3: Not Adjusting for Thickness

**Problem:** Using same settings for 3mm and 6mm material.

**Why it's bad:**

- 6mm needs much more energy (lower speed or more passes)
- 3mm settings won't cut through 6mm

**Solution:** Adjust speed (slower) or passes (more) for thicker materials.

---

### Mistake 4: Focusing on Power Only

**Problem:** Only adjusting power, never speed.

**Why it's bad:**

- Speed is equally important
- Sometimes speed adjustment is more effective

**Solution:** Adjust both power AND speed to find optimal settings.

---

## Quick Reference

### Starting Points (40W CO2 Laser)

| Material      | Thickness | Power | Speed       | Passes |
| ------------- | --------- | ----- | ----------- | ------ |
| **Plywood**   | 3mm       | 70%   | 300 mm/min  | 1-2    |
| **Plywood**   | 6mm       | 80%   | 200 mm/min  | 2-3    |
| **Acrylic**   | 3mm       | 90%   | 150 mm/min  | 1      |
| **Acrylic**   | 6mm       | 100%  | 80 mm/min   | 1-2    |
| **Cardboard** | 2mm       | 15%   | 1500 mm/min | 1      |
| **Leather**   | 2mm       | 50%   | 400 mm/min  | 1      |

**Note:** These are starting points only. Always test on your specific machine and material.

---

### Starting Points (5W Diode Laser)

| Material            | Thickness | Power | Speed      | Passes |
| ------------------- | --------- | ----- | ---------- | ------ |
| **Wood engrave**    | -         | 70%   | 800 mm/min | 1      |
| **Wood cut**        | 3mm       | 90%   | 100 mm/min | 3-5    |
| **Cardboard**       | 2mm       | 50%   | 500 mm/min | 2      |
| **Leather engrave** | -         | 60%   | 600 mm/min | 1      |

**Note:** Diode lasers struggle with thick materials. Multi-pass is often required.

---

## Best Practices

1. **Always test new materials** with Material Test Grid
2. **Start conservative** (lower power, moderate speed) and increase gradually
3. **Use multi-pass** for thick materials rather than maxing out power
4. **Adjust one parameter at a time** (power OR speed, not both)
5. **Document successful settings** for future reference
6. **Account for material variation** (different batches may need adjustment)
7. **Monitor first few cuts** of any job to catch issues early

---

## Related Pages

- [Material Test Grid](../features/operations/material-test-grid.md) - Automated power/speed testing
- [Understanding Operations](understanding-operations.md) - Operation types and settings
- [Kerf](../features/kerf.md) - Compensation techniques
- [Laser Safety](laser-safety.md) - Safe power/speed practices
