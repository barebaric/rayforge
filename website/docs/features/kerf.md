# Kerf Compensation

Kerf is the material removed by the laser beam during cutting. Kerf compensation adjusts toolpaths to account for this, ensuring cut parts match their designed dimensions.

## What is Kerf?

**Kerf** = the width of material removed by the cutting process.

**Example:**
- Laser spot size: 0.2mm
- Material interaction: adds ~0.1mm on each side
- **Total kerf:** ~0.4mm

---

## How Kerf Compensation Works

Kerf compensation **offsets the toolpath** inward or outward to account for material removal:

**For outside cuts (cutting a part):**
- Offset path **outward** by half the kerf width
- Result: Final part is the correct size

**For inside cuts (cutting a hole):**
- Offset path **inward** by half the kerf width
- Result: Final hole is the correct size

**Example with 0.4mm kerf:**

```
Original path:  50mm square
Compensation:   Offset outward by 0.2mm (half kerf)
Laser follows:  50.4mm square
After cutting:  Part measures 50.0mm (perfect!)
```

---

## Measuring Kerf

**Accurate kerf measurement procedure:**

1. **Create a test file:**
   - Draw a 50mm x 50mm square
   - Draw a circle (any size, for inside cut test)

2. **Cut the test:**
   - Use your normal cutting settings
   - Cut completely through
   - Let material cool

3. **Measure:**
   - **Outer square (part):** Measure with calipers
     - If < 50mm, kerf was removed outward
     - Kerf = (50 - measured) x 2
   - **Inner circle (hole):** Measure diameter
     - If > designed diameter, kerf was removed inward
     - Kerf = (measured - designed) / 2

4. **Average:** Use the average of multiple measurements

**Variables affecting kerf:**
- Laser power (higher = wider)
- Cutting speed (slower = wider)
- Material type and density
- Focus distance
- Air assist pressure

---

## Manual Kerf Compensation

If automated kerf compensation isn't available, compensate in your design software:

**Inkscape:**

1. **Select the path**
2. **Path → Dynamic Offset** (Ctrl+J)
3. **Drag to offset** by half your kerf measurement
   - Outward for parts (to make path larger)
   - Inward for holes (to make path smaller)
4. **Path → Object to Path** to finalize

**Illustrator:**

1. **Select the path**
2. **Object → Path → Offset Path**
3. **Enter offset value:** (kerf / 2)
   - Positive for outward, negative for inward
4. **OK** to apply

**Fusion 360 / CAD:**

- Offset sketch entities before export
- Use the kerf/offset dimension

---

## Related Pages

- [Contour Operation](operations/contour) - Cutting operations
- [Material Test Grid](operations/material-test-grid) - Find optimal settings
