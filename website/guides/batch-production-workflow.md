# Batch Production Workflow

Efficiently produce multiple identical parts using RayForge's production features and best practices.

## Goal

Set up a repeatable workflow for cutting multiple identical parts, whether it's 10 coasters for a craft fair or 100 brackets for a product run.

## Prerequisites

- Design finalized and tested
- Material settings optimized (use Material Test Grid if needed)
- Sufficient material for production run
- Understanding of RayForge layers and operations

## Production Workflow Overview

**Efficient production requires:**

1. Optimized design (nesting, spacing, minimal waste)
2. Consistent material positioning (jigs, fixtures, or reliable origin)
3. Streamlined job setup (save settings, templates)
4. Quality control (inspect first piece, monitor for drift)
5. Safety procedures (never leave unattended, fire watch)

## Step 1: Optimize Your Design for Production

Prepare your design for efficient batch cutting.

### Part Layout and Nesting

**Maximize material usage:**

1. **Duplicate your part** to fill available material area
2. **Arrange in grid** with appropriate spacing
3. **Minimize waste** - pack parts efficiently
4. **Leave margin** for securing material (tape, clamps)

**Spacing considerations:**

- **Part-to-part spacing:** 3-5mm minimum
  - Too close: Heat affects adjacent parts
  - Too far: Wastes material
- **Edge spacing:** 10-15mm from material edges
  - Room for securing material
  - Avoids edge irregularities

**Example layout (300mm x 400mm material, 50mm x 50mm parts):**

```
Margin ->  [==========================]
           [  Part  Part  Part  Part  ]
           [  Part  Part  Part  Part  ]
           [  Part  Part  Part  Part  ]
           [  Part  Part  Part  Part  ]
           [==========================]
```

Fits: 16 parts with 5mm spacing between parts, 10mm edge margin

### Alignment Features

**Add registration or alignment marks:**

- Small crosses or circles at corners
- Help position material consistently
- Useful for multi-step processes (cut, then engrave)

**Don't cut through alignment marks:**
- Use lower power or separate layer
- Keep marks visible for next sheet

### Optimize Toolpaths

**Reduce travel time:**

- Use RayForge's path optimization (if available)
- Group adjacent parts together
- Minimize laser-off travel moves

**Organize by operation:**

- Layer 1: All engraving operations
- Layer 2: All cutting operations
- Runs all engraving first, then all cuts (prevents parts from falling out mid-job)

## Step 2: Create Production Template

Save your optimized design as a reusable template.

**Save project file:**

1. Design your full sheet layout (all parts arranged)
2. Configure all operation settings (power, speed, passes)
3. Save as: `ProductName-ProductionSheet.rf` (or similar)

**Template benefits:**
- Quickly start new sheets
- Consistent settings every time
- No need to reconfigure for each run

**Version control:**
- Save versions when changing settings
- Name files: `Coaster-v1.rf`, `Coaster-v2.rf`
- Document what changed in each version

## Step 3: Set Up Material Positioning System

Establish a reliable way to position material consistently.

### Method 1: Absolute Positioning (Simplest)

**Use job origin in absolute mode:**

1. Set job origin to specific coordinates (e.g., X=10, Y=10)
2. Always place material in same position on bed
3. Use tape markers or physical stops to guide placement

**Pros:**
- Simple, no fixtures needed
- Works for any material size

**Cons:**
- Manual positioning, less precise
- Material must be placed carefully each time

### Method 2: Physical Jigs

**Create alignment jigs:**

1. Cut a positioning jig from scrap material:
   - Rectangle with cutout for your material sheet
   - Place jig on laser bed, material fits inside
   - Material is now positioned consistently

**Example jig:**
```
[================================]
[                                ]
[     +-----------------+        ]
[     |  Your material  |        ]
[     |   fits here     |        ]
[     +-----------------+        ]
[                                ]
[================================]
```

**Pros:**
- Fast, consistent positioning
- No measuring needed

**Cons:**
- Requires creating jig
- Different jig for different material sizes

### Method 3: Registration Pins

**Use registration pins for repeatable positioning:**

1. Cut small holes in corner of material sheet
2. Install pins on laser bed at matching positions
3. Drop material onto pins - perfect alignment every time

**Pros:**
- Extremely precise
- Fast material changes

**Cons:**
- Requires modifying material (cutting holes)
- Need to install pins on bed

### Method 4: Laser-Bed Grid Markings

**Mark laser bed with ruler markings:**

1. Use tape or engraved marks to create reference grid
2. Align material edges to grid marks
3. Consistent positioning without jigs

**Pros:**
- No jigs needed
- Works for various sizes

**Cons:**
- Less precise than jigs
- Manual alignment each time

## Step 4: First Article Inspection

Always inspect the first piece before running full batch.

**Why inspect first?**
- Verify settings are correct
- Catch design errors before wasting material
- Confirm dimensions are accurate

**Run one part:**

1. Run job to cut just the first part (or first few parts)
2. Pause job or run a single-part version of the design

**Inspect thoroughly:**
- **Dimensions:** Measure with calipers - are parts correct size?
- **Quality:** Clean cuts, no excessive charring?
- **Fit/function:** If parts assemble, test fit now
- **Alignment:** Is positioning on material correct?

**If issues found:**
- Stop production
- Adjust settings (power, speed, kerf compensation, etc.)
- Re-run first article test
- Don't proceed until first part is perfect

**If first part is good:**
- Proceed with full production run
- Periodically inspect parts to catch any drift

## Step 5: Run Production Batches

Execute your production run efficiently and safely.

### Safety for Long Runs

**Never leave laser unattended:**
- Production runs can be long (hours)
- Stay near the machine at all times
- Watch for fires (more cuts = more fire risk)

**Fire watch:**
- Keep fire extinguisher ready
- Clear debris between sheets (prevents accumulated debris from igniting)
- Monitor high-risk materials closely

**Ventilation:**
- Ensure exhaust runs continuously
- Check airflow periodically
- Replace filters if using filtration system

### Material Loading

**Efficient material changes:**

1. **Wait for completion** - let current sheet finish
2. **Remove cut parts** - clear bed completely
3. **Clean debris** - remove small pieces and dust (fire hazard)
4. **Load new sheet** - use jig or positioning method
5. **Verify position** - frame job to confirm alignment
6. **Restart job**

**Material prep:**
- Pre-cut sheets to size (saves time between jobs)
- Stack sheets nearby for quick loading
- Keep sheets flat (warped material causes issues)

### Monitoring Production

**Watch for issues:**

- **Quality drift:** Parts start charring more or cutting less deeply
  - Cause: Lens/mirror getting dirty from smoke
  - Solution: Pause and clean optics
- **Position drift:** Parts shifting on bed
  - Cause: Material warping from heat or poor securing
  - Solution: Re-secure material, use better hold-downs
- **Mechanical issues:** Strange sounds, skipped steps
  - Cause: Belt slipping, motor overheating
  - Solution: Stop, inspect mechanics, let cool

**Periodic inspection:**
- Every 5-10 parts (or every sheet), inspect one part
- Measure critical dimensions
- Check for quality degradation
- Stop if issues arise

## Step 6: Quality Control and Final Inspection

Ensure all parts meet quality standards.

### Sorting and Inspection

**Inspect each part:**

- **Visual inspection:** Check for burn marks, incomplete cuts, damage
- **Dimensional check:** Spot-check dimensions on random parts
- **Functional test:** If parts assemble or serve specific function, test a sample

**Sort parts:**
- **Grade A:** Perfect, ready to use/sell
- **Grade B:** Minor flaws, usable for less critical applications
- **Scrap:** Defects, unusable

### Record Keeping

**Track production metrics:**

- Number of parts produced
- Material used (sheets, area)
- Time taken
- Defect rate
- Settings used

**Why track?**
- Improve future runs
- Calculate costs accurately
- Identify trends (quality degrading over time?)

## Advanced Production Techniques

### Multi-Material Projects

**Different materials in same design:**

**Challenge:** Each material needs different power/speed settings.

**Solution:**

1. **Separate layers by material:**
   - Layer 1: Plywood parts (settings for plywood)
   - Layer 2: Acrylic parts (settings for acrylic)
2. **Run each layer separately:**
   - Load plywood, run Layer 1
   - Load acrylic, run Layer 2
3. **Use consistent positioning** (jigs, registration marks)

### Two-Step Production (Cut + Engrave)

**Cut parts, then engrave details:**

**Workflow:**

1. **Step 1: Cut all parts**
   - Batch cut 20 parts from sheets
   - Remove parts from sheets
2. **Step 2: Engrave all parts**
   - Load individual parts into jig
   - Engrave details on each part
   - Faster than cutting + engraving each part individually

**Benefits:**
- Efficient material usage (cut from sheets)
- Precise engraving on individual parts
- Quality control between steps

### Automated Part Numbering

**Engrave unique numbers on each part:**

**Use case:** Serialization, inventory tracking, kit assembly

**Implementation:**

1. Use RayForge's variable substitution (if supported) or macros
2. Increment part number for each piece
3. Engrave number automatically during production

**Alternative:** Manually update part number between batches.

## Troubleshooting Production Issues

### Parts Not Cutting Through Consistently

**Problem:** Some parts cut cleanly, others don't.

**Causes:**
- Material thickness variation
- Dirty optics (power drops as lens gets dirty)
- Focus height wrong
- Warped material (parts of material closer/farther from laser)

**Solutions:**
- Use higher quality, consistent material
- Clean optics between sheets
- Verify focus before each sheet
- Use multi-pass to compensate for variation

### Material Warping During Cut

**Problem:** Material curls or warps as laser cuts, causing later cuts to fail.

**Causes:**
- Heat stress
- Thin material
- Large cuts release tension

**Solutions:**
- Use hold-down pins or weights
- Cut in multiple passes (less heat per pass)
- Cut parts in smaller sections (less stress)
- Use thicker material if possible

### Slow Production Speed

**Problem:** Production taking longer than expected.

**Causes:**
- Inefficient toolpaths (long travel moves)
- Conservative settings (low speed)
- Manual steps taking too long

**Solutions:**
- Optimize part nesting and path order
- Increase speed (if quality allows)
- Streamline material loading process
- Use jigs for faster positioning

### Defect Rate Too High

**Problem:** Too many parts are scrap or Grade B.

**Causes:**
- Settings not optimized
- Material quality issues
- Design flaws

**Solutions:**
- Re-run Material Test Grid to optimize settings
- Source better quality material
- Refine design (add fillets to prevent cracking, adjust tolerances)

## Best Practices Summary

1. **Optimize before production:**
   - Perfect your design on scrap material first
   - Use Material Test Grid to dial in settings
   - Run multiple test pieces until settings are perfect

2. **Use templates and jigs:**
   - Save production template files
   - Create physical jigs for repeatable positioning
   - Document settings for future runs

3. **First article inspection:**
   - Always inspect first part before full run
   - Verify dimensions and quality
   - Adjust if needed

4. **Safety first:**
   - Never leave laser unattended
   - Watch for fires continuously
   - Clear debris between sheets

5. **Monitor and adapt:**
   - Inspect periodically during run
   - Clean optics when quality degrades
   - Stop and fix issues immediately

6. **Track and improve:**
   - Record metrics (time, defects, material usage)
   - Identify improvement opportunities
   - Refine process over time

## Related Pages

- [Material Test Grid](creating-material-test-grid.md) - Optimize settings before production
- [Multi-Layer Workflow](../features/multi-layer.md) - Organize complex jobs
- [Calibrating Your Workspace](calibrating-your-workspace.md) - Ensure dimensional accuracy
- [Power vs Speed](../concepts/power-vs-speed.md) - Understand settings for consistency
