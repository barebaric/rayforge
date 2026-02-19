# Shrink Wrap

Shrink Wrap creates an efficient cutting path around multiple objects by generating a boundary that "shrinks" around them. It's useful for cutting multiple parts from a sheet with minimal waste.

## Overview

Shrink Wrap operations:

- Create boundary paths around groups of objects
- Minimize material waste
- Reduce cutting time by combining paths
- Support offset distances for clearance
- Work with any combination of vector shapes

## When to Use Shrink Wrap

Use shrink wrap for:

- Cutting multiple small parts from a sheet
- Minimizing material waste
- Creating efficient nesting boundaries
- Separating groups of parts
- Reducing total cutting time

**Don't use shrink wrap for:**

- Single objects (use [Contour](contour) instead)
- Parts that need individual boundaries
- Precise rectangular cuts

## How Shrink Wrap Works

Shrink wrap creates a boundary using a computational geometry algorithm:

1. **Start** with a convex hull around all objects
2. **Shrink** the boundary inward toward the objects
3. **Wrap** tightly around the object group
4. **Offset** outward by the specified distance

The result is an efficient cutting path that follows the overall shape of your parts while maintaining clearance.

## Creating a Shrink Wrap Operation

### Step 1: Arrange Objects

1. Place all parts you want to wrap on the canvas
2. Position them with desired spacing
3. Multiple separate groups can be shrink-wrapped together

### Step 2: Select Objects

1. Select all objects to include in the shrink wrap
2. Can be different shapes, sizes, and types
3. All selected objects will be wrapped together

### Step 3: Add Shrink Wrap Operation

- **Menu:** Operations Add Shrink Wrap
- **Right-click:** Context menu Add Operation Shrink Wrap

### Step 4: Configure Settings

![Shrink Wrap step settings](/images/step-shrink-wrap-step-settings.png)

## Key Settings

### Power & Speed

Like other cutting operations:

**Power (%):**

- Laser intensity for cutting
- Same as you'd use for [Contour](contour) cutting

**Speed (mm/min):**

- How fast the laser moves
- Match your material's cutting speed

**Passes:**

- Number of times to cut the boundary
- Usually 1-2 passes
- Same as contour cutting for your material

### Offset Distance

**Offset (mm):**

- How much clearance around the parts
- Distance from objects to the shrink-wrap boundary
- Larger offset = more material left around parts

**Typical values:**

- **2-3mm:** Tight wrap, minimal waste
- **5mm:** Comfortable clearance
- **10mm+:** Extra material for handling

**Why offset matters:**

- Too small: Risk cutting into parts
- Too large: Wastes material
- Consider: Kerf width, cutting accuracy

### Smoothness

Controls how closely the boundary follows object shapes:

**High smoothness:**

- Follows objects more closely
- More complex path
- Longer cutting time
- Less material waste

**Low smoothness:**

- Simpler, more rounded path
- Shorter cutting time
- Slightly more material waste

**Recommended:** Medium smoothness for most cases

## Use Cases

### Batch Part Production

**Scenario:** Cutting 20 small parts from a large sheet

**Without shrink wrap:**

- Cut full sheet boundary
- Waste all material around parts
- Long cutting time

**With shrink wrap:**

- Cut tight boundary around part group
- Save material for other projects
- Faster cutting (shorter perimeter)

### Nesting Optimization

**Workflow:**

1. Nest parts efficiently on sheet
2. Group parts into sections
3. Shrink wrap each section
4. Cut sections separately

**Benefits:**

- Can remove finished sections while continuing
- Easier handling of cut parts
- Reduced risk of part movement

### Material Conservation

**Example:** Small parts on expensive material

**Process:**

1. Arrange parts tightly
2. Shrink wrap with 3mm offset
3. Cut free from sheet
4. Save remaining material

**Result:** Maximum material efficiency

## Combining with Other Operations

### Shrink Wrap + Contour

Common workflow:

1. **Contour** operations on individual parts (cut details)
2. **Shrink wrap** around the group (cut free from sheet)

**Execution order:**

- First: Cut details in parts (while secured)
- Last: Shrink wrap cuts group free

See [Multi-Layer Workflow](../multi-layer) for details.

### Shrink Wrap + Raster

**Example:** Engraved and cut parts

1. **Raster** engrave logos on parts
2. **Contour** cut part outlines
3. **Shrink wrap** around entire group

**Benefits:**

- All engraving happens while material is secured
- Final shrink wrap cuts entire batch free

## Tips & Best Practices

![Shrink Wrap post-processing settings](/images/step-shrink-wrap-post-processing.png)

### Part Spacing

**Optimal spacing:**

- 5-10mm between parts
- Enough for shrink wrap to distinguish separate objects
- Not so much that you waste material

**Too close:**

- Parts may be wrapped together
- Shrink wrap may bridge gaps
- Difficult to separate after cutting

**Too far:**

- Wastes material
- Longer cutting time
- Inefficient use of sheet

### Material Considerations

**Best for:**

- Production runs (many identical parts)
- Small parts from large sheets
- Expensive materials (minimize waste)
- Batch cutting jobs

**Not ideal for:**

- Single large parts
- Parts filling entire sheet
- When you need full sheet cut

### Safety

**Always:**

- Check that boundary doesn't overlap parts
- Verify offset is sufficient
- Preview in [Simulation Mode](../simulation-mode)
- Test on scrap first

**Watch for:**

- Shrink wrap cutting into parts (increase offset)
- Parts moving before shrink wrap completes
- Material warping pulling parts out of position

## Advanced Techniques

### Multiple Shrink Wraps

Create separate boundaries for different groups:

**Process:**

1. Arrange parts into logical groups
2. Shrink wrap Group 1 (top parts)
3. Shrink wrap Group 2 (bottom parts)
4. Cut groups separately

**Benefits:**

- Remove finished groups during job
- Better organization
- Easier part retrieval

### Nested Shrink Wraps

Shrink wrap within a larger boundary:

**Example:**

1. Inner shrink wrap: Small detailed parts
2. Outer shrink wrap: Includes larger parts
3. Contour: Full sheet boundary

**Use for:** Complex multi-part layouts

### Clearance Testing

Before production run:

1. Create shrink wrap
2. Preview with [Simulation Mode](../simulation-mode)
3. Verify clearance is adequate
4. Check no parts are intersected
5. Run test on scrap material

## Troubleshooting

### Shrink wrap cuts into parts

- **Increase:** Offset distance
- **Check:** Parts aren't too close together
- **Verify:** Shrink wrap path in preview
- **Account for:** Kerf width (laser beam width)

### Boundary doesn't follow shapes

- **Increase:** Smoothness setting
- **Check:** Parts are properly selected
- **Try:** Smaller offset (might be wrapping too far out)

### Parts are wrapped together

- **Increase:** Spacing between parts
- **Add:** Manual contours around individual parts
- **Split:** Into multiple shrink wrap operations

### Cutting takes too long

- **Decrease:** Smoothness (simpler path)
- **Increase:** Offset (straighter boundaries)
- **Consider:** Multiple smaller shrink wraps

### Parts move during cutting

- **Add:** Small tabs to hold parts (see [Holding Tabs](../holding-tabs))
- **Use:** Cutting order: inside to outside
- **Ensure:** Material is flat and secured
- **Check:** Sheet isn't warped

## Technical Details

### Algorithm

Shrink wrap uses computational geometry:

1. **Convex hull** - Find outer boundary
2. **Alpha shape** - Shrink toward objects
3. **Offset** - Expand by offset distance
4. **Simplify** - Based on smoothness setting

### Path Optimization

The boundary path is optimized for:

- Minimum total length
- Smooth curves (based on smoothness)
- Efficient start/end points

### Coordinate System

- **Units:** Millimeters (mm)
- **Precision:** 0.01mm typical
- **Coordinates:** Same as workspace

## Related Topics

- **[Contour Cutting](contour)** - Cutting individual object outlines
- **[Multi-Layer Workflow](../multi-layer)** - Combining operations effectively
- **[Holding Tabs](../holding-tabs)** - Keeping parts secure during cutting
- **[Simulation Mode](../simulation-mode)** - Previewing cutting paths
- **[Material Test Grid](material-test-grid)** - Finding optimal cutting settings
