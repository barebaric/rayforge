# Material Test Grid

The Material Test Grid generator creates parametric test patterns to help you find optimal laser settings for different materials. Instead of manually creating test squares, Rayforge automatically generates a grid with varying speed and power combinations.

## Overview

Material testing is essential for laser work - different materials require different power and speed settings. The Material Test Grid automates this process by:

- Generating test grids with configurable speed/power ranges
- Providing presets for common laser types (Diode, CO2)
- Optimizing execution order for safety (fastest speeds first)
- Adding labels to identify each test cell's settings

<!-- SCREENSHOT
id: feature-material-test-grid-example
type: screenshot
size: full-window
description: |
  Material test grid displayed on canvas showing a 5x5 grid of test squares
  with speed and power labels. Grid shows gradient shading indicating varying
  power levels from light (low power) to dark (high power).
setup:
  - action: menu_click
    path: Tools > Material Test Grid
  - action: set_parameters
    preset: diode_engrave
    grid_rows: 5
    grid_cols: 5
  - action: click_element
    text: "Generate"
  - action: zoom_to_fit
  - action: capture
    region: main_window
filename: Ref-MaterialTestGrid-Example.png
alt: "Material test grid example showing 5x5 test pattern with labels"
-->

<!-- ![Material test grid example](../../images/Ref-MaterialTestGrid-Example.png) -->

## Creating a Material Test Grid

### Step 1: Open the Generator

Access the Material Test Grid generator:

- Menu: **Tools → Material Test Grid**
- This creates a special workpiece that generates the test pattern

### Step 2: Choose a Preset (Optional)

Rayforge includes presets for common scenarios:

| Preset            | Speed Range       | Power Range | Use For               |
| ----------------- | ----------------- | ----------- | --------------------- |
| **Diode Engrave** | 1000-10000 mm/min | 10-100%     | Diode laser engraving |
| **Diode Cut**     | 100-5000 mm/min   | 50-100%     | Diode laser cutting   |
| **CO2 Engrave**   | 3000-20000 mm/min | 10-50%      | CO2 laser engraving   |
| **CO2 Cut**       | 1000-20000 mm/min | 30-100%     | CO2 laser cutting     |

Presets are starting points - you can adjust all parameters after selecting one.

<!-- SCREENSHOT: feature-material-test-presets
description: Material Test Grid settings dialog showing preset dropdown with Diode Engrave selected
filename: UI-MaterialTestGrid-Presets.png
-->

<!-- ![Material test grid presets](../../images/UI-MaterialTestGrid-Presets.png) -->

### Step 3: Configure Parameters

Adjust the test grid parameters:

#### Test Type

- **Engrave**: Fills squares with raster pattern
- **Cut**: Cuts outline of squares

#### Speed Range

- **Min Speed**: Slowest speed to test (mm/min)
- **Max Speed**: Fastest speed to test (mm/min)
- Columns in grid represent different speeds

#### Power Range

- **Min Power**: Lowest power to test (%)
- **Max Power**: Highest power to test (%)
- Rows in grid represent different power levels

#### Grid Dimensions

- **Columns**: Number of speed variations (typically 3-7)
- **Rows**: Number of power variations (typically 3-7)

#### Size & Spacing

- **Shape Size**: Size of each test square in mm (default: 20mm)
- **Spacing**: Gap between squares in mm (default: 5mm)

#### Labels

- **Include Labels**: Enable/disable axis labels showing speed and power values
- Labels appear on left and top edges
- Labels are engraved at 10% power, 1000 mm/min

<!-- SCREENSHOT
id: feature-material-test-settings
type: screenshot
size: dialog
description: |
  Material Test Grid settings dialog showing all configuration options:
  - Preset selector set to "Diode Engrave"
  - Test type: Engrave
  - Speed range: 1000-10000 mm/min
  - Power range: 10-100%
  - Grid: 5 columns x 5 rows
  - Shape size: 20mm
  - Spacing: 5mm
  - Labels: enabled
setup:
  - action: menu_click
    path: Tools > Material Test Grid
  - action: set_parameters
    preset: diode_engrave
    grid_rows: 5
    grid_cols: 5
    shape_size: 20
    spacing: 5
    include_labels: true
  - action: capture
    region: dialog
filename: UI-MaterialTestGrid-SettingsDialog.png
alt: "Material Test Grid settings dialog with all parameters configured"
-->

<!-- ![Material test grid settings](../../images/UI-MaterialTestGrid-SettingsDialog.png) -->

### Step 4: Generate the Grid

Click **Generate** to create the test pattern. The grid appears on your canvas as a special workpiece.

## Understanding the Grid Layout

### Grid Organization

```
Power (%)     Speed (mm/min) →
    ↓      1000   2500   5000   7500   10000
  100%     [  ]   [  ]   [  ]   [  ]   [  ]
   75%     [  ]   [  ]   [  ]   [  ]   [  ]
   50%     [  ]   [  ]   [  ]   [  ]   [  ]
   25%     [  ]   [  ]   [  ]   [  ]   [  ]
   10%     [  ]   [  ]   [  ]   [  ]   [  ]
```

- **Columns**: Speed increases from left to right
- **Rows**: Power increases from bottom to top
- **Labels**: Show exact values for each row/column

### Grid Size Calculation

**Without labels:**

- Width = columns × (shape_size + spacing) - spacing
- Height = rows × (shape_size + spacing) - spacing

**With labels:**

- Add 15mm margin to left and top for label space

**Example:** 5×5 grid with 20mm squares and 5mm spacing:

- Without labels: 120mm × 120mm
- With labels: 135mm × 135mm

## Execution Order (Risk Optimization)

Rayforge executes test cells in a **risk-optimized order** to prevent material damage:

1. **Highest speed first**: Fast speeds are safer (less heat buildup)
2. **Lowest power within speed**: Minimizes risk at each speed level

This prevents charring or fire from starting with slow, high-power combinations.

**Example execution order for 3×3 grid:**

```
Order:  1  2  3
        4  5  6  ← Highest speed, power increasing
        7  8  9

(Fastest speed/lowest power executed first)
```

## Using Material Test Results

### Step 1: Run the Test

1. Load your material in the laser
2. Focus the laser properly
3. Run the material test grid job
4. Monitor the test - stop if any cell causes problems

### Step 2: Evaluate Results

After the test completes, examine each cell:

- **Too light**: Increase power or decrease speed
- **Too dark/charred**: Decrease power or increase speed
- **Perfect**: Note the speed/power combination

### Step 3: Record Settings

Create a material settings reference:

| Material        | Thickness | Operation | Speed       | Power | Notes            |
| --------------- | --------- | --------- | ----------- | ----- | ---------------- |
| Birch Plywood   | 3mm       | Engrave   | 5000 mm/min | 40%   | Perfect contrast |
| Birch Plywood   | 3mm       | Cut       | 500 mm/min  | 80%   | 2 passes         |
| Acrylic (clear) | 3mm       | Cut       | 300 mm/min  | 90%   | Clean edge       |

!!! tip "Material Database"
Consider documenting your material test results for quick reference in future projects.

## Advanced Usage

### Combining with Other Operations

Material test grids are regular workpieces - you can combine them with other operations:

**Example workflow:**

1. Create material test grid
2. Add contour cut around the entire grid
3. Run test, cut free, evaluate results

This is useful for cutting the test piece free from stock material.

### Custom Test Ranges

For fine-tuning, create narrow-range tests:

**Coarse test** (find ballpark):

- Speed: 1000-10000 mm/min (5 columns)
- Power: 10-100% (5 rows)

**Fine test** (optimize):

- Speed: 4000-6000 mm/min (5 columns)
- Power: 35-45% (5 rows)

### Different Materials, Same Grid

Run the same grid configuration on different materials to build your material library faster.

## Tips & Best Practices

### Grid Design

✅ **Start with presets** - Good starting points for common scenarios
✅ **Use 5×5 grids** - Good balance of detail and test time
✅ **Enable labels** - Essential for identifying results
✅ **Keep squares ≥20mm** - Easier to see and measure results

### Testing Strategy

✅ **Test scrap first** - Never test on final material
✅ **One variable at a time** - Test speed OR power range, not both extremes
✅ **Allow cooldown** - Wait between tests on same material
✅ **Consistent focus** - Same focus distance for all tests

### Safety

⚠️ **Monitor tests** - Never leave running tests unattended
⚠️ **Start conservative** - Begin with lower power ranges
⚠️ **Check ventilation** - Ensure proper fume extraction
⚠️ **Fire watch** - Have fire extinguisher ready

## Troubleshooting

### Grid doesn't appear on canvas

- **Check**: Make sure you clicked "Generate" in the settings dialog
- **Check**: Zoom out - the grid might be outside the visible area
- **Try**: Use **View → Zoom to Fit** (++ctrl+0++)

### Labels are missing

- **Check**: "Include Labels" checkbox is enabled
- **Check**: Labels extend beyond the grid - zoom out to see them
- **Note**: Labels add 15mm margin to left and top

### Test cells execute in wrong order

- Rayforge uses risk-optimized order (fastest speeds first)
- This is intentional and cannot be changed
- See [Execution Order](#execution-order-risk-optimization) above

### Grid is too large for material

- **Reduce**: Number of columns/rows
- **Reduce**: Shape size (try 15mm or 10mm)
- **Reduce**: Spacing (try 3mm or 2mm)
- **Calculate**: Use formula in [Grid Size Calculation](#grid-size-calculation)

### Results are inconsistent

- **Check**: Material is flat and properly secured
- **Check**: Focus is consistent across entire test area
- **Check**: Laser power is stable (check power supply)
- **Try**: Smaller grid to reduce test area

## Technical Details

### File Format

Material test grids are stored as:

- **Import Source**: Virtual path `[material-test]` with JSON parameters
- **Workpiece**: References the import source
- **Layer**: Regular layer (can be mixed with other operations)

### Serialization

The test grid parameters are stored as JSON:

```json
{
  "type": "MaterialTestGridProducer",
  "params": {
    "test_type": "Engrave",
    "speed_range": [1000, 10000],
    "power_range": [10, 100],
    "grid_dimensions": [5, 5],
    "shape_size": 20.0,
    "spacing": 5.0,
    "include_labels": true
  }
}
```

## Related Topics

- **[Simulation Mode](../simulation-mode.md)** - Preview test execution before running
- **[Raster Engraving](raster.md)** - Understanding engrave operations
- **[Contour Cutting](contour.md)** - Understanding cut operations
