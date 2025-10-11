# Simulating Your Job

Learn how to use Rayforge's simulation mode to preview your laser job, identify potential issues, and estimate completion time before running on real hardware.

## Overview

Simulation mode allows you to visualize your laser job execution without actually running the machine. This helps catch errors, optimize settings, and plan your workflow.

## Benefits of Simulation

- **Preview job execution**: See exactly how the laser will move
- **Estimate time**: Get accurate job duration estimates
- **Identify issues**: Spot overlaps, gaps, or unexpected behavior
- **Optimize path order**: Visualize cutting sequence
- **Learn G-code**: Understand how operations translate to machine commands

## Starting a Simulation

1. **Load or create your design** in Rayforge
2. **Configure operations** with desired settings
3. **Click the Simulate button** in the toolbar (or use keyboard shortcut)
4. **Watch the simulation** play through your job

## Simulation Controls

### Playback Controls

- **Play/Pause**: Start or pause the simulation
- **Step Forward/Back**: Move through the job one command at a time
- **Speed Control**: Adjust playback speed (0.5x to 10x)
- **Jump to Position**: Skip to specific percentage of job
- **Restart**: Begin simulation from the start

### Visualization Options

- **Show toolpath**: Display the path the laser head will follow
- **Show travel moves**: Visualize rapid positioning moves
- **Show laser power**: Color-code paths by power level
- **Heatmap mode**: Visualize dwell time and power density

### Information Display

During simulation, monitor:

- **Current position**: X, Y coordinates of laser head
- **Job progress**: Percentage complete
- **Estimated time remaining**: Based on current progress
- **Current operation**: Which operation is executing
- **Power and speed**: Current laser parameters

## Interpreting Simulation Results

### What to Look For

- **Path efficiency**: Are there unnecessary travel moves?
- **Overlapping cuts**: Unintended double-cutting of paths
- **Operation order**: Does the sequence make sense?
- **Power distribution**: Is power applied consistently?
- **Unexpected moves**: Any jerky or strange motion patterns

### Heatmap Visualization

The heatmap shows cumulative laser exposure:

- **Cool colors (blue/green)**: Low exposure
- **Warm colors (yellow/orange)**: Moderate exposure
- **Hot colors (red)**: High exposure or dwell time

Use this to identify:

- **Hotspots**: Areas that may over-burn
- **Gaps**: Areas that may be under-exposed
- **Overlap issues**: Unintended double-exposure

See [Simulation Mode](../features/simulation-mode.md) for detailed information.

## Using Simulation for Optimization

### Optimize Cut Order

If simulation reveals inefficient path order:

1. **Enable path optimization** in operation settings
2. **Choose optimization method** (nearest neighbor, TSP)
3. **Re-simulate** to verify improvement

### Adjust Timing

Simulation provides accurate time estimates:

- **Long job times**: Consider optimizing paths or increasing speed
- **Very short times**: Verify settings are correct for material
- **Unexpected duration**: Check for hidden operations or duplicates

### Verify Multi-Layer Jobs

For complex multi-layer projects:

1. **Simulate each layer** independently
2. **Verify operation order** across layers
3. **Check for conflicts** between layers
4. **Estimate total time** for complete job

## Simulation vs. Real Execution

### Differences to Note

Simulation is highly accurate but:

- **Doesn't account for**: Mechanical imperfections, backlash, vibration
- **May differ slightly**: Actual acceleration/deceleration vs. simulated
- **Doesn't show**: Material interaction, smoke, fumes
- **Time estimates**: Usually accurate within 5-10%

### When to Re-simulate

- **After changing settings**: Power, speed, or operation parameters
- **After editing design**: Any design changes
- **Before expensive materials**: Double-check before committing
- **When troubleshooting**: Verify fixes to identified issues

## Tips for Effective Simulation

- **Always simulate** before running important jobs
- **Use slower playback** to catch subtle issues
- **Enable heatmap** for engraving jobs
- **Compare multiple settings** by simulating variations
- **Document results**: Screenshot or note issues found

## Troubleshooting Simulation

**Simulation won't start**: Check that operations are properly configured

**Simulation runs too fast**: Adjust playback speed to slower setting

**Can't see details**: Zoom in on specific areas of interest

**Time estimate seems wrong**: Verify machine profile has correct max speeds

## Related Topics

- [Simulation Mode Feature](../features/simulation-mode.md)
- [Optimizing Large Jobs](../guides/optimizing-large-jobs.md)
- [Multi-Layer Workflow](../features/multi-layer.md)
