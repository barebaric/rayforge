---
description: "Simulate your laser job before cutting. Preview toolpaths in 2D or full 3D, catch mistakes early, and save materials with Rayforge's simulation mode."
---

# Simulation Mode

![Simulation Mode](/screenshots/main-simulation.png)

Simulation Mode shows how your laser job will execute before running it on the machine. You can step through the G-code and see exactly what will happen.

## Activating Simulation Mode

- **Keyboard**: Press <kbd>F11</kbd>
- **Menu**: Go to **View → Simulate Execution**
- **Toolbar**: Click the simulation button

## Visualization

### Speed Heatmap

Operations are colored by speed:

| Speed   | Color  |
| ------- | ------ |
| Slowest | Blue   |
| Slow    | Cyan   |
| Medium  | Green  |
| Fast    | Yellow |
| Fastest | Red    |

The colors are relative to your job's speed range - blue is the minimum, red is the maximum.

### Power Transparency

Line opacity shows laser power:

- **Faint lines** = Low power (travel moves, light engraving)
- **Solid lines** = High power (cutting)

## Playback Controls

Use the controls at the bottom of the canvas:

- **Play/Pause** (<kbd>Space</kbd>): Start or stop automatic playback
- **Progress slider**: Drag to scrub through the job
- **Arrow keys**: Step through instructions one at a time

The simulation and G-code view stay in sync — stepping through the simulation highlights the
corresponding G-code line, and clicking a G-code line jumps the simulation to that point.

The 3D view also has its own simulation with synchronized playback. Stepping through the 3D
simulator highlights the matching line in the G-code viewer, and vice versa.

## Editing During Simulation

You can edit workpieces while simulating. Move, scale, or rotate objects, and the simulation updates automatically.

## Related Topics

- **[3D Preview](../ui/3d-preview)** - 3D toolpath visualization
- **[Material Test Grid](operations/material-test-grid)** - Use simulation to validate tests
