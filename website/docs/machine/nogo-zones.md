---
description: "Define no-go zones in Rayforge to protect sensitive areas of your laser cutter. Prevent the laser head from colliding with clamps and fixtures."
---

# No-Go Zones

No-go zones define restricted areas on the work surface that the laser should
not enter. When enabled, they are checked as part of the
[job sanity checks](../features/sanity-checks) before running or exporting.

![No-Go Zones](/screenshots/machine-nogo-zones.png)

## Adding a No-Go Zone

Open **Settings → Machine** and navigate to the **No-Go Zones** page. Click
the add button to create a new zone, then choose its shape and position.

Each zone has the following settings:

- **Name**: A descriptive label for the zone
- **Enabled**: Toggle the zone on or off without deleting it
- **Shape**: Rectangle, Box, or Cylinder
- **Position (X, Y, Z)**: Where the zone is placed on the work surface
- **Dimensions**: Width, height, and depth (or radius for cylinders)

## Visibility

No-go zones are displayed on both the 2D and 3D canvas as semi-transparent
overlays. Use the no-go zone toggle button in the canvas overlay to show or
hide them. The visibility setting is remembered between sessions.

---

## Related Pages

- [Hardware Settings](hardware) - Machine dimensions and axis configuration
- [Job Sanity Checks](../features/sanity-checks) - Pre-job validation
- [3D View](../ui/3d-preview) - 3D toolpath visualization
