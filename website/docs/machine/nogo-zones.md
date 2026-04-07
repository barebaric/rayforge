# No-Go Zones

No-go zones define restricted areas on the work surface that the laser should
not enter. Before running or exporting a job, Rayforge checks whether any
toolpaths enter an enabled no-go zone and warns you if a collision is detected.

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

## Collision Warnings

When you run or export a job, Rayforge checks all toolpaths against enabled
no-go zones. If a toolpath passes through a zone, a warning dialog appears
with the option to cancel or proceed at your own risk.

## Visibility

No-go zones are displayed on both the 2D and 3D canvas as semi-transparent
overlays. Use the no-go zone toggle button in the canvas overlay to show or
hide them. The visibility setting is remembered between sessions.

---

## Related Pages

- [Hardware Settings](hardware) - Machine dimensions and axis configuration
- [3D View](../ui/3d-preview) - 3D toolpath visualization
