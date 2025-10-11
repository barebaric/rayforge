# Common Problems

This page provides quick solutions to frequently encountered issues in RayForge.

## File Import Issues

### SVG Import Problems

**Problem:** SVG file imports with missing or incorrect elements.

**Common causes:**

1. **Unsupported SVG features** - Some advanced SVG features may not import correctly
2. **Embedded raster images** - Images inside SVG files need special handling
3. **Text not converted to paths** - Text must be converted to paths in your design software
4. **Grouped or nested elements** - Deep nesting can sometimes cause issues

**Solutions:**

- In Inkscape: Select all text and use `Path  Object to Path`
- Simplify complex paths: `Path  Simplify`
- Ungroup nested groups: `Object  Ungroup` (multiple times if needed)
- Save as "Plain SVG" or "Optimized SVG" rather than "Inkscape SVG"
- Ensure document units match your intended units (mm, inches)

### DXF Import Problems

**Problem:** DXF file imports with wrong scale or missing elements.

**Solutions:**

1. **Check units** - Ensure the DXF file uses the correct units (mm vs inches)
2. **Verify layer visibility** - Some layers may be hidden or on non-printing layers
3. **Simplify the file** - Remove unnecessary layers and objects in your CAD software
4. **Save as R12/LT2 DXF** - This older format has better compatibility

### PDF Import Issues

**Problem:** PDF imports as raster image instead of vector paths.

**Solution:**

PDFs containing raster images will import as images. To get vector paths:

1. Ensure the PDF contains actual vector graphics, not embedded images
2. Use vector design software (Inkscape, Illustrator) to open and re-export as SVG
3. Flatten layers and convert text to paths before exporting

---

## Job Execution Problems

### Job Runs in Wrong Location

**Problem:** The laser cuts/engraves in the wrong position on the material.

**Diagnosis:**

This is usually an origin or positioning issue.

**Solutions:**

1. **Frame the job first** - Use the Frame command to preview where the job will run
2. **Check job origin settings** - Verify the origin is set correctly (typically top-left or center)
3. **Home the machine** - Some machines require homing before accurate positioning
4. **Check work offset** - Ensure G54 work coordinate system is set correctly
5. **Verify machine position** - Move the laser to the desired start position manually first

### Cuts Don't Go Through Material

**Problem:** The laser doesn't cut all the way through the material.

**Common causes:**

1. **Insufficient power** - Power setting too low for material thickness
2. **Speed too fast** - Not enough time for the laser to cut through
3. **Poor focus** - Laser beam not focused at material surface
4. **Multiple passes needed** - Material too thick for single pass
5. **Air assist off** - Removes debris and improves cutting efficiency

**Solutions:**

- Use [Material Test Grid](../features/operations/material-test-grid.md) to find optimal settings
- Reduce speed or increase power incrementally
- Ensure proper focus distance (usually 2-5mm depending on lens)
- Enable multiple passes for thick materials
- Enable air assist if available
- Clean laser lens and mirrors

### Engraving Too Light or Too Dark

**Problem:** Raster engraving output is too faint or too dark.

**Solutions:**

**Too light:**
- Increase power
- Reduce speed
- Ensure proper focus
- Clean optics
- Check if material surface is reflective (some metals/plastics won't mark well)

**Too dark/charred:**
- Reduce power
- Increase speed
- Enable overscan to reduce acceleration marks
- Adjust DPI/line spacing
- Consider multiple light passes instead of one heavy pass

### Disconnection During Job

**Problem:** Machine disconnects in the middle of a job.

**Possible causes:**

1. **USB cable issues** - Loose connection or faulty cable
2. **Power supply problems** - Voltage drops under load
3. **EMI interference** - Motor cables too close to USB
4. **Firmware crash** - Controller firmware hung
5. **Computer sleep/power saving** - OS suspending USB

**Solutions:**

- Use a high-quality, short (<2m) USB cable
- Secure USB connections (tape if necessary)
- Route USB cables away from stepper motor wires
- Disable USB selective suspend in power settings (Windows)
- Disable computer sleep mode while running jobs
- Update controller firmware

---

## Preview and Simulation Issues

### 3D Preview Looks Wrong

**Problem:** The 3D preview doesn't match expected output.

**Common issues:**

1. **Operation order** - Operations execute in a specific order
2. **Overlapping operations** - Multiple operations affect the same area
3. **Material height incorrect** - Set material thickness in settings

**Solutions:**

- Check operation order in the layers panel
- Use [Simulation Mode](../features/simulation-mode.md) to see execution order and timing
- Adjust Z-height and material thickness settings
- Review operation settings for each layer

### Simulation Mode Not Starting

**Problem:** Simulation mode (F7) doesn't activate or shows nothing.

**Solutions:**

1. Ensure you have at least one operation in your document
2. Check that operations have valid G-code (preview them first)
3. Try regenerating the G-code (make a small change to force regeneration)
4. Check for errors in the console/logs

---

## Performance Issues

### Slow UI Response

**Problem:** RayForge feels sluggish or unresponsive.

**Solutions:**

1. **Reduce preview quality** - Lower 3D preview resolution in settings
2. **Simplify paths** - Use fewer nodes in complex paths
3. **Limit operations** - Reduce number of active operations/layers
4. **Close unused documents** - Multiple open documents consume memory
5. **Update graphics drivers** - Especially on Linux

See [Performance Troubleshooting](performance.md) for detailed optimization tips.

### Long Import Times

**Problem:** Importing large files takes a very long time.

**Solutions:**

- Simplify paths in your design software before importing
- Remove unnecessary detail (nodes, guides, hidden layers)
- Split very large jobs into multiple smaller files
- For raster images, reduce resolution to what's actually needed (typically 300-500 DPI)

---

## Material and Quality Issues

### Burn Marks on Cuts

**Problem:** Visible char marks or discoloration along cut edges.

**Solutions:**

1. **Enable overscan** - Allows laser to reach full speed before cutting
2. **Use air assist** - Blows away smoke and debris
3. **Mask the material** - Apply masking tape to protect surface
4. **Reduce power slightly** - Find minimum power that still cuts through
5. **Increase speed** - Faster cutting = less heat buildup
6. **Multiple light passes** - Instead of one heavy pass

See [Reducing Burn Marks](../guides/reducing-burn-marks.md) guide.

### Inconsistent Engraving Depth

**Problem:** Raster engraving has uneven depth across the work area.

**Possible causes:**

1. **Uneven material surface** - Material not flat
2. **Bed not level** - Cutting bed tilted
3. **Variable focus** - Focal point changes across work area
4. **Inconsistent material** - Material density varies

**Solutions:**

- Use a flat, rigid material surface
- Check bed levelness with a straightedge
- Ensure material is secured flat (not warped)
- Use consistent, high-quality materials
- Test with a known-good material to isolate hardware vs material issues

### Jobs Look Distorted or Skewed

**Problem:** Circles appear oval, squares are parallelograms.

**Causes:**

1. **Mechanical issues** - Loose belts, worn wheels
2. **Step calibration wrong** - Steps-per-mm incorrectly configured
3. **Acceleration too high** - Motors skipping steps

**Solutions:**

- Tighten belts and check for mechanical play
- Calibrate steps-per-mm: [Calibrating Your Workspace](../guides/calibrating-your-workspace.md)
- Reduce acceleration settings in GRBL config
- Check for binding or friction in axes
- Verify pulleys are secure on motor shafts

---

## G-code Issues

### Warning: Out of Bounds

**Problem:** RayForge warns that the job exceeds machine boundaries.

**Solutions:**

1. **Scale down your design** - Reduce size to fit work area
2. **Reposition the job** - Move it within bounds
3. **Check machine dimensions** - Verify work area size in machine settings
4. **Adjust job origin** - Change origin point (e.g., from center to corner)

### G-code Contains Errors

**Problem:** Generated G-code has warnings or errors.

**Solutions:**

1. Check operation settings for invalid values (negative speeds, zero power)
2. Ensure all paths are valid (no unclosed paths for cutting operations)
3. Review operations in the correct order
4. Check for extremely small or degenerate geometry
5. Try removing and re-adding problematic operations

---

## Layer and Organization Issues

### Can't See My Objects

**Problem:** Imported objects aren't visible in the canvas.

**Possible causes:**

1. **Outside viewport** - Objects imported at large coordinates
2. **Layer hidden** - Layer visibility toggled off
3. **Extremely small** - Object too small to see at current zoom
4. **Behind other objects** - Obscured by other elements

**Solutions:**

- Use `View  Zoom to Fit` or zoom out significantly
- Check layer panel for hidden layers (eye icon)
- Select all (`Ctrl+A`) and zoom to selection
- Check the layer stack order

### Operations Not Generating G-code

**Problem:** Operation exists but produces no G-code output.

**Diagnosis:**

1. **No objects assigned** - Operation has no geometry to process
2. **Empty layer** - Layer contains no paths
3. **Invalid operation settings** - Settings that produce no output

**Solutions:**

- Ensure the operation's layer contains visible paths
- Check that paths are assigned to the correct layer
- Verify operation settings (especially power > 0)
- Review operation type matches geometry (e.g., Contour needs closed paths for cutting)

---

## Installation and Startup Issues

### Application Won't Start

=== "Linux (Snap)"
    ```bash
    # Check snap installation
    snap list | grep rayforge

    # View logs
    snap logs rayforge

    # Reinstall if needed
    sudo snap refresh rayforge
    ```

=== "Linux (From Source)"
    ```bash
    # Check Python version
    python --version  # Should be 3.10+

    # Install dependencies
    pixi install

    # Run with verbose logging
    RAYFORGE_LOG_LEVEL=DEBUG pixi run rayforge
    ```

=== "Windows"
    - Ensure you have the latest Visual C++ Redistributable
    - Check that antivirus isn't blocking the executable
    - Try running as administrator
    - Check Windows Event Viewer for error details

### Missing Icons or UI Elements

**Problem:** UI shows missing icons or broken layout.

**Solutions:**

- Ensure GTK libraries are properly installed
- On Linux, install `libadwaita-1-0` package
- Check theme compatibility (try default system theme)
- For Snap users, ensure you have `gtk-common-themes` connected

---

## When to File a Bug Report

If you've tried the solutions above and still have issues, please file a bug report on GitHub with:

1. **RayForge version** and installation method (Snap, source, AppImage, etc.)
2. **Operating system** and version
3. **Steps to reproduce** the issue
4. **Expected vs actual behavior**
5. **Logs** (run with `RAYFORGE_LOG_LEVEL=DEBUG`)
6. **Sample files** if relevant (SVG, project files)
7. **Screenshots** showing the problem

---

## Related Pages

- [Connection Issues](connection.md) - Serial connection troubleshooting
- [Performance](performance.md) - Optimization and performance tuning
- [Snap Permissions](snap-permissions.md) - Linux Snap-specific issues
- [Machine Setup](../machine/index.md) - Machine configuration guide
