# Performance Optimization

This page helps you diagnose and resolve performance issues in Rayforge, including slow UI, sluggish rendering, and long processing times.

## Common Performance Issues

### Slow Canvas Response

**Symptoms:**
- Panning and zooming feels laggy
- Selecting objects takes time
- Moving objects stutters

**Solutions:**

#### Simplify Complex Paths

Complex vector paths with thousands of nodes can slow down rendering:

1. **In Inkscape before import:**
   - Select path
   - `Path  Simplify` (or Ctrl+L)
   - Adjust smoothing threshold to reduce nodes while preserving shape

2. **Check node count:**
   - Very detailed paths (>5000 nodes) should be simplified
   - For engraving, detail beyond what the laser can resolve is wasted

3. **Remove hidden objects:**
   - Delete guides, construction lines, and hidden layers
   - Only import what needs to be cut/engraved

#### Reduce Active Operations

Each operation adds processing overhead:

- Combine similar operations when possible
- Disable or delete unused operations
- Break very large jobs into multiple smaller files

#### Optimize Raster Images

Large raster images impact performance:

- Reduce resolution to what's actually needed (300-500 DPI is typical)
- Crop images to remove unnecessary areas
- Convert to grayscale if color isn't needed
- Avoid extremely large image files (>10MB)

---

### Slow 3D Preview

**Symptoms:**
- Long delay when opening 3D preview
- Choppy rotation and navigation
- High CPU/GPU usage

**Solutions:**

#### Reduce Preview Quality

Rayforge may allow adjusting preview quality in preferences:

1. Open Settings/Preferences
2. Look for 3D Preview or Rendering settings
3. Reduce:
   - Mesh resolution/quality
   - Anti-aliasing
   - Texture detail

#### Simplify Operations

The 3D preview must render all operations:

- Reduce operation count
- Lower raster DPI for preview (doesn't affect output)
- Disable preview auto-update if available

#### Update Graphics Drivers

Especially on Linux:

```bash
# Check current driver
glxinfo | grep "OpenGL renderer"

# Ubuntu/Debian: Update Mesa drivers
sudo apt update && sudo apt upgrade

# For NVIDIA cards, use proprietary drivers
sudo ubuntu-drivers autoinstall
```

---

### Long G-code Generation Time

**Symptoms:**
- Significant delay before job starts
- "Generating G-code..." takes minutes
- Application becomes unresponsive during generation

**Causes:**

1. **Complex raster engraving** - High DPI on large areas
2. **Many small operations** - Hundreds of small paths
3. **Inefficient path ordering** - Optimization calculations

**Solutions:**

#### Optimize Raster Settings

For engraving operations:

- **Reduce DPI** - 300-500 DPI is usually sufficient, not 1200+
- **Crop images** - Only engrave the necessary area
- **Increase line spacing** - Slightly wider spacing significantly reduces G-code
- **Use appropriate operation type** - Depth engraving for 3D effects, raster for photos

#### Simplify Vector Operations

For cutting/contour operations:

- **Merge adjacent paths** - Combine small segments
- **Remove tiny details** - Features smaller than laser spot size won't show
- **Use appropriate tolerance** - Don't over-optimize path ordering for simple jobs

#### Example: Large Photo Engraving

Before (slow):
- Image: 3000x2000 pixels (6MP)
- DPI: 1000
- Result: 10+ minutes generation, 500MB G-code

After (optimized):
- Image: Scaled to 1500x1000 pixels
- DPI: 500
- Cropped to subject area only
- Result: <1 minute generation, 50MB G-code
- **Visual quality: Nearly identical**

---

### Application Startup Slow

**Symptoms:**
- Rayforge takes a long time to launch
- Splash screen visible for extended period

**Solutions:**

=== "Linux (Snap)"
    Snaps have some startup overhead due to sandboxing:

    ```bash
    # Ensure snap is updated
    sudo snap refresh rayforge

    # Check for disk space issues
    df -h

    # Clear snap cache if needed
    sudo snap set system refresh.retain=2
    ```

=== "Linux (From Source)"
    ```bash
    # Precompile Python files
    python -m compileall rayforge/

    # Use faster Python interpreter (if available)
    python3.11  # or latest version
    ```

=== "General"
    - Close unused documents before quitting (they reopen on startup)
    - Clear recent file history if very long
    - Check for large log files that may slow startup

---

### Import Takes Too Long

**Symptoms:**
- Importing SVG/DXF/PDF files is very slow
- Application freezes during import

**Solutions:**

#### Prepare Files in Design Software

**Before exporting to Rayforge:**

1. **Simplify paths:**
   - Inkscape: `Path  Simplify`
   - Illustrator: `Object  Path  Simplify`

2. **Remove unnecessary elements:**
   - Delete guides, grids, construction geometry
   - Remove hidden layers
   - Flatten nested groups

3. **Convert text to paths:**
   - Select all text
   - `Path  Object to Path`

4. **Optimize SVG export:**
   - Use "Optimized SVG" or "Plain SVG"
   - Don't include metadata, editor data

#### File Size Guidelines

For smooth import:

- **SVG files:** <5MB ideal, <20MB acceptable
- **Raster images:** <10MB, reduce resolution if larger
- **DXF files:** <1000 entities for fast import

#### Split Large Jobs

Instead of one massive file:

- Break into multiple smaller sections
- Import and process separately
- Combine if needed after optimization

---

## System Requirements

### Recommended Hardware

For smooth performance:

- **CPU:** Modern multi-core processor (4+ cores)
- **RAM:** 8GB minimum, 16GB+ recommended for large files
- **GPU:** Dedicated graphics recommended for 3D preview
- **Storage:** SSD preferred (faster file I/O)

### Operating System

#### Linux

**Best performance:**
- Recent kernel (5.15+)
- Up-to-date Mesa graphics drivers
- Wayland or X11 (both supported)

**For GTK4/Libadwaita:**
```bash
# Ensure required libraries installed
sudo apt install libadwaita-1-0 libgtk-4-1
```

#### Windows

- Windows 10/11 recommended
- Latest Visual C++ Redistributable
- Updated graphics drivers

---

## Monitoring Performance

### Check Resource Usage

=== "Linux"
    ```bash
    # Monitor CPU and memory while running Rayforge
    htop

    # Check GPU usage (if NVIDIA)
    nvidia-smi

    # Monitor disk I/O
    iotop
    ```

=== "Windows"
    - Open Task Manager (Ctrl+Shift+Esc)
    - Check "Performance" tab
    - Look for high CPU, Memory, or Disk usage

### Enable Debug Logging

To identify bottlenecks:

```bash
# Run with debug logging
RAYFORGE_LOG_LEVEL=DEBUG rayforge 2>&1 | tee rayforge.log

# Look for slow operations in the log
grep -i "slow\|delay\|timeout" rayforge.log
```

---

## Optimization Checklist

Before reporting performance issues, try:

- [ ] Simplify paths (reduce node count)
- [ ] Reduce raster image resolution
- [ ] Lower 3D preview quality
- [ ] Remove unused operations/layers
- [ ] Update graphics drivers
- [ ] Close other applications
- [ ] Ensure adequate free disk space (>10GB)
- [ ] Ensure adequate free RAM (>2GB available)
- [ ] Check for system updates
- [ ] Try a known-simple file (single square) to isolate issue

---

## Known Performance Limitations

### Large Raster Engravings

Very large photo engravings (>100cm at high DPI) will inherently take time to process:

- **This is normal** - Millions of G-code commands must be generated
- **Mitigation:** Use appropriate DPI, don't over-sample

### Complex Path Optimization

Optimization algorithms for ordering thousands of paths:

- **This is normal** - Traveling salesman problem is computationally expensive
- **Mitigation:** Reduce path count, disable optimization if speed isn't critical

### First Operation After Import

Initial processing of a new file may be slower:

- **This is normal** - Caching and initial calculations
- **Subsequent operations should be faster**

---

## Platform-Specific Issues

### Linux: Snap Performance

Snap sandboxing adds some overhead:

**If performance is critical:**

1. Consider building from source instead:
   ```bash
   git clone https://github.com/kylemartin57/rayforge
   cd rayforge
   pixi install
   pixi run rayforge
   ```

2. Or use Flatpak (if available) - often faster than Snap

### Linux: Wayland vs X11

Some users report better performance on X11:

```bash
# Force X11 session
# (Configure in your display manager before login)
```

### Windows: Anti-virus Scanning

Real-time scanning can slow file operations:

- Add Rayforge installation directory to antivirus exclusions
- Add temporary directory to exclusions
- Don't disable antivirus entirely - use targeted exclusions

---

## Reporting Performance Issues

If performance is still poor after optimization:

1. **Provide system specs:**
   - CPU model and speed
   - RAM amount
   - GPU model
   - OS and version

2. **Describe the specific operation:**
   - File import? G-code generation? UI interaction?
   - How long does it take?
   - With what file (provide sample if possible)

3. **Include benchmarks:**
   ```bash
   # Example timing information
   time rayforge --import test.svg --export test.gcode
   ```

4. **Attach debug log:**
   ```bash
   RAYFORGE_LOG_LEVEL=DEBUG rayforge 2>&1 | tee performance-issue.log
   ```

---

## Related Pages

- [Common Problems](common.md) - General troubleshooting
- [Connection Issues](connection.md) - Serial connection performance
- [3D Preview](../ui/3d-preview.md) - Preview settings
- [Operations](../features/operations/index.md) - Operation optimization
