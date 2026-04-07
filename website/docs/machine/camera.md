# Camera Integration

Rayforge supports USB camera integration for precise material alignment and positioning.
The camera overlay feature allows you to see exactly where your laser will cut or
engrave on the material, eliminating guesswork and reducing material waste.

![Camera Settings](/screenshots/machine-camera.png)

## Overview

The camera integration provides:

- **Live video overlay** on the canvas showing your material in real-time
- **Image alignment** to calibrate camera position relative to the laser
- **Visual positioning** to accurately place jobs on irregular or pre-marked materials
- **Material preview** before running jobs
- **Multiple camera support** for different machine setups

:::tip Use Cases

- Aligning cuts on pre-printed materials
- Working with irregularly shaped materials
- Precise placement of engravings on existing objects
- Reducing test cuts and material waste
  :::

---

## Camera Setup

### Hardware Requirements

**Compatible cameras:**

- USB webcams (most common)
- Laptop built-in cameras (if running Rayforge on laptop near machine)
- Any camera supported by Video4Linux2 (V4L2) on Linux or DirectShow on Windows

**Recommended setup:**

- Camera mounted above the work area with clear view of material
- Consistent lighting conditions
- Camera positioned to capture the laser work area
- Secure mounting to prevent camera movement

### Adding a Camera

1. **Connect your camera** to your computer via USB

2. **Open Camera Settings:**
   - Navigate to **Settings → Preferences → Camera**
   - Or use the camera toolbar button

3. **Add a new camera:**
   - Click the "+" button to add a camera
   - Enter a descriptive name (e.g., "Top Camera", "Work Area Cam")
   - Select the device from the dropdown
     - On Linux: `/dev/video0`, `/dev/video1`, etc.
     - On Windows: Camera 0, Camera 1, etc.

4. **Enable the camera:**
   - Toggle the camera enable switch
   - The live feed should appear on your canvas

5. **Adjust camera settings:**
   - **Brightness:** Adjust if material is too dark/bright
   - **Contrast:** Enhance edge visibility
   - **Transparency:** Control overlay opacity (20-50% recommended)
   - **White Balance:** Auto or manual Kelvin temperature

---

## Camera Alignment

Camera alignment calibrates the relationship between camera pixels and real-world
coordinates, enabling accurate positioning.

### Why Alignment is Necessary

The camera sees the work area from above, but the image may be:

- Rotated relative to the machine axes
- Scaled differently in X and Y directions
- Distorted by lens perspective

Alignment creates a transformation matrix that maps camera pixels to machine coordinates.

### Alignment Procedure

1. **Open the Alignment Dialog:**
   - Click the camera alignment button in the toolbar
   - Or go to **Camera → Align Camera**

2. **Place alignment markers:**
   - You need at least 3 reference points (4 recommended for better accuracy)
   - Alignment points should be spread across the work area
   - Use known positions like:
     - Machine home position
     - Ruler markings
     - Pre-cut alignment holes
     - Calibration grid

3. **Mark image points:**
   - Click on the camera image to place a point at a known location
   - The bubble widget appears showing point coordinates
   - Repeat for each reference point

4. **Enter world coordinates:**
   - For each image point, enter the real-world X/Y coordinates in mm
   - These are the actual machine coordinates where each point is located
   - Measure accurately with a ruler or use known machine positions

5. **Apply alignment:**
   - Click "Apply" to calculate the transformation
   - The camera overlay will now be properly aligned

6. **Verify alignment:**
   - Move the laser head to a known position
   - Check that the laser dot aligns with the expected position in the camera view
   - Fine-tune by re-aligning if needed

### Alignment Tips

:::tip Best Practices

- Use points at the corners of your work area for maximum coverage
- Avoid clustering points in one area
- Measure world coordinates carefully - accuracy here determines overall alignment quality
- Re-align if you move the camera or change the focus distance
- Save your alignment - it persists across sessions
  :::

**Example alignment workflow:**

1. Move laser to home position (0, 0) and mark in camera
2. Move laser to (100, 0) and mark in camera
3. Move laser to (100, 100) and mark in camera
4. Move laser to (0, 100) and mark in camera
5. Enter exact coordinates for each point
6. Apply and verify

---

## Using the Camera Overlay

Once aligned, the camera overlay helps position jobs accurately.

### Enabling/Disabling the Overlay

- **Toggle camera:** Click the camera icon in the toolbar
- **Adjust transparency:** Use the slider in camera settings (20-50% works well)
- **Refresh image:** Camera updates continuously while enabled

### Positioning Jobs with the Camera

**Workflow for precise placement:**

1. **Enable the camera overlay** to see your material

2. **Import your design** (SVG, DXF, etc.)

3. **Position the design** on the canvas:
   - Drag the design to align with features visible in the camera
   - Use zoom to see fine details
   - Rotate/scale as needed

4. **Preview the alignment:**
   - Use the [Simulation Mode](../features/simulation-mode) to visualize
   - Check that cuts/engravings will be where you expect

5. **Frame the job** to verify positioning before running

6. **Run the job** with confidence

### Example: Engraving on a Pre-Printed Card

1. Place the printed card on the laser bed
2. Enable camera overlay
3. Import your engraving design
4. Drag and position the design to align with printed features
5. Fine-tune position using arrow keys
6. Frame to verify
7. Run the job

---

## Camera Settings Reference

### Device Settings

| Setting       | Description                     | Values                               |
| ------------- | ------------------------------- | ------------------------------------ |
| **Name**      | Descriptive name for the camera | Any text                             |
| **Device ID** | System device identifier        | `/dev/video0` (Linux), `0` (Windows) |
| **Enabled**   | Camera active state             | On/Off                               |

### Image Adjustment

| Setting           | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| **Brightness**    | Overall image brightness (-100 to +100)                                  |
| **Contrast**      | Edge definition and contrast (0 to 100)                                  |
| **Prefer YUYV**   | Use uncompressed YUYV instead of MJPEG. Slower but can fix some glitches |
| **Transparency**  | Overlay opacity on canvas (0% opaque to 100% transparent)                |
| **White Balance** | Color temperature correction (Auto or 2500-10000K)                       |
| **Denoise**       | Temporal noise reduction (0.0 to 0.95)                                   |

The YUYV option is useful if your camera produces green-tinted images with the
default MJPEG format. Note that YUYV is uncompressed and may reduce the
available resolution or frame rate on USB 2.0 connections.

### Alignment Data

| Property                  | Description                         |
| ------------------------- | ----------------------------------- |
| **Image Points**          | Pixel coordinates in camera image   |
| **World Points**          | Real-world machine coordinates (mm) |
| **Transformation Matrix** | Calculated mapping (internal)       |

---

## Advanced Features

### Camera Calibration (Lens Distortion Correction)

If your camera has a wide-angle lens or is mounted at an angle, the image may show
visible curvature — straight lines appear bent, especially near the edges of the frame.
This is called lens distortion, and it can throw off alignment even if your alignment
points are carefully measured.

Rayforge includes a guided calibration wizard that corrects this distortion
automatically. Here is how it works:

1. **Print the calibration card** — Rayforge provides a printable pattern (a grid of
   markers) that you place on your laser bed
2. **Follow the wizard** — The calibration wizard walks you through capturing several
   images of the card from different positions on the bed
3. **Apply the correction** — Rayforge computes a distortion model from the captured
   images and uses it to straighten the camera overlay

Once calibrated, the camera overlay will show a noticeably more accurate representation
of what is on the bed. This is especially helpful for wide-angle lenses, cameras mounted
off-center, or work that requires tight alignment tolerances.

:::note When to Calibrate
Calibration is most useful when you notice the camera overlay does not line up well with
the real bed, even after careful alignment. If your current alignment looks good, you may
not need it. But if things seem slightly off — especially toward the edges of the frame —
running through the calibration wizard usually helps.
:::

### Multiple Cameras

Rayforge supports multiple cameras for different views or machines:

- Add multiple cameras in preferences
- Each camera can have independent alignment
- Switch between cameras using the camera selector
- Use cases:
  - Top view + side view for 3D objects
  - Different cameras for different machines
  - Wide angle + detail camera

---

## Troubleshooting

### Camera Not Detected

**Problem:** Camera doesn't appear in device list.

**Solutions:**

**Linux:**
Check if the camera is recognized by the system:

```bash
# List video devices
ls -l /dev/video*

# Check camera with v4l2
v4l2-ctl --list-devices

# Test with another application
cheese  # or VLC, etc.
```

**For Snap users:**

```bash
# Grant camera access
sudo snap connect rayforge:camera
```

**Windows:**

- Check Device Manager for camera under "Cameras" or "Imaging devices"
- Ensure no other application is using the camera (close Zoom, Skype, etc.)
- Try a different USB port
- Update camera drivers

### Camera Shows Black Screen

**Problem:** Camera detected but shows no image.

**Possible causes:**

1. **Camera in use by another application** - Close other video apps
2. **Incorrect device selected** - Try different device IDs
3. **Camera permissions** - On Linux Snap, ensure camera interface connected
4. **Hardware issue** - Test camera with another application

**Solutions:**

```bash
# Linux: Release camera device
sudo killall cheese  # or other camera apps

# Check which process is using the camera
sudo lsof /dev/video0
```

### Alignment Not Accurate

**Problem:** Camera overlay doesn't match real laser position.

**Diagnosis:**

1. **Insufficient alignment points** - Use at least 4 points
2. **Measurement errors** - Double-check world coordinates
3. **Camera moved** - Re-align if camera position changed
4. **Non-linear distortion** - May need lens calibration

**Improve accuracy:**

- Use more alignment points (6-8 for very large areas)
- Spread points across entire work area
- Measure world coordinates very carefully
- Use machine movement commands to precisely position laser at known coordinates
- Re-align after any camera adjustments

### Poor Image Quality

**Problem:** Camera image is blurry, dark, or washed out.

**Solutions:**

1. **Adjust brightness/contrast** in camera settings
2. **Improve lighting** - Add consistent work area lighting
3. **Clean camera lens** - Dust and debris reduce clarity
4. **Check focus** - Auto-focus may not work well; use manual if possible
5. **Reduce transparency** temporarily to see camera image more clearly
6. **Try different white balance** settings
7. **Adjust denoise setting** if image appears grainy

### Camera Lag or Stuttering

**Problem:** Live camera feed is choppy or delayed.

**Solutions:**

- Lower camera resolution in device settings (if accessible)
- Close other applications using CPU/GPU
- Update graphics drivers
- On Linux, ensure using V4L2 backend (automatic in Rayforge)
- Disable camera when not needed to save resources

---

## Related Pages

- [Simulation Mode](../features/simulation-mode) - Preview execution with camera overlay
- [3D Preview](../ui/3d-preview) - Visualize jobs in 3D
- [Framing Jobs](../features/framing-your-job) - Verify job position
- [General Settings](general) - Machine configuration
