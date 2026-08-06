---
description: "Set up camera calibration in Rayforge for accurate workpiece alignment. Use your camera to preview and position designs on materials."
---

# Camera Integration

Rayforge supports USB camera integration for precise material alignment and
positioning. The camera overlay feature allows you to see exactly where your
laser will cut or engrave on the material, eliminating guesswork and reducing
material waste.

![Camera Settings](/screenshots/machine-settings-camera.png)

## Setup Workflow

Setting up a camera can be done either through the guided
[Camera Wizard](#step-2-camera-wizard) — a single flow covering image
settings, lens calibration, and alignment — or by configuring each area
manually from the camera properties panel. Either way, the setup covers four
areas:

1. **Add a camera** — Connect your camera and add it to the machine
   configuration
2. **Adjust image settings** — Tune brightness, contrast, white balance, and
   noise reduction
3. **Calibrate the lens** — Correct distortion with the camera wizard's
   automatic calibration or manual coefficients
4. **Align the camera** — Map camera pixels to machine coordinates for
   accurate positioning

The camera properties panel shows status icons for calibration and alignment
at a glance:

- ✓ **Lens Calibration** — Calibration has been performed
- ⚠ **Image Alignment** — Warning when alignment must be redone (e.g., after
  lens calibration)
- ✓ **Image Alignment** — Alignment is current and valid

---

## Step 1: Add a Camera

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
   - Click the **+** button to add a camera
   - Enter a descriptive name (e.g., "Top Camera", "Work Area Cam")
   - Select the device from the dropdown
     - On Linux: `/dev/video0`, `/dev/video1`, etc.
     - On Windows: Camera 0, Camera 1, etc.

4. **Enable the camera:**
   - Toggle the camera enable switch
   - The live feed should appear on your canvas

---

## Step 2: Camera Wizard

The **Camera Wizard** runs the full camera setup in a single guided flow,
covering all three areas in order — image settings, lens calibration, and
image alignment. It is started from:

- The **Camera Wizard** row in the camera properties panel — click **Start**
- Automatically from the [configuration
  wizard](../getting-started/first-time-setup.md) when you enable a camera on
  its camera step and continue

### Step 2.1: Adjust Image Settings

![Image Settings Dialog](/screenshots/machine-settings-camera-image-settings.png)

Image settings is the camera wizard's first stage — it opens there and lets
you set the resolution, white balance, brightness, contrast, and noise
reduction. If you didn't run the wizard, or want to tweak the values it set,
click **Configure** next to **Image Settings** in the camera properties to
open the image settings dialog. Adjust these parameters to get a clear
camera view:

| Setting           | Description                                                              |
| ----------------- | ------------------------------------------------------------------------ |
| **Brightness**    | Overall image brightness (-100 to +100)                                  |
| **Contrast**      | Edge definition and contrast (0 to 100)                                  |
| **Prefer YUYV**   | Use uncompressed YUYV instead of MJPEG. Slower but can fix some glitches |
| **Transparency**  | Overlay opacity on canvas (0% opaque to 100% transparent)                |
| **White Balance** | Color temperature correction (Auto or 2500–10000K)                       |
| **Denoise**       | Temporal noise reduction (0.0 to 0.95)                                   |

The YUYV option is useful if your camera produces green-tinted images with the
default MJPEG format. Note that YUYV is uncompressed and may reduce the
available resolution or frame rate on USB 2.0 connections.

### Step 2.2: Lens Calibration

If your camera has a wide-angle lens or is mounted at an angle, the image
may show visible curvature — straight lines appear bent, especially near
the edges of the frame. This is called lens distortion, and it can throw
off alignment even if your alignment points are carefully measured.

Lens calibration is the camera wizard's second stage. It lets you choose how
to correct the distortion:

- **Automatic** — capture frames of a printed calibration card; the wizard
  computes the distortion model for you
- **Manual** — enter the radial (k1–k3) and tangential (p1–p2) coefficients
  by hand
- **Skip** — leave the distortion uncorrected; you can calibrate later

#### Automatic Calibration

For **Automatic** calibration, the wizard walks you through capturing
several images of a printed calibration card from different positions on the
bed, then computes a distortion model automatically.

![Wizard — Card Settings](/screenshots/machine-settings-camera-lens-calibration-wizard-card.png)

1. Set the **Width** and **Height** of your printed card. The preview updates
   in real-time — the card should cover about 70% of the camera view.
2. Click **Save to PDF** to export the card for printing, then print it and
   place it on the laser bed.

![Wizard — Capture](/screenshots/machine-settings-camera-lens-calibration-wizard-capture.png)

3. Move the card to different positions and angles within the camera view and
   click **Capture Frame** for each position. Aim for at least 8 captures
   covering the entire frame, including corners and edges. The progress bar
   and status indicators show capture quality.
4. When enough frames are captured, the wizard computes the distortion model
   and applies it — the camera overlay now shows a corrected, straight
   image.

#### Manual Calibration

![Lens Calibration Dialog](/screenshots/machine-settings-camera-lens-calibration.png)

For manual coefficients or to fine-tune the result after an automatic
calibration, open the lens calibration dialog by clicking **Configure** next
to **Lens Calibration** in the camera properties. From here you can adjust
the distortion coefficients manually — fine-tune the radial (k1–k3) and
tangential (p1–p2) parameters.

### Step 2.3: Image Alignment

![Image Alignment Dialog](/screenshots/machine-settings-camera-image-alignment.png)

Image alignment is the camera wizard's final stage. Camera alignment
calibrates the relationship between camera pixels and real-world
coordinates, enabling accurate positioning. The wizard uses the same
procedure described here, and applying the alignment finishes the wizard.

#### Why Alignment is Necessary

The camera sees the work area from above, but the image may be:

- Rotated relative to the machine axes
- Scaled differently in X and Y directions
- Distorted by lens perspective

Alignment creates a transformation matrix that maps camera pixels to machine
coordinates.

#### Alignment Procedure

1. **Open the Alignment Dialog:**
   - Click the **Configure** button next to **Image Alignment** in the camera
     properties
   - The dialog shows the camera feed with the current alignment overlay

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
   - Click **Apply** to calculate the transformation
   - The camera overlay will now be properly aligned

6. **Verify alignment:**
   - Move the laser head to a known position
   - Check that the laser dot aligns with the expected position in the camera
     view
   - Fine-tune by re-aligning if needed

#### Alignment Status

The camera properties panel shows the alignment status with an icon:

- **Checkmark** — Alignment is current and valid
- **Warning** — Alignment must be redone. This happens when lens calibration
  is updated, because the distortion correction changes the camera image and
  invalidates the existing alignment. Your alignment points are preserved —
  simply open the dialog and click
  **Apply** again.

#### Example Workflow

1. Move laser to home position (0, 0) and mark in camera
2. Move laser to (100, 0) and mark in camera
3. Move laser to (100, 100) and mark in camera
4. Move laser to (0, 100) and mark in camera
5. Enter exact coordinates for each point
6. Click **Apply** and verify

:::tip Best Practices

- Use points at the corners of your work area for maximum coverage
- Avoid clustering points in one area
- Measure world coordinates carefully — accuracy here determines overall
  alignment quality
- Re-align if you move the camera or change the focus distance
- Re-align after updating lens calibration
- Save your alignment — it persists across sessions
  :::

---

## Using the Camera Overlay

Once aligned, the camera overlay helps position jobs accurately. Toggle it
by clicking the camera icon in the main window toolbar.

---

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

1. **Camera in use by another application** — Close other video apps
2. **Incorrect device selected** — Try different device IDs
3. **Camera permissions** — On Linux Snap, ensure camera interface connected
4. **Hardware issue** — Test camera with another application

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

1. **Insufficient alignment points** — Use at least 4 points
2. **Measurement errors** — Double-check world coordinates
3. **Camera moved** — Re-align if camera position changed
4. **Non-linear distortion** — May need lens calibration

**Improve accuracy:**

- Use more alignment points (6–8 for very large areas)
- Spread points across entire work area
- Measure world coordinates very carefully
- Use machine movement commands to precisely position laser at known
  coordinates
- Re-align after any camera adjustments

### Poor Image Quality

**Problem:** Camera image is blurry, dark, or washed out.

**Solutions:**

1. **Adjust brightness/contrast** in camera settings
2. **Improve lighting** — Add consistent work area lighting
3. **Clean camera lens** — Dust and debris reduce clarity
4. **Check focus** — Auto-focus may not work well; use manual if possible
5. **Reduce transparency** temporarily to see camera image more clearly
6. **Try different white balance** settings
7. **Adjust denoise setting** if image appears grainy

### Camera Lag or Stuttering

**Problem:** Live camera feed is choppy or delayed.

**Solutions:**

- Lower camera resolution in device settings (if accessible)
- Close other applications using CPU/GPU
- Update graphics drivers

---

## Related Pages

- [3D Preview](../ui/3d-preview.md) — Preview execution with camera overlay
- [Framing Jobs](../features/framing-your-job.md) — Verify job position
- [General Settings](general) — Machine configuration
