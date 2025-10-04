# Using Camera Alignment

Align your laser work to existing material or previously cut pieces using RayForge's camera feature.

## Goal

Use a camera to precisely position your design on material, allowing you to:

- Engrave on pre-cut or irregularly shaped objects
- Align work to existing features on material
- Precisely place designs on specific locations
- Resume work on previously cut pieces

## Prerequisites

- Camera installed and configured in RayForge
- Material placed on laser bed
- Basic understanding of RayForge workpiece operations
- Camera calibration completed (if required by your setup)

## When to Use Camera Alignment

**Good use cases:**
- Engraving on pre-made objects (coasters, boxes, etc.)
- Aligning to printed artwork or existing designs
- Cutting around irregularly shaped material
- Multi-step processes (cut, then engrave specific areas)
- Precise placement on small or expensive materials

**Not ideal for:**
- Simple rectangular cuts on blank material (use job origin instead)
- Production runs of identical parts (use fixtures instead)
- Materials where camera view is blocked

## Camera Alignment Workflow Overview

**Process summary:**

1. Configure camera in RayForge settings
2. Place material on laser bed
3. Capture camera image
4. Mark alignment points on image
5. Align design to alignment points
6. Run job with camera-aligned positioning

## Step 1: Configure Camera

Ensure your camera is set up correctly in RayForge.

**Check camera settings:**

1. Navigate to **Settings > Camera**
2. Verify camera is detected and selected
3. Test camera feed - you should see live video
4. Adjust camera settings if needed:
   - **Resolution:** Higher is better (720p or 1080p)
   - **Exposure:** Adjust for clear image of material
   - **Focus:** Should be focused at laser bed height

**Camera position:**
- Camera should be mounted to view the laser bed
- Typically mounted on laser head (moves with head) or fixed above bed
- Clear view of work area without obstructions

## Step 2: Place Material on Laser Bed

Position your material where you want to work.

**Material placement tips:**

- Secure material so it won't move during cutting
- Ensure material is flat (use tape, magnets, or hold-downs)
- Position material within camera's field of view
- Avoid reflective materials that cause glare

**For pre-cut objects:**
- Identify features you want to align to
- Make sure those features are visible to camera
- Consider marking reference points with tape or marker

## Step 3: Capture Camera Image

Capture a reference image of your material.

**Open camera view:**

1. In RayForge, navigate to **Camera** menu or toolbar
2. Select **Capture Image** (or similar option)
3. Camera captures current view of laser bed

**Move laser head if needed:**
- If camera is mounted on head, jog the head to position camera over your material
- Use manual jog controls or move to specific coordinates
- Ensure the area you want to align to is visible in camera view

**Capture settings:**
- Use good lighting for clear image
- Avoid shadows or glare
- Capture at position where laser will start working

## Step 4: Mark Alignment Points

Define reference points that link camera image to real-world coordinates.

**What are alignment points?**
- Points you mark on the camera image
- Correspond to known positions on your material
- RayForge uses these to calculate image-to-workspace transformation

**How many points?**
- **Minimum 2 points:** Basic alignment (translation and rotation)
- **3+ points:** Better accuracy, compensates for camera distortion
- **4 points:** Recommended for best results

**Marking alignment points:**

1. **Identify features on material:**
   - Corners of pre-cut object
   - Edges or intersections
   - Pre-marked reference dots
   - Any clear, identifiable feature

2. **Click on camera image** to place alignment point
3. **Enter real-world coordinates** for each point:
   - Where is this point in machine coordinates?
   - Example: Top-left corner at (50mm, 50mm)

**Example alignment:**

```
Material: Pre-cut 100mm x 100mm coaster at position (50, 50)

Point 1: Top-left corner of coaster
  - Click on image at coaster's top-left corner
  - Enter coordinates: (50, 50)

Point 2: Top-right corner of coaster
  - Click on image at coaster's top-right corner
  - Enter coordinates: (150, 50)

Point 3: Bottom-right corner of coaster
  - Click on image at coaster's bottom-right corner
  - Enter coordinates: (150, 150)

Point 4: Bottom-left corner of coaster
  - Click on image at coaster's bottom-left corner
  - Enter coordinates: (50, 150)
```

**Accuracy tips:**
- Zoom in on camera image for precise clicking
- Use distinct features (sharp corners, intersections)
- Measure actual coordinates if possible (jog laser to feature)
- More points = better accuracy

## Step 5: Align Your Design

Import or create your design and position it relative to camera image.

**Import design:**

1. Import SVG, DXF, or other design file into RayForge
2. Design appears on canvas

**Align to camera image:**

1. Camera image is now visible on canvas as an overlay or reference
2. Move/rotate/scale your design to align with features in camera image
3. Use camera alignment tools to snap design to alignment points

**Alignment methods:**

**Visual alignment:**
- Drag design to match camera image
- Rotate to align angles
- Scale if needed

**Point-to-point alignment:**
- Click design feature (e.g., corner)
- Click corresponding camera image feature
- RayForge aligns automatically

**Example:**
- You want to engrave a logo in center of pre-cut coaster
- Camera image shows coaster position
- Drag logo to center of coaster in camera image
- Logo is now positioned correctly for engraving

## Step 6: Verify Alignment

Check that alignment is correct before cutting.

**Frame the job:**

1. Use **Frame Job** feature to trace job boundary with laser
2. Watch laser trace outline - does it match your material?
3. Verify position is correct

**If alignment is off:**
- Re-check alignment points (did you mark them correctly?)
- Re-check real-world coordinates (did you enter correct values?)
- Re-align design to camera image

**If alignment is good:**
- Proceed to cutting

## Step 7: Run the Job

Execute your aligned job.

**Safety check:**
- Ensure material is still secured (hasn't moved)
- Ventilation running
- Fire extinguisher ready
- Emergency stop accessible

**Run job:**

1. Click **Start Job**
2. Monitor cutting closely
3. Verify first few cuts are landing in correct position
4. Emergency stop if alignment is wrong

**If first cuts are misaligned:**
- Stop job immediately
- Re-check alignment points and coordinates
- Repeat alignment process

## Advanced Camera Techniques

### Using Registration Marks

**What are registration marks?**
- Small reference marks (crosses, circles, etc.) cut or marked on material
- Used for precise multi-step alignment

**Workflow:**

1. **First pass:** Cut registration marks (small crosses at corners)
2. **Capture camera image** showing registration marks
3. **Mark alignment points** on registration marks
4. **Align second design** to registration marks
5. **Run second pass** - perfectly aligned to first pass

**Use cases:**
- Multi-color engraving (engrave different layers aligned to each other)
- Cut, then engrave (cut outline, then engrave detail inside)
- Front/back alignment (flip material, align to registration marks)

### Camera Calibration

Some camera setups require calibration to compensate for lens distortion.

**When to calibrate:**
- Wide-angle camera lenses (cause barrel distortion)
- Alignment accuracy is consistently off
- Different error at different positions in camera view

**Calibration process:**

1. Print or cut a calibration pattern (grid of points at known spacing)
2. Capture camera image of calibration pattern
3. Use RayForge's camera calibration tool to mark grid points
4. RayForge calculates distortion correction
5. Save calibration profile

**After calibration:** Camera alignment should be more accurate across entire view.

### Multiple Camera Positions

For large work areas, capture multiple camera images:

1. Capture image at position 1 (e.g., left side of bed)
2. Capture image at position 2 (e.g., right side of bed)
3. Mark alignment points on each image
4. Align design to stitched camera view

**Benefits:**
- Cover larger work area than single camera view
- Better accuracy across large bed

## Troubleshooting

### Camera Not Detected

**Problem:** RayForge doesn't see your camera.

**Solutions:**

1. Check camera is connected (USB or network)
2. Verify camera permissions (Linux: user in video group)
3. Check camera is not in use by another application
4. Restart RayForge
5. See [Camera Setup](../features/camera.md) for detailed troubleshooting

### Image Blurry or Out of Focus

**Problem:** Camera image is not clear enough to mark alignment points.

**Solutions:**

1. Adjust camera focus (physical focus ring or software setting)
2. Ensure camera is at correct height (focused at laser bed level)
3. Improve lighting in workspace
4. Clean camera lens

### Alignment Consistently Off by Same Amount

**Problem:** Alignment is wrong, but always off by same distance/direction.

**Cause:** Camera offset not configured correctly.

**Solutions:**

1. Check camera offset settings in RayForge
   - If camera is mounted on laser head, offset from laser focal point must be configured
2. Re-measure offset distance
3. Update settings and test again

### Alignment Off in Different Directions

**Problem:** Alignment error varies depending on position in camera view.

**Cause:** Camera lens distortion not corrected.

**Solutions:**

1. Perform camera calibration (see Advanced Techniques above)
2. Use alignment points closer to center of camera view (less distortion)
3. Use narrower camera lens (less distortion)

### Design Rotated Incorrectly

**Problem:** Design angle doesn't match material angle.

**Cause:** Not enough alignment points, or points not accurately marked.

**Solutions:**

1. Use at least 3 alignment points (4 recommended)
2. Spread alignment points far apart (better angle calculation)
3. Mark alignment points more precisely

### Material Moved After Capture

**Problem:** Material shifted after camera image was captured.

**Cause:** Material not secured properly.

**Solutions:**

1. Secure material better (tape, magnets, hold-downs)
2. Re-capture camera image after material moves
3. Frame job before running to verify position

## Tips for Best Results

1. **Use distinct alignment features:**
   - Sharp corners, not rounded edges
   - High-contrast features (dark on light background)
   - Pre-marked dots or crosses

2. **Spread alignment points apart:**
   - Use corners of work area
   - Wide spacing improves rotation accuracy
   - Avoid clustering points in one area

3. **Measure real-world coordinates accurately:**
   - Jog laser to feature to get exact coordinates
   - Use calipers to measure from known origin
   - Double-check coordinate entry

4. **Secure material firmly:**
   - Material must not move between capture and cutting
   - Use tape, magnets, vacuum table, or fixtures
   - Verify material is flat

5. **Good lighting is crucial:**
   - Even lighting across work area
   - Avoid shadows and glare
   - Consider adding supplemental lighting

6. **Frame before running:**
   - Always frame the job to verify alignment
   - Stop immediately if frame doesn't match material

## Related Pages

- [Camera Feature](../features/camera.md) - Camera setup and configuration
- [Coordinates and Origin](../concepts/coordinates-and-origin.md) - Understanding coordinate systems
- [Machine Setup](../machine/device-config.md) - Configure camera offset
- [Multi-Layer Workflow](../features/multi-layer.md) - Organizing multi-step jobs
