# Canvas Tools

The canvas provides a comprehensive set of tools for manipulating designs, measuring, and preparing your laser jobs.

## Selection Tool

Select, move, and transform objects on the canvas.

**Usage:**

- Click to select single object
- <kbd>ctrl</kbd> + click to select multiple objects
- Drag to create selection box
- Click and drag selected objects to move

**Transform Handles:**

- **Corner handles**: Scale proportionally (hold <kbd>shift</kbd> to scale non-proportionally)
- **Edge handles**: Scale in one direction
- **Rotation handle**: Rotate around center point

**Keyboard Shortcuts:**

- <kbd>arrow-keys</kbd>: Move selected objects by 1 unit
- <kbd>shift+arrow-keys</kbd>: Move by 10 units
- <kbd>ctrl+d</kbd>: Duplicate selection

## Pan Tool

Navigate around the canvas without accidentally moving objects.

**Usage:**

- Click and drag to pan
- Alternatively, hold <kbd>space</kbd> with any tool active and drag

## Zoom Tool

Zoom in on specific areas of your design.

**Usage:**

- Click to zoom in at point
- <kbd>alt</kbd> + click to zoom out
- Click and drag to zoom to specific area

**Shortcuts:**

- <kbd>ctrl+"+"</kbd>: Zoom in
- <kbd>ctrl+"-"</kbd>: Zoom out
- <kbd>ctrl+0</kbd>: Reset zoom (fit to window)
- Mouse wheel: Zoom in/out at cursor position

## Measurement Tool

Measure distances and angles on the canvas.

**Usage:**

- Click starting point
- Click ending point to complete measurement
- Measurement displays in current units (mm or inches)

**Features:**

- Distance between two points
- Angle relative to horizontal
- Real-time preview while measuring

## Alignment Tools

Align and distribute multiple objects.

**Alignment Options:**

- Align left edges
- Align right edges
- Align top edges
- Align bottom edges
- Align horizontal centers
- Align vertical centers

**Distribution Options:**

- Distribute horizontally
- Distribute vertically
- Equal spacing

**Usage:**

1. Select multiple objects
2. Choose alignment/distribution option from toolbar or Edit menu
3. Objects align/distribute immediately

## Grid and Snapping

Assist precise positioning with grid and snapping.

**Grid:**

- Toggle: View → Show Grid (<kbd>ctrl+g</kbd>)
- Configure spacing in Preferences
- Visual guide only (optional snapping)

**Snapping:**

- **Snap to Grid**: Align objects to grid points
- **Snap to Objects**: Align to edges of other objects
- **Snap to Origin**: Align to machine origin (0,0)

Toggle snapping: View → Snap (<kbd>ctrl+shift+g</kbd>)

## Object Transformation

Transform objects numerically for precision.

**Accessible via Properties Panel:**

- **Position (X, Y)**: Exact coordinates
- **Size (W, H)**: Exact dimensions
- **Rotation**: Degrees from horizontal
- **Scale**: Percentage of original size

## Boolean Operations

Combine or subtract shapes:

- **Union**: Merge overlapping shapes
- **Difference**: Subtract one shape from another
- **Intersection**: Keep only overlapping areas
- **Exclusion**: Remove overlapping areas

**Usage:**

1. Select two or more vector objects
2. Choose boolean operation from Edit menu
3. Result replaces selected objects

## Tips for Efficient Canvas Use

1. **Use keyboard shortcuts**: Much faster than toolbar clicks
2. **Master pan and zoom**: Essential for large or detailed designs
3. **Snap to grid**: Speeds up alignment for rectangular layouts
4. **Measure first**: Verify dimensions before generating G-code
5. **Group related objects**: Easier to move and organize (<kbd>ctrl+g</kbd> to group)

---

**Next**: [3D Preview →](3d-preview)
