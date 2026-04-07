# Main Window

The Rayforge main window is your primary workspace for creating and managing
laser jobs.

## Window Layout

![Main Window](/screenshots/main-standard.png)

### 1. Menu Bar

Access all Rayforge functions through organized menus:

- **File**: Open, save, import, export, and recent files
- **Edit**: Undo, redo, copy, paste, preferences
- **View**: Zoom, grid, rulers, panels, and view modes
- **Object**: Add, edit, and manage operations
- **Machine**: Connect, jog, home, start/stop jobs
- **Help**: About, Donate, Save Debug Log

### 2. Toolbar

Quick access to frequently used controls:

- **Machine dropdown**: Select your machine, view connection status, and see
  ETA during jobs
- **WCS dropdown**: Select the active Work Coordinate System (G53-G59)
- **Simulation toggle**: Enable/disable job simulation mode
- **Focus laser**: Toggle laser focusing mode
- **Job controls**: Home, Frame, Send, Hold, and Cancel buttons

The machine dropdown shows your machine's connection status and current state
(e.g. Idle, Run) directly in the toolbar. During job execution, it also
displays an estimated time remaining.

The WCS dropdown allows you to quickly switch between coordinate systems.
See [Work Coordinate Systems](../general-info/coordinate-systems) for
more information.

Visibility toggles for workpieces, tabs, camera feed, travel moves, and
other elements have moved to overlay buttons on the canvas itself, so they
are always close at hand while you work.

### 3. Canvas

The main workspace where you:

- Import and arrange designs
- Preview toolpaths
- Position objects relative to machine origin
- Test frame boundaries

**Canvas Controls:**

- **Pan**: Middle-click drag or <kbd>space</kbd> + drag
- **Zoom**: Mouse wheel or <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Reset View**: <kbd>ctrl+0</kbd> or View → Reset Zoom

### 4. Layers Panel

Manage operations and layer assignments:

- View all operations in your project
- Assign operations to design elements
- Reorder operation execution
- Enable/disable individual operations
- Configure operation parameters

### 5. Properties Panel

Configure settings for selected objects or operations:

- Operation type (Contour, Raster, etc.)
- Power and speed settings
- Number of passes
- Advanced options (overscan, kerf, tabs)

### 6. Bottom Panel

The Bottom Panel provides tabs for the Console, G-code Viewer, and your
document assets (stock and sketches). Jog controls and WCS management are
always visible on the right side. The estimated job time is shown in the
layer list header above the layers panel.

See [Bottom Panel](bottom-panel) for detailed information.

## Window Management

### Panels

Show/hide panels as needed:

- **Bottom Panel**: View → Bottom Panel (<kbd>ctrl+l</kbd>)

### Full Screen Mode

Focus on your work with full screen:

- Enter: <kbd>f11</kbd> or View → Full Screen
- Exit: <kbd>f11</kbd> or <kbd>esc</kbd>

## Customization

Customize the interface in **Edit → Settings**:

- **Theme**: Light, dark, or system
- **Units**: Millimeters or inches
- **Grid**: Show/hide and configure grid spacing
- **Rulers**: Show/hide rulers on canvas

---

**Related Pages:**

- [Work Coordinate Systems](../general-info/coordinate-systems) - WCS
- [Canvas Tools](canvas-tools) - Tools for manipulating designs
- [Bottom Panel](bottom-panel) - Manual machine control, status, and logs
- [3D View](3d-preview) - Visualize toolpaths in 3D
