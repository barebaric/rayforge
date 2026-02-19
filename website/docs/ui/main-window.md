# Main Window

The Rayforge main window is your primary workspace for creating and managing
laser jobs.

## Window Layout

![Main Window](/images/ss-main.png)

### 1. Menu Bar

Access all Rayforge functions through organized menus:

- **File**: Open, save, import, export, and recent files
- **Edit**: Undo, redo, copy, paste, preferences
- **View**: Zoom, grid, rulers, panels, and view modes
- **Operations**: Add, edit, and manage operations
- **Machine**: Connect, jog, home, start/stop jobs
- **Help**: Documentation, about, and support

### 2. Toolbar

Quick access to frequently used tools:

- **Selection tool**: Select and move objects
- **Pan tool**: Navigate the canvas
- **Zoom tool**: Zoom in/out on specific areas
- **Measurement tool**: Measure distances and angles
- **Alignment tools**: Align and distribute objects
- **WCS dropdown**: Select the active Work Coordinate System (G53-G59)

The WCS dropdown allows you to quickly switch between coordinate systems.
See [Work Coordinate Systems (WCS)](work-coordinate-systems) for more
information.

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

### 6. Control Panel

The Control Panel at the bottom of the window provides:

- **Jog Controls**: Manual machine movement and positioning
- **Machine Status**: Real-time position and connection state
- **Log View**: G-code communication and operation history
- **WCS Management**: Work coordinate system selection and zeroing

See [Control Panel](control-panel) for detailed information.

## Window Management

### Panels

Show/hide panels as needed:

- **Layers Panel**: View → Layers Panel (<kbd>ctrl+l</kbd>)
- **Properties Panel**: View → Properties Panel (<kbd>ctrl+i</kbd>)

### Full Screen Mode

Focus on your work with full screen:

- Enter: <kbd>f11</kbd> or View → Full Screen
- Exit: <kbd>f11</kbd> or <kbd>esc</kbd>

## Customization

Customize the interface in **Edit → Preferences**:

- **Theme**: Light, dark, or system
- **Units**: Millimeters or inches
- **Grid**: Show/hide and configure grid spacing
- **Rulers**: Show/hide rulers on canvas
- **Toolbar**: Customize visible buttons

---

**Related Pages:**

- [Work Coordinate Systems (WCS)](work-coordinate-systems) - Managing WCS
- [Canvas Tools](canvas-tools) - Tools for manipulating designs
- [Control Panel](control-panel) - Manual machine control, status, and logs
- [3D Preview](3d-preview) - Visualize toolpaths in 3D
