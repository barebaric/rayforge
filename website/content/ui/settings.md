# Settings & Preferences

Customize Rayforge to match your workflow and preferences.

## Accessing Settings

- **Menu**: Edit → Preferences
- **Keyboard**: ++ctrl+comma++

## General Settings

### Interface

- **Theme**: Choose between System, Light, or Dark theme
- **Language**: Select interface language (English, Portuguese, Spanish, German)
- **Window Behavior**: Restore window size and position on startup

### Units

- **Measurement Units**: Millimeters or Inches
- **Speed Units**: mm/min, mm/sec, or inches/min
- **Display Precision**: Number of decimal places

### Canvas

- **Grid**:
  - Show/hide grid
  - Grid spacing
  - Grid color and opacity

- **Rulers**:
  - Show/hide rulers
  - Ruler units (inherit from general units)

- **Background Color**: Canvas background color

### Snapping

- **Snap to Grid**: Enable/disable grid snapping
- **Snap Distance**: Distance threshold for snapping (pixels)
- **Snap to Objects**: Enable edge and center point snapping

## Machine Settings

Configure machine profiles and hardware. See [Machine Setup](../machine/index.md) for detailed information.

- **Machine Profiles**: Create and manage multiple machine configurations
- **Default Profile**: Select which profile loads on startup

## Performance

### Rendering

- **Hardware Acceleration**: Enable/disable GPU acceleration
- **Anti-aliasing**: Smooth line rendering (may impact performance)
- **Maximum Path Points**: Limit for complex path rendering

### G-code Generation

- **Arc Interpolation**: Enable arc (G2/G3) commands vs. line segments
- **Decimal Places**: Precision for coordinates in G-code
- **Optimize Path Order**: Reduce travel time by reordering paths

### Preview

- **3D Preview Quality**: Low, Medium, High
- **Auto-open Preview**: Automatically show 3D preview after G-code generation
- **Preview Update Frequency**: Real-time vs. manual update

## File Handling

### Import

- **Default DPI**: For raster images without embedded DPI information
- **SVG Import**:
  - Flatten layers
  - Convert text to paths
  - Import hidden layers

### Export

- **G-code File Encoding**: UTF-8, ASCII
- **Line Endings**: LF (Linux/Mac) or CRLF (Windows)
- **Add Comments**: Include operation information in G-code

### Autosave

- **Enable Autosave**: Automatically save project at intervals
- **Autosave Interval**: Time between autosaves (minutes)
- **Autosave Location**: Directory for autosave files

## Keyboard Shortcuts

Customize keyboard shortcuts for common actions. See [Keyboard Shortcuts](../reference/shortcuts.md) for the complete list.

- **Reset to Defaults**: Restore all shortcuts to default bindings
- **Import/Export**: Share shortcut configurations

## Advanced

### Logging

- **Log Level**: Debug, Info, Warning, Error
- **Log Location**: View and change log file directory
- **Enable Crash Reports**: Help improve Rayforge by sending anonymous crash reports

### Experimental Features

Enable features in development:

- **New Feature Flags**: Access cutting-edge features
- **Beta Testing**: Opt-in to test new functionality

!!! warning "Experimental Features"
    Experimental features may be unstable or change without notice. Use with caution in production environments.

### Developer Tools

- **Show Developer Menu**: Access debugging and profiling tools
- **Enable Debug Output**: Verbose console logging
- **Performance Metrics**: Display FPS and memory usage

## Resetting Settings

### Reset to Defaults

Restore all settings to default values:

1. Open Preferences
2. Click "Reset All Settings"
3. Confirm the action
4. Restart Rayforge

!!! caution
    This will reset all preferences, including machine profiles. Export your machine profiles first if you want to keep them.

### Reset Specific Sections

Reset only certain settings:

- **Reset Window Layout**: Restore default panel positions
- **Reset Keyboard Shortcuts**: Restore default key bindings
- **Reset Theme**: Return to system theme

---

**Next**: [Features Overview →](../features/index.md)
