# Screenshot Placeholder Specification

**Version:** 1.0
**Date:** October 3, 2025
**Purpose:** Define a standardized format for screenshot placeholders in RayForge documentation that enables automated screenshot generation via MCP tools.

## Overview

Screenshot placeholders are HTML comments embedded in Markdown files that describe:
1. What the screenshot should show
2. How to set up the application state to capture it
3. What annotations (arrows, callouts, highlights) to apply
4. Where to save the resulting image

These placeholders serve as both documentation for manual screenshot creation and machine-readable instructions for automated screenshot generation.

## Basic Format

```markdown
<!-- SCREENSHOT
id: unique-identifier
type: screenshot|video|gif
size: full-window|canvas-only|dialog|toolbar|overlay-widget
description: |
  Multi-line description of what should be visible in the screenshot.
  Include important details about UI state, content, and context.
setup:
  - action: action_name
    parameter: value
  - action: another_action
    parameter: value
annotations:
  - type: annotation_type
    property: value
filename: OutputFileName.png
alt: "Alt text for accessibility"
-->

![Alt text](../images/OutputFileName.png)
```

## Required Fields

### `id` (string)
Unique identifier for this screenshot within the documentation set. Use kebab-case.

**Format:** `{section}-{feature}-{variant}`

**Examples:**
- `ui-settings-machine-profile`
- `guide-simulation-mode-heatmap`
- `feature-material-test-grid-presets`

### `type` (enum)
Type of media to capture.

**Values:**
- `screenshot` - Static image capture
- `video` - Short video clip (MP4)
- `gif` - Animated GIF for simple interactions

### `description` (multi-line string)
Detailed description of what the screenshot should contain. Use the `|` YAML literal block scalar for multi-line content.

**Should include:**
- UI elements that must be visible
- Application state (e.g., "with simulation at 60% progress")
- Content details (e.g., "showing a material test grid with 5x5 cells")
- Important visual characteristics

### `filename` (string)
Output filename following the naming convention.

**Convention:** `{Category}-{Topic}-{Element}{Number}.{ext}`

**Categories:**
- `Guide` - Tutorial/guide screenshots
- `UI` - User interface reference
- `Ref` - Feature reference
- `Diag` - Diagrams and illustrations
- `Example` - Example outputs

**Examples:**
- `Guide-MaterialTest-SettingsDialog.png`
- `UI-Toolbar-SimulationToggle.png`
- `Ref-SimulationMode-SpeedHeatmap.png`

### `alt` (string)
Accessibility alt text for the image. Should be concise but descriptive.

## Optional Fields

### `size` (enum)
Capture region specification.

**Values:**
- `full-window` - Entire application window (default)
- `canvas-only` - Just the canvas/workspace area
- `dialog` - Active dialog window
- `toolbar` - Toolbar area only
- `overlay-widget` - Specific overlay widget
- `custom` - Custom region (requires `region` parameter)

### `setup` (array of actions)
Ordered list of actions to set up the application state before capture.

Each action is an object with:
- `action` (required): Action type
- Additional parameters specific to the action

See "Setup Actions Reference" section below.

### `annotations` (array of annotation objects)
Visual annotations to add to the screenshot after capture.

See "Annotations Reference" section below.

### `region` (object)
Custom capture region when `size: custom`.

**Format:**
```yaml
region:
  x: 100      # X coordinate from top-left
  y: 200      # Y coordinate from top-left
  width: 800  # Width in pixels
  height: 600 # Height in pixels
```

### `wait_for` (object)
Condition to wait for before capturing.

**Examples:**
```yaml
wait_for:
  element: "#simulation-controls"
  visible: true
  timeout: 5000  # milliseconds
```

```yaml
wait_for:
  state: simulation_playing
  timeout: 3000
```

## Setup Actions Reference

### File Operations

#### `open_file`
Open a file from the filesystem.

```yaml
- action: open_file
  path: /path/to/file.svg
```

#### `open_example`
Open a file from the examples directory.

```yaml
- action: open_example
  name: material-test-grid
```

#### `create_new`
Create a new blank project.

```yaml
- action: create_new
  width: 500
  height: 500
```

### Menu Operations

#### `menu_click`
Click a menu item by path.

```yaml
- action: menu_click
  path: Tools > Material Test Grid
```

```yaml
- action: menu_click
  path: View > Simulation Mode
  shortcut: F7  # Optional: document the shortcut
```

### Keyboard/Mouse Input

#### `press_key`
Press a keyboard key or key combination.

```yaml
- action: press_key
  key: F7
```

```yaml
- action: press_key
  key: Ctrl+G
```

#### `click_element`
Click a UI element by selector or ID.

```yaml
- action: click_element
  selector: "#simulation-toggle"
```

```yaml
- action: click_element
  text: "Generate Grid"
```

### Application State

#### `set_parameters`
Set parameters in the active dialog or settings.

```yaml
- action: set_parameters
  speed_min: 1000
  speed_max: 10000
  power_min: 10
  power_max: 100
  grid_rows: 5
  grid_cols: 5
```

#### `activate_simulation`
Enter simulation mode.

```yaml
- action: activate_simulation
```

#### `toggle_simulation`
Toggle simulation mode on/off.

```yaml
- action: toggle_simulation
```

#### `play_simulation`
Start simulation playback.

```yaml
- action: play_simulation
```

#### `pause_simulation`
Pause simulation playback.

```yaml
- action: pause_simulation
```

#### `set_simulation_progress`
Set simulation progress to specific position.

```yaml
- action: set_simulation_progress
  percent: 60
```

```yaml
- action: set_simulation_progress
  seconds: 5.5
```

### Canvas Operations

#### `zoom_to_fit`
Zoom the canvas to fit all content.

```yaml
- action: zoom_to_fit
```

#### `set_zoom`
Set specific zoom level.

```yaml
- action: set_zoom
  level: 1.5  # 150%
```

#### `pan_to`
Pan canvas to specific coordinates.

```yaml
- action: pan_to
  x: 100
  y: 200
```

### Timing

#### `wait`
Wait for a specific duration.

```yaml
- action: wait
  seconds: 0.5
```

```yaml
- action: wait
  milliseconds: 250
```

#### `wait_for_element`
Wait for an element to appear.

```yaml
- action: wait_for_element
  selector: "#simulation-controls"
  timeout: 5000
```

### Capture

#### `capture`
Take the screenshot (usually the final action).

```yaml
- action: capture
  region: main_window
```

```yaml
- action: capture
  region: dialog
  delay: 100  # ms delay before capture
```

## Annotations Reference

Annotations are visual elements added to screenshots after capture to highlight or explain specific areas.

### Callout

Text callout with leader line.

```yaml
- type: callout
  x: 100          # Anchor point X
  y: 150          # Anchor point Y
  text: "Speed heatmap: Blue=slow, Red=fast"
  position: top-right  # Optional: callout box position relative to anchor
  color: "#FF6B00"     # Optional: custom color
```

**Position values:** `top-left`, `top-right`, `bottom-left`, `bottom-right`, `auto` (default)

### Arrow

Directional arrow between two points.

```yaml
- type: arrow
  from: [200, 300]  # [x, y]
  to: [250, 320]    # [x, y]
  text: "Laser head position"  # Optional label
  color: "#FF6B00"              # Optional
  width: 3                      # Optional: line width in pixels
```

### Highlight

Highlight a UI element with a border or background.

```yaml
- type: highlight
  element: simulation_toggle_button  # Element ID
  color: "#FF6B00"
  style: border  # border | background | glow
  width: 3       # Border width for border style
```

```yaml
- type: highlight
  rect: [100, 150, 200, 50]  # [x, y, width, height]
  color: "#FF6B00"
  style: glow
  opacity: 0.3
```

### Box

Draw a rectangle around an area.

```yaml
- type: box
  x: 100
  y: 150
  width: 200
  height: 100
  color: "#FF6B00"
  style: solid  # solid | dashed | dotted
  width: 2      # Line width
  fill: false   # Optional: fill the box
```

### Text

Add standalone text label.

```yaml
- type: text
  x: 300
  y: 200
  text: "Selected layer"
  size: 14           # Font size
  color: "#FFFFFF"   # Text color
  background: "#000000"  # Optional background
  opacity: 0.8       # Background opacity
```

### Circle

Draw a circle or ellipse.

```yaml
- type: circle
  x: 250       # Center X
  y: 300       # Center Y
  radius: 30
  color: "#FF6B00"
  style: solid
  width: 2
  fill: false
```

### Blur

Blur a region (for privacy/focus).

```yaml
- type: blur
  rect: [100, 150, 200, 50]  # [x, y, width, height]
  strength: 10  # Blur radius
```

## Simplified Format

For simple screenshots without complex setup or annotations, use the short form:

```markdown
<!-- SCREENSHOT: unique-id
description: Brief description of the screenshot
filename: OutputFileName.png
-->

![Alt text](../images/OutputFileName.png)
```

This is equivalent to:

```markdown
<!-- SCREENSHOT
id: unique-id
type: screenshot
size: full-window
description: Brief description of the screenshot
filename: OutputFileName.png
alt: "Alt text"
-->

![Alt text](../images/OutputFileName.png)
```

## Complete Examples

### Example 1: Basic Dialog Screenshot

```markdown
<!-- SCREENSHOT: ui-settings-machine-profile
description: Machine settings dialog showing the Profiles tab with a diode laser configuration selected
filename: UI-Settings-MachineProfile.png
alt: "Machine profile settings dialog"
-->

![Machine profile settings](../images/UI-Settings-MachineProfile.png)
```

### Example 2: Complex Feature Screenshot

```markdown
<!-- SCREENSHOT
id: guide-simulation-mode-speed-heatmap
type: screenshot
size: full-window
description: |
  Simulation mode active in 2D view showing speed heatmap visualization.
  The canvas displays a material test grid (5x5) with color gradient from
  blue (slow speeds at 1000 mm/min) to red (fast speeds at 10000 mm/min).
  The laser head indicator is visible at approximately 60% progress through
  the job. The playback controls overlay shows the scrubbing slider at 60%
  and displays the speed range "1000-10000 mm/min".
setup:
  - action: create_new
    width: 300
    height: 300
  - action: menu_click
    path: Tools > Material Test Grid
  - action: set_parameters
    test_type: engrave
    speed_min: 1000
    speed_max: 10000
    power_min: 10
    power_max: 100
    grid_rows: 5
    grid_cols: 5
    cell_size: 20
    spacing: 5
  - action: click_element
    text: "Generate"
  - action: wait
    seconds: 0.5
  - action: press_key
    key: F7
  - action: wait_for_element
    selector: "#simulation-controls"
    timeout: 3000
  - action: play_simulation
  - action: wait
    seconds: 2
  - action: set_simulation_progress
    percent: 60
  - action: pause_simulation
  - action: wait
    milliseconds: 200
  - action: zoom_to_fit
  - action: capture
    region: main_window
annotations:
  - type: callout
    x: 150
    y: 100
    text: "Speed heatmap: Blue=slow (1000 mm/min), Red=fast (10000 mm/min)"
    position: top-right
  - type: arrow
    from: [250, 320]
    to: [280, 340]
    text: "Laser head position (60% complete)"
    color: "#FFFFFF"
  - type: highlight
    element: simulation_playback_controls
    color: "#FF6B00"
    style: glow
    opacity: 0.5
filename: Guide-SimulationMode-SpeedHeatmap.png
alt: "Simulation mode displaying speed heatmap with laser head indicator at 60% progress"
-->

![Simulation mode speed heatmap](../images/Guide-SimulationMode-SpeedHeatmap.png)
```

### Example 3: Video Capture

```markdown
<!-- SCREENSHOT
id: guide-simulation-playback-scrubbing
type: video
size: full-window
description: |
  Video showing simulation mode playback with user scrubbing through the
  timeline. Start with simulation paused at 0%, then drag the scrubber
  from 0% to 100% over 3 seconds, showing the laser head moving through
  the material test grid and the heatmap being rendered progressively.
setup:
  - action: open_example
    name: material-test-grid-5x5
  - action: activate_simulation
  - action: set_simulation_progress
    percent: 0
  - action: capture
    region: main_window
    duration: 5
    actions:
      - at: 1.0
        action: drag_slider
        element: simulation_scrubber
        from: 0
        to: 100
        duration: 3
filename: Guide-SimulationMode-Scrubbing.mp4
alt: "Video demonstration of simulation timeline scrubbing"
-->

![Simulation scrubbing demonstration](../images/Guide-SimulationMode-Scrubbing.mp4)
```

### Example 4: GIF Animation

```markdown
<!-- SCREENSHOT
id: ui-toolbar-simulation-toggle
type: gif
size: toolbar
description: |
  Animated GIF showing the simulation mode toggle button being clicked.
  Start with simulation off, cursor moves to button, clicks it, button
  activates and simulation overlay appears in the background.
setup:
  - action: open_example
    name: simple-square
  - action: zoom_to_fit
  - action: capture
    region: toolbar
    duration: 2
    actions:
      - at: 0.5
        action: move_cursor
        to_element: simulation_toggle
      - at: 1.0
        action: click_element
        element: simulation_toggle
filename: UI-Toolbar-SimulationToggle.gif
alt: "Animated demonstration of simulation mode toggle"
-->

![Simulation toggle animation](../images/UI-Toolbar-SimulationToggle.gif)
```

## MCP Tool Integration

An MCP tool for automated screenshot generation should:

1. **Parse Documentation Files**
   - Scan all `.md` files in `docs-site/`
   - Extract `<!-- SCREENSHOT ... -->` blocks
   - Parse YAML content within comments

2. **Validate Placeholders**
   - Check required fields are present
   - Validate action types and parameters
   - Verify filename follows naming convention
   - Check that image path matches filename

3. **Execute Setup Actions**
   - Launch RayForge application
   - Execute actions sequentially
   - Handle errors and timeouts gracefully
   - Verify expected state is reached

4. **Capture Media**
   - Take screenshot/video/GIF based on type
   - Use specified region or element
   - Wait for animations to complete

5. **Apply Annotations**
   - Add callouts, arrows, highlights
   - Use consistent styling
   - Ensure annotations are visible and clear

6. **Save and Verify**
   - Save to specified filename in `docs-site/images/`
   - Optimize file size (compress PNG, optimize GIF)
   - Verify image was created successfully
   - Update placeholder status if tracking

7. **Report Results**
   - List successfully generated screenshots
   - Report failures with error messages
   - Provide statistics (total, success, failed, skipped)

## File Organization

Screenshots should be organized in `docs-site/images/` following this structure:

```
docs-site/images/
├── Guide-*.png          # Guide/tutorial screenshots
├── UI-*.png             # User interface reference screenshots
├── Ref-*.png            # Feature reference screenshots
├── Diag-*.svg           # Diagrams (prefer SVG for diagrams)
├── Example-*.png        # Example outputs
└── *.mp4, *.gif         # Videos and animations
```

## Best Practices

### Writing Descriptions
- Be specific about UI state and content
- Include important details (percentages, values, selections)
- Describe visual characteristics that must be present
- Mention if specific colors/gradients should be visible

### Setup Actions
- Keep actions minimal but complete
- Use waits to ensure UI is ready
- Prefer menu paths over direct element clicks (more stable)
- Include zoom/pan actions for consistency

### Annotations
- Use sparingly - don't clutter the screenshot
- Choose high-contrast colors (consider dark/light themes)
- Keep callout text concise
- Position annotations to avoid covering important UI elements

### Filenames
- Follow the naming convention consistently
- Use descriptive names that indicate content
- Keep names under 60 characters
- Use PascalCase for multi-word components

### Accessibility
- Always provide meaningful alt text
- Describe what's shown, not just "screenshot of X"
- Include important information visible in the image

## Version History

- **1.0** (2025-10-03): Initial specification
