# Screenshot Automation System

This directory contains tools and specifications for automated screenshot generation in RayForge documentation.

## Overview

The screenshot automation system allows documentation writers to embed machine-readable placeholders in Markdown files that describe:
- What the screenshot should show
- How to set up the application state
- What annotations to apply
- Where to save the result

A tool can then parse these placeholders and automatically generate screenshots by:
1. Launching RayForge
2. Executing setup actions via GUI automation
3. Capturing screenshots/videos
4. Applying annotations
5. Saving to the correct location

## Files

### [markdown_screenshot_spec.md](markdown_screenshot_spec.md)
Complete specification for screenshot placeholder format, including:
- Required and optional fields
- Setup action reference
- Annotation types
- Naming conventions
- Best practices


## Quick Start

### 1. Add Screenshot Placeholders to Documentation

In your Markdown file, add an HTML comment placeholder:

```markdown
<!-- SCREENSHOT: ui-settings-dialog
description: Settings dialog showing the Machine tab
filename: UI-Settings-MachineTab.png
-->

![Settings dialog](../images/UI-Settings-MachineTab.png)
```

For more complex screenshots with automation:

```markdown
<!-- SCREENSHOT
id: guide-simulation-mode-active
type: screenshot
description: Simulation mode showing speed heatmap
setup:
  - action: open_example
    name: material-test-grid
  - action: press_key
    key: F7
  - action: play_simulation
  - action: pause_at
    progress: 0.6
annotations:
  - type: callout
    text: "Speed heatmap"
filename: Guide-SimulationMode-Active.png
-->

![Simulation mode active](../images/Guide-SimulationMode-Active.png)
```

### 2. Validate Placeholders

Check that all placeholders are correctly formatted:

```bash
python screenshot_generator.py --dry-run --verbose
```

This will:
- Find all screenshot placeholders
- Validate required fields
- Check naming conventions
- Report any errors

### 3. Generate Screenshots

Generate all missing screenshots:

```bash
python screenshot_generator.py
```

Regenerate all screenshots:

```bash
python screenshot_generator.py --regenerate-all
```

Generate specific screenshots:

```bash
python screenshot_generator.py --filter "simulation"
```

## Placeholder Format

### Simple Format

For basic screenshots:

```markdown
<!-- SCREENSHOT: unique-id
description: What the screenshot shows
filename: OutputFileName.png
-->
```

### Detailed Format

For screenshots with automation and annotations:

```markdown
<!-- SCREENSHOT
id: unique-id
type: screenshot|video|gif
size: full-window|canvas-only|dialog|toolbar|custom
description: |
  Multi-line description of what should be visible
setup:
  - action: action_name
    parameter: value
annotations:
  - type: annotation_type
    property: value
filename: OutputFileName.png
alt: "Accessibility alt text"
-->
```

## Naming Convention

Follow this convention for consistency:

```
{Category}-{Topic}-{Element}.{ext}
```

**Categories:**
- `Guide` - Tutorial/guide screenshots
- `UI` - User interface reference
- `Ref` - Feature reference
- `Diag` - Diagrams
- `Example` - Example outputs

**Examples:**
- `Guide-MaterialTest-SettingsDialog.png`
- `UI-Toolbar-SimulationToggle.png`
- `Ref-SimulationMode-SpeedHeatmap.png`

## Setup Actions

Common setup actions for automation:

### File Operations
- `open_file` - Open a file
- `open_example` - Open example file
- `create_new` - Create new project

### Menu/UI Interaction
- `menu_click` - Click menu item
- `press_key` - Press keyboard key
- `click_element` - Click UI element
- `set_parameters` - Set form values

### Application State
- `activate_simulation` - Enter simulation mode
- `play_simulation` - Start playback
- `pause_simulation` - Pause playback
- `set_simulation_progress` - Set playback position

### Canvas
- `zoom_to_fit` - Zoom to fit content
- `set_zoom` - Set zoom level
- `pan_to` - Pan to coordinates

### Timing
- `wait` - Wait for duration
- `wait_for_element` - Wait for element to appear

See [markdown_screenshot_spec.md](markdown_screenshot_spec.md) for complete action reference.

## Annotations

Add visual annotations to highlight features:

### Callout
```yaml
- type: callout
  x: 100
  y: 150
  text: "Explanation text"
```

### Arrow
```yaml
- type: arrow
  from: [200, 300]
  to: [250, 320]
  text: "Label"
```

### Highlight
```yaml
- type: highlight
  element: element_id
  color: "#FF6B00"
  style: glow
```

See [markdown_screenshot_spec.md](markdown_screenshot_spec.md) for complete annotation reference.

## Integration with MCP

This system is designed to integrate with an MCP (Model Context Protocol) tool that can:

1. **Parse placeholders** from documentation files
2. **Control RayForge** via GUI automation (e.g., using pyautogui, playwright, or similar)
3. **Capture media** (screenshots, videos, GIFs)
4. **Apply annotations** using image processing libraries
5. **Save results** to the correct location

The current `screenshot_generator.py` is a proof-of-concept that handles parsing and validation. The actual screenshot generation logic needs to be implemented based on your automation framework of choice.

## Future Enhancements

Potential improvements to the system:

- [ ] Implement actual screenshot capture using GUI automation
- [ ] Add support for different themes (light/dark mode)
- [ ] Support for multiple resolutions/DPI settings
- [ ] Parallel screenshot generation for speed
- [ ] Screenshot diffing to detect visual regressions
- [ ] Integration with CI/CD to auto-update screenshots on releases
- [ ] Web-based preview tool for reviewing placeholders
- [ ] Template system for common screenshot types

## Contributing

When adding documentation:

1. **Always include placeholders** for screenshots, even if you can't generate them yet
2. **Write detailed descriptions** so others (or automation) can recreate the screenshot
3. **Follow naming conventions** for consistency
4. **Test placeholders** with `--dry-run` before committing
5. **Add annotations** to highlight important UI elements

## Questions?

See [markdown_screenshot_spec.md](markdown_screenshot_spec.md) for detailed specification or check [screenshot_examples.md](screenshot_examples.md) for usage examples.
