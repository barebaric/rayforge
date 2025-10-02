# Execution Preview - Design Specification

## Overview

The Execution Preview feature provides a real-time simulation of laser operation execution. It displays operations with speed-based color mapping (heatmap) and power-based transparency, along with a laser head position indicator and playback controls.

## Purpose

The preview feature serves multiple use cases:

1. **Understanding Execution Order**: Visualize the sequence of operations before sending to the laser
2. **Speed Visualization**: See operation speed through color heatmap (blue=slow, red=fast)
3. **Power Visualization**: See power levels through transparency (transparent=low, opaque=high)
4. **Material Test Validation**: Preview the order, speed, and power of test squares
5. **Job Debugging**: Verify operations execute in the expected sequence
6. **Safety**: Catch potential issues before running the actual job

## Architecture

### View Mode System

Execution Preview is one of three mutually exclusive view modes:

```
View Modes (Radio Group)
├─ 2D View (F5) - Edit workpieces in 2D workspace
├─ 3D View (F6) - Preview toolpath in 3D
└─ Execution Preview (F7) - Simulate execution with visualization
```

### Component Structure

```
WorkSurface (Canvas Integration)
    └── PreviewOverlay (CanvasElement)
        ├── OpsTimeline (Data Model)
        │   ├── Timeline steps with state tracking
        │   └── Speed range tracking
        └── Speed heatmap rendering

PreviewControls (GTK Overlay Widget)
    ├── Play/Pause button
    ├── Scrubbing slider
    └── Progress/speed range labels

MainWindow
    └── _refresh_execution_preview_if_active()
        └── Auto-regenerates on settings changes
```

### Data Flow

```
Document Operations
    ↓
OpsTimeline.set_ops(ops)
    ↓ (parse commands, track speed range)
Timeline Steps: [(cmd, state, start_pos), ...]
    ↓
User interaction (play/scrub)
    ↓
PreviewOverlay.draw(ctx)
    ├── Render ops with heatmap colors
    ├── Apply power transparency
    └── Draw laser head indicator
    ↓
Canvas redraws
```

## Components

### 1. OpsTimeline

**File**: `rayforge/workbench/elements/preview_overlay.py`

**Responsibility**: Converts operations into timeline of discrete steps with state tracking.

**Key Features**:
- Tracks current power and speed for each step
- Calculates min/max speed range across all operations
- Stores immutable state snapshots

**Timeline Structure**:
```python
steps: List[Tuple[Command, State, Tuple[float, float]]]
# Each tuple:
# - Command: Move command (LineTo, etc.)
# - State: Machine state (power, cut_speed)
# - Start position: (x, y) where command begins
```

**Speed Range Tracking**:
- Scans all steps during construction
- Records minimum and maximum speeds
- Used for heatmap normalization
- Available via `get_speed_range()`

### 2. PreviewOverlay

**File**: `rayforge/workbench/elements/preview_overlay.py`

**Responsibility**: Canvas element that renders operations with visualization features.

**Rendering Features**:

#### Speed Heatmap
Maps speed to color gradient:
- **Blue** (slowest) → **Cyan** → **Green** → **Yellow** → **Red** (fastest)
- 5-color gradient with smooth transitions
- Normalized to actual speed range of operations
- Function: `speed_to_heatmap_color(speed, min_speed, max_speed)`

#### Power Transparency
Maps power to alpha channel:
- 0% power → 10% opacity (alpha = 0.1)
- 100% power → 100% opacity (alpha = 1.0)
- Linear interpolation: `alpha = 0.1 + (power/100.0) * 0.9`
- Ensures low-power lines remain visible

#### Laser Head Indicator
Shows current laser position:
- Red crosshair (6mm lines in each direction)
- Circle outline (3mm radius)
- Center dot (0.5mm)
- Drawn at current step's end position
- Moves during playback

**Key Methods**:
- `set_ops(ops)` - Updates operations and rebuilds timeline
- `set_current_step(step)` - Sets playback position
- `get_step_count()` - Returns total steps
- `get_speed_range()` - Returns (min_speed, max_speed)
- `get_current_position()` - Returns laser head (x, y)

### 3. PreviewControls

**File**: `rayforge/workbench/preview_controls.py`

**Responsibility**: GTK overlay widget providing playback controls.

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ ▶️  ═══════●════════════                │
│           42 / 250                       │
│ Speed range: 100 - 5000 mm/min          │
└─────────────────────────────────────────┘
```

**Features**:
- **Play/Pause Button**: Toggle playback (icon changes)
- **Slider**: Scrub through operations (supports fractional steps)
- **Progress Label**: "current / total"
- **Speed Range Label**: "min - max mm/min"

**Playback Behavior**:
- **Auto-start**: Begins playing when preview mode entered
- **Adaptive speed**: Completes full playback in 5 seconds
- **Frame rate**: 24 FPS

**Configuration**:
- `target_duration_sec`: Default 5.0 seconds for full playback

### 4. WorkSurface Integration

**File**: `rayforge/workbench/surface.py`

**Preview Mode**:
- Flag: `_preview_mode` (boolean)
- Disables workpiece interaction (selection, transformation)
- Keeps zoom/pan gestures active
- Shows grid and axis as normal
- Hides geometry preview lines (magenta ops surfaces)

**Key Methods**:
- `is_preview_mode()` - Returns preview mode state
- `set_preview_mode(enabled, preview_overlay)` - Enables/disables preview
  - Unselects all when entering
  - Adds/removes preview overlay
  - Triggers redraw

## View Menu Integration

### Menu Structure

```
View
├─ 2D View (F5) - Radio
├─ 3D View (F6) - Radio
├─ Execution Preview (F7) - Radio
├─ [separator]
├─ Top View (1) - Radio [enabled in 3D mode]
├─ Front View (2) - Radio [enabled in 3D mode]
├─ Isometric View (3) - Radio [enabled in 3D mode]
├─ [separator]
└─ Toggle Perspective (p) [enabled in 3D mode]
```

### Actions

**File**: `rayforge/actions.py`

- **Action**: `win.view_mode` (radio action with string parameter)
- **Values**: "2d", "3d", "preview"
- **Hotkeys**:
  - F5 → 2D View
  - F6 → 3D View
  - F7 → Execution Preview

### Handlers

**File**: `rayforge/mainwindow.py`

**`on_view_mode_changed(action, value)`**:
- Switches between 2D/3D/Preview modes
- Calls `_enter_preview_mode()` when switching to preview
- Calls `_exit_preview_mode()` when leaving preview
- Updates action state

**`_enter_preview_mode()`**:
1. Creates PreviewOverlay with work area size
2. Aggregates operations from all layers
3. Sets ops on preview overlay
4. Enables preview mode on WorkSurface
5. Creates and shows PreviewControls
6. **Auto-starts playback**

**`_exit_preview_mode()`**:
1. Disables preview mode on WorkSurface
2. Removes preview controls from overlay
3. Cleans up preview objects

**`_refresh_execution_preview_if_active()`**:
- Called when document changes (settings updates)
- Checks if currently in preview mode
- Regenerates ops and updates overlay
- Resets playback to beginning
- Resumes playing if was playing
- Updates speed range label

## Auto-Refresh System

When settings change that affect generated ops:

1. **Trigger**: `Step.updated` signal sent
2. **Handler**: `MainWindow.on_doc_changed()`
3. **Check**: Is preview mode active?
4. **Action**: Regenerate preview with new ops
5. **Playback**: Reset to beginning, resume if was playing

## Visual Design

### Color Scheme

Five-segment linear gradient:
1. **0-25%**: Blue → Cyan
2. **25-50%**: Cyan → Green
3. **50-75%**: Green → Yellow
4. **75-100%**: Yellow → Red

Normalized to actual speed range of operations (not fixed values).

## Usage Workflow

### Basic Usage

1. **Enter Preview Mode**: Press F7
2. **Auto-playback begins**: Operations animate automatically
3. **Observe**:
   - Speed changes through color
   - Power changes through transparency
   - Laser head movement
   - Execution order
4. **Interact**:
   - Pause to examine specific point
   - Scrub slider to specific operation
   - Let loop for continuous observation
5. **Exit**: Press F5 (2D) or F6 (3D) to switch views


## Design Decisions

### Why Canvas Integration?

**Advantages over separate widget**:
- ✅ Inherits grid, zoom, pan from WorkSurface
- ✅ Consistent coordinate system
- ✅ Familiar navigation controls
- ✅ Seamless view switching

### Why Speed Heatmap?

**Alternatives considered**:
1. Uniform color (all operations same)
2. Power-based color
3. Layer-based color

**Chosen: Speed-based heatmap**

**Rationale**:
- Speed changes more dramatic than power in material tests
- Easy to spot speed variations
- Complements power transparency
- Intuitive: hot colors = fast, cool colors = slow



## Related Files

- `rayforge/workbench/elements/preview_overlay.py` - Timeline and rendering
- `rayforge/workbench/preview_controls.py` - Playback UI
- `rayforge/workbench/surface.py` - Canvas integration
- `rayforge/mainwindow.py` - Mode switching and refresh
- `rayforge/actions.py` - Action registration
- `rayforge/main_menu.py` - Menu structure
