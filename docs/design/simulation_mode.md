# Simulation Mode - Design Specification

## Overview

The Simulation Mode provides a real-time visualization of laser operation execution as a toggle overlay within the 2D view. It displays operations with speed-based color mapping (heatmap) and power-based transparency, along with a laser head position indicator and playback controls.

## Purpose

The simulation feature serves multiple use cases:

1. **Understanding Execution Order**: Visualize the sequence of operations before sending to the laser
2. **Speed Visualization**: See operation speed through color heatmap (blue=slow, red=fast)
3. **Power Visualization**: See power levels through transparency (transparent=low, opaque=high)
4. **Material Test Validation**: Simulate the order, speed, and power
5. **Job Debugging**: Verify operations execute in the expected sequence
6. **Safety**: Catch potential issues before running the actual job

## Architecture

### View System

Simulation Mode is a **toggle overlay** within the 2D view, not a separate view mode:

```
View Modes (Radio Group)
├─ 2D View (F5) - Edit workpieces in 2D workspace
│   └─ Simulate Execution (F7) - Toggle overlay with visualization
└─ 3D View (F6) - View toolpath in 3D

Simulation is independent of view mode and works within 2D view.
```

**Key Design**:
- Simulation toggle is **separate** from view mode switching
- Users can **edit workpieces** while simulation is active
- **No context switching** required - stay in 2D view

### Component Structure

```
WorkSurface (Canvas Integration)
    └── PreviewOverlay (CanvasElement) [legacy name: "SimulationOverlay"]
        ├── OpsTimeline (Data Model)
        │   ├── Timeline steps with state tracking
        │   └── Speed range tracking
        └── Speed heatmap rendering

PreviewControls (GTK Overlay Widget) [legacy name: "SimulationControls"]
    ├── Play/Pause button
    ├── Scrubbing slider
    └── Progress/speed range labels

MainWindow
    └── _refresh_execution_preview_if_active() [monitors simulation state]
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

**File**: `rayforge/workbench/elements/simulation_overlay.py`

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

### 2. PreviewOverlay (Simulation Overlay)

**File**: `rayforge/workbench/elements/simulation_overlay.py`

**Note**: Class is named `PreviewOverlay` for legacy reasons, but it implements simulation mode visualization.

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

### 3. PreviewControls (Simulation Controls)

**File**: `rayforge/workbench/simulation_controls.py`

**Note**: Class is named `PreviewControls` for legacy reasons, but it implements simulation playback controls.

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
- **Auto-start**: Begins playing when simulation mode is enabled
- **Adaptive speed**: Completes full playback in 5 seconds
- **Frame rate**: 24 FPS

**Configuration**:
- `target_duration_sec`: Default 5.0 seconds for full playback

### 4. WorkSurface Integration

**File**: `rayforge/workbench/surface.py`

**Simulation Mode**:
- Flag: `_preview_mode` (boolean) - **Note**: Legacy name, actually controls "simulation mode"
- **Keeps workpiece interaction enabled** (selection, transformation) ✅
- Keeps zoom/pan gestures active
- Shows grid and axis as normal
- Simulation overlay renders on top of workpieces
- Hides geometry lines (magenta ops surfaces) during simulation

**Key Methods** (legacy names from when it was "preview mode"):
- `is_preview_mode()` - Returns simulation mode state
- `set_preview_mode(enabled, preview_overlay)` - Enables/disables simulation
  - Adds/removes simulation overlay
  - **Does NOT disable workpiece interaction** (users can edit while simulating)
  - Triggers redraw

## View Menu Integration

### Menu Structure

```
View
├─ Show Workpieces (h)
├─ Show Tabs (t)
├─ Show Camera Image (Alt+c)
├─ Show Travel Moves (Ctrl+Shift+t)
├─ [separator]
├─ 2D View (F5) - Radio
├─ 3D View (F6) - Radio
├─ [separator]
├─ Simulate Execution (F7) - Toggle (independent of view mode) ✅
├─ [separator]
├─ Top View (1) - Radio [enabled in 3D mode]
├─ Front View (2) - Radio [enabled in 3D mode]
├─ Isometric View (3) - Radio [enabled in 3D mode]
├─ [separator]
└─ Toggle Perspective (p) [enabled in 3D mode]
```

### Actions

**File**: `rayforge/actions.py`

**View Mode**:
- **Action**: `win.view_mode` (radio action with string parameter)
- **Values**: "2d", "3d" (only two modes)
- **Hotkeys**:
  - F5 → 2D View
  - F6 → 3D View

**Simulation Toggle**:
- **Action**: `win.simulate_mode` (stateful boolean toggle)
- **Values**: true/false
- **Hotkey**: F7 → Toggle simulation on/off

### Handlers

**File**: `rayforge/mainwindow.py`

**`on_view_mode_changed(action, value)`**:
- Switches between 2D and 3D views only
- No longer handles preview/simulation
- Simplified logic

**`on_simulate_mode_state_change(action, value)`**:
- New handler for simulation toggle
- Calls `_enter_simulate_mode()` when enabled
- Calls `_exit_simulate_mode()` when disabled
- **Works within 2D view** (does not change view mode)

**`_enter_simulate_mode()`**:
1. Creates simulation overlay (PreviewOverlay class) with work area size
2. Aggregates operations from all layers
3. Sets ops on simulation overlay
4. Enables simulation on WorkSurface (but keeps interaction enabled)
5. Creates and shows simulation controls (PreviewControls class)
6. **Auto-starts playback**

**`_exit_simulate_mode()`**:
1. Disables simulation on WorkSurface
2. Removes simulation controls from overlay
3. Cleans up simulation objects

**`_refresh_execution_preview_if_active()`**:
- Called when document changes (settings updates)
- Checks if simulation toggle is active
- Regenerates ops and updates overlay
- Resets playback to beginning
- Resumes playing if was playing
- Updates speed range label

## Auto-Refresh System

When settings change that affect generated ops:

1. **Trigger**: `Step.updated` signal sent
2. **Handler**: `MainWindow.on_doc_changed()`
3. **Check**: Is simulation mode active?
4. **Action**: Regenerate simulation with new ops
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

1. **Ensure in 2D View**: Press F5 if needed
2. **Toggle Simulation**: Press F7 or click simulate button in toolbar
3. **Auto-playback begins**: Operations animate automatically
4. **Observe**:
   - Speed changes through color
   - Power changes through transparency
   - Laser head movement
   - Execution order
5. **Interact**:
   - **Edit workpieces** while simulation plays (resize, move, select) ✅
   - Pause to examine specific point
   - Scrub slider to specific operation
   - Let loop for continuous observation
6. **Exit**: Press F7 again to toggle off simulation (stays in 2D view)


## Design Decisions

### Why Toggle Instead of View Mode?

**Original Design**: Simulation was a separate view mode (2D/3D/Execution Preview)
**New Design**: Simulation is a toggle within 2D view

**Advantages of toggle approach**:
- ✅ **No context switching** - Stay in 2D view
- ✅ **Edit while simulating** - Resize/move workpieces during playback
- ✅ **Simpler mental model** - Only 2 view modes (2D/3D)
- ✅ **Better UX** - Users can now edit workpieces while simulation is active

### Why Canvas Integration?

**Advantages over separate widget**:
- ✅ Inherits grid, zoom, pan from WorkSurface
- ✅ Consistent coordinate system
- ✅ Familiar navigation controls
- ✅ Allows workpiece interaction while simulating

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



## Toolbar Integration

**File**: `rayforge/toolbar.py`

The simulation toggle is accessible via a toolbar button for quick access:

**Button Properties**:
- **Type**: `Gtk.ToggleButton`
- **Icon**: `media-playback-start-symbolic` (play icon)
- **Tooltip**: "Toggle execution simulation"
- **Action**: `win.simulate_mode`
- **Position**: After 3D view button, before tabs visibility button

**Button Behavior**:
- Toggles on/off with visual state change
- Synced with menu item and F7 keyboard shortcut
- Active state shows simulation is running

## Related Files

- `rayforge/workbench/elements/simulation_overlay.py` - Timeline and rendering
- `rayforge/workbench/simulation_controls.py` - Playback UI
- `rayforge/workbench/simulation_widget.py` - Timeline and renderer utilities
- `rayforge/workbench/surface.py` - Canvas integration
- `rayforge/mainwindow.py` - Simulation toggle handlers and refresh
- `rayforge/actions.py` - Action registration and keyboard shortcuts
- `rayforge/main_menu.py` - Menu structure
- `rayforge/toolbar.py` - Toolbar button integration
