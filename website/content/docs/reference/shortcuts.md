# Keyboard Shortcuts

Complete reference of keyboard shortcuts in Rayforge.

!!! note "Platform Conventions"
    - **Linux/Windows:** `Ctrl` key
    - **macOS:** ` (Command)` key
    - Documentation uses `Ctrl` - macOS users substitute with ``

---

## File Operations

| Shortcut | Action      | Description             |
| -------- | ----------- | ----------------------- |
| `Ctrl+N` | New Sketch  | Create new sketch       |
| `Ctrl+I` | Import      | Open file import dialog |
| `Ctrl+E` | Export      | Export G-code           |
| `Ctrl+Q` | Quit        | Exit Rayforge           |
| `Ctrl+,` | Preferences | Open preferences dialog |

---

## Edit & Clipboard

| Shortcut            | Action           | Description                      |
| ------------------- | ---------------- | -------------------------------- |
| `Ctrl+Z`            | Undo             | Undo last action                 |
| `Ctrl+Y`            | Redo             | Redo last undone action          |
| `Ctrl+Shift+Z`      | Redo (alternate) | Alternative redo shortcut        |
| `Ctrl+X`            | Cut              | Cut selection to clipboard       |
| `Ctrl+C`            | Copy             | Copy selection to clipboard      |
| `Ctrl+V`            | Paste            | Paste from clipboard             |
| `Ctrl+A`            | Select All       | Select all items in active layer |
| `Ctrl+D`            | Duplicate        | Duplicate selected items         |
| `Delete`            | Remove           | Delete selected items            |
| `Ctrl+Shift+Delete` | Clear            | Clear all items from document    |

---

## View & Display

| Shortcut       | Action            | Description                    |
| -------------- | ----------------- | ------------------------------ |
| `H`            | Toggle Workpieces | Show/hide workpiece visibility |
| `T`            | Toggle Tabs       | Show/hide holding tabs         |
| `Alt+C`        | Toggle Camera     | Show/hide camera overlay       |
| `F12`          | 3D View           | Toggle 3D preview window       |
| `Ctrl+Shift+S` | Simulation Mode   | Toggle simulation mode         |

### 3D View Controls

| Shortcut | Action             | Description                                 |
| -------- | ------------------ | ------------------------------------------- |
| `1`      | Top View           | Switch to top-down view                     |
| `2`      | Front View         | Switch to front view                        |
| `7`      | Isometric View     | Switch to isometric view                    |
| `P`      | Toggle Perspective | Switch between perspective and orthographic |

---

## Layer & Organization

| Shortcut         | Action          | Description              |
| ---------------- | --------------- | ------------------------ |
| `Ctrl+G`         | Group           | Group selected items     |
| `Ctrl+U`         | Ungroup         | Ungroup selected group   |
| `Ctrl+Page Up`   | Move Layer Up   | Move layer up in stack   |
| `Ctrl+Page Down` | Move Layer Down | Move layer down in stack |
| `Alt+S`          | Add Stock       | Add stock/material item  |

---

## Tabs (Holding Tabs)

| Shortcut | Action                 | Description            |
| -------- | ---------------------- | ---------------------- |
| `Alt+T`  | Add Tabs (Equidistant) | Add evenly-spaced tabs |
| `T`      | Toggle Tab Visibility  | Show/hide tabs overlay |

---

## Alignment & Distribution

### Alignment

| Shortcut      | Action         | Description                                 |
| ------------- | -------------- | ------------------------------------------- |
| `Shift+Left`  | Align Left     | Align selected items to left                |
| `Shift+Right` | Align Right    | Align selected items to right               |
| `Shift+Up`    | Align Top      | Align selected items to top                 |
| `Shift+Down`  | Align Bottom   | Align selected items to bottom              |
| `Shift+Home`  | Align H-Center | Align selected items horizontally to center |
| `Shift+End`   | Align V-Center | Align selected items vertically to center   |

### Distribution

| Shortcut       | Action              | Description                          |
| -------------- | ------------------- | ------------------------------------ |
| `Ctrl+Shift+H` | Spread Horizontally | Distribute items evenly horizontally |
| `Ctrl+Shift+V` | Spread Vertically   | Distribute items evenly vertically   |

### Layout

| Shortcut | Action        | Description        |
| -------- | ------------- | ------------------ |
| `Alt+A`  | Pixel Perfect | Snap to pixel grid |

---

## Transform

| Shortcut  | Action          | Description                        |
| --------- | --------------- | ---------------------------------- |
| `Shift+H` | Flip Horizontal | Mirror selected items horizontally |
| `Shift+V` | Flip Vertical   | Mirror selected items vertically   |

---

## Machine Control

| Shortcut | Action           | Description                    |
| -------- | ---------------- | ------------------------------ |
| `Ctrl+J` | Jog Dialog       | Open manual jog control dialog |
| `Ctrl+<` | Machine Settings | Open machine settings dialog   |
| `F1`     | About            | Show about dialog              |

!!! note "Machine Operations"
    Machine control operations (Home, Frame, Send, etc.) currently don't have default shortcuts but can be accessed via toolbar buttons or menus.

---

## Canvas Navigation

### Mouse Controls

| Input                 | Action         | Description           |
| --------------------- | -------------- | --------------------- |
| **Left Click**        | Select         | Select item           |
| **Left Drag**         | Move           | Move selected items   |
| **Ctrl+Left Drag**    | Box Select     | Select multiple items |
| **Middle Click Drag** | Pan            | Pan the canvas        |
| **Scroll Wheel**      | Zoom           | Zoom in/out           |
| **Ctrl+Scroll**       | Precision Zoom | Finer zoom control    |

### Arrow Keys

| Shortcut      | Action      | Description                     |
| ------------- | ----------- | ------------------------------- |
| ` ` ` `       | Nudge       | Move selected items by 1 unit   |
| `Shift+Arrow` | Large Nudge | Move selected items by 10 units |

---

## Text Editor (G-code Editor)

When editing G-code or text fields:

| Shortcut | Action     | Description              |
| -------- | ---------- | ------------------------ |
| `Ctrl+Z` | Undo       | Undo text edit           |
| `Ctrl+Y` | Redo       | Redo text edit           |
| `Ctrl+A` | Select All | Select all text          |
| `Ctrl+X` | Cut        | Cut selected text        |
| `Ctrl+C` | Copy       | Copy selected text       |
| `Ctrl+V` | Paste      | Paste text               |
| `Ctrl+F` | Find       | Find text (if supported) |

---

## Quick Reference by Category

### Most Used (Top 10)

1. `Ctrl+Z` / `Ctrl+Y` - Undo/Redo
2. `Ctrl+C` / `Ctrl+V` - Copy/Paste
3. `Ctrl+D` - Duplicate
4. `Delete` - Remove
5. `Ctrl+A` - Select All
6. `Ctrl+N` - New Sketch
7. `Ctrl+I` - Import
8. `Ctrl+E` - Export
9. `H` - Toggle Workpieces
10. `F12` - 3D View

### View & Visualization

- `H` - Hide/show workpieces
- `T` - Hide/show tabs
- `Alt+C` - Toggle camera
- `F12` - 3D view
- `Ctrl+Shift+S` - Simulation mode
- `1`, `2`, `7` - 3D view presets
- `P` - Perspective toggle

### Edit & Transform

- `Ctrl+Z` / `Ctrl+Y` - Undo/Redo
- `Ctrl+X` / `Ctrl+C` / `Ctrl+V` - Cut/Copy/Paste
- `Ctrl+D` - Duplicate
- `Delete` - Remove
- `Shift+H` / `Shift+V` - Flip H/V
- `Ctrl+G` / `Ctrl+U` - Group/Ungroup

### Alignment

- `Shift+Arrow Keys` - Align to edges
- `Shift+Home` / `Shift+End` - Center align
- `Ctrl+Shift+H` / `Ctrl+Shift+V` - Distribute
- `Alt+A` - Pixel perfect

---

## Customizing Shortcuts

!!! note "Custom Shortcuts"
    Keyboard shortcuts are currently hardcoded. Custom shortcut configuration may be added in future versions.

**Current limitations:**

- Shortcuts cannot be changed without modifying source code
- No GUI for shortcut customization
- Some actions may not have shortcuts assigned

**Feature request:** If you need custom shortcuts, please open an issue on GitHub.

---

## Tips & Tricks

### Efficiency Tips

1. **Learn the edit shortcuts first** - `Ctrl+Z/Y/C/V/D` are used constantly
2. **Use single-key toggles** - `H`, `T`, `P` for quick view changes
3. **3D view shortcuts** - `1`, `2`, `7` for instant view switching
4. **Alignment shortcuts** - `Shift+Arrow` faster than clicking alignment buttons
5. **Simulation mode** - `Ctrl+Shift+S` to quickly check execution

### Workflow Shortcuts

**Quick edit cycle:**

```
1. New Sketch (Ctrl+N) or Import (Ctrl+I)
2. Arrange items (arrow keys, Shift+Arrow for alignment)
3. Duplicate parts (Ctrl+D)
4. Check in 3D (F12)
5. Simulate (Ctrl+Shift+S)
6. Export (Ctrl+E)
```

**Precision positioning:**

```
1. Select item
2. Arrow keys for 1mm nudges
3. Shift+Arrow for 10mm jumps
4. Shift+Home/End for centering
```

### Hidden Shortcuts

Some lesser-known shortcuts:

- `Ctrl+,` - Quick access to preferences (standard on macOS, works here too)
- `Ctrl+Shift+Z` - Alternative redo (for users who prefer this over Ctrl+Y)
- `Alt+A` - Pixel-perfect snapping for precise placement
- `Alt+T` - Quick equidistant tab placement

---

## Platform Differences

### Linux

- Uses standard `Ctrl` modifier
- All shortcuts work as documented
- GTK standard shortcuts apply in text fields

### Windows

- Uses standard `Ctrl` modifier
- Identical to Linux shortcuts
- Windows-specific keys (Win key) not used

### macOS

- `Ctrl` maps to ` (Command)`
- `Alt` maps to `% (Option)`
- Standard macOS conventions apply
- `Cmd+Q` to quit (instead of Ctrl+Q)

---

## Troubleshooting Shortcuts

### Shortcut Not Working

**Common issues:**

1. **Focus on wrong element** - Ensure canvas or main window has focus, not a text field
2. **Conflicting application** - Another app may be intercepting the shortcut
3. **Desktop environment shortcut** - System shortcuts may override (e.g., Alt+F4)
4. **Numpad vs number row** - Use number row for 3D view shortcuts, not numpad

**Solutions:**

- Click on the canvas to ensure it has focus
- Check system keyboard shortcuts for conflicts
- Try the menu action instead to verify functionality

### Modifier Keys

**If Ctrl doesn't work:**

- Verify Caps Lock is off (can interfere on some systems)
- Try Ctrl on both sides of keyboard
- Check keyboard layout settings

**If Alt doesn't work:**

- Some window managers capture Alt for window dragging
- Try disabling window manager Alt shortcuts
- Use menu items as alternative

---

## Shortcut Cheat Sheet

**Print this quick reference:**

| Category      | Shortcut     | Action           |
| ------------- | ------------ | ---------------- |
| **File**      | Ctrl+N       | New Sketch       |
|               | Ctrl+I       | Import           |
|               | Ctrl+E       | Export           |
|               | Ctrl+Q       | Quit             |
|               | Ctrl+,       | Preferences      |
| **Edit**      | Ctrl+Z       | Undo             |
|               | Ctrl+Y       | Redo             |
|               | Ctrl+C       | Copy             |
|               | Ctrl+V       | Paste            |
|               | Ctrl+D       | Duplicate        |
|               | Delete       | Remove           |
| **View**      | H            | Workpieces       |
|               | T            | Tabs             |
|               | Alt+C        | Camera           |
|               | F12          | 3D View          |
|               | 1/2/7        | View Presets     |
|               | P            | Perspective      |
| **Align**     | Shift+Left   | Align Left       |
|               | Shift+Right  | Align Right      |
|               | Shift+Up     | Align Top        |
|               | Shift+Down   | Align Bottom     |
|               | Shift+Home   | H-Center         |
|               | Shift+End    | V-Center         |
| **Simulate**  | Ctrl+Shift+S | Mode             |
| **Transform** | Shift+H      | Flip H           |
|               | Shift+V      | Flip V           |
|               | Ctrl+G       | Group            |
|               | Ctrl+U       | Ungroup          |
| **Machine**   | Ctrl+J       | Jog Dialog       |
|               | Ctrl+<       | Machine Settings |
|               | F1           | About            |

---

## Related Pages

- [Main Window](../ui/main-window.md) - UI overview
- [Canvas Tools](../ui/canvas-tools.md) - Canvas interaction
- [3D Preview](../ui/3d-preview.md) - 3D view controls
- [Simulation Mode](../features/simulation-mode.md) - Simulation features
