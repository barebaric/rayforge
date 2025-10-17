# Drag-and-Drop File Import Feature

**Status:** Proposed
**Date:** 2025-10-16

## Overview

Add drag-and-drop functionality to the Rayforge canvas, allowing users to import files by dragging them from their file manager and dropping them onto the WorkSurface. Imported items will be centered at the drop location on the canvas. This feature is already documented in the user documentation but not yet implemented.

## Motivation

### Current State
- Users can only import files via **File → Open** (Ctrl+I) menu option
- Imported files always appear at the default position (0,0)
- Documentation at `website/content/docs/files/importing.md` describes drag-and-drop as "Method 2" for importing files
- Feature gap between documentation and implementation

### Benefits
1. **Improved UX**: Faster, more intuitive file import workflow
2. **Spatial Control**: Users can place imported items exactly where they want them
3. **Industry Standard**: Common pattern in creative/design applications (Inkscape, Illustrator, Figma)
4. **Documentation Alignment**: Matches existing user-facing documentation
5. **Workflow Efficiency**: Reduces clicks and eliminates need to reposition after import

## Requirements

### Functional Requirements

1. **File Drop Acceptance**
   - Accept single or multiple files dropped onto the canvas (WorkSurface)
   - Support all currently supported file formats:
     - Vector: SVG, DXF, PDF
     - Raster: PNG, JPG, JPEG, BMP

2. **Import Behavior**
   - Maintain identical import logic to existing File → Open workflow
   - For SVG files: Show the import options dialog (Direct Import vs. Trace Bitmap)
   - For raster files: Automatically trace using default TraceConfig
   - For unsupported files: Show appropriate error notification

3. **Positioning**
   - Import files centered at the drop location (x, y coordinates on canvas)
   - Convert widget coordinates to canvas world coordinates (mm)
   - Handle multi-file drops by positioning each file at the same drop point

4. **Visual Feedback**
   - Show temporary overlay message "Drop files to import" when dragging files over canvas
   - Remove overlay when drag leaves canvas or drop occurs
   - Use existing toast notification system for import success/errors

5. **Multi-File Handling**
   - Process multiple dropped files sequentially
   - Import all files into the current document/active layer
   - All files positioned at the same drop location (stacked)
   - When dropping multiple raster files, show a single configuration dialog
   - Apply the same tracing settings to all raster files in the batch

6. **Clipboard Paste Support**
   - Detect when image data is on the clipboard
   - Import image from clipboard via Ctrl+V or Edit → Paste
   - Position pasted image at canvas center (or last cursor position)
   - Support common clipboard image formats (PNG, JPEG, BMP)
   - Apply same tracing workflow as file imports

### Non-Functional Requirements

1. **Code Reuse**: Leverage existing `import_handler` module - no duplication
2. **Error Handling**: Gracefully handle invalid files, missing permissions, etc.
3. **Performance**: Non-blocking for large files (use existing async import infrastructure)
4. **GTK 4 Compliance**: Use modern `Gtk.DropTarget` API

## Technical Design

### Architecture

```
┌─────────────────────────────────────────┐
│         WorkSurface (Canvas)            │
│  ┌───────────────────────────────────┐  │
│  │   _setup_file_drop_target()       │  │
│  │   - Creates Gtk.DropTarget        │  │
│  │   - Registers for Gio.File drops  │  │
│  │   - Connects to "drop" signal     │  │
│  └───────────────────────────────────┘  │
│                 │                        │
│                 ▼                        │
│  ┌───────────────────────────────────┐  │
│  │   _on_file_dropped()              │  │
│  │   - Extracts file paths           │  │
│  │   - Converts widget → world coords│  │
│  │   - Determines MIME types         │  │
│  │   - Passes drop position to       │  │
│  │     import handler                │  │
│  └───────────────────────────────────┘  │
│                 │                        │
└─────────────────┼────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   MainWindow / DocEditor                │
│  ┌───────────────────────────────────┐  │
│  │  Import delegation methods        │  │
│  │  - Handle SVG import dialog       │  │
│  │  - Call load_file_from_path()     │  │
│  │  - Position workpiece at drop loc │  │
│  └───────────────────────────────────┘  │
│                 │                        │
└─────────────────┼────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│   doceditor/ui/import_handler.py        │
│  ┌───────────────────────────────────┐  │
│  │  Existing import infrastructure   │  │
│  │  (reused as-is)                   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### GTK 4 Drag-and-Drop API

Rayforge uses GTK 4, which provides a controller-based drag-and-drop API:

```python
# Create drop target controller
drop_target = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
drop_target.set_gtypes([Gio.File, Gdk.FileList])  # Single or multiple files

# Connect to drop signal
drop_target.connect("drop", self._on_file_dropped)

# Visual feedback during drag
drop_target.connect("enter", self._on_drag_enter)
drop_target.connect("leave", self._on_drag_leave)

# Attach to canvas widget
self.add_controller(drop_target)
```

**Key GTK 4 Concepts:**
- `Gtk.DropTarget`: Controller attached to a widget to receive drops
- `Gio.File`: Represents a file URI
- `Gdk.FileList`: Represents multiple dropped files
- `drop` signal: Triggered when files are dropped, provides (x, y) coordinates
- Support for file content types via `Gio.File` and `Gdk.FileList`

### Implementation Location

**Primary File:** `rayforge/workbench/surface.py` (WorkSurface class)

**Methods to Add:**

1. **`_setup_file_drop_target(self)`**
   - Called from `__init__()` after other UI setup
   - Creates and configures `Gtk.DropTarget` for file drops
   - Registers signal handlers

2. **`_on_file_dropped(self, drop_target, value, x, y) -> bool`**
   - Main drop handler
   - Extracts file list from `value` (handles `Gio.File` and `Gdk.FileList`)
   - Converts widget coordinates (x, y) to world coordinates using `widget_to_world_coords()`
   - Queries MIME type for each file
   - For multiple raster files: Shows single tracing configuration dialog
   - Delegates to MainWindow with drop position
   - Returns `True` on success, `False` on failure

3. **`_on_drag_enter(self, drop_target, x, y) -> Gdk.DragAction`**
   - Shows overlay message "Drop files to import" when drag enters canvas
   - Returns `Gdk.DragAction.COPY` to indicate acceptance

4. **`_on_drag_leave(self, drop_target)`**
   - Removes overlay message when drag leaves canvas

5. **`_show_drop_overlay(self)`** and **`_hide_drop_overlay(self)`**
   - Helper methods to manage the overlay message display
   - Overlay should be semi-transparent, centered on canvas
   - Use existing canvas overlay infrastructure if available

**Supporting Changes in MainWindow:**

Need to add clipboard monitoring and paste handler:

1. **Clipboard Paste Handler:**
```python
def on_paste_action(self, action, param):
    """
    Handle paste action - check if clipboard contains image data.
    If so, import it. Otherwise, delegate to existing paste logic.
    """
    clipboard = self.get_clipboard()
    # Check for image data on clipboard
    # If image: extract, save to temp file, import
    # Otherwise: existing workpiece paste logic
```

2. **Import with position parameter:**

```python
def import_file_at_position(
    self,
    file_path: Path,
    mime_type: str,
    position_mm: Optional[Tuple[float, float]] = None
):
    """
    Import a file and optionally position it at specified coordinates.

    Args:
        file_path: Path to file to import
        mime_type: MIME type of the file
        position_mm: Optional (x, y) tuple in world coordinates (mm)
    """
    # Existing import logic...
    # After import, if position_mm provided:
    #   - Get the imported workpiece
    #   - Set its position to center at position_mm
```

### Coordinate Transformation

The canvas has a method `widget_to_world_coords(widget_x, widget_y)` that converts pixel coordinates to canvas world coordinates (mm):

```python
def _on_file_dropped(self, drop_target, value, x, y) -> bool:
    # x, y are widget coordinates (pixels)
    world_x_mm, world_y_mm = self.widget_to_world_coords(x, y)

    # Now import the file and position at (world_x_mm, world_y_mm)
    ...
```

This method uses the canvas's `view_transform` matrix to handle zoom and pan correctly.

### Visual Feedback Overlay

The overlay message will be implemented using a simple approach:

```python
def _show_drop_overlay(self):
    """Display 'Drop files to import' overlay on canvas."""
    # Create a semi-transparent label or text element
    # Position it centered on the canvas
    # Make it visible
    if not hasattr(self, '_drop_overlay_label'):
        self._drop_overlay_label = Gtk.Label(label="Drop files to import")
        self._drop_overlay_label.add_css_class("drop-overlay")
        # Style with CSS: semi-transparent background, large text

    # Show the overlay (implementation depends on canvas overlay system)
    # Could use Gtk.Overlay if MainWindow uses one
    # Or draw directly on canvas during render

def _hide_drop_overlay(self):
    """Remove the drop overlay from canvas."""
    # Hide or remove the overlay element
```

**Styling:**
- Semi-transparent dark background with white text
- Large, readable font size
- Centered on canvas viewport
- Subtle rounded corners
- Low opacity to not obscure canvas content

### Data Flow

```
User drags file from file manager
         │
         ▼
User hovers file over canvas
         │
         ▼
GTK emits "enter" signal on WorkSurface's DropTarget
         │
         ├─── _on_drag_enter() called
         │    └─── _show_drop_overlay() displays "Drop files to import"
         │
         ▼
User drops file onto canvas at pixel position (x, y)
  (OR user drags away → "leave" signal → _hide_drop_overlay())
         │
         ▼
GTK emits "drop" signal on WorkSurface's DropTarget
         │
         ▼
_on_file_dropped(drop_target, value, x, y) receives:
  - value: Gio.File or Gdk.FileList
  - x, y: Widget coordinates (pixels)
         │
         ├─── _hide_drop_overlay() removes overlay message
         │
         ├─── Convert (x, y) to world coordinates (mm)
         │    using widget_to_world_coords()
         │
         ├─── Extract file paths from value
         │
         ├─── Query MIME types
         │
         └─── For each file:
              │
              ├─── SVG? → Show import dialog → Import at position
              │
              └─── Raster? → Import with TraceConfig → Position at drop point
                              │
                              └─── Set workpiece.position to (world_x_mm, world_y_mm)
```



## Error Handling

### Expected Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Unsupported file type | Toast notification: "Unsupported file format: {mime_type}" |
| File permission denied | Toast notification: "Cannot read file: Permission denied" |
| File not found | Toast notification: "File not found: {filename}" |
| Corrupt/invalid file | Delegate to existing import error handling |
| Import exception | Log exception, show generic toast: "Failed to import {filename}" |
| Coordinate transformation fails | Fallback to (0, 0) position, log warning |
| Empty clipboard | No action, silent (or could show subtle message) |
| Clipboard has unsupported format | Toast notification: "Clipboard does not contain a supported image format" |
| Temp file creation fails | Toast notification: "Failed to process clipboard image" |


## Documentation Updates

## Design Decisions

### Why Canvas-Level Drop Target (Not Window-Level)?

**Rationale:**
1. **Spatial Control**: Allows users to position imported items precisely
2. **Industry Standard**: Matches behavior of design tools (Inkscape, Figma, Illustrator)
3. **Better UX**: Direct manipulation - drop where you want it
4. **Canvas Context**: File drops are logically related to canvas, not window chrome

**Trade-offs:**
- Slightly smaller drop target (only canvas area, not toolbar/sidebars)
- More complex implementation (coordinate transformation required)
- **Accepted**: Benefits outweigh complexity

### Why Position at Drop Location (Not Default 0,0)?

**Rationale:**
1. **User Intent**: If user drops at specific location, they likely want it there
2. **Workflow Efficiency**: Eliminates repositioning step
3. **Spatial Awareness**: Helps users organize their workspace during import
4. **Expected Behavior**: Matches how drag-and-drop works in similar applications

**Trade-offs:**
- Need coordinate transformation logic
- Multi-file drops all stack at same point (could be confusing)
- **Accepted**: Matches user expectations, solves real workflow problem

### How to Handle Multi-File Drops?

**Decision: Stack all files at drop point**

**Rationale:**
- Simple and predictable behavior
- User can drag files apart after import if needed
- Alternative (spread files in grid) adds complexity and assumptions about spacing

**Could enhance later:**
- Automatically space files in a grid pattern
- Show preview before finalizing positions

### Clipboard Paste Integration

**Decision: Hook into existing paste action, check for images first**

**Rationale:**
- Existing paste action already handles workpiece paste
- Need to check clipboard for image data before delegating to workpiece paste
- If image found: import as new workpiece
- If no image: use existing paste logic (paste copied workpieces)

**Implementation approach:**
```python
def on_paste_action(self, action, param):
    clipboard = self.get_clipboard()

    # Priority 1: Check for image data
    if clipboard.has_image():
        self._import_from_clipboard()
        return

    # Priority 2: Existing workpiece paste logic
    self.on_paste_requested(...)
```

**Positioning for pasted images:**
- Canvas center (most predictable)
- Alternative: Last cursor/click position (more flexible)
- **Decision:** Use canvas center for simplicity and consistency
