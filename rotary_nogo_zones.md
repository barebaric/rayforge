# Remaining Work: No-Go Zones & Warnings

## 1. No-Go Zone Data Model

### 1a. `NoGoZone` — `rayforge/machine/models/nogo_zone.py` (new)

```python
class NoGoShape(Enum):
    RECT = "rect"          # axis-aligned 2D box (x, y, w, h)
    CYLINDER = "cylinder"  # 3D cylinder (x, y, z, radius, height)
    BOX = "box"            # 3D box (x, y, z, w, h, d)

class NoGoZone:
    """Follow the Laser/RotaryModule pattern: class with changed Signal,
    setter methods, to_dict/from_dict, pickle support, extra dict."""
    uid: str
    name: str
    shape: NoGoShape
    params: Dict[str, float]   # shape-specific dimensions
    enabled: bool = True
    color: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.3)
```

- **2D zones** use `RECT` with `{x, y, w, h}` — checked against workpiece
  bounding boxes and toolpath XY coordinates.
- **3D zones** use `CYLINDER` or `BOX` — additionally checked against Z travel
  (e.g. rotary chuck extends upward).
- All coordinates in **machine space** (mm), within `axis_extents`.

### 1b. Integration into `Machine`

Add to `Machine`:

```python
self.nogo_zones: Dict[str, NoGoZone] = {}

# Serialization keys in to_dict/from_dict:
# "nogo_zones": [...]
```

Follow the same dict-by-uid pattern as `rotary_modules`.

### 1c. Integration into `MachineProfile`

Add optional fields to `MachineProfile` so preset machines can ship with
default no-go zones (e.g. Carvera Air knows where its rotary chuck sits).

---

## 2. Zone Checking

### 2a. `ZoneChecker` — `rayforge/machine/models/zone_checker.py` (new)

Pure-logic class, no UI dependency:

```python
class ZoneCheckResult:
    zone: NoGoZone
    severity: str          # "warning" | "error"
    message: str
    intersection_type: str  # "bbox" | "toolpath"

class ZoneChecker:
    def __init__(self, machine: Machine):
        self.machine = machine

    def get_active_zones(self) -> List[NoGoZone]:
        """Returns enabled zones + rotary module volumes if configured."""

    def check_bbox(self, bbox: Rect) -> List[ZoneCheckResult]:
        """Check a 2D bounding box against all 2D zones."""

    def check_toolpath(self, ops: Ops) -> List[ZoneCheckResult]:
        """Check ops vertices against zones (2D + optional Z)."""

    def check_rotary_wcs_alignment(self, layer: Layer) -> Optional[str]:
        """If layer is rotary, verify WCS points to rotary axis position."""
```

**Intersection logic**: Reuse `rayforge/core/geo/polygon.py`
(`polygons_intersect`) for 2D. For 3D (Z-axis check against rotary module
height), a simple range check suffices.

### 2b. Where checks run

| Check                    | Trigger                                     | Location                                                           |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------ |
| **BBox vs. zones**       | Workpiece moved/resized, zone changed       | Pipeline `WorkPiecePipelineStage` or `DocEditor` command post-hook |
| **Toolpath vs. zones**   | Pipeline produces `Ops`                     | `StepPipelineStage` — after ops production, before encoding        |
| **Rotary WCS alignment** | Layer `rotary_enabled` toggled, WCS changed | `Layer` signal handler in `DocEditor`                              |

Results emitted via signal:

```python
zone_warnings_changed = Signal()  # sends List[ZoneCheckResult]
```

---

## 3. Warning UI

### 3a. Toolbar warnings

The app has `Toolbar.machine_warning_box` for persistent warnings and
`MainWindow._add_toast()` for transient messages. Extend:

- **Rotary WCS mismatch**: persistent warning in toolbar. Clicking opens
  hardware page to fix WCS.
- **Zone violation**: toast with severity. For `severity="error"`, block job
  execution (disable "Generate G-code" button, like existing `can_export`).

### 3b. Canvas visual feedback

When a zone violation is active, the intersecting zone renderer pulses or turns
red. Follows how the jog widget uses `add_css_class("warning")` for limit
violations.

### 3c. `ZoneWarningBanner` (optional)

A collapsible `Gtk.InfoBar` above the canvas listing active zone violations.
More detailed than toasts, less intrusive than a dialog.

---

## 4. No-Go Zones Settings UI

New `NogoZonesPage` in `rayforge/ui_gtk/machine/`, registered in the machine
settings dialog sidebar:

```
No-Go Zones
├── [ListBox] List of zones (name, shape, enabled toggle, delete button)
├── [Button] "Add Zone"
│   └── Dialog: choose shape → set dimensions & position
└── [SpinRow] per-zone: X, Y, W, H (rect) or X, Y, Z, Radius, Height (cylinder/box)
```

Follows `PreferencesGroupWithButton` + `TrackedPreferencesPage` pattern.

---

## 5. Key Design Decisions

1. **Machine-scoped**: Zones belong to the physical machine, persist in its
   YAML config. Documents are portable across machines.

2. **Coordinate space**: All zone coordinates in machine space. Existing
   `MachineSpace` transforms handle conversion.

3. **3D-first collision**: Full 3D collision detection from day one. 2D rect
   zones are degenerate 3D zones (Z height = 0), not a separate 2D system.

4. **Reuse geometry**: `polygons_intersect()` from `core/geo/polygon.py` for
   XY-plane intersection. Simple range checks for Z-axis bounds. No new 3D
   geometry library needed.

5. **Non-blocking warnings**: Zone violations warn but don't prevent design.
   Only block at job execution time.
