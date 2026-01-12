# Ops API Documentation

The `Ops` (Operations) module provides a framework for representing and
manipulating laser cutting and engraving operations. It defines a sequence of
commands that describe toolpaths, machine state changes, and logical markers
for job generation.

---

## Class: Ops

`rayforge.core.ops.container.Ops` is the primary class for representing a
sequence of laser operations.

### Constructor

```python
Ops()
```

Initializes a new, empty Ops object.

### Properties

- **`commands`** (`List[Command]`): The list of commands in this Ops object.
- **`last_move_to`** (`Tuple[float, float, float]`): The endpoint of the last
  `move_to` command.

### Serialization

- **`to_dict() -> Dict[str, Any]`**
  Serializes the Ops object to a dictionary for JSON storage.

- **`from_dict(cls, data: Dict[str, Any]) -> Ops`**
  Deserializes a dictionary into an Ops instance.

- **`to_numpy_arrays() -> Dict[str, np.ndarray]`**
  Serializes the command list into a dictionary of NumPy arrays for efficient
  storage and transfer. Uses a Struct-of-Arrays approach.

- **`from_numpy_arrays(cls, arrays: Dict[str, np.ndarray]) -> Ops`**
  Reconstructs an Ops object from a dictionary of NumPy arrays.

- **`from_geometry(cls, geometry: Geometry) -> Ops`**
  Creates an Ops object from a Geometry object, converting its path.

- **`to_geometry() -> Geometry`**
  Creates a Geometry path from this Ops object, including only the geometric
  commands.

### Path Construction

- **`move_to(x: float, y: float, z: float = 0.0)`**
  Starts a new subpath at the specified coordinates.

- **`line_to(x: float, y: float, z: float = 0.0)`**
  Adds a straight line segment to the specified coordinates.

- **`close_path()`**
  Closes the current subpath by drawing a line to the last `move_to` point.

- **`arc_to(x: float, y: float, i: float, j: float, clockwise: bool = True, z: float = 0.0)`**
  Adds a circular arc. `i` and `j` are offsets from the start point to the
  center.

- **`bezier_to(c1: Tuple[float, float, float], c2: Tuple[float, float, float], end: Tuple[float, float, float], num_steps: int = 20)`**
  Adds a cubic Bézier curve approximated by a series of LineToCommands.

- **`scan_to(x: float, y: float, z: float = 0.0, power_values: Optional[bytearray] = None)`**
  Adds a scan line command with variable power values for raster engraving.

### State Commands

- **`set_power(power: float)`**
  Sets the intended laser power (0.0 to 1.0) for subsequent commands.

- **`set_cut_speed(speed: float)`**
  Sets the intended feed rate for subsequent cutting commands.

- **`set_travel_speed(speed: float)`**
  Sets the intended feed rate for subsequent travel commands.

- **`enable_air_assist(enable: bool = True)`**
  Sets the intended state of the air assist for subsequent commands.

- **`disable_air_assist()`**
  Disables the air assist for subsequent commands.

- **`set_laser(laser_uid: str)`**
  Sets the intended active laser for subsequent commands.

### Marker Commands

- **`job_start()`**
  Adds a job start marker command.

- **`job_end()`**
  Adds a job end marker command.

- **`layer_start(layer_uid: str)`**
  Adds a layer start marker command.

- **`layer_end(layer_uid: str)`**
  Adds a layer end marker command.

- **`workpiece_start(workpiece_uid: str)`**
  Adds a workpiece start marker command.

- **`workpiece_end(workpiece_uid: str)`**
  Adds a workpiece end marker command.

- **`ops_section_start(section_type: SectionType, workpiece_uid: str)`**
  Adds an ops section start marker command for semantically distinct blocks.

- **`ops_section_end(section_type: SectionType)`**
  Adds an ops section end marker command.

### Query & Analysis

- **`rect(include_travel: bool = False) -> Tuple[float, float, float, float]`**
  Returns the bounding box `(min_x, min_y, max_x, max_y)` of the occupied
  area in the XY plane.

- **`distance() -> float`**
  Calculates the total 2D path length for all moving commands.

- **`cut_distance() -> float`**
  Calculates the total 2D cut distance (excluding travel moves).

- **`estimate_time(default_cut_speed: float = 1000.0, default_travel_speed: float = 3000.0, acceleration: float = 1000.0) -> float`**
  Estimates the execution time of the operations in seconds.

- **`is_empty() -> bool`**
  Returns `True` if the Ops object contains no commands.

- **`get_frame(power: Optional[float] = None, speed: Optional[float] = None) -> Ops`**
  Returns a new Ops object containing four move_to operations forming a frame
  around the occupied area.

### Modifications & Transformations

- **`transform(matrix: np.ndarray) -> Ops`**
  Applies a 4x4 affine transformation matrix to all geometric commands.

- **`translate(dx: float, dy: float, dz: float = 0.0) -> Ops`**
  Translates all geometric commands.

- **`scale(sx: float, sy: float, sz: float = 1.0) -> Ops`**
  Scales all geometric commands.

- **`rotate(angle_deg: float, cx: float, cy: float) -> Ops`**
  Rotates all points around a center `(cx, cy)` in the XY plane.

- **`linearize_all()`**
  Replaces all complex commands (e.g., Arcs) with simple LineToCommands.

- **`clip(rect: Tuple[float, float, float, float]) -> Ops`**
  Clips the Ops to the given rectangle, returning a new Ops object.

- **`clip_at(x: float, y: float, width: float) -> bool`**
  Creates a gap of the given width centered at the closest point to `(x, y)`.

### State Management

- **`preload_state()`**
  Walks through all commands, enriching each by the intended state of the
  machine. Useful for post-processors that need to re-order commands.

### Operations

- **`copy() -> Ops`**
  Creates a deep copy of the Ops object.

- **`clear()`**
  Resets the Ops object to an empty state.

- **`add(command: Command)`**
  Appends a single command to the Ops object.

- **`extend(other_ops: Ops)`**
  Appends all commands from another Ops object to this one.

- **`segments() -> Generator[List[Command], None, None]`**
  Yields segments of commands grouped by travel moves and state changes.

---

## Command Classes

All commands inherit from the abstract `Command` base class.

### MovingCommand

Abstract base class for commands that move the tool.

- **`MoveToCommand`**: Starts a new subpath (travel move).
- **`LineToCommand`**: A straight line segment (cut move).
- **`ArcToCommand`**: A circular arc segment (cut move).
- **`ScanLinePowerCommand`**: A line segment with variable power for raster
  engraving.

### State Commands

Commands that modify the machine state:

- **`SetPowerCommand(power: float)`**: Sets the laser power (0.0 to 1.0).
- **`SetCutSpeedCommand(speed: int)`**: Sets the cutting feed rate.
- **`SetTravelSpeedCommand(speed: int)`**: Sets the travel feed rate.
- **`EnableAirAssistCommand`**: Enables the air assist.
- **`DisableAirAssistCommand`**: Disables the air assist.
- **`SetLaserCommand(laser_uid: str)`**: Selects the active laser.

### Marker Commands

Logical markers for the job structure:

- **`JobStartCommand`**: Marks the job start.
- **`JobEndCommand`**: Marks the job end.
- **`LayerStartCommand(layer_uid: str)`**: Marks the layer start.
- **`LayerEndCommand(layer_uid: str)`**: Marks the layer end.
- **`WorkpieceStartCommand(workpiece_uid: str)`**: Marks the workpiece start.
- **`WorkpieceEndCommand(workpiece_uid: str)`**: Marks the workpiece end.
- **`OpsSectionStartCommand(section_type: SectionType, workpiece_uid: str)`**:
  Marks the semantically distinct block start.
- **`OpsSectionEndCommand(section_type: SectionType)`**: Marks the block end.

### SectionType Enum

Defines the semantic types for Ops sections:

- **`VECTOR_OUTLINE`**: Vector cutting operations.
- **`RASTER_FILL`**: Raster engraving operations.

---

## State Class

`rayforge.core.ops.commands.State` represents the machine state during
execution.

### Properties

- **`power`** (`float`): Normalized power from 0.0 to 1.0.
- **`air_assist`** (`bool`): Air assist enabled state.
- **`cut_speed`** (`Optional[int]`): Cutting feed rate.
- **`travel_speed`** (`Optional[int]`): Travel feed rate.
- **`active_laser_uid`** (`Optional[str]`): Currently selected laser.

### Methods

- **`allow_rapid_change(target_state: State) -> bool`**
  Returns True if a change to the target state should be allowed in a rapid
  manner (e.g., for each gcode instruction). Changing the air-assist rapidly
  is prevented to protect the air pump.
