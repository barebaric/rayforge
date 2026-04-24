# Changelog

All notable changes to Rayforge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.6

### Added

- Device profiles replace machine profiles with declarative packages that bundle
  machine config and G-code dialect together
- Export and import UI for sharing device profiles between machines or users
- Per-layer Work Coordinate System (WCS) assignment with edit button in layer
  settings
- Redesigned layer system with visual workflow indicators in each layer column
- Drag-and-drop layer reordering in the layer list
- Workpieces are reorderable in the layer list to change z-order
- Middle-click pan in the layer list
- Layer columns now show subtitles and have limited default width
- New documents start with a default of 3 layers
- Rotary mode now supports true 4th axis and axis replacement (switching X or
  Y for rotary)
- Machine settings offer settings for roller-type rotary axis
- Material test grid supports new parameter combinations with extra speed or
  power labels in multipass mode
- GRBL Telnet driver for networked grblHAL and ESP3D controllers (thanks to
  gyordanov)
- Update checker notifies when a new Rayforge version is available
- Right panel is now a floating overlay for more canvas space
- Values next to sliders are now editable entry fields
- Setting to choose whether ops use layer color or laser color
- Double clicking a workpiece in the layer box opens its properties
- Print and cut addon for aligning laser cuts with printed material (#180)
- Device profile for the Creality Falcon A1
- Right-click context menu for empty canvas space
- Sketcher: support changing text color using the fill tool
- Site search on the Rayforge website
- Sponsor page on the Rayforge website

### Changed

- Major refactoring of kinematics and 3D simulator for better rotary support
- Improved error messages for Grbl alarms
- Kinematics now build dynamically from AxisSet and AxisMapper
- Massively decreased memory usage of multi-layer PDFs
- Reduced memory required for vertex storage by storing power values instead
  of colors
- 2D canvas adapts to rotary mode automatically
- 3D canvas displays rotaries correctly in all configurations
- GRBL serial driver: improved deadlock recovery, comment stripping before
  sending G-code, improved buffer handling
- Air assist state no longer resets between workpieces
- 2D simulator removed (superseded by the 3D simulator)
- Layer settings dialog is now non-modal
- Bottom panel is now visible by default
- Status polling is disabled during job execution by default

### Fixed

- WCS marker not updating in 2D canvas when changing WCS in layer settings
- WCS synchronization fails if device does not report Z coordinates
- Re-syncing WCS with the machine did not clear stale offsets
- G0 and G1 feedrate is shared (#210)
- Air assist disabled after workpiece (#208)
- Sketcher shortcut shadowed by New Project shortcut
- 1 key shortcut shadowed dimension input (#207)
- Canvas not centered when opening a project with rotary layer
- Opening a layer in rotary replacement mode overwrites machine Y dimensions
- Textures not drawn with proper opacity in 3D canvas
- Textures stretched too wide around the cylinder in rotary mode
- 3D canvas axis extent frame with inverted margins when origin is top-right
- 3D canvas not updating ops when rotary config changes
- Laser head not moving in Y in flat mode
- Cylinder not rotating during playback
- Drawing trails behind laser in rotary simulation
- Clipping when zooming in 3D canvas in orthographic view
- No Z in G-code for rotary in Z replacement mode
- G-code for rotary missing rotary command
- Terminal window visible on Windows
- Debouncing caused material test not to update when switching presets (#187)
- Deleting the active machine could lead to no machine being active
- Stale ops in 3D canvas after deleting a layer
- Multiple GRBL serial driver robustness improvements
- GRBL alarm codes were mapped to wrong error descriptions
- Snap fails to launch with gpu-2404 slot not connected error (#196)
- Pipeline applied gear ratio even for visual representation (#195)
- Delayed main thread callbacks not cancellable via handle.cancel()
- GRBL sending initial $I as realtime command though it is not one
- Base image of inverted SVG not inverted after import
- Rotary cylinder rendered with diameter of chuck instead of workpiece (#195)
- Beta version string parsing for debian releases
- Zero axis not working over GRBL network connection (#220)
- Jog distance not applying when entered via keyboard (#221)
- Gtk imported in worker subprocesses (#224)
- Various Gtk warnings

## 1.5.2

### Fixed

- G-code production could fail if no rotary axis commands were defined
- G5 commands in LinuxCNC and Marlin templates did not respect the omit
  unchanged axis flag
- 2D canvas showing stale ops when operation generates zero ops
- Model preview showing models in wrong orientation by default
- Point light not turning off when laser is off
- Potential race conditions in the pipeline and 3D canvas
- Model preview now displays colors correctly

## 1.5.1

### Changed

- Post processor page is more compact by putting each post processor into an expander
- 3D canvas: better line width for rendered ops

### Fixed

- GRBL serial not connecting (#196)
- Auto brightness toggle setting not remembered in raster step settings

## 1.5

### Added

- 3D simulator with full playback: play/pause, step forward/backward, scrubber,
  and speed control (1x to 16x)
- End-to-end bezier curve support (G5) through the entire pipeline, from import
  to G-code output
- No-go zones: define restricted areas in machine settings with collision checking
- 3D model support for rotary axes (GLB format) with shading and proper coloring
- Global model manager for storing and reusing 3D models across machines
- Import dialog now offers three layer modes: flatten, merge to existing, or
  create new layers
- Imported layers automatically get sensible default workflow steps
- Dockable bottom panel: tabs can be rearranged freely and split into separate
  columns
- Layer list moved into the bottom panel with drag-and-drop from asset list
- Asset browser overhaul: all assets visible, multi-selection, drag to canvas,
  thumbnails for most image formats, helpful empty-state placeholder
- Lead-in / lead-out postprocessor for zero-power approach and exit moves
- Material test: labels engraved first for cleaner results
- Material test: overscan transformer support
- Material test: independent label engraving speed setting
- Ctrl+F search in console and G-code viewer
- Addons can register their own toggles in the View menu
- YUYV camera protocol support
- Canvas remembers state of view toggles between sessions
- Double clicking a stock asset opens its properties
- Drag assets from the asset list to the layer list
- Continuous laser mode and modal feedrate G-code options
- GRBL raster dialect that omits unnecessary M4/M5 spindle commands
- Grbl MKS DLC32 machine profile
- Laser head model rendered in the 3D simulator

### Changed

- Canvas is significantly more responsive during drag, pan, and zoom operations
  by suppressing expensive path rendering until interaction stops
- Multi-step workflows composite into a single surface for faster rendering
- Large images are automatically scaled to prevent multi-gigabyte memory spikes
- Image handling rewritten to avoid unnecessary copies, reducing overall RAM usage
- Smarter caching: base images cached on source assets and reused across workpieces
- Cache memory limit in the 2D canvas prevents unbounded memory growth
- Time estimation updates instantly instead of recomputing from scratch
- Pipeline recalculation can now be toggled off in settings
- Status bar removed; machining time moved to layer list header, ETA to machine
  dropdown, status messages to canvas overlay
- Visibility toggles for perspective, model, and no-go zones moved to canvas
  overlays
- Rows in main window expanders are more compact
- Bottom panel layout of coordinate controls and jog widget is responsive
- Decreased default merge lines tolerance to 0.01
- Crop-to-stock now crops to workarea if no stock is defined in the document
- Illustrator files now use correct 72 DPI instead of 70

### Fixed

- Race condition causing 2D canvas re-renders to hang
- Race condition in doceditor.wait_until_settled_sync()
- Status overlay displayed even when no message was set
- Stale job shown in 3D canvas after changes
- 3D canvas not showing dimmed versions of travel moves
- Wide strokes rendered incorrectly in import dialog preview
- Item layers not added when importing from the command line
- Tabs cutting paths in more than one place
- Addon installation failing in Snap packages
- 2D and 3D canvas stealing focus from the sketcher
- Sketch parameters could not be edited in the properties panel
- Crop-to-stock linearized arcs unnecessarily
- Deadlock when switching WCS while 3D canvas visible
- 3D canvas not updating fully after hardware changes
- Toggling no-go zones off undimmed vertices in the 3D canvas
- Mapping of stepped down vertices in rotary mode
- Drag and drop bug in the asset browser
- No G-code output if document contained an empty layer
- Editing machine settings resetting the canvas perspective
- Single instance lock: second window now gracefully exits
- Two memory leaks in the 3D simulator (shared memory and shader)
- 3D canvas empty if G-code viewer was not open

## 1.4.1

### Changed

- PDF import now falls back to `fitz` when `pymupdf` is not installed (#186)
- DPI setting in the SVG import dialog is now persistent between sessions

## 1.4

### Added

- Full rotary axis support with 3D visualization
- Support for multiple rotary modules
- Configurable rotary mode per layer
- Rotary icon displayed on rotary layers
- PDF direct vector import with layer support
- Improved five factor camera de-distortion algorithm
- Charuco card based calibration wizard with guided setup process
- Sketcher: ellipse tool replaces circle tool for more flexibility
- Sketcher: many tools automatically constrain geometry during creation
- Sketcher: magnetic snap now works while creating geometry
- Sketcher: replaced snap to grid with smarter magnetic snap
- Sketcher: equality constraint now works on ellipses
- Configure frame speed in laser head settings
- Corner dwell time setting for framing
- Repeat count setting for framing
- New merge lines post-processor to avoid double cutting
- Machine profile for Acmer S1 added
- Dialects support separate laser on command for focusing
- Add a DPI setting to the SVG import dialog if the SVG is unitless

### Changed

- More compact left panel layout with add buttons moved into group headers
- G-code viewer moved into the bottom panel
- 3D canvas performance improvements
- Dialects are now isolated copies (templates)
- GRBL buffer size tracking improved
- Texture dimension limit prevents memory exhaustion
- Addon manager: safer threading approach

### Fixed

- Material test producer bugs (issues #181 and #182)
- GRBL position reporting for machines with only X and Y axes (#179)
- Sketcher: distance constraint shadowing the line
- Texture renderer memory exhaustion on large images
- Race condition in worker initialization

## 1.3.2

### Added

- Raster step settings now display angle numerically

### Fixed

- Import issues on Windows
- Unknown G-code dialects now fall back to Grbl instead of causing errors
- Duplicate error notifications from the driver

## 1.3.1

### Fixed

- Icon sizing issues on systems with Gtk 4.21
- Windows and macOS build issues
- Conflicting menu shortcuts in the sketcher

## 1.3

### Added

- Sketcher: full support for bezier curves with intuitive handle-based editing
- Sketcher: new unified path tool that combines lines and curves
- Sketcher: grid tool for visual reference and alignment
- Sketcher: toggle buttons to show/hide construction geometry and constraints
- Sketcher: hold Shift to constrain movement to the nearest axis
- Sketcher: endpoints connected by coincident constraint can be made smooth or symmetric
- Sketcher: "straighten" tool to convert bezier curves to straight lines
- Sketcher: path edit tool can now connect to existing points
- Sketcher: conflicting constraints now shown in the panel
- Raster operation: sample interval and power levels settings
- G-code viewer now shows line count and byte size

### Changed

- Sketcher moved to an add-on (installed and enabled by default)
- G-code now omits unchanged coordinates for more compact output
- Removed obsolete "no-Z" G-code dialect variants
- G-code viewer always shows at least the first 20,000 lines
- Sketcher: hide gray background area for cleaner editing
- AI workpiece generator now creates sketches when geometry can be mapped

### Fixed

- Most icons not displayed on systems with Gtk 4.21 or higher
- Fillet and chamfer tools not working correctly
- Texture encoder drawing spaced dots instead of lines in some cases
- G1 emitted without coordinates when all coordinates unchanged
- Loading project files with unknown assets

## 1.2.1

### Added

- Buttons to move to center, bottom/left and top/right of workpiece
- Support for setting a "tab power"
- Sketcher: allow entering dimensions while adding geometry

### Changed

- Better button layout in the control panel
- Sketcher: circle now uses diameter constraint consistently, not sometimes radius
- Laser dot now always drawn on top, not obscured by workpieces

### Fixed

- Builtin addon yaml files not included in .snap
- Imprecise tab location while dragging the tab handle
- Tabs not working on beziers

## 1.2

### Added

- AI workpiece generation
- Camera image enhancement with temporal noise reduction (thanks
  to MausRundung)
- Fisheye lens distortion correction with radial and tangential parameters
  (thanks to MausRundung)
- Zoom, pan and keyboard navigation in camera alignment dialog
  (thanks to MausRundung)
- Complete addon system rewrite and refactoring
- Tons of new materials in core materials addon
- Sketcher: live preview for line tool
- Sketcher: show dimensions while adding geometry
- Sketcher: snap-to-grid on Ctrl press
- Sketcher: highlight hovered entities and constraints
- Sketcher: toolbar shows currently available shortcuts
- Sketcher: Shift+Double click selects connected geometry
- Sketcher: allow arc radius changes while adding second arc endpoint
- Path optimizer now also optimizes inter-workpiece travel
- Mach4 G-code dialect
- Post-processor: crop-to-stock
- Click canvas to set zero feature
- Support for configuring cut/raster colors per laser
- Support for duplicating stock and non-rectangular stock
- Convert workpiece to stock (right-click menu)
- RAYFORGE_DISABLE_3D environment variable
- Machine profile for OMTech K40+
- Navigation and zoom icons for UI controls

### Changed

- Stock is now a document-level concept - no more stock per layer
- Improved camera selection dialog (left/right indicators, keyboard support)
- Imported items and stock now aligned with WCS origin by default
- Stock placed at WCS origin by default
- Sketcher: preserve selection when creating lines, arcs, or circles
- Sketcher symmetry constraint click order now aligns with FreeCAD
- Maximum laser G-code power increased to 100.000
- Addon terminology changed from "plugin" to "addon"
- Context now lazy loads most services for better performance

### Fixed

- Undoing text box left entries in history manager stack
- Y axis drawn on wrong side of canvas
- Auto layout for rotated stock
- SVG exporter stacking multiple workpieces on top of each other
- Text color in Sketcher while editing
- When opening project file referencing non-existent laser, assign default laser
- Text box rotates while typing on Windows
- Addon handling issues on Windows

### macOS Specific

- Narrow macOS app menu window to MainWindow - pgilfernandez
- macOS app menu actions fix - pgilfernandez

## 1.1.2

### Fixed

- Text box rotating while typing in sketcher
- Assets cannot be deleted after loading from a project file
- Empty sketches showing as black squares after loading from project
- Maybe: Pie menu not opening in correct location on Windows

## 1.1.1

- Fix screenshot link in appstream file causing Flatpak build to fail.

## 1.1

### Added

- Major: MacOS support was added thanks to Github user pgilfernandez!
- Major: Complete data pipeline backend rewrite with improved performance
  and memory management
- Major: Unified raster operations - the old "raster", "depth", and
  "dither" operations are now replaced by a single, unified raster operation
- Major: G-code console replacing the log view with a fully featured terminal
- SVG and DXF export support - export your documents or selected objects
- Angle constraints now available in the sketcher
- New "Raster (Dither)" operation type with configurable algorithms
- Built-in G-code dialect supporting dynamic power mode
- Auto thresholding for Variable Power and Multipass raster modes
- Histogram in raster engraver to help setting thresholds
- Symbolic visualization of raster direction in rasterizer
- Configurable line distance in all raster operations
- Axis extents, work surface, and soft limits can now be configured per machine
- RAYFORGE_MAX_WORKERS environment variable to limit the number of processes
- macOS packaging scripts and resources for better platform support
- `--uiscript` CLI argument for executing scripts after the UI is up
- Optional (opt-in) anonymous usage statistic collection
- Material files can now contain translations

### Changed

- Machine selector moved to the window header for easier access
- G-code viewer and control panel toggles are now independent
- Machine settings dialog reorganized for better clarity
- Pressing "reset position" on a workpiece places it at WCS origin,
  not machine origin
- Workpiece properties now show the position in WCS coordinates
- Contour and shrinkwrap operations use proper tolerance instead of
  laser dot size
- Increased maximum laser spot size to 10 mm
- File dialog now selects BMP by default instead of all supported files
- Imported images are now placed at reference origin by default
- Added Ukrainian translations
- Re-assigned Alt key bindings to avoid clashing with main menu actions

### Fixed

- Multiple memory leaks in the data pipeline
- Race conditions between worker pool task completion and pipeline shutdown
- Camera device scanning crash on Windows
- Multipass post-processor exception
- Rasterizer angle causing distortion
- Various macOS issues: shm_open length errors, SVG rendering fallback,
  keyboard shortcuts
- Workpiece stage not emitting correct node state
- Zoom resolution stuck until resizing at least once
- Blurry view overlay on startup
- Race condition leads to stale vectors for cancelled tasks
- Memory for view artifacts released late
- Unresponsive serial ports now handled gracefully with log message
- Simulation preview strokes now stay constant across zoom levels
- Speed entered in Jog dialog is now correctly converted to base units
- Switching to 3D view sometimes showed no operations
- Some icons not appearing when running the snap on non-GNOME environments
- GRBL connection handshake timeout handled with backoff retry

## [1.0.1] - 2026-02-01

### Fixed

- Some strings were not translatable
- Reformatted the 1.0 appstream release notes to not trip the Flatpak build up

## [1.0] - 2026-02-01

### Added

- Major: Project save/load support
- Major: The sketcher now supports text, with many bells and whistles
- Major: The jog dialog has been merged together with the log view into a bottom panel
- Sketcher supports aspect ratio constraints
- Engraving steps now have an invert setting
- The raster engraver now supports setting the engraving angle in degrees
- A recent files menu entry was added
- Installers for all platforms now register the .ryp (project) and .rfs (sketch) file extensions

### Changed

- Command line interface: `--direct-vector` renamed to `--vector`; added
  `--trace` to force trace mode. Default is now to try vector import first,
  falling back to trace if not supported
- Import errors now collected and displayed in import dialog
- Importers almost completely rewritten for testability
- Sketcher: The solver now biases points to their previous position for more stable dragging
- Simulation mode keyboard shortcut changed to F11 to avoid conflic with "Save as..."
- Remember G-code view and control panel visibility across sessions
- Increased precision of power sliders and display digits everywhere
- Import dialog now shows number of vectors per layer
- Error message shown when attempting to delete a dialect that is in use
- Chinese translations were added
- When opening bitmap images from CLI, default to import the whole image, not tracing

### Fixed

- Traceback in the sketcher when adding a constraint (affected Windows build only)
- Fixed a potential memory leak and stale ops display
- Multi layer DXF import
- Numerous alignment bugs in importers
- Traceback when using invert switch in import dialog
- Sketches not properly centered on the surface after import

### Documentation

- Updated importer developer documentation

### Build

- Added hicolor-icon-theme dependency to snap build

## [0.28.4] - 2026-01-22

### Fixed

- Traceback when dialect contained deprecated attributes
- Debian package missing asyncudp dependency

## [0.28.3] - 2026-01-18

### Added

- Option to enable/disable WCS injection in G-code dialect

### Fixed

- Depth engraver not respecting master power setting
- Traceback when using invert image in import dialog

## [0.28.2] - 2026-01-17

### Fixed

- Test that depended on specific version string

## [0.28.1] - 2026-01-17

### Fixed

- Invalid ampersand in appstream XML

## [0.28] - Work Coordinate Systems, True Arcs, and a New Package Manager

### Added

- Full support for Work Coordinate Systems (WCS) G54-G59
  - "Set Origin Here" button to define temporary work zero at any point
  - Visual feedback with active WCS origin marked on 2D canvas
  - 3D view renders geometry relative to active WCS
  - WCS integrated across G-code encoder, 2D/3D views, and GRBL drivers
  - Ability to create set G-code offsets in machine settings
  - Offline configuration of WCS settings and offsets
- True arc support and superior geometry handling
  - DXF and SVG importers now preserve arcs and bezier curves
  - Machine settings to configure arc support and tolerance
  - Non-uniform scaling of designs with arcs handled gracefully
- Package Manager for extensibility
  - Install, update, and manage extensions
  - Automatic update checks on startup
- New Sketcher tools
  - Rectangle tool with support for rounded rectangles
  - Fillet tool for rounded corners
  - Chamfer tool for beveled corners
- Machine connectivity improvements
  - Configurable GRBL polling (disable during job runs)
  - GRBL corruption detection
  - GRBL error messages more descriptive
  - Support for reading WCO and extended status fields
- User interface enhancements
  - "Import whole image" checkbox for direct raster import
  - Multi-layer SVG import option
  - Supporter recognition section in About dialog
  - Maintenance counter alerts link to maintenance counter page
  - Sketch instances automatically added to document on creation

### Changed

- G-code generator now strips unneeded trailing zeros
- Diagonal jogging now jogs in a direct line instead of two separate commands
- Connection and device errors displayed prominently next to device selector
- Debug log button moved from log view to help menu
- Stock list and sketch list merged into unified asset list
- Importing complex DXF and SVG files is now significantly faster
- Makera Air G-code now uses inline power commands
- G-code dialect editor now checks variable existence and bracket balance

### Fixed

- Job Control & Safety
  - Application getting stuck in "Running" state after job cancellation
  - Race condition where driver alarms might self-clear
- Framing
  - Runaway issue with framing position drifting cumulatively
  - Framing bounds calculation for full circles
- Machine & Drivers
  - Position reporting not updating until machine settings saved
- Import Fixes
  - SVG import dialog not allowing direct vector import of SVGs without layers
  - Vector misalignment when importing certain SVG files
  - DXF import failing on files with blocks containing solid fills
  - Raster image with tracing threshold set to 1 not importing full image
- Platform-Specific Fixes
  - (Windows) Dialog closing passing focus to wrong window
  - (Windows) PNG files not opening via file selector
- General Fixes
  - Duplicate axis labels drawn on canvas
  - Depth Engraver treating semi-transparent pixels as black
  - Material test grid including invalid power on/off toggles
  - Laser position indicator updating infrequently or moving wrong direction
  - Incorrect position readout in Jog dialog
  - Inconsistent button states during G-code job execution
  - Recipes not saved correctly when creating from step settings dialog
  - Multi-layer SVGs incorrectly imported as single layer
  - Coordinate systems issues for machines with negative axes
  - G-code dialect changes not applied without restart
  - Selected laser parameters not loading initially in machine settings UI
  - Imported SVGs not scaled correctly when resized non-uniformly
  - 3D canvas turntable rotation broken
  - Axis grid not aligned for machines without bottom left origin
  - Tracebacks and crashes in machine settings dialog and G-code generation
  - `pluggy` and `GitPython` dependencies not included in Debian package

## [0.27.1] - Maintenance Release

### Changed

- Importing complex SVG files is now significantly faster through intelligent
  path simplification

### Fixed

- SVG files with complex or invalid clipping paths rendered incorrectly
- Fills in sketches loaded from disk could not be toggled on or off
- On-screen position of laser dot not taking machine's configured origin into
  account

## [0.27] - Enhanced Sketching, Machine Control & UI Refinements

### Added

- Expressions and parameters in sketches
  - Expression editor with syntax highlighting and auto-completion
  - Instance parameters for each sketch instance
- Filled shapes support in sketcher
- Rounded rectangle tool
- Drag-select for multiple sketch elements
- Variable substitutions in preamble and postscript G-code sections
- Support for machines with top-right and bottom-right origins
- Negative axis support
- Configurable single-axis homing option
- Machine hours recording support
- Intro video to homepage

### Changed

- Sketches treated as "templates" that can be placed multiple times
- Sketch parameters have dedicated section in properties panel
- Construction line dash lengths measured in pixels for consistent look
- Double-clicking stock opens stock properties
- `Ctrl+N` shortcut for creating new sketch
- "Preferences" renamed to "Settings"
- "Edit Recipe" dialog uses three tabs: General, Applicability, Settings to Apply
- `ESC` and `Ctrl+W` close machine settings
- Machine settings menu entry cleaned up by removing "..."
- Raster import dialog renamed to "Import Dialog"
- Export sketch default location is original import location
- Many icons replaced with built-in icons
- Machine settings dialog redesigned for clearer layout
- Dark mode text readability improvements
- Asset list unified (stock and sketch lists merged)
- "New Sketch" button removed from main toolbar
- Main window no longer updates unnecessarily on workpiece transforms

### Fixed

- Import dialog not working correctly for many file formats
- Focus-related bugs in sketcher
- Sketches not resizable using drag and drop
- Editing sketch resetting instance's size on canvas
- Distance constraints not selectable or highlighting on hover
- Arcs going in wrong direction in sketch-generated geometry
- Titles from varset not properly escaped
- "Reverse axis" setting affecting G-code output
- Race condition in GRBL serial driver
- Boolean variables in device settings not applied
- Key text in simulation mode not readable in dark mode
- Popover not readable in dark mode in camera alignment dialog
- Laser dot drawn too large
- Varset variables with type Var not subclassed showing as "unsupported"
- Menu not closing when clicking surface
- Missing icons on some Linux distributions
- Race condition in tasker test
- Expression editor not closing when pressing enter
- Sketch not using input parameters
- Varset out of sync after var key rename
- Dragging sketches to surface failing
- Step list drag & drop reordering broken
- Sketches not positioned at center of sketcher surface when re-editing
- Excessive linearization precision causing lag
- Pyright detecting invalid `str` argument in machine dialect definition
- Perpendicular constraint hit detection
- Radius constraints not always recognized
- Constrained geometry not being green after loading sketch

## [0.26] - Parametric 2D Sketcher & Major Performance Upgrades

### Added

- Parametric 2D Sketcher
  - Create 2D geometry with lines, circles, and arcs
  - Constraint system: Coincident, Vertical, Horizontal, Tangent,
    Perpendicular, Point on Line/Shape, Symmetry, Distance, Diameter,
    Radius, Equal Length/Radius
  - Import/Export sketches with parametric constraints preserved
  - Context-aware pie menu for quick tool access
  - Keyboard shortcuts inspired by FreeCAD
  - Full undo/redo support
- Adaptive precision grid based on zoom level
- Machine profile for Makera Carvera Air CNC machine

### Changed

- Default cut mode for vector CAM operations changed to 'Centerline'
- Significant performance optimizations for moving, scaling, rotating geometry
- Significantly reduced memory consumption
- Complex vector files handled more smoothly during toolpath generation

### Fixed

- 3D view and G-code output mirrored when using Y-down machine configuration
- Smoothieware driver issues
- `.dxf` files not visible in import dialog
- Save button incorrectly enabled when workflow is empty

## [0.25] - Import Workflow Overhaul

### Added

- Interactive import dialog with pre-canvas configuration
- Enhanced tracing configuration in import dialog
- Threshold slider for contour operation
- "Object -> Split Workpiece" command
- Groups now have "natural size" property
- Support for configuring custom G-code dialects
- DXF layer support for importing as Groups

### Changed

- Tracing logic improved for transparent images
- Contour operation now processes inner edges before outer edges (configurable)
- Debouncing in processing pipeline for snappier UI

### Fixed

- Geometry incorrectly removed from vector inputs if cut side not "centerline"
- Items scaled down on import even when fitting machine area
- Step box displaying 0% power regardless of actual setting
- Workpiece operations not drawn correctly when grouping/ungrouping
- Imported images displayed with masks
- Notifications not cleared when user begins editing

## [0.24] - 2025-11-07

### Added

- Operation Recipes: Save and reuse operation settings (laser head, speed,
  power, kerf) for specific operation types
- Automatic recipe selection based on operation type, machine, stock material,
  and thickness
- Dedicated profile for xTool D1 Pro with updated start G-code macros (M17 and
  M106)
- Support for configuring specific port numbers for Grbl connections
- Session-based log file system for easier troubleshooting and debugging
- Dialog to display metadata associated with imported images
- Dedicated option to toggle laser on at low power for easy focusing

### Changed

- Step settings dialog now has tabs, separating post-processing settings
- Windows distribution now uses "onedir" installer bundle for faster startup
- Machine branding strings updated from "Xtool" to "xTool" capitalization
- Macros can now be executed directly from main application menu
  (Machine -> Macros)

### Fixed

- Various issues causing application crashes on Windows systems
- Race condition in logging setup that could lead to duplicate log entries
- Workpieces retaining stale operation settings after cut and paste to new layer
- "Remove inner edges" function failing to execute correctly
- Tracebacks when copying elements with tabs
- Incorrect tab placement during shrinkwrap or frame operations
- Segmentation fault during certain import scenarios
- Task bar remaining visible after completing import operations
- Global shortcuts incorrectly captured while editing text fields in workpiece
  properties panel
- Logging reliability by ensuring all logs flushed before application exits

## [0.23.2] - 2025-11-03

### Fixed

- Crashes on Windows due to Gtk 4 API incompatibility
- Crash when sending job due to not running event loop

## [0.23.1] - 2025-11-02

### Added

- New Windows installer
- French translation

### Changed

- RayforgeContext object introduced for future API support
- ArtifactStore refactored

### Fixed

- Git command line briefly opening on Windows startup
- Test suite issues on Windows
- "Reset to natural size" buttons not working for DXF and Ruida imports
- Warning for resizing on import now persistent until dismissed
- Debouncing for some step settings
- Bug where two events in rapid succession could cause stale operations

## [0.23] - 2025-10-25

### Added

- G-code Viewer & Simulator with full playback and synchronized G-code text
- Depth Engraver operation for multi-pass engravings with varying depths
- Shrink Wrap operation for form-fitting contours
- Frame operation for rectangular frames around workpieces
- Material Test Grid tool for finding optimal power and speed settings
- Material Manager in settings for managing material libraries and materials
- Flip & Mirror tools for horizontal/vertical object flipping
- Engraving overscan option for maintaining constant velocity
- Offset and kerf compensation for cutting operations
- Native importers for JPEG, full-color PNG, and BMP files
- Jog controls dialog for manual laser positioning
- Multi-head laser support with step assignment to specific heads
- Cross-hatch fill option for Raster Engraving operation
- Stock material assignment to individual layers
- Machining time estimate in status bar with progress and ETA during execution
- Preferred length unit setting in preferences
- Machine acceleration values in machine profiles
- Snap to grid functionality (Ctrl key while moving/rotating)
- Adjustable rasterizer threshold with undo/redo support
- French translation

### Changed

- Backend almost completely redesigned for performance
- Task manager redesigned around process pool
- Toolpath optimizer significantly faster and enabled by default
- Rendering pipeline uses shared memory for faster data transfer
- Tracing engine switched to vtracer for higher quality
- Stock handling completely redesigned
- "Edge" and "Outline" producers merged into single "Contour" operation
- Higher baud rates supported for serial connections
- GRBL streaming protocol supported

### Fixed

- Dependencies not correctly installed when installing with pip
- 3D view not updating correctly when toggling step visibility
- Contour operation now produces path even if offsetting fails
- Traceback when zooming into large workpiece
- Alignment issues across all importers
- PDF importer clipping direction bug
- SVG importer vector alignment problems

## [0.22] - 2025-10-01

### Added

- Tabbing system for holding parts in place during cutting
  - Flexible configuration (global or per-step)
  - Automatic placement
  - Manual control via context menu
  - Interactive editing with drag handles
- G-code macros and hooks with variable substitution
- Direct SVG vector import option

### Changed

- Work surface panning smoother
- Grouping and ungrouping faster
- Main menu and toolbar reorganized with keyboard shortcuts
- Global preference for UI speed units
- Step settings dialog closes with Esc or Ctrl+W
- Workpieces automatically scaled down if too large for surface
- DXF and Ruida imports pre-split into component parts
- Auto-layouter respects stock boundaries
- PDF importer auto-crops to content

### Fixed

- Select All (Ctrl+A) not selecting groups correctly
- Misleading error for non-existent serial ports
- Task manager warning on shutdown
- Auto-layouter 90° rotation bug

## [0.21] - 2025-09-14

### Added

- Micrometric resolution support for imports and G-code generation
- Configurable decimal places in G-code output
- Stock material area definition
- Device alarm reset button for GRBL machines
- Automatic alarm reset on connection option
- Official PPA for Ubuntu

### Changed

- Snap package now supports GRBL serial port connections
- Camera backend on Linux defaults to V4L2
- New icons for layers
- Workpiece properties panel reorganized
- Connection and device status messages now translated

### Fixed

- 3D editor switch not working
- Device drivers not shutting down correctly on close
- Flickering "RUN" status with GrblSerial devices

## [0.20.2] - 2025-08-20

### Fixed

- ImportError when opening the app

## [0.20.1] - 2025-08-20

### Added

- Context menu on work surface (right-click)

### Changed

- Baud Rate field now a dropdown

### Fixed

- Crash during job execution
- Serial Grbl driver issues
- USB port selection lost when machine disconnected
- Layer list rendering artifacts
- Camera view alignment

## [0.20] - 2025-08-19

### Added

- 3D G-Code Previewer with orbit, pan, zoom controls
- Z Step Down per Pass option
- DXF importer with full geometry support
- Ruida (.rd) importer
- Auto-Layout tool
- Shear tool for skewing workpieces
- Grouping and ungrouping support

### Changed

- ESC key deselects items on canvas
- Smoothing algorithm replaced with more effective version
- Operation generation more efficient

### Fixed

- Invalid G-code when travel speed not set
- On-screen laser dot position slightly misplaced
- Resizing multiple selected objects incorrectly in Y-up mode
- Export Debug Log not working
- Creating new machine from profile failing
- Windows installer issues

## [0.19.1] - 2025-08-08

### Fixed

- Traceback on Windows startup
- Missing icons on Windows

## [0.19] - 2025-08-???

### Added

- New GRBL drivers (Network and Serial Port) with firmware settings UI
- Canvas alignment tools (top, bottom, left, right, center)
- Object distribution tools (horizontal, vertical)
- Ctrl+PageUp/PageDown for moving objects between layers
- Themed icon support for light/dark themes
- Ctrl + < shortcut for machine settings
- Spanish translation

### Changed

- Canvas and camera stream performance improved
- Default camera overlay opacity 20%
- Status icons replaced with modern symbolic icons

### Fixed

- "Remove All Workpieces" button not working
- Worksteps unnecessarily regenerated when unrelated step removed
- "Smoothness" slider not functional
- Canvas grid aspect ratio issue
- Rendering failure with large on-screen dimensions
- Camera toggle not updating canvas immediately
- On-screen laser dot position

## [0.18.4] - 2025-08-04

### Fixed

- Custom postscript used even if disabled
- Machine config lost on startup (race condition)
- Home, Pause and Cancel buttons not working
- Laser dot shown in wrong position when zoomed

## [0.18.3] - 2025-08-03

### Added

- --version CLI flag

### Changed

- Sliders in workstep settings now smoother (debouncing)

### Fixed

- Ops not removed when deleting workpiece
- Performance: removing step no longer re-generates all steps
- Subtitle not showing in driver selection if no driver initially selected

## [0.18.2] - 2025-08-03

### Added

- Experimental GRBL driver

### Fixed

- Performance regression for canvas rendering

### Changed

- Workstep settings dialog can be moved

## [0.18.1] - 2025-08-03

### Fixed

- Camera stream not disabling when switching machines
- Performance regression for canvas rendering

## [0.18] - 2025-08-03

### Added

- Layer support
- Multi-machine support
- Machine profiles (Sculpfun iCube, Other)
- G-code dialects (Marlin, GRBL, Smoothieware)
- Theme preferences with dark mode improvements
- Debug information collection button in machine view

### Changed

- Main window panel redesigned
- Paths more precise with reduced rounding errors
- Smoothing algorithm polished
- Camera rendering speed massively improved

### Fixed

- Boundary alignment for rastering with chunked images
- Smoothing angle threshold up to 179 degrees
- Travel optimizer running when disabled
- Driver description not shown in dropdown subtitle

## [0.17] - 2025-07-28

### Added

- Undo and redo support for all actions
- Main menu at top of window
- Copy, cut, paste support
- Ctrl+D duplicate shortcut
- Multiple selection in canvas
- Select-by-frame support
- Flipped Y-axis support
- About dialog with version info

## [0.16.2] - 2025-07-25

### Fixed

- Non-square work surfaces shown as square (now proper aspect ratio)

## [0.16.1] - 2025-07-25

### Fixed

- Path disappearing when zooming

## [0.16] - 2025-07-25

### Added

- Improved resize tool
- Workpiece rotation
- Path smoothing option in work step dialog
- Better progress bar with status messages
- Progress shown during export operations

### Changed

- Canvas now displays travel move optimization result
- Worksteps processed in parallel
- Larger surfaces supported via tiling (removes 32,000 x 32,000 limit)

### Fixed

- Numerous Windows EXE bugs

## [0.15] - 2025-07-???

### Added

- Camera alignment UI with on-screen editing
- German and Portuguese languages
- Smoothieware support (via Telnet)

### Fixed

- Many Windows EXE bugs, test suite now passes in CI/CD

## [0.14] - 2025-07-12

### Added

- Camera configuration (USB cameras via OpenCV)
- Live feed picture overlay on canvas
- Image settings (white balance, brightness, contrast, transparency)
- Image alignment and de-distortion support

## [0.13] - 2025-07-10

### Added

- GRBL serial connection support
- Workpiece properties panel for precise position and dimensions
- Experimental Windows installer
