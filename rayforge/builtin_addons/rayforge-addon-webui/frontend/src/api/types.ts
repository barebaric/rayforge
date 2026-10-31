/**
 * Types mirroring the documented /api/v1 contract. This module is the
 * single place to replace hand-written types with generated OpenAPI
 * types later on.
 */

export interface SystemInfo {
  name: string;
  version: string;
  api_version: string;
  python_version: string;
  machine_count: number;
}

export type PanelOrientation =
  | "native"
  | "rotated_left"
  | "rotated_right";

/** 2x3 affine matrix: x' = a*x + b*y + c, y' = d*x + e*y + f. */
export type Affine2D = [number, number, number, number, number, number];

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** A position report whose components may individually be unknown. */
export interface Vec3Nullable {
  x: number | null;
  y: number | null;
  z: number | null;
}

export interface Size2 {
  w: number;
  h: number;
}

export interface ViewportExtents {
  width_mm: number;
  height_mm: number;
}

export interface ViewportMargins {
  left: number;
  right: number;
  top: number;
  bottom: number;
  front: number;
  back: number;
}

export interface WorkareaRect {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/** Rect as [x, y, w, h] — the shape the API uses for rects. */
export interface RectXYWH {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ActiveWcs {
  name: string;
  label: string;
  offset_panel: Vec2;
}

export interface ViewportContract {
  panel_orientation: PanelOrientation;
  world_to_panel: Affine2D;
  x_axis_right: boolean;
  y_axis_down: boolean;
  x_axis_negative: boolean;
  y_axis_negative: boolean;
  extents: ViewportExtents;
  margins: ViewportMargins;
  workarea: WorkareaRect;
  workarea_origin_panel: Vec2;
  work_origin_panel: Vec2;
  active_wcs: ActiveWcs;
  beam_position_panel: Vec2 | null;
  soft_limits: Record<string, number> | null;
  soft_limits_panel: RectXYWH | null;
  /** No-go zones as closed PANEL-space polygons [x, y, x, y, ...]. */
  nogo_zones: number[][];
  rotary_active: boolean;
}

export interface WorkpieceInfo {
  uid: string;
  name: string;
  pos_mm: Vec2;
  size_mm: Size2;
  angle_deg: number;
  geometry_revision: number;
  transform_revision: number;
}

/** A stock material instance from GET /doc/tree. */
export interface StockInfo {
  uid: string;
  name: string;
  visible: boolean;
  color: string | null;
  pos_mm: Vec2;
  size_mm: Size2;
  angle_deg: number;
  geometry_revision: number;
  transform_revision: number;
}

export interface StepInfo {
  uid: string;
  step_type: string;
  typelabel?: string;
  name: string;
  enabled: boolean;
  visible?: boolean;
  workpiece_uids: string[];
  params: Record<string, unknown>;
}

export interface LayerInfo {
  uid: string;
  name: string;
  visible: boolean;
  color: string;
  is_active: boolean;
  wcs: string | null;
  rotary_enabled?: boolean;
  steps: StepInfo[];
  workpieces: WorkpieceInfo[];
}

export interface DocTree {
  rev: number;
  layers: LayerInfo[];
  assets_count: number;
  stock_items: StockInfo[];
}

/** Editable application settings from GET/PUT /settings. */
export interface AppSettings {
  theme: string;
  language: string | null;
  import_dpi: number;
  cache_budget_bytes: number;
  watch_source_files: boolean;
  check_for_app_updates: boolean;
  ops_color_mode: string;
  unit_preferences: Record<string, string>;
}

/** Editable machine geometry/motion fields from GET/PUT /machines/{id}/settings. */
export interface HeadInfo {
  uid: string;
  name: string;
  type: string;
  max_power?: number;
  frame_power?: number;
  focus_power?: number;
  frame_speed?: number;
  frame_repeat_count?: number;
  frame_corner_pause?: number;
  spot_size_mm?: number[];
  focal_distance?: number;
  pointer_offset_enabled?: boolean;
  pointer_offset_x?: number;
  pointer_offset_y?: number;
}

export interface RotaryModuleInfo {
  uid: string;
  name: string;
  axis: string;
  mode: string;
  mm_per_rotation: number;
  default_diameter: number;
  max_workpiece_length: number;
  reverse_axis: boolean;
}

export interface NogoZoneInfo {
  uid: string;
  name: string;
  shape: string;
  params: Record<string, number>;
  enabled: boolean;
}

export interface WcsInfo {
  name: string;
  label: string;
  offset: number[];
}

export interface MachineSettings {
  name: string;
  axis_extents: number[];
  origin: string;
  panel_orientation: string;
  work_margins: number[];
  reverse_x_axis: boolean;
  reverse_y_axis: boolean;
  reverse_z_axis: boolean;
  soft_limits_enabled: boolean;
  max_travel_speed: number;
  max_cut_speed: number;
  acceleration: number;
  jog_speed: number;
  dialect_uid: string | null;
  heads: HeadInfo[];
  rotary_modules: RotaryModuleInfo[];
  rotary_enabled_default: boolean;
  nogo_zones: NogoZoneInfo[];
  coordinate_systems: WcsInfo[];
}

export interface DriverInfo {
  name: string;
  maturity: string;
  maturity_warning: string;
}

/** Patch for one laser head in PUT /machines/{id}/heads. */
export interface HeadPatch extends Partial<HeadInfo> {
  uid: string;
}

export interface DriverSetupPayload {
  driver: string;
  setup: SchemaGroup;
  values: Record<string, unknown>;
  precheck_error: string | null;
  device_settings: SchemaGroup[] | null;
}

export interface MacroInfo {
  uid: string;
  name: string;
  code: string[];
  enabled: boolean;
}

export interface GeometryOutline {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/** Polylines in WORLD space (mm, Y-up): [[x, y, x, y, ...], ...] */
export interface GeometryResponse {
  rev: number;
  polylines: number[][];
  outline: GeometryOutline;
}

export interface DocHistory {
  can_undo: boolean;
  can_redo: boolean;
  undo_labels: string[];
  redo_labels: string[];
  dirty: boolean;
}

export interface MachineListItem {
  id: string;
  name: string;
  driver: string;
  connected: boolean;
  is_active: boolean;
}

/** Driver-reported error attached to the machine state. */
export interface MachineError {
  code: number;
  title: string;
  description: string;
}

export interface MachineState {
  status: string;
  connected: boolean;
  machine_pos: Vec3Nullable | null;
  work_pos: Vec3Nullable | null;
  wco: Vec3Nullable | null;
  beam_position_panel: Vec2 | null;
  error: MachineError | null;
  feed_rate: number | null;
  spindle_speed: number | null;
  active_wcs: string | null;
}

export interface JobStatus {
  active: boolean;
  phase: string;
  progress_fraction: number;
  eta_seconds: number | null;
  elapsed_seconds: number | null;
}

/** One toolpath segment; points are flat WORLD mm [x, y, x, y, ...]. */
export interface OpsSegment {
  points: number[];
  power: number;
  rate: number | null;
  cutting: boolean;
}

/** GET /ops response: world-space segments of the current job artifact. */
export interface OpsResponse {
  segments: OpsSegment[];
  estimated_seconds: number | null;
  distance_mm: number;
  generation_id: string;
}

export interface ItemTransform {
  pos_mm: Vec2;
  size_mm: Size2;
  angle_deg: number;
}

export type FlipAxis = "horizontal" | "vertical";

export type AlignMode =
  | "left"
  | "right"
  | "top"
  | "bottom"
  | "center_horizontal"
  | "center_vertical";

/** A registered layout strategy from GET /layouts. */
export interface LayoutOption {
  id: string;
  label: string;
}

export interface LayerPatch {
  visible?: boolean;
  name?: string;
  color?: string;
  wcs?: string | null;
  rotary_enabled?: boolean;
}

export interface StepPatch {
  name?: string;
  params?: Record<string, unknown>;
}

export interface StepAddRequest {
  step_type: string;
  name?: string;
}

/** A registered step type from GET /steps. */
export interface StepTypeEntry {
  step_type: string;
  typelabel: string;
  icon?: string | null;
  required_machine_caps?: string[];
}

export type SchemaFieldType = "boolean" | "number" | "string";

export type SchemaFieldKind =
  | "input"
  | "switch"
  | "slider"
  | "length"
  | "speed"
  | "angle"
  | "textarea"
  | "choice";

/** One form-field descriptor from the VarSet schema. */
export interface SchemaField {
  key: string;
  type: SchemaFieldType;
  kind: SchemaFieldKind;
  label: string;
  description?: string | null;
  default?: unknown;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: Array<{ value: string; label: string }> | null;
  unit?: string | null;
  optional?: boolean;
}

export interface SchemaGroup {
  title: string;
  fields: SchemaField[];
}

export interface StepSchema {
  step_type: string;
  groups: SchemaGroup[];
}

export interface ImportManifestLayer {
  id: string;
  name: string;
  default_active: boolean;
  feature_count: number;
}

export interface ImportManifest {
  title: string;
  natural_size_mm: [number, number] | null;
  layers: ImportManifestLayer[];
  warnings: string[];
  errors: string[];
}

export type ImportAction =
  | "direct_load"
  | "interactive_config"
  | "unsupported";

export interface ImportAnalysis {
  supported: boolean;
  action: ImportAction;
  importer: string | null;
  features: string[];
  manifest: ImportManifest | null;
}

export interface ImportPlacement {
  spec?: Record<string, unknown>;
  x?: number;
  y?: number;
}

export interface ImportResult {
  created: string[];
  layer_uid: string | null;
}

export interface DocWsEvent {
  type: "doc";
  kind: string;
  uids: string[];
  revisions: Record<string, number>;
}

export type WsEvent =
  | DocWsEvent
  | { type: "viewport"; data: ViewportContract }
  | { type: "machine"; data: MachineState }
  | { type: "job"; data: JobStatus }
  | { type: "notify"; level: string; message: string }
  | { type: "snapshot"; data: unknown };
