import type {
  Affine2D,
  ActiveWcs,
  DocTree,
  GeometryResponse,
  ImportAction,
  ImportAnalysis,
  JobStatus,
  LayerInfo,
  LayoutOption,
  MachineError,
  MachineListItem,
  MachineState,
  OpsResponse,
  OpsSegment,
  RectXYWH,
  SchemaField,
  SchemaGroup,
  StepInfo,
  StepSchema,
  StepTypeEntry,
  StockInfo,
  Vec2,
  Vec3Nullable,
  ViewportContract,
  WorkareaRect,
  WorkpieceInfo,
} from "./types";

/**
 * Maps server payloads onto the documented /api/v1 contract shapes.
 * The canonical forms pass through unchanged; older or partially
 * migrated field layouts (arrays instead of objects, alternate field
 * names) are folded in so the rest of the app only sees the contract.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

function asRecord(value: Any): Record<string, Any> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, Any>)
    : null;
}

function num(value: Any, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : typeof value === "string" && value !== "" && Number.isFinite(Number(value))
      ? Number(value)
      : fallback;
}

function vec2(value: Any, fallback: Vec2 = { x: 0, y: 0 }): Vec2 {
  if (Array.isArray(value) && value.length >= 2) {
    return { x: num(value[0]), y: num(value[1]) };
  }
  const rec = asRecord(value);
  if (rec) return { x: num(rec.x), y: num(rec.y) };
  return fallback;
}

function numOrNull(value: Any): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function vec3Nullable(value: Any): Vec3Nullable | null {
  if (Array.isArray(value) && value.length >= 3) {
    return {
      x: numOrNull(value[0]),
      y: numOrNull(value[1]),
      z: numOrNull(value[2]),
    };
  }
  const rec = asRecord(value);
  if (rec) {
    return {
      x: numOrNull(rec.x),
      y: numOrNull(rec.y),
      z: numOrNull(rec.z),
    };
  }
  return null;
}

function affine(value: Any): Affine2D {
  if (Array.isArray(value) && value.length === 6) {
    return [num(value[0]), num(value[1]), num(value[2]), num(value[3]), num(value[4]), num(value[5])];
  }
  if (Array.isArray(value) && value.length === 2) {
    const row0 = Array.isArray(value[0]) ? value[0] : null;
    const row1 = Array.isArray(value[1]) ? value[1] : null;
    if (row0 && row1) {
      return [
        num(row0[0]),
        num(row0[1]),
        num(row0[2] ?? 0),
        num(row1[0]),
        num(row1[1]),
        num(row1[2] ?? 0),
      ];
    }
  }
  return [1, 0, 0, 0, 1, 0];
}

function workarea(value: Any, fallback: WorkareaRect): WorkareaRect {
  if (Array.isArray(value) && value.length >= 4) {
    const [x, y, w, h] = value.map(num) as [number, number, number, number];
    return { x_min: x, y_min: y, x_max: x + w, y_max: y + h };
  }
  const rec = asRecord(value);
  if (rec) {
    return {
      x_min: num(rec.x_min),
      y_min: num(rec.y_min),
      x_max: num(rec.x_max),
      y_max: num(rec.y_max),
    };
  }
  return fallback;
}

function minMaxRect(value: Any): WorkareaRect | null {
  if (Array.isArray(value) && value.length >= 4) {
    return {
      x_min: num(value[0]),
      y_min: num(value[1]),
      x_max: num(value[2]),
      y_max: num(value[3]),
    };
  }
  const rec = asRecord(value);
  if (rec) {
    return {
      x_min: num(rec.x_min),
      y_min: num(rec.y_min),
      x_max: num(rec.x_max),
      y_max: num(rec.y_max),
    };
  }
  return null;
}

function xywhRect(value: Any): RectXYWH | null {
  if (Array.isArray(value) && value.length >= 4) {
    return {
      x: num(value[0]),
      y: num(value[1]),
      w: num(value[2]),
      h: num(value[3]),
    };
  }
  const rec = asRecord(value);
  if (rec) {
    return { x: num(rec.x), y: num(rec.y), w: num(rec.w), h: num(rec.h) };
  }
  return null;
}

function extentsOf(v: Record<string, Any>): {
  width_mm: number;
  height_mm: number;
} {
  const rec = asRecord(v.extents);
  if (rec) {
    return { width_mm: num(rec.width_mm, 200), height_mm: num(rec.height_mm, 200) };
  }
  if (Array.isArray(v.extents) && v.extents.length >= 2) {
    return { width_mm: num(v.extents[0], 200), height_mm: num(v.extents[1], 200) };
  }
  return { width_mm: 200, height_mm: 200 };
}

export function normalizeViewport(raw: Any): ViewportContract {
  const v = asRecord(raw) ?? {};
  const extents = extentsOf(v);
  const fallbackWorkarea: WorkareaRect = {
    x_min: 0,
    y_min: 0,
    x_max: extents.width_mm,
    y_max: extents.height_mm,
  };
  const margins = asRecord(v.margins);
  const softLimits = minMaxRect(v.soft_limits);
  const softLimitsPanel = xywhRect(v.soft_limits_panel);
  const wcsRec = asRecord(v.active_wcs) ?? asRecord(v.wcs);

  const activeWcs: ActiveWcs | null = wcsRec
    ? {
        name: typeof wcsRec.name === "string" ? wcsRec.name : "G54",
        label:
          typeof wcsRec.label === "string"
            ? wcsRec.label
            : typeof wcsRec.name === "string"
              ? wcsRec.name
              : "G54",
        offset_panel: vec2(wcsRec.offset_panel),
      }
    : null;

  return {
    panel_orientation:
      v.panel_orientation === "rotated_left" ||
      v.panel_orientation === "rotated_right" ||
      v.panel_orientation === "native"
        ? v.panel_orientation
        : v.orientation === "rotated_left" || v.orientation === "rotated_right"
          ? v.orientation
          : "native",
    world_to_panel: affine(v.world_to_panel),
    x_axis_right: v.x_axis_right !== undefined ? Boolean(v.x_axis_right) : true,
    y_axis_down: Boolean(v.y_axis_down),
    x_axis_negative: Boolean(v.x_axis_negative),
    y_axis_negative: Boolean(v.y_axis_negative),
    extents,
    margins: {
      left: num(margins?.left),
      right: num(margins?.right),
      top: num(margins?.top),
      bottom: num(margins?.bottom),
      front: num(margins?.front),
      back: num(margins?.back),
    },
    workarea: workarea(v.workarea, fallbackWorkarea),
    workarea_origin_panel: vec2(
      v.workarea_origin_panel ?? v.workarea_origin,
    ),
    work_origin_panel: vec2(
      v.work_origin_panel ?? v.work_origin ?? v.workarea_origin,
    ),
    active_wcs: activeWcs ?? {
      name: "G54",
      label: "G54",
      offset_panel: { x: 0, y: 0 },
    },
    beam_position_panel:
      v.beam_position_panel !== undefined && v.beam_position_panel !== null
        ? vec2(v.beam_position_panel)
        : v.beam
          ? vec2(v.beam)
          : null,
    soft_limits: (softLimits as Record<string, number> | null),
    soft_limits_panel: softLimitsPanel,
    nogo_zones: (Array.isArray(v.nogo_zones) ? v.nogo_zones : [])
      .filter((poly: Any) => Array.isArray(poly))
      .map((poly: Any[]) => poly.map((n) => num(n))),
    rotary_active: Boolean(v.rotary_active),
  };
}

export function normalizeLayouts(raw: Any): LayoutOption[] {
  const list: Any[] = Array.isArray(raw)
    ? raw
    : Array.isArray(asRecord(raw)?.layouts)
      ? asRecord(raw)!.layouts
      : [];
  return list
    .map((entry: Any): LayoutOption => {
      const rec = asRecord(entry) ?? {};
      return {
        id: String(rec.id ?? ""),
        label: String(rec.label ?? rec.id ?? ""),
      };
    })
    .filter((layout) => layout.id !== "");
}

function normalizeError(raw: Any): MachineError | null {
  const rec = asRecord(raw);
  if (!rec) return null;
  return {
    code: num(rec.code),
    title: typeof rec.title === "string" ? rec.title : "Error",
    description:
      typeof rec.description === "string" ? rec.description : "",
  };
}

export function normalizeMachineState(raw: Any): MachineState {
  const v = asRecord(raw) ?? {};
  const beam =
    v.beam_position_panel !== undefined && v.beam_position_panel !== null
      ? vec2(v.beam_position_panel)
      : v.beam
        ? vec2(v.beam)
        : null;
  return {
    status: typeof v.status === "string" ? v.status : "unknown",
    connected: Boolean(v.connected),
    machine_pos: vec3Nullable(v.machine_pos),
    work_pos: vec3Nullable(v.work_pos),
    wco: vec3Nullable(v.wco),
    beam_position_panel: beam,
    error: normalizeError(v.error),
    feed_rate: numOrNull(v.feed_rate),
    spindle_speed: numOrNull(v.spindle_speed),
    active_wcs: typeof v.active_wcs === "string" ? v.active_wcs : null,
  };
}

export function normalizeMachines(raw: Any): MachineListItem[] {
  const list: Any[] = Array.isArray(raw)
    ? raw
    : Array.isArray(asRecord(raw)?.machines)
      ? asRecord(raw)!.machines
      : [];
  return list.map((entry) => {
    const rec = asRecord(entry) ?? {};
    return {
      id: String(rec.id ?? ""),
      name: String(rec.name ?? rec.id ?? ""),
      driver: rec.driver === null || rec.driver === undefined ? "" : String(rec.driver),
      connected: Boolean(rec.connected),
      is_active: Boolean(rec.is_active ?? rec.active),
    };
  });
}

export function normalizeJobStatus(raw: Any): JobStatus {
  const v = asRecord(raw) ?? {};
  const inner = asRecord(v.job);
  const job = inner ?? v;
  const active =
    v.active !== undefined
      ? Boolean(v.active)
      : job.phase === "running" || job.phase === "paused";
  const phase =
    typeof job.phase === "string" && job.phase !== ""
      ? job.phase
      : active
        ? "running"
        : "idle";
  return {
    active,
    phase,
    progress_fraction: num(job.progress_fraction),
    eta_seconds:
      job.eta_seconds === null || job.eta_seconds === undefined
        ? null
        : num(job.eta_seconds),
    elapsed_seconds:
      job.elapsed_seconds === null || job.elapsed_seconds === undefined
        ? null
        : num(job.elapsed_seconds),
  };
}

function normalizeStep(raw: Any): StepInfo {
  const step = asRecord(raw) ?? {};
  return {
    uid: String(step.uid ?? ""),
    step_type: String(step.step_type ?? step.type ?? "step"),
    typelabel:
      typeof step.typelabel === "string" ? step.typelabel : undefined,
    name: String(step.name ?? "Step"),
    enabled: step.enabled !== undefined ? Boolean(step.enabled) : true,
    visible:
      step.visible !== undefined ? Boolean(step.visible) : undefined,
    workpiece_uids: Array.isArray(step.workpiece_uids)
      ? step.workpiece_uids.map(String)
      : [],
    params: asRecord(step.params) ?? {},
  };
}

function size2(value: Any): { w: number; h: number } {
  if (Array.isArray(value) && value.length >= 2) {
    return { w: num(value[0]), h: num(value[1]) };
  }
  const rec = asRecord(value);
  return { w: num(rec?.w), h: num(rec?.h) };
}

function normalizeWorkpiece(raw: Any): WorkpieceInfo {
  const wp = asRecord(raw) ?? {};
  return {
    uid: String(wp.uid ?? ""),
    name: String(wp.name ?? "Workpiece"),
    pos_mm: vec2(wp.pos_mm),
    size_mm: size2(wp.size_mm),
    angle_deg: num(wp.angle_deg),
    geometry_revision: num(wp.geometry_revision, 0),
    transform_revision: num(wp.transform_revision, 0),
  };
}

export function normalizeDocTree(raw: Any): DocTree {
  const v = asRecord(raw) ?? {};
  const layersRaw: Any[] = Array.isArray(v.layers) ? v.layers : [];
  const activeLayerUid =
    typeof v.active_layer_uid === "string" ? v.active_layer_uid : null;

  const workpieceById = new Map<string, WorkpieceInfo>();
  for (const wp of Array.isArray(v.workpieces) ? v.workpieces : []) {
    const normalized = normalizeWorkpiece(wp);
    workpieceById.set(normalized.uid, normalized);
  }
  const stepById = new Map<string, StepInfo>();
  for (const step of Array.isArray(v.steps) ? v.steps : []) {
    const normalized = normalizeStep(step);
    stepById.set(normalized.uid, normalized);
  }

  const layers = layersRaw.map((entry): LayerInfo => {
    const l = asRecord(entry) ?? {};
    const embeddedWorkpieces = Array.isArray(l.workpieces)
      ? l.workpieces.map(normalizeWorkpiece)
      : [];
    const embeddedSteps = Array.isArray(l.steps)
      ? l.steps.map(normalizeStep)
      : [];
    const contentUids: string[] = Array.isArray(l.content_uids)
      ? l.content_uids.map(String)
      : [];
    const stepUids: string[] = Array.isArray(l.step_uids)
      ? l.step_uids.map(String)
      : [];
    const workpieces = embeddedWorkpieces.length
      ? embeddedWorkpieces
      : contentUids
          .map((uid) => workpieceById.get(uid))
          .filter((wp): wp is WorkpieceInfo => wp !== undefined);
    const steps = embeddedSteps.length
      ? embeddedSteps
      : stepUids
          .map((uid) => stepById.get(uid))
          .filter((step): step is StepInfo => step !== undefined);
    return {
      uid: String(l.uid ?? ""),
      name: String(l.name ?? "Layer"),
      visible: l.visible !== undefined ? Boolean(l.visible) : true,
      color: typeof l.color === "string" ? l.color : "#4d9fff",
      is_active: Boolean(l.is_active) || l.uid === activeLayerUid,
      wcs: typeof l.wcs === "string" ? l.wcs : null,
      rotary_enabled:
        l.rotary_enabled !== undefined ? Boolean(l.rotary_enabled) : undefined,
      steps,
      workpieces,
    };
  });
  const stocks: StockInfo[] = (Array.isArray(v.stock_items)
    ? v.stock_items
    : []
  ).map((entry: Any): StockInfo => {
    const s = asRecord(entry) ?? {};
    return {
      uid: String(s.uid ?? ""),
      name: String(s.name ?? "Stock"),
      visible: s.visible !== undefined ? Boolean(s.visible) : true,
      color:
        typeof s.color === "string" || s.color === null
          ? (s.color as string | null)
          : null,
      pos_mm: vec2(s.pos_mm),
      size_mm: size2(s.size_mm),
      angle_deg: num(s.angle_deg),
      geometry_revision: num(s.geometry_revision, 0),
      transform_revision: num(s.transform_revision, 0),
    };
  });
  return {
    rev: num(v.rev, -1),
    assets_count: num(v.assets_count, workpieceById.size),
    layers,
    stock_items: stocks,
  };
}

export function normalizeGeometry(raw: Any): GeometryResponse {
  const v = asRecord(raw) ?? {};
  const outlineRec = asRecord(v.outline);
  return {
    rev: num(v.rev, 0),
    polylines: (Array.isArray(v.polylines) ? v.polylines : [])
      .filter((p: Any) => Array.isArray(p))
      .map((p: Any) => (p as Any[]).map((n) => num(n))),
    outline: {
      x_min: num(outlineRec?.x_min),
      y_min: num(outlineRec?.y_min),
      x_max: num(outlineRec?.x_max),
      y_max: num(outlineRec?.y_max),
    },
  };
}

const SCHEMA_KINDS: SchemaField["kind"][] = [
  "input",
  "switch",
  "slider",
  "length",
  "speed",
  "angle",
  "textarea",
  "choice",
];

function normalizeSchemaField(raw: Any): SchemaField {
  const f = asRecord(raw) ?? {};
  const type: SchemaField["type"] =
    f.type === "boolean" || f.type === "number"
      ? f.type
      : "string";
  const kind = SCHEMA_KINDS.includes(f.kind) ? f.kind : "input";
  const options = Array.isArray(f.options)
    ? f.options
        .map((opt: Any) => {
          const rec = asRecord(opt) ?? {};
          return {
            value: String(rec.value ?? ""),
            label: String(rec.label ?? rec.value ?? ""),
          };
        })
        .filter((opt) => opt.value !== "")
    : null;
  const field: SchemaField = {
    key: String(f.key ?? ""),
    type,
    kind,
    label: String(f.label ?? f.key ?? ""),
    description:
      typeof f.description === "string" ? f.description : null,
    default: f.default,
    min: f.min === null || f.min === undefined ? null : num(f.min),
    max: f.max === null || f.max === undefined ? null : num(f.max),
    step: f.step === null || f.step === undefined ? null : num(f.step),
    options,
    unit: typeof f.unit === "string" ? f.unit : null,
    optional: Boolean(f.optional),
  };
  return field;
}

export function normalizeStepSchema(raw: Any): StepSchema {
  const v = asRecord(raw) ?? {};
  const groups: SchemaGroup[] = (Array.isArray(v.groups) ? v.groups : []).map(
    (entry: Any): SchemaGroup => {
      const g = asRecord(entry) ?? {};
      return {
        title: String(g.title ?? ""),
        fields: (Array.isArray(g.fields) ? g.fields : []).map(
          normalizeSchemaField,
        ),
      };
    },
  );
  return {
    step_type: String(v.step_type ?? ""),
    groups,
  };
}

export function normalizeStepTypeEntry(raw: Any): StepTypeEntry {
  const v = asRecord(raw) ?? {};
  return {
    step_type: String(v.step_type ?? ""),
    typelabel: String(v.typelabel ?? v.step_type ?? ""),
    icon: typeof v.icon === "string" ? v.icon : null,
    required_machine_caps: Array.isArray(v.required_machine_caps)
      ? v.required_machine_caps.map(String)
      : [],
  };
}

const IMPORT_ACTIONS: ImportAction[] = [
  "direct_load",
  "interactive_config",
  "unsupported",
];

export function normalizeImportAnalysis(raw: Any): ImportAnalysis {
  const v = asRecord(raw) ?? {};
  const action: ImportAction = IMPORT_ACTIONS.includes(v.action)
    ? v.action
    : "unsupported";
  const manifestRec = asRecord(v.manifest);
  return {
    supported: Boolean(v.supported),
    action,
    importer: typeof v.importer === "string" ? v.importer : null,
    features: Array.isArray(v.features) ? v.features.map(String) : [],
    manifest: manifestRec
      ? {
          title: String(manifestRec.title ?? ""),
          natural_size_mm: Array.isArray(manifestRec.natural_size_mm)
            ? [num(manifestRec.natural_size_mm[0]), num(manifestRec.natural_size_mm[1])]
            : null,
          layers: (Array.isArray(manifestRec.layers)
            ? manifestRec.layers
            : []
          ).map((entry: Any) => {
            const l = asRecord(entry) ?? {};
            return {
              id: String(l.id ?? ""),
              name: String(l.name ?? ""),
              default_active: Boolean(l.default_active),
              feature_count: num(l.feature_count, 0),
            };
          }),
          warnings: Array.isArray(manifestRec.warnings)
            ? manifestRec.warnings.map(String)
            : [],
          errors: Array.isArray(manifestRec.errors)
            ? manifestRec.errors.map(String)
            : [],
        }
      : null,
  };
}

/** Folds a points payload into flat [x, y, x, y, ...] coordinates. */
function flattenPoints(value: Any): number[] {
  if (!Array.isArray(value)) return [];
  const first = value[0];
  if (typeof first === "number" || typeof first === "string") {
    return value.map((n) => num(n));
  }
  const out: number[] = [];
  for (const entry of value) {
    if (Array.isArray(entry) && entry.length >= 2) {
      out.push(num(entry[0]), num(entry[1]));
      continue;
    }
    const rec = asRecord(entry);
    if (rec) out.push(num(rec.x), num(rec.y));
  }
  return out;
}

export function normalizeOps(raw: Any): OpsResponse {
  const v = asRecord(raw) ?? {};
  const segments: OpsSegment[] = (Array.isArray(v.segments) ? v.segments : [])
    .map((entry: Any): OpsSegment => {
      const s = asRecord(entry) ?? {};
      return {
        points: flattenPoints(s.points),
        power: num(s.power),
        rate: numOrNull(s.rate),
        cutting: Boolean(s.cutting),
      };
    })
    .filter(
      (segment) =>
        segment.points.length >= 4 ||
        (!segment.cutting && segment.points.length >= 2),
    );
  return {
    segments,
    estimated_seconds: numOrNull(v.estimated_seconds),
    distance_mm: num(v.distance_mm),
    generation_id: String(v.generation_id ?? ""),
  };
}
