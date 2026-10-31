import type { ApiBackend } from "../api/client";
import type {
  Affine2D,
  DocHistory,
  DocTree,
  GeometryResponse,
  ImportAnalysis,
  ImportPlacement,
  ImportResult,
  ItemTransform,
  JobStatus,
  LayerInfo,
  LayerPatch,
  MachineListItem,
  MachineState,
  OpsResponse,
  OpsSegment,
  StepAddRequest,
  StepInfo,
  StepPatch,
  StepSchema,
  StepTypeEntry,
  SystemInfo,
  Vec2,
  ViewportContract,
  WsEvent,
  WorkpieceInfo,
} from "../api/types";

const IDENTITY: Affine2D = [1, 0, 0, 0, 1, 0];

function rectPoints(pos: Vec2, size: { w: number; h: number }, deg: number): number[] {
  const rad = (deg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const hw = size.w / 2;
  const hh = size.h / 2;
  const corners: Array<[number, number]> = [
    [-hw, -hh],
    [hw, -hh],
    [hw, hh],
    [-hw, hh],
    [-hw, -hh],
  ];
  const pts: number[] = [];
  for (const [cx, cy] of corners) {
    pts.push(pos.x + cx * cos - cy * sin, pos.y + cx * sin + cy * cos);
  }
  return pts;
}

function ellipsePoints(
  pos: Vec2,
  rx: number,
  ry: number,
  segments = 40,
): number[] {
  const pts: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts.push(pos.x + rx * Math.cos(a), pos.y + ry * Math.sin(a));
  }
  return pts;
}

function starPoints(pos: Vec2, outer: number): number[] {
  const pts: number[] = [];
  for (let i = 0; i <= 10; i++) {
    const r = i % 2 === 0 ? outer : outer * 0.45;
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
    pts.push(pos.x + r * Math.cos(a), pos.y + r * Math.sin(a));
  }
  return pts;
}

interface MockWorkpiece extends WorkpieceInfo {
  build(): number[];
}

const mockWorkpieces: MockWorkpiece[] = [
  {
    uid: "wp-frame",
    name: "Outer frame",
    pos_mm: { x: 225, y: 170 },
    size_mm: { w: 220, h: 140 },
    angle_deg: 0,
    geometry_revision: 1,
    transform_revision: 1,
    build: () =>
      rectPoints(
        mockWorkpieces[0]!.pos_mm,
        mockWorkpieces[0]!.size_mm,
        mockWorkpieces[0]!.angle_deg,
      ),
  },
  {
    uid: "wp-circle",
    name: "Boss circle",
    pos_mm: { x: 60, y: 90 },
    size_mm: { w: 60, h: 60 },
    angle_deg: 0,
    geometry_revision: 1,
    transform_revision: 1,
    build: () =>
      ellipsePoints(
        mockWorkpieces[1]!.pos_mm,
        mockWorkpieces[1]!.size_mm.w / 2,
        mockWorkpieces[1]!.size_mm.h / 2,
      ),
  },
  {
    uid: "wp-star",
    name: "Star badge",
    pos_mm: { x: 340, y: 60 },
    size_mm: { w: 60, h: 60 },
    angle_deg: 0,
    geometry_revision: 1,
    transform_revision: 1,
    build: () =>
      starPoints(mockWorkpieces[2]!.pos_mm, mockWorkpieces[2]!.size_mm.w / 2),
  },
];

const mockLayers: LayerInfo[] = [
  {
    uid: "layer-cut",
    name: "Cut",
    visible: true,
    color: "#4d9fff",
    is_active: true,
    wcs: "G54",
    rotary_enabled: false,
    steps: [
      {
        uid: "step-cut",
        step_type: "Contour",
        typelabel: "Contour",
        name: "Cut outline",
        enabled: true,
        workpiece_uids: ["wp-frame", "wp-circle", "wp-star"],
        params: { power: 80, speed: 3000, passes: 1, kerf: 0.1 },
      },
    ],
    workpieces: [],
  },
  {
    uid: "layer-mark",
    name: "Mark",
    visible: true,
    color: "#ff9f40",
    is_active: false,
    wcs: "G54",
    rotary_enabled: false,
    steps: [
      {
        uid: "step-mark",
        step_type: "Line",
        typelabel: "Line fill",
        name: "Mark badges",
        enabled: false,
        workpiece_uids: ["wp-star"],
        params: { power: 30, speed: 5000 },
      },
    ],
    workpieces: [],
  },
];

mockLayers[0]!.workpieces = [mockWorkpieces[0]!, mockWorkpieces[1]!];
mockLayers[1]!.workpieces = [mockWorkpieces[2]!];

const viewport: ViewportContract = {
  panel_orientation: "native",
  world_to_panel: IDENTITY,
  x_axis_right: false,
  y_axis_down: false,
  x_axis_negative: false,
  y_axis_negative: false,
  extents: { width_mm: 400, height_mm: 300 },
  margins: { left: 10, right: 10, top: 10, bottom: 10, front: 20, back: 20 },
  workarea: { x_min: 10, y_min: 20, x_max: 390, y_max: 280 },
  workarea_origin_panel: { x: 10, y: 20 },
  work_origin_panel: { x: 10, y: 20 },
  active_wcs: { name: "G54", label: "WCS 1", offset_panel: { x: 0, y: 0 } },
  beam_position_panel: null,
  soft_limits: { x_min: 0, y_min: 0, x_max: 400, y_max: 300 },
  soft_limits_panel: { x: 0, y: 0, w: 400, h: 300 },
  nogo_zones: [
    [300, 200, 380, 200, 380, 280, 300, 280, 300, 200],
  ],
  rotary_active: false,
};

const machineList: MachineListItem[] = [
  {
    id: "mock-grbl",
    name: "Workbench GRBL",
    driver: "grbl",
    connected: true,
    is_active: true,
  },
  {
    id: "mock-dummy",
    name: "Basement Diode",
    driver: "dummy",
    connected: false,
    is_active: false,
  },
];

let treeRev = 7;
let undoAvailable = false;

function findWorkpiece(uid: string): MockWorkpiece | undefined {
  return mockWorkpieces.find((wp) => wp.uid === uid);
}

function findStep(uid: string): StepInfo | undefined {
  for (const layer of mockLayers) {
    const step = layer.steps.find((s) => s.uid === uid);
    if (step) return step;
  }
  return undefined;
}

const mockStepTypes: StepTypeEntry[] = [
  { step_type: "Contour", typelabel: "Contour", icon: null },
  { step_type: "Line", typelabel: "Line fill", icon: null },
  { step_type: "Raster", typelabel: "Raster fill", icon: null },
];

const mockStepSchemas: Record<string, StepSchema> = {
  Contour: {
    step_type: "Contour",
    groups: [
      {
        title: "Laser",
        fields: [
          {
            key: "power",
            type: "number",
            kind: "slider",
            label: "Power",
            description: null,
            default: 80,
            min: 0,
            max: 100,
            step: 1,
            options: null,
            unit: null,
            optional: false,
          },
          {
            key: "speed",
            type: "number",
            kind: "speed",
            label: "Speed",
            description: null,
            default: 3000,
            min: 1,
            max: 60000,
            step: 10,
            options: null,
            unit: "mm/min",
            optional: false,
          },
        ],
      },
    ],
  },
};

function mockLayerPatch(uid: string, patch: LayerPatch): void {
  const layer = mockLayers.find((l) => l.uid === uid);
  if (!layer) throw new Error(`mock: unknown layer ${uid}`);
  if (patch.visible !== undefined) layer.visible = patch.visible;
  if (patch.name !== undefined) layer.name = patch.name;
  if (patch.color !== undefined) layer.color = patch.color;
  if (patch.wcs !== undefined) layer.wcs = patch.wcs;
  if (patch.rotary_enabled !== undefined) {
    layer.rotary_enabled = patch.rotary_enabled;
  }
}

let mockUidSeq = 0;

function nextMockUid(prefix: string): string {
  mockUidSeq += 1;
  return `${prefix}-mock-${mockUidSeq}`;
}

const machineState: MachineState = {
  status: "IDLE",
  connected: true,
  machine_pos: { x: 10, y: 20, z: 0 },
  work_pos: { x: 0, y: 0, z: 0 },
  wco: { x: 10, y: 20, z: 0 },
  beam_position_panel: null,
  error: null,
  feed_rate: 0,
  spindle_speed: 0,
  active_wcs: "G54",
};

const jobStatus: JobStatus = {
  active: false,
  phase: "idle",
  progress_fraction: 0,
  eta_seconds: null,
  elapsed_seconds: null,
};

/** Listeners registered by startMockEvents (see publishMockEvent). */
const mockListeners = new Set<(ev: WsEvent) => void>();

function publishMockEvent(ev: WsEvent): void {
  for (const listener of mockListeners) listener(ev);
}

function publishMachineState(): void {
  publishMockEvent({ type: "machine", data: { ...snapshotMachineState() } });
}

function publishJobStatus(): void {
  publishMockEvent({ type: "job", data: { ...jobStatus } });
}

function snapshotMachineState(): MachineState {
  const beam = machineState.beam_position_panel;
  return {
    ...machineState,
    machine_pos: machineState.machine_pos && { ...machineState.machine_pos },
    work_pos: machineState.work_pos && { ...machineState.work_pos },
    wco: machineState.wco && { ...machineState.wco },
    beam_position_panel: beam && { ...beam },
    error: machineState.error && { ...machineState.error },
  };
}

function requireConnectedMachine(): MachineListItem {
  const machine = machineList.find((m) => m.is_active);
  if (!machine || !machine.connected) {
    throw new Error("Machine is not connected");
  }
  return machine;
}

const beamCenter = { x: 200, y: 150 };

function applyBeam(x: number, y: number): void {
  beamCenter.x = x;
  beamCenter.y = y;
  machineState.beam_position_panel = { x, y };
  machineState.machine_pos = { x, y, z: 0 };
  const wcoX = machineState.wco?.x ?? 0;
  const wcoY = machineState.wco?.y ?? 0;
  machineState.work_pos = { x: x - wcoX, y: y - wcoY, z: 0 };
}

const JOG_DELTAS: Record<string, [number, number]> = {
  "+x": [1, 0],
  "-x": [-1, 0],
  "+y": [0, 1],
  "-y": [0, -1],
};

let jobTimer: number | null = null;

function stopJobSimulation(): void {
  if (jobTimer !== null) {
    window.clearInterval(jobTimer);
    jobTimer = null;
  }
}

function startJobSimulation(): void {
  stopJobSimulation();
  jobStatus.active = true;
  jobStatus.phase = "running";
  jobStatus.progress_fraction = 0;
  jobStatus.eta_seconds = 20;
  jobStatus.elapsed_seconds = 0;
  publishJobStatus();
  jobTimer = window.setInterval(() => {
    if (jobStatus.phase === "paused") return;
    jobStatus.progress_fraction = Math.min(
      1,
      jobStatus.progress_fraction + 0.02,
    );
    jobStatus.eta_seconds = Math.max(
      0,
      Math.round(20 * (1 - jobStatus.progress_fraction)),
    );
    jobStatus.elapsed_seconds = Math.round(20 * jobStatus.progress_fraction);
    if (jobStatus.progress_fraction >= 1) {
      stopJobSimulation();
      jobStatus.active = false;
      jobStatus.phase = "idle";
      machineState.status = "IDLE";
    }
    publishJobStatus();
  }, 200);
}

function polylineBounds(points: number[]) {
  let xMin = Infinity;
  let yMin = Infinity;
  let xMax = -Infinity;
  let yMax = -Infinity;
  for (let i = 0; i < points.length; i += 2) {
    xMin = Math.min(xMin, points[i]!);
    xMax = Math.max(xMax, points[i]!);
    yMin = Math.min(yMin, points[i + 1]!);
    yMax = Math.max(yMax, points[i + 1]!);
  }
  return { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax };
}

const MOCK_CUT_RATE = 1800;
const MOCK_TRAVEL_RATE = 5000;

function polylineLength(points: number[]): number {
  let length = 0;
  for (let i = 2; i + 1 < points.length; i += 2) {
    length += Math.hypot(
      points[i]! - points[i - 2]!,
      points[i + 1]! - points[i - 1]!,
    );
  }
  return length;
}

function travelTo(points: number[]): OpsSegment {
  return {
    points: [points[0]!, points[1]!],
    power: 0,
    rate: MOCK_TRAVEL_RATE,
    cutting: false,
  };
}

function cutSegment(points: number[], power: number): OpsSegment {
  return {
    points,
    power,
    rate: MOCK_CUT_RATE,
    cutting: true,
  };
}

/** Dev-only synthetic ops artifact matching the documented /ops shape. */
function mockOps(): OpsResponse {
  const segments: OpsSegment[] = [];
  const add = (
    points: number[],
    power: number,
    cutting: boolean,
  ): void => {
    if (cutting) {
      segments.push(cutSegment(points, power));
    } else {
      segments.push(travelTo(points));
    }
  };
  add(mockWorkpieces[0]!.build(), 1.0, true);
  add(mockWorkpieces[1]!.build().slice(0, 2), 1.0, false);
  add(mockWorkpieces[1]!.build(), 0.6, true);
  add(mockWorkpieces[2]!.build().slice(0, 2), 0.6, false);
  add(mockWorkpieces[2]!.build(), 0.25, true);
  let distance = 0;
  let seconds = 0;
  for (const segment of segments) {
    const length = polylineLength(segment.points);
    distance += length;
    const rate = (segment.cutting ? MOCK_CUT_RATE : MOCK_TRAVEL_RATE) / 60;
    seconds += length / rate;
  }
  return {
    segments,
    estimated_seconds: Number(seconds.toFixed(2)),
    distance_mm: Number(distance.toFixed(2)),
    generation_id: "mock-gen-1",
  };
}

function currentTree(): DocTree {
  return {
    rev: treeRev,
    layers: mockLayers.map((layer) => ({
      ...layer,
      workpieces: layer.workpieces.map((wp) => ({ ...wp })),
      steps: layer.steps.map((step) => ({ ...step })),
    })),
    assets_count: 2,
    stock_items: [],
  };
}

function currentHistory(): DocHistory {
  return {
    can_undo: undoAvailable,
    can_redo: false,
    undo_labels: undoAvailable ? ["Move Star badge"] : [],
    redo_labels: [],
    dirty: undoAvailable,
  };
}

export function createMockBackend(): ApiBackend {
  return {
    async getSystemInfo(): Promise<SystemInfo> {
      return {
        name: "rayforge",
        version: "0.1.0.dev0",
        api_version: "1",
        python_version: "3.13.1",
        machine_count: machineList.length,
      };
    },
    async getViewport() {
      return { ...viewport, world_to_panel: [...IDENTITY] as Affine2D };
    },
    async getDocTree() {
      return currentTree();
    },
    async getItemGeometry(uid, _tolerance = 0.05): Promise<GeometryResponse> {
      await delay(120);
      const wp = findWorkpiece(uid);
      if (!wp) throw new Error(`mock: unknown item ${uid}`);
      const points = wp.build();
      return { rev: treeRev, polylines: [points], outline: polylineBounds(points) };
    },
    async getItemTexture() {
      return null;
    },
    async getDocHistory() {
      return currentHistory();
    },
    async postUndo() {
      undoAvailable = false;
    },
    async postRedo() {},
    async postNewDoc() {
      treeRev++;
      undoAvailable = false;
    },
    async putItemTransform(uid, transform: ItemTransform) {
      const wp = findWorkpiece(uid);
      if (!wp) throw new Error(`mock: unknown item ${uid}`);
      wp.pos_mm = { ...transform.pos_mm };
      wp.size_mm = { ...transform.size_mm };
      wp.angle_deg = transform.angle_deg;
      wp.transform_revision++;
      wp.geometry_revision++;
      treeRev++;
      undoAvailable = true;
    },
    async putLayer(uid, patch: LayerPatch) {
      mockLayerPatch(uid, patch);
      treeRev++;
    },
    async postLayer() {
      const uid = nextMockUid("layer");
      const layer: LayerInfo = {
        uid,
        name: `Layer ${mockLayers.length + 1}`,
        visible: true,
        color: "#3ddc84",
        is_active: true,
        wcs: null,
        rotary_enabled: false,
        steps: [],
        workpieces: [],
      };
      for (const other of mockLayers) other.is_active = false;
      mockLayers.push(layer);
      treeRev++;
      return { ...layer };
    },
    async deleteLayer(uid) {
      if (mockLayers.length <= 1) {
        throw new Error("mock: cannot delete the last layer");
      }
      const index = mockLayers.findIndex((l) => l.uid === uid);
      if (index < 0) throw new Error(`mock: unknown layer ${uid}`);
      mockLayers.splice(index, 1);
      treeRev++;
    },
    async activateLayer(uid) {
      const layer = mockLayers.find((l) => l.uid === uid);
      if (!layer) throw new Error(`mock: unknown layer ${uid}`);
      for (const other of mockLayers) other.is_active = other.uid === uid;
      treeRev++;
    },
    async reorderLayers(uids) {
      const byUid = new Map(mockLayers.map((l) => [l.uid, l]));
      const ordered: LayerInfo[] = [];
      for (const uid of uids) {
        const layer = byUid.get(uid);
        if (!layer) throw new Error(`mock: unknown layer ${uid}`);
        ordered.push(layer);
      }
      for (const layer of mockLayers) {
        if (!ordered.includes(layer)) ordered.push(layer);
      }
      mockLayers.splice(0, mockLayers.length, ...ordered);
      treeRev++;
    },
    async getStepTypes() {
      return mockStepTypes.map((t) => ({ ...t }));
    },
    async getStepSchema(uid) {
      const step = findStep(uid);
      const schema = mockStepSchemas[step?.step_type ?? ""];
      if (!schema) throw new Error(`mock: no schema for step ${uid}`);
      return schema;
    },
    async postStep(layerUid, body: StepAddRequest) {
      const layer = mockLayers.find((l) => l.uid === layerUid);
      if (!layer) throw new Error(`mock: unknown layer ${layerUid}`);
      const step = {
        uid: nextMockUid("step"),
        step_type: body.step_type,
        typelabel: body.step_type,
        name: body.name ?? body.step_type,
        enabled: true,
        workpiece_uids: layer.workpieces.map((wp) => wp.uid),
        params: {},
      };
      layer.steps.push(step);
      treeRev++;
      return { ...step, layer_uid: layerUid };
    },
    async putStep(uid, body: StepPatch) {
      const step = findStep(uid);
      if (!step) throw new Error(`mock: unknown step ${uid}`);
      if (body.name !== undefined) step.name = body.name;
      if (body.params) step.params = { ...step.params, ...body.params };
      treeRev++;
      return { ...step };
    },
    async deleteStep(uid) {
      for (const layer of mockLayers) {
        const index = layer.steps.findIndex((s) => s.uid === uid);
        if (index >= 0) {
          layer.steps.splice(index, 1);
          treeRev++;
          return;
        }
      }
      throw new Error(`mock: unknown step ${uid}`);
    },
    async deleteItems(uids) {
      for (const layer of mockLayers) {
        layer.workpieces = layer.workpieces.filter(
          (wp) => !uids.includes(wp.uid),
        );
      }
      treeRev++;
    },
    async duplicateItems(uids) {
      const created: string[] = [];
      for (const layer of mockLayers) {
        for (const wp of [...layer.workpieces]) {
          if (!uids.includes(wp.uid)) continue;
          const copy: WorkpieceInfo = {
            ...wp,
            uid: nextMockUid("wp"),
            name: `${wp.name} copy`,
            pos_mm: { x: wp.pos_mm.x + 10, y: wp.pos_mm.y + 10 },
          };
          layer.workpieces.push(copy);
          created.push(copy.uid);
        }
      }
      treeRev++;
      return { created };
    },
    async analyzeImport(file): Promise<ImportAnalysis> {
      const name = file.name.toLowerCase();
      const supported = name.endsWith(".svg") || name.endsWith(".png");
      return {
        supported,
        action: supported ? "interactive_config" : "unsupported",
        importer: supported ? "MockImporter" : null,
        features: supported ? ["BITMAP_TRACING"] : [],
        manifest: supported
          ? {
              title: file.name,
              natural_size_mm: [100, 80],
              layers: [],
              warnings: [],
              errors: [],
            }
          : null,
      };
    },
    async importFile(file, placement: ImportPlacement = {}): Promise<ImportResult> {
      const active = mockLayers.find((l) => l.is_active) ?? mockLayers[0]!;
      const wp: WorkpieceInfo = {
        uid: nextMockUid("wp"),
        name: file.name,
        pos_mm: { x: placement.x ?? 100, y: placement.y ?? 100 },
        size_mm: { w: 40, h: 30 },
        angle_deg: 0,
        geometry_revision: 1,
        transform_revision: 1,
      };
      active.workpieces.push(wp);
      treeRev++;
      return { created: [wp.uid], layer_uid: active.uid };
    },
    async openProject(file) {
      const uid = nextMockUid("layer");
      const layer: LayerInfo = {
        uid,
        name: file.name.replace(/\.[^.]+$/, ""),
        visible: true,
        color: "#4d9fff",
        is_active: true,
        wcs: null,
        rotary_enabled: false,
        steps: [],
        workpieces: [],
      };
      mockLayers.splice(0, mockLayers.length, layer);
      treeRev++;
      return currentTree();
    },
    async saveProject() {
      return new Blob(["mock project"], { type: "application/octet-stream" });
    },
    async exportGcode() {
      return "; mock gcode\n";
    },
    async flipItems(uids, axis) {
      for (const wp of mockWorkpieces) {
        if (!uids.includes(wp.uid)) continue;
        wp.angle_deg =
          axis === "horizontal" ? -wp.angle_deg : -wp.angle_deg + 180;
        wp.transform_revision++;
        wp.geometry_revision++;
      }
      treeRev++;
      undoAvailable = true;
    },
    async alignItems(uids, mode) {
      const targets = mockWorkpieces.filter((wp) => uids.includes(wp.uid));
      if (targets.length < 2) return;
      const horizontal =
        mode === "left" ||
        mode === "right" ||
        mode === "center_horizontal";
      const size = (wp: WorkpieceInfo) =>
        horizontal ? wp.size_mm.w : wp.size_mm.h;
      const low = (wp: WorkpieceInfo) =>
        horizontal ? wp.pos_mm.x : wp.pos_mm.y;
      const setLow = (wp: WorkpieceInfo, value: number) => {
        if (horizontal) wp.pos_mm.x = value;
        else wp.pos_mm.y = value;
      };
      const lows = targets.map(low);
      const highs = targets.map((wp) => low(wp) + size(wp));
      if (mode === "left" || mode === "top") {
        const edge = Math.min(...lows);
        for (const wp of targets) setLow(wp, edge);
      } else if (mode === "right" || mode === "bottom") {
        const edge = Math.max(...highs);
        for (const wp of targets) setLow(wp, edge - size(wp));
      } else {
        const centers = targets.map((wp) => low(wp) + size(wp) / 2);
        const center =
          centers.reduce((sum, c) => sum + c, 0) / centers.length;
        for (const wp of targets) setLow(wp, center - size(wp) / 2);
      }
      for (const wp of targets) {
        wp.transform_revision++;
        wp.geometry_revision++;
      }
      treeRev++;
      undoAvailable = true;
    },
    async getLayouts() {
      return [{ id: "pixel-perfect", label: "Auto Layout" }];
    },
    async runLayout() {
      let cursorX = 20;
      let cursorY = 60;
      for (const layer of mockLayers) {
        for (const wp of layer.workpieces) {
          wp.pos_mm = { x: cursorX, y: cursorY };
          wp.angle_deg = 0;
          wp.transform_revision++;
          cursorX += wp.size_mm.w + 10;
          if (cursorX > 360) {
            cursorX = 20;
            cursorY += wp.size_mm.h + 10;
          }
        }
      }
      treeRev++;
      undoAvailable = true;
      publishMockEvent({ type: "doc", kind: "history", uids: [], revisions: { tree: treeRev } });
    },
    async getMachines() {
      return machineList.map((m) => ({ ...m }));
    },
    async getSettings() {
      return {
        theme: "dark",
        language: null,
        import_dpi: 96,
        cache_budget_bytes: 2147483648,
        watch_source_files: true,
        check_for_app_updates: true,
        ops_color_mode: "laser",
        unit_preferences: { length: "mm", speed: "mm/min" },
      };
    },
    async putSettings(values) {
      return {
        theme: "dark",
        language: null,
        import_dpi: 96,
        cache_budget_bytes: 2147483648,
        watch_source_files: true,
        check_for_app_updates: true,
        ops_color_mode: "laser",
        unit_preferences: { length: "mm", speed: "mm/min" },
        ...values,
      };
    },
    async getMachineSettings(id: string) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      return {
        name: "Workbench GRBL",
        axis_extents: [400, 300],
        origin: "bottom_left",
        panel_orientation: "native",
        work_margins: [10, 10, 10, 10],
        reverse_x_axis: false,
        reverse_y_axis: false,
        reverse_z_axis: false,
        soft_limits_enabled: true,
        max_travel_speed: 3000,
        max_cut_speed: 1000,
        acceleration: 1000,
        jog_speed: 1000,
        dialect_uid: "grbl",
        heads: [
          {
            uid: "mock-head",
            name: "Laser Head",
            type: "LaserHead",
            max_power: 1000,
            frame_power: 0,
            focus_power: 0,
            frame_speed: 0,
            frame_repeat_count: 20,
            frame_corner_pause: 0,
            spot_size_mm: [0.1, 0.1],
            focal_distance: 0,
            pointer_offset_enabled: false,
            pointer_offset_x: 0,
            pointer_offset_y: 0,
          },
        ],
        rotary_modules: [],
        rotary_enabled_default: false,
        nogo_zones: [
          {
            uid: "mock-zone",
            name: "Clamp",
            shape: "rect",
            params: { x: 300, y: 200, w: 80, h: 80 },
            enabled: true,
          },
        ],
        coordinate_systems: [
          { name: "G54", label: "WCS 1", offset: [0, 0, 0] },
          { name: "G55", label: "", offset: [0, 0, 0] },
        ],
      };
    },
    async putMachineSettings(id: string, values) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      return {
        name: "Workbench GRBL",
        axis_extents: [400, 300],
        origin: "bottom_left",
        panel_orientation: "native",
        work_margins: [10, 10, 10, 10],
        reverse_x_axis: false,
        reverse_y_axis: false,
        reverse_z_axis: false,
        soft_limits_enabled: true,
        max_travel_speed: 3000,
        max_cut_speed: 1000,
        acceleration: 1000,
        jog_speed: 1000,
        dialect_uid: "grbl",
        heads: [],
        rotary_modules: [],
        rotary_enabled_default: false,
        nogo_zones: [],
        coordinate_systems: [],
        ...values,
      };
    },
    async getDrivers() {
      return [
        { name: "NoDeviceDriver", maturity: "stable", maturity_warning: "" },
        {
          name: "GrblSerialDriver",
          maturity: "stable",
          maturity_warning: "",
        },
      ];
    },
    async getDriverSetup(id: string) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      return {
        driver: "GrblSerialDriver",
        setup: {
          title: "Setup",
          fields: [
            {
              key: "port",
              type: "string",
              kind: "input",
              label: "Port",
              description: null,
              default: "/dev/ttyUSB0",
              min: null,
              max: null,
              step: null,
              options: null,
              unit: null,
              optional: false,
            },
            {
              key: "baudrate",
              type: "number",
              kind: "input",
              label: "Baud Rate",
              description: null,
              default: 115200,
              min: 9600,
              max: 250000,
              step: 9600,
              options: null,
              unit: null,
              optional: false,
            },
          ],
        },
        values: { port: "/dev/ttyUSB0", baudrate: 115200 },
        precheck_error: null,
        device_settings: null,
      };
    },
    async putDriverSetup(id, values) {
      await this.getDriverSetup(id);
      return this.getDriverSetup(id).then((payload) => ({
        ...payload,
        values: { ...payload.values, ...values },
      }));
    },
    async putDriver(id, driver) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      if (driver !== "NoDeviceDriver" && driver !== "GrblSerialDriver") {
        throw new Error(`mock: unknown driver ${driver}`);
      }
      return this.getDriverSetup(id);
    },
    async putHeads(id, heads) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      return heads.map((patch, index) => ({
        uid: patch.uid,
        name: `Laser Head ${index + 1}`,
        type: "LaserHead",
        max_power: patch.max_power ?? 1000,
        frame_power: patch.frame_power ?? 0,
        focus_power: patch.focus_power ?? 0,
        frame_speed: patch.frame_speed ?? 0,
        frame_repeat_count: patch.frame_repeat_count ?? 20,
        frame_corner_pause: patch.frame_corner_pause ?? 0,
        spot_size_mm: patch.spot_size_mm ?? [0.1, 0.1],
        focal_distance: patch.focal_distance ?? 0,
        pointer_offset_enabled: patch.pointer_offset_enabled ?? false,
        pointer_offset_x: patch.pointer_offset_x ?? 0,
        pointer_offset_y: patch.pointer_offset_y ?? 0,
      }));
    },
    async getMacros(id) {
      if (!machineList.some((m) => m.id === id)) {
        throw new Error(`mock: unknown machine ${id}`);
      }
      return [
        { uid: "mock-macro-1", name: "Home", code: ["G28"], enabled: true },
      ];
    },
    async createMacro(id, macro) {
      await this.getMacros(id);
      return {
        uid: `mock-macro-${Date.now()}`,
        name: macro.name,
        code: macro.code.split("\n"),
        enabled: macro.enabled,
      };
    },
    async updateMacro(id, macroUid, patch) {
      const macros = await this.getMacros(id);
      const found = macros.find((m) => m.uid === macroUid);
      if (!found) throw new Error("mock: unknown macro");
      return {
        ...found,
        ...patch,
        code: patch.code ? patch.code.split("\n") : found.code,
      };
    },
    async deleteMacro(id, macroUid) {
      const macros = await this.getMacros(id);
      if (!macros.some((m) => m.uid === macroUid)) {
        throw new Error("mock: unknown macro");
      }
    },
    async putWcs(id, slot, offset, label) {
      await this.getMachineSettings(id);
      return { name: slot, label: label ?? "", offset };
    },
    async createNogoZone(id, zone) {
      await this.getMachineSettings(id);
      return {
        uid: `mock-zone-${Date.now()}`,
        name: zone.name ?? "No-Go Zone",
        shape: zone.shape,
        params: zone.params,
        enabled: zone.enabled ?? true,
      };
    },
    async deleteNogoZone(id, zoneUid) {
      const settings = await this.getMachineSettings(id);
      if (!settings.nogo_zones.some((z) => z.uid === zoneUid)) {
        throw new Error("mock: unknown zone");
      }
    },
    async putRotaryModule(id, moduleUid, patch) {
      const settings = await this.getMachineSettings(id);
      const found = settings.rotary_modules.find((m) => m.uid === moduleUid);
      if (!found) throw new Error("mock: unknown rotary module");
      return [{ ...found, ...patch }];
    },
    async getMachineState(): Promise<MachineState> {
      return snapshotMachineState();
    },
    async connectMachine(id: string): Promise<void> {
      const machine = machineList.find((m) => m.id === id);
      if (!machine) throw new Error(`mock: unknown machine ${id}`);
      machine.connected = true;
      if (machine.is_active) {
        machineState.connected = true;
        publishMachineState();
      }
    },
    async disconnectMachine(id: string): Promise<void> {
      const machine = machineList.find((m) => m.id === id);
      if (!machine) throw new Error(`mock: unknown machine ${id}`);
      machine.connected = false;
      if (machine.is_active) {
        machineState.connected = false;
        machineState.status = "UNKNOWN";
        publishMachineState();
      }
    },
    async jog(direction, distanceMm): Promise<void> {
      requireConnectedMachine();
      const delta = JOG_DELTAS[direction];
      if (!delta) {
        throw new Error(`mock: invalid jog direction ${direction}`);
      }
      const pos = machineState.machine_pos;
      if (!pos) throw new Error("mock: beam position unknown");
      applyBeam(
        Math.min(400, Math.max(0, (pos.x ?? 0) + delta[0] * distanceMm)),
        Math.min(300, Math.max(0, (pos.y ?? 0) + delta[1] * distanceMm)),
      );
      publishMachineState();
    },
    async homeMachine(): Promise<void> {
      requireConnectedMachine();
      machineState.status = "IDLE";
      applyBeam(0, 0);
      publishMachineState();
    },
    async moveTo(x, y): Promise<void> {
      requireConnectedMachine();
      applyBeam(x, y);
      publishMachineState();
    },
    async setPower(percent: number): Promise<void> {
      requireConnectedMachine();
      machineState.spindle_speed = Math.round(percent * 100);
      publishMachineState();
    },
    async setFocusPower(): Promise<void> {
      requireConnectedMachine();
    },
    async runRaw(): Promise<void> {
      requireConnectedMachine();
    },
    async clearAlarm(): Promise<void> {
      machineState.status = "IDLE";
      machineState.error = null;
      publishMachineState();
    },
    async setWorkOriginHere(): Promise<void> {
      requireConnectedMachine();
      const px = machineState.machine_pos?.x ?? 0;
      const py = machineState.machine_pos?.y ?? 0;
      machineState.wco = { x: px, y: py, z: 0 };
      machineState.work_pos = { x: 0, y: 0, z: 0 };
      viewport.active_wcs = {
        name: machineState.active_wcs ?? "G54",
        label: `WCS ${machineState.active_wcs ?? "G54"}`,
        offset_panel: { x: px, y: py },
      };
      publishMachineState();
      publishMockEvent({ type: "viewport", data: { ...viewport } });
    },
    async selectWcs(slot: string): Promise<void> {
      requireConnectedMachine();
      machineState.active_wcs = slot;
      viewport.active_wcs = {
        name: slot,
        label: `WCS ${slot}`,
        offset_panel: { ...viewport.active_wcs.offset_panel },
      };
      publishMockEvent({ type: "viewport", data: { ...viewport } });
    },
    async syncWcs(): Promise<void> {
      requireConnectedMachine();
    },
    async frame(): Promise<void> {
      requireConnectedMachine();
    },
    async getJobStatus(): Promise<JobStatus> {
      return { ...jobStatus };
    },
    async getOps(): Promise<OpsResponse> {
      await delay(60);
      return mockOps();
    },
    async startJob(): Promise<void> {
      requireConnectedMachine();
      if (jobStatus.active) {
        throw new Error("A job is already running");
      }
      const hasContent = mockLayers.some((l) => l.workpieces.length > 0);
      if (!hasContent) {
        throw new Error("Document has no content to run");
      }
      machineState.status = "RUN";
      publishMachineState();
      startJobSimulation();
    },
    async holdJob(hold: boolean): Promise<void> {
      requireConnectedMachine();
      if (!jobStatus.active) throw new Error("No job is running");
      jobStatus.phase = hold ? "paused" : "running";
      machineState.status = hold ? "HOLD" : "RUN";
      publishMachineState();
      publishJobStatus();
    },
    async cancelJob(): Promise<void> {
      requireConnectedMachine();
      stopJobSimulation();
      jobStatus.active = false;
      jobStatus.phase = "idle";
      jobStatus.progress_fraction = 0;
      jobStatus.eta_seconds = null;
      machineState.status = "IDLE";
      publishMachineState();
      publishJobStatus();
    },
    getWsUrl(): string {
      return "mock://ws";
    },
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface MockEventConnection {
  close(): void;
}

/** Emits contract-shaped WS events so the UI is fully live in mock mode. */
export function startMockEvents(onEvent: (ev: WsEvent) => void): MockEventConnection {
  mockListeners.add(onEvent);
  let t = 0;
  const machineTimer = window.setInterval(() => {
    t += 0.2;
    const beam: Vec2 = {
      x: beamCenter.x + 20 * Math.sin(t / 6),
      y: beamCenter.y + 15 * Math.sin(t / 3 + 0.7),
    };
    machineState.beam_position_panel = beam;
    machineState.machine_pos = { x: beam.x, y: beam.y, z: 0 };
    machineState.work_pos = {
      x: beam.x - (machineState.wco?.x ?? 0),
      y: beam.y - (machineState.wco?.y ?? 0),
      z: 0,
    };
    onEvent({ type: "machine", data: { ...snapshotMachineState() } });
  }, 200);
  const notifyTimer = window.setInterval(() => {
    onEvent({
      type: "notify",
      level: "info",
      message: "Mock mode: simulated machine telemetry",
    });
  }, 15000);
  onEvent({ type: "job", data: { ...jobStatus } });
  return {
    close() {
      mockListeners.delete(onEvent);
      window.clearInterval(machineTimer);
      window.clearInterval(notifyTimer);
      stopJobSimulation();
    },
  };
}
