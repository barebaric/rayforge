import {
  normalizeDocTree,
  normalizeGeometry,
  normalizeImportAnalysis,
  normalizeJobStatus,
  normalizeLayouts,
  normalizeMachineState,
  normalizeMachines,
  normalizeOps,
  normalizeStepSchema,
  normalizeStepTypeEntry,
  normalizeViewport,
} from "./normalize";
import type {
  AlignMode,
  AppSettings,
  DocHistory,
  DocTree,
  DriverInfo,
  DriverSetupPayload,
  FlipAxis,
  GeometryResponse,
  HeadInfo,
  HeadPatch,
  ImportAnalysis,
  ImportPlacement,
  ImportResult,
  ItemTransform,
  JobStatus,
  LayerPatch,
  LayoutOption,
  MachineListItem,
  MachineSettings,
  MachineState,
  MacroInfo,
  NogoZoneInfo,
  OpsResponse,
  RotaryModuleInfo,
  StepAddRequest,
  StepPatch,
  StepSchema,
  StepTypeEntry,
  SystemInfo,
  ViewportContract,
  WcsInfo,
} from "./types";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Transport-agnostic description of the API surface. The HTTP client
 * implements it against the real backend; the dev mock implements it
 * in-memory.
 */
export interface ApiBackend {
  getSystemInfo(): Promise<SystemInfo>;
  getViewport(): Promise<ViewportContract>;
  getDocTree(): Promise<DocTree>;
  getItemGeometry(uid: string, tolerance?: number): Promise<GeometryResponse>;
  getItemTexture(uid: string, rev: number): Promise<string | null>;
  getDocHistory(): Promise<DocHistory>;
  postUndo(): Promise<void>;
  postRedo(): Promise<void>;
  postNewDoc(): Promise<void>;
  putItemTransform(uid: string, transform: ItemTransform): Promise<void>;
  flipItems(uids: string[], axis: FlipAxis): Promise<void>;
  alignItems(uids: string[], mode: AlignMode): Promise<void>;
  getLayouts(): Promise<LayoutOption[]>;
  runLayout(strategy: string, params?: Record<string, number>): Promise<void>;
  getSettings(): Promise<AppSettings>;
  putSettings(values: Partial<AppSettings>): Promise<AppSettings>;
  getMachineSettings(id: string): Promise<MachineSettings>;
  putMachineSettings(
    id: string,
    values: Partial<MachineSettings>,
  ): Promise<MachineSettings>;
  getDrivers(): Promise<DriverInfo[]>;
  getDriverSetup(id: string): Promise<DriverSetupPayload>;
  putDriverSetup(
    id: string,
    values: Record<string, unknown>,
  ): Promise<DriverSetupPayload>;
  putDriver(id: string, driver: string): Promise<DriverSetupPayload>;
  putHeads(id: string, heads: HeadPatch[]): Promise<HeadInfo[]>;
  getMacros(id: string): Promise<MacroInfo[]>;
  createMacro(
    id: string,
    macro: { name: string; code: string; enabled: boolean },
  ): Promise<MacroInfo>;
  updateMacro(
    id: string,
    macroUid: string,
    patch: { name?: string; code?: string; enabled?: boolean },
  ): Promise<MacroInfo>;
  deleteMacro(id: string, macroUid: string): Promise<void>;
  putWcs(
    id: string,
    slot: string,
    offset: number[],
    label?: string,
  ): Promise<WcsInfo>;
  createNogoZone(
    id: string,
    zone: {
      name?: string;
      shape: string;
      params: Record<string, number>;
      enabled?: boolean;
    },
  ): Promise<NogoZoneInfo>;
  deleteNogoZone(id: string, zoneUid: string): Promise<void>;
  putRotaryModule(
    id: string,
    moduleUid: string,
    patch: {
      name?: string;
      default_diameter?: number;
      mm_per_rotation?: number;
      reverse_axis?: boolean;
    },
  ): Promise<RotaryModuleInfo[]>;
  putLayer(uid: string, patch: LayerPatch): Promise<void>;
  postLayer(): Promise<unknown>;
  deleteLayer(uid: string): Promise<void>;
  activateLayer(uid: string): Promise<void>;
  reorderLayers(uids: string[]): Promise<void>;
  getStepTypes(): Promise<StepTypeEntry[]>;
  getStepSchema(uid: string): Promise<StepSchema>;
  postStep(layerUid: string, body: StepAddRequest): Promise<unknown>;
  putStep(uid: string, body: StepPatch): Promise<unknown>;
  deleteStep(uid: string): Promise<void>;
  deleteItems(uids: string[]): Promise<void>;
  duplicateItems(uids: string[]): Promise<{ created: string[] }>;
  analyzeImport(file: File): Promise<ImportAnalysis>;
  importFile(file: File, placement?: ImportPlacement): Promise<ImportResult>;
  openProject(file: File): Promise<DocTree>;
  saveProject(): Promise<Blob>;
  exportGcode(): Promise<string>;
  getMachines(): Promise<MachineListItem[]>;
  getMachineState(): Promise<MachineState>;
  connectMachine(id: string): Promise<void>;
  disconnectMachine(id: string): Promise<void>;
  jog(direction: string, distanceMm: number, speed?: number): Promise<void>;
  homeMachine(axes?: string[]): Promise<void>;
  moveTo(x: number, y: number, z?: number, speed?: number): Promise<void>;
  setPower(percent: number): Promise<void>;
  setFocusPower(percent: number): Promise<void>;
  runRaw(gcode: string): Promise<void>;
  clearAlarm(): Promise<void>;
  setWorkOriginHere(axes?: string[]): Promise<void>;
  selectWcs(slot: string): Promise<void>;
  syncWcs(): Promise<void>;
  frame(): Promise<void>;
  getJobStatus(): Promise<JobStatus>;
  getOps(tolerance?: number): Promise<OpsResponse>;
  startJob(): Promise<void>;
  holdJob(hold: boolean): Promise<void>;
  cancelJob(): Promise<void>;
  getWsUrl(): string;
}

const API_BASE = "/api/v1";

export class ApiClient implements ApiBackend {
  private readonly token: string | null;

  constructor() {
    this.token =
      typeof window === "undefined"
        ? null
        : new URLSearchParams(window.location.search).get("token");
  }

  getSystemInfo(): Promise<SystemInfo> {
    return this.get("/system/info");
  }

  getViewport(): Promise<ViewportContract> {
    return this.get<unknown>("/viewport").then(normalizeViewport);
  }

  getDocTree(): Promise<DocTree> {
    return this.get<unknown>("/doc/tree").then(normalizeDocTree);
  }

  getItemGeometry(uid: string, tolerance = 0.05): Promise<GeometryResponse> {
    const t = encodeURIComponent(tolerance);
    return this.get<unknown>(`/items/${uid}/geometry?tolerance=${t}`).then(
      normalizeGeometry,
    );
  }

  async getItemTexture(uid: string, rev: number): Promise<string | null> {
    const res = await fetch(
      this.url(`/items/${uid}/texture.png?rev=${rev}`),
      { headers: this.headers() },
    );
    if (!res.ok) return null;
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }

  getDocHistory(): Promise<DocHistory> {
    return this.get("/doc/history");
  }

  postUndo(): Promise<void> {
    return this.post("/doc/undo");
  }

  postRedo(): Promise<void> {
    return this.post("/doc/redo");
  }

  postNewDoc(): Promise<void> {
    return this.post("/doc/new");
  }

  putItemTransform(uid: string, transform: ItemTransform): Promise<void> {
    return this.put(`/items/${uid}/transform`, {
      pos_mm: [transform.pos_mm.x, transform.pos_mm.y],
      size_mm: [transform.size_mm.w, transform.size_mm.h],
      angle_deg: transform.angle_deg,
    });
  }

  flipItems(uids: string[], axis: FlipAxis): Promise<void> {
    return this.postJson("/items/flip", { uids, axis }).then(() => undefined);
  }

  alignItems(uids: string[], mode: AlignMode): Promise<void> {
    return this.postJson("/items/align", { uids, mode }).then(
      () => undefined,
    );
  }

  getLayouts(): Promise<LayoutOption[]> {
    return this.get<unknown>("/layouts").then(normalizeLayouts);
  }

  runLayout(
    strategy: string,
    params?: Record<string, number>,
  ): Promise<void> {
    return this.postJson("/layout", {
      strategy,
      ...(params ? { params } : {}),
    }).then(() => undefined);
  }

  getSettings(): Promise<AppSettings> {
    return this.get("/settings");
  }

  putSettings(values: Partial<AppSettings>): Promise<AppSettings> {
    return this.putJson("/settings", { values });
  }

  getMachineSettings(id: string): Promise<MachineSettings> {
    return this.get(
      `/machines/${encodeURIComponent(id)}/settings`,
    );
  }

  putMachineSettings(
    id: string,
    values: Partial<MachineSettings>,
  ): Promise<MachineSettings> {
    return this.putJson(`/machines/${encodeURIComponent(id)}/settings`, {
      values,
    });
  }

  getDrivers(): Promise<DriverInfo[]> {
    return this.get("/drivers");
  }

  getDriverSetup(id: string): Promise<DriverSetupPayload> {
    return this.get(`/machines/${encodeURIComponent(id)}/driver-setup`);
  }

  putDriverSetup(
    id: string,
    values: Record<string, unknown>,
  ): Promise<DriverSetupPayload> {
    return this.putJson(`/machines/${encodeURIComponent(id)}/driver-setup`, {
      values,
    });
  }

  putDriver(id: string, driver: string): Promise<DriverSetupPayload> {
    return this.putJson(`/machines/${encodeURIComponent(id)}/driver`, {
      driver,
    });
  }

  putHeads(id: string, heads: HeadPatch[]): Promise<HeadInfo[]> {
    return this.putJson(`/machines/${encodeURIComponent(id)}/heads`, {
      heads,
    });
  }

  getMacros(id: string): Promise<MacroInfo[]> {
    return this.get<unknown>(`/machines/${encodeURIComponent(id)}/macros`).then(
      (raw: unknown) => {
        const record = raw as { macros?: MacroInfo[] };
        return Array.isArray(record?.macros) ? record.macros : [];
      },
    );
  }

  createMacro(
    id: string,
    macro: { name: string; code: string; enabled: boolean },
  ): Promise<MacroInfo> {
    return this.postJson(`/machines/${encodeURIComponent(id)}/macros`, macro);
  }

  updateMacro(
    id: string,
    macroUid: string,
    patch: { name?: string; code?: string; enabled?: boolean },
  ): Promise<MacroInfo> {
    return this.putJson(
      `/machines/${encodeURIComponent(id)}/macros/${encodeURIComponent(macroUid)}`,
      patch,
    );
  }

  deleteMacro(id: string, macroUid: string): Promise<void> {
    return this.requestVoid(
      "DELETE",
      `/machines/${encodeURIComponent(id)}/macros/${encodeURIComponent(macroUid)}`,
    );
  }

  putWcs(
    id: string,
    slot: string,
    offset: number[],
    label?: string,
  ): Promise<WcsInfo> {
    return this.putJson(
      `/machines/${encodeURIComponent(id)}/wcs/${encodeURIComponent(slot)}`,
      { offset, ...(label !== undefined ? { label } : {}) },
    );
  }

  createNogoZone(
    id: string,
    zone: {
      name?: string;
      shape: string;
      params: Record<string, number>;
      enabled?: boolean;
    },
  ): Promise<NogoZoneInfo> {
    return this.postJson(`/machines/${encodeURIComponent(id)}/nogo-zones`, zone);
  }

  deleteNogoZone(id: string, zoneUid: string): Promise<void> {
    return this.requestVoid(
      "DELETE",
      `/machines/${encodeURIComponent(id)}/nogo-zones/${encodeURIComponent(zoneUid)}`,
    );
  }

  putRotaryModule(
    id: string,
    moduleUid: string,
    patch: {
      name?: string;
      default_diameter?: number;
      mm_per_rotation?: number;
      reverse_axis?: boolean;
    },
  ): Promise<RotaryModuleInfo[]> {
    return this.putJson(
      `/machines/${encodeURIComponent(id)}/rotary-modules/${encodeURIComponent(moduleUid)}`,
      patch,
    );
  }

  putLayer(uid: string, patch: LayerPatch): Promise<void> {
    return this.put(`/layers/${uid}`, patch);
  }

  postLayer(): Promise<unknown> {
    return this.postJson("/layers");
  }

  deleteLayer(uid: string): Promise<void> {
    return this.requestVoid("DELETE", `/layers/${uid}`);
  }

  activateLayer(uid: string): Promise<void> {
    return this.post(`/layers/${uid}/activate`);
  }

  reorderLayers(uids: string[]): Promise<void> {
    return this.postJson("/layers/reorder", { uids }).then(() => undefined);
  }

  getStepTypes(): Promise<StepTypeEntry[]> {
    return this.get<unknown>("/steps").then((raw) => {
      const list = Array.isArray((raw as { steps?: unknown[] }).steps)
        ? (raw as { steps: unknown[] }).steps
        : [];
      return list.map(normalizeStepTypeEntry);
    });
  }

  getStepSchema(uid: string): Promise<StepSchema> {
    return this.get<unknown>(`/steps/${uid}/schema`).then(
      normalizeStepSchema,
    );
  }

  postStep(layerUid: string, body: StepAddRequest): Promise<unknown> {
    return this.postJson(`/layers/${layerUid}/steps`, body);
  }

  putStep(uid: string, body: StepPatch): Promise<unknown> {
    return this.putJson(`/steps/${uid}`, body);
  }

  deleteStep(uid: string): Promise<void> {
    return this.requestVoid("DELETE", `/steps/${uid}`);
  }

  deleteItems(uids: string[]): Promise<void> {
    return this.postJson("/items/delete", { uids }).then(() => undefined);
  }

  duplicateItems(uids: string[]): Promise<{ created: string[] }> {
    return this.postJson<{ created: string[] }>("/items/duplicate", {
      uids,
    });
  }

  analyzeImport(file: File): Promise<ImportAnalysis> {
    return this.postForm("/import/analyze", file).then(
      normalizeImportAnalysis,
    );
  }

  importFile(
    file: File,
    placement: ImportPlacement = {},
  ): Promise<ImportResult> {
    const params = new URLSearchParams();
    if (placement.spec) {
      params.set("spec", JSON.stringify(placement.spec));
    }
    if (placement.x !== undefined && placement.y !== undefined) {
      params.set("x", String(placement.x));
      params.set("y", String(placement.y));
    }
    const query = params.toString();
    return this.postForm(`/import${query ? `?${query}` : ""}`, file);
  }

  openProject(file: File): Promise<DocTree> {
    return this.postForm("/project/open", file).then(normalizeDocTree);
  }

  async saveProject(): Promise<Blob> {
    const res = await fetch(this.url("/project/save"), {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new ApiError(res.status, `GET /project/save failed (${res.status})`);
    }
    return res.blob();
  }

  async exportGcode(): Promise<string> {
    const res = await fetch(this.url("/project/gcode"), {
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new ApiError(res.status, `GET /project/gcode failed (${res.status})`);
    }
    return res.text();
  }

  getMachines(): Promise<MachineListItem[]> {
    return this.get<unknown>("/machines").then(normalizeMachines);
  }

  getMachineState(): Promise<MachineState> {
    return this.get<unknown>("/machine/state").then(normalizeMachineState);
  }

  connectMachine(id: string): Promise<void> {
    return this.post(`/machines/${encodeURIComponent(id)}/connect`);
  }

  disconnectMachine(id: string): Promise<void> {
    return this.post(`/machines/${encodeURIComponent(id)}/disconnect`);
  }

  jog(direction: string, distanceMm: number, speed?: number): Promise<void> {
    return this.postJson("/machine/jog", {
      direction,
      distance_mm: distanceMm,
      ...(speed !== undefined && speed > 0 ? { speed: Math.round(speed) } : {}),
    }).then(() => undefined);
  }

  homeMachine(axes?: string[]): Promise<void> {
    return this.postJson("/machine/home", { axes: axes ?? [] }).then(
      () => undefined,
    );
  }

  moveTo(x: number, y: number, z?: number, speed?: number): Promise<void> {
    return this.postJson("/machine/move-to", {
      x,
      y,
      ...(z !== undefined ? { z } : {}),
      ...(speed !== undefined && speed > 0 ? { speed } : {}),
    }).then(() => undefined);
  }

  setPower(percent: number): Promise<void> {
    return this.postJson("/machine/set-power", { percent }).then(
      () => undefined,
    );
  }

  setFocusPower(percent: number): Promise<void> {
    return this.postJson("/machine/set-focus-power", { percent }).then(
      () => undefined,
    );
  }

  runRaw(gcode: string): Promise<void> {
    return this.postJson("/machine/run-raw", { gcode }).then(() => undefined);
  }

  clearAlarm(): Promise<void> {
    return this.post("/machine/clear-alarm");
  }

  setWorkOriginHere(axes?: string[]): Promise<void> {
    return this.postJson("/machine/set-work-origin-here", {
      axes: axes ?? [],
    }).then(() => undefined);
  }

  selectWcs(slot: string): Promise<void> {
    return this.postJson("/machine/select-wcs", { slot }).then(
      () => undefined,
    );
  }

  syncWcs(): Promise<void> {
    return this.post("/machine/sync-wcs");
  }

  frame(): Promise<void> {
    return this.postJson("/machine/frame", { include_overscan: false }).then(
      () => undefined,
    );
  }

  getJobStatus(): Promise<JobStatus> {
    return this.get<unknown>("/job/status").then(normalizeJobStatus);
  }

  getOps(tolerance = 0.05): Promise<OpsResponse> {
    const t = encodeURIComponent(tolerance);
    return this.get<unknown>(`/ops?tolerance=${t}`).then(normalizeOps);
  }

  startJob(): Promise<void> {
    return this.post("/job/start");
  }

  holdJob(hold: boolean): Promise<void> {
    return this.postJson("/job/hold", { hold }).then(() => undefined);
  }

  cancelJob(): Promise<void> {
    return this.post("/job/cancel");
  }

  getWsUrl(): string {
    const proto =
      typeof window !== "undefined" && window.location.protocol === "https:"
        ? "wss:"
        : "ws:";
    const host = typeof window === "undefined" ? "" : window.location.host;
    const suffix = this.token ? `?token=${encodeURIComponent(this.token)}` : "";
    return `${proto}//${host}${API_BASE}/ws${suffix}`;
  }

  private url(path: string): string {
    return `${API_BASE}${path}`;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    return headers;
  }

  private async get<T>(path: string): Promise<T> {
    const res = await fetch(this.url(path), { headers: this.headers() });
    return this.parse<T>(res, "GET", path);
  }

  private async post(path: string): Promise<void> {
    const res = await fetch(this.url(path), {
      method: "POST",
      headers: this.headers(),
    });
    await this.parse<unknown>(res, "POST", path);
  }

  private async postJson<T>(path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.url(path), {
      method: "POST",
      headers: { "Content-Type": "application/json", ...this.headers() },
      body: JSON.stringify(body ?? {}),
    });
    return this.parse<T>(res, "POST", path);
  }

  private async put(path: string, body: unknown): Promise<void> {
    await this.putJson(path, body);
  }

  private async putJson<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(this.url(path), {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...this.headers() },
      body: JSON.stringify(body),
    });
    return this.parse<T>(res, "PUT", path);
  }

  private async requestVoid(
    method: string,
    path: string,
  ): Promise<void> {
    const res = await fetch(this.url(path), {
      method,
      headers: this.headers(),
    });
    await this.parse<unknown>(res, method, path);
  }

  private async postForm<T>(path: string, file: File): Promise<T> {
    const form = new FormData();
    form.append("file", file, file.name);
    const res = await fetch(this.url(path), {
      method: "POST",
      headers: this.headers(),
      body: form,
    });
    return this.parse<T>(res, "POST", path);
  }

  private async parse<T>(
    res: Response,
    method: string,
    path: string,
  ): Promise<T> {
    if (!res.ok) {
      throw await this.errorFrom(res, method, path);
    }
    const text = await res.text();
    if (!text) return undefined as T;
    return JSON.parse(text) as T;
  }

  /** Prefers the backend's `detail` message over a generic phrase. */
  private async errorFrom(
    res: Response,
    method: string,
    path: string,
  ): Promise<ApiError> {
    let detail = "";
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((item) =>
            typeof item === "string"
              ? item
              : typeof (item as { msg?: unknown })?.msg === "string"
                ? (item as { msg: string }).msg
                : JSON.stringify(item),
          )
          .join("; ");
      }
    } catch {
      return new ApiError(
        res.status,
        `${method} ${path} failed (${res.status})`,
      );
    }
    const message = detail
      ? `${detail} (${res.status})`
      : `${method} ${path} failed (${res.status})`;
    return new ApiError(res.status, message);
  }
}
