import { api } from "./stores/api";
import {
  normalizeJobStatus,
  normalizeMachineState,
  normalizeViewport,
} from "./api/normalize";
import { ApiError } from "./api/client";
import {
  clearSelection,
  docStore,
  findWorkpiece,
  planDocEvent,
  selectUids,
  setDocHistory,
  setDocTree,
} from "./stores/docStore";
import { DocActions } from "./stores/docActions";
import {
  clearTransformPreview,
  setTransformPreview,
} from "./stores/interactionStore";
import {
  machineStore,
  setMachineState,
  setMachines,
  setViewport,
} from "./stores/machineStore";
import { jobStore, setJobStatus } from "./stores/jobStore";
import {
  opsStore,
  refreshOps,
  shouldRefetchOps,
  startPlaybackTicker,
} from "./stores/opsStore";
import { sampleAt, type Track } from "./stores/playback";
import { worldToPanel } from "./scene/coords";
import {
  pushToast,
  setProjectName,
  setShowToolpaths,
  setSystemInfo,
  setWsStatus,
  uiStore,
} from "./stores/uiStore";
import { connectWs, dispatchWsEvent, type EventConnection } from "./ws";
import { startMockEvents } from "./dev/mock";
import { downloadBlob } from "./download";
import { buildAppToolbar } from "./components/appToolbar";
import { buildLeftPanel } from "./components/leftPanel";
import { buildRightPanel } from "./components/rightPanel";
import { buildTopBar } from "./components/topBar";
import { openImportDialog } from "./components/importDialog";
import { buildAlarmBanner } from "./components/alarmBanner";
import { buildJobBar } from "./components/jobBar";
import { buildHistoryTimeline } from "./components/historyTimeline";
import {
  decideAutoLayout,
  pickLayout,
} from "./components/autoLayout";
import {
  openSettingsDialog,
  type SettingsSection,
} from "./components/settingsDialog";
import { openMachineSettingsDialog } from "./components/machineSettingsDialog";
import { openStepPicker } from "./components/stepPicker";
import { buildStatusBar, buildTransport } from "./components/statusBar";
import { toastContainer } from "./components/ui/Toast";
import { el } from "./components/ui/dom";
import type { ViewPreset } from "./scene/CameraRig";
import { SceneManager, type CommitItem } from "./scene/SceneManager";
import { xformOf } from "./scene/interaction/TransformController";
import type { AlignMode, FlipAxis, LayoutOption } from "./api/types";

export interface ShellActions {
  newDoc: () => void;
  openProject: () => void;
  saveProject: () => void;
  exportGcode: () => void;
  undo: () => void;
  redo: () => void;
  deleteSelected: () => void;
  duplicateSelected: () => void;
  flip: (axis: FlipAxis) => void;
  align: (mode: AlignMode) => void;
  autoLayout: (anchor: HTMLElement) => void;
  addStep: () => void;
  machineSettings: () => void;
  openSettings: () => void;
  importFile: () => void;
  setTool: (tool: string) => void;
  viewPreset: (preset: ViewPreset) => void;
  frameSelection: () => void;
  zoom: (deltaY: number) => void;
  toggleToolpaths: () => void;
  startJob: () => void;
  homeAll: () => void;
  setOriginHere: () => void;
  goToOrigin: () => void;
  refreshDoc: () => Promise<void>;
  about: () => void;
}

export interface ShellRefs {
  root: HTMLElement;
  viewportHost: HTMLElement;
  topbar: ReturnType<typeof buildTopBar>;
  toolbar: ReturnType<typeof buildAppToolbar>;
  leftPanel: ReturnType<typeof buildLeftPanel>;
  rightPanel: ReturnType<typeof buildRightPanel>;
  statusbar: ReturnType<typeof buildStatusBar>;
  transport: ReturnType<typeof buildTransport>;
}

export interface AppHandle {
  destroy(): void;
}

/** Debounce window for tree refetches during a history roll. */
const ROLL_TREE_REFETCH_MS = 300;

function noopActions(): ShellActions {
  return {
    newDoc: () => undefined,
    openProject: () => undefined,
    saveProject: () => undefined,
    exportGcode: () => undefined,
    undo: () => undefined,
    redo: () => undefined,
    deleteSelected: () => undefined,
    duplicateSelected: () => undefined,
    flip: () => undefined,
    align: () => undefined,
    autoLayout: () => undefined,
    addStep: () => undefined,
    machineSettings: () => undefined,
    openSettings: () => undefined,
    importFile: () => undefined,
    setTool: () => undefined,
    viewPreset: () => undefined,
    frameSelection: () => undefined,
    zoom: () => undefined,
    toggleToolpaths: () => undefined,
    startJob: () => undefined,
    homeAll: () => undefined,
    setOriginHere: () => undefined,
    goToOrigin: () => undefined,
    refreshDoc: async () => undefined,
    about: () => undefined,
  };
}

/** Nudges the orbit camera by a synthetic wheel event on the canvas. */
function zoomCanvas(deltaY: number): void {
  const canvas = document.querySelector<HTMLElement>(
    ".viewport-host canvas",
  );
  if (!canvas) return;
  canvas.dispatchEvent(
    new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true }),
  );
}

/** Pure DOM shell construction (no networking, no WebGL). */
export function buildShell(
  root: HTMLElement,
  actions: ShellActions = noopActions(),
): ShellRefs {
  root.classList.add("app");

  const topbar = buildTopBar({
    newDoc: actions.newDoc,
    openProject: actions.openProject,
    saveProject: actions.saveProject,
    exportGcode: actions.exportGcode,
    undo: actions.undo,
    redo: actions.redo,
    importFile: actions.importFile,
    duplicateSelected: actions.duplicateSelected,
    deleteSelected: actions.deleteSelected,
    viewPreset: actions.viewPreset,
    frameSelection: actions.frameSelection,
    toggleToolpaths: actions.toggleToolpaths,
    startJob: actions.startJob,
    openSettings: actions.openSettings,
    about: actions.about,
  });
  const toolbar = buildAppToolbar({
    setTool: actions.setTool,
    newDoc: actions.newDoc,
    openProject: actions.openProject,
    saveProject: actions.saveProject,
    importFile: actions.importFile,
    exportGcode: actions.exportGcode,
    duplicateSelected: actions.duplicateSelected,
    deleteSelected: actions.deleteSelected,
    flip: actions.flip,
    align: actions.align,
    autoLayout: (anchor) => actions.autoLayout(anchor),
    addStep: () => actions.addStep(),
    machineSettings: () => actions.machineSettings(),
    frameSelection: actions.frameSelection,
    viewPreset: actions.viewPreset,
    zoom: actions.zoom,
    startJob: actions.startJob,
    homeAll: actions.homeAll,
    setOriginHere: actions.setOriginHere,
    goToOrigin: actions.goToOrigin,
    about: actions.about,
  });
  const leftPanel = buildLeftPanel({
    refreshDoc: actions.refreshDoc,
  });
  const rightPanel = buildRightPanel({
    refreshDoc: actions.refreshDoc,
    newDoc: actions.newDoc,
    deleteSelected: actions.deleteSelected,
    duplicateSelected: actions.duplicateSelected,
    flip: actions.flip,
    align: actions.align,
  });
  const statusbar = buildStatusBar();

  const viewportHost = el("div", "viewport-host");
  const transport = buildTransport();
  viewportHost.appendChild(transport.root);
  viewportHost.appendChild(buildJobBar());
  viewportHost.appendChild(buildAlarmBanner());

  root.append(
    viewportHost,
    topbar.root,
    toolbar.root,
    leftPanel.root,
    leftPanel.stub,
    rightPanel.root,
    rightPanel.stub,
    statusbar.root,
  );
  root.appendChild(toastContainer());

  return {
    root,
    viewportHost,
    topbar,
    toolbar,
    leftPanel,
    rightPanel,
    statusbar,
    transport,
  };
}

export function startApp(root: HTMLElement): AppHandle {
  let scene: SceneManager | null = null;
  let connection: EventConnection | null = null;

  async function refetchViewport(): Promise<void> {
    try {
      setViewport(await api.getViewport());
    } catch (err) {
      pushToast("error", `Viewport fetch failed: ${describe(err)}`);
    }
  }

  async function refetchTree(): Promise<void> {
    try {
      setDocTree(await api.getDocTree());
      await refetchHistory();
    } catch (err) {
      pushToast("error", `Document fetch failed: ${describe(err)}`);
    }
  }

  async function refetchHistory(): Promise<void> {
    try {
      setDocHistory(await api.getDocHistory());
    } catch {
      return;
    }
  }

  async function refetchMachine(): Promise<void> {
    try {
      setMachineState(await api.getMachineState());
    } catch {
      return;
    }
  }

  async function refetchJob(): Promise<void> {
    try {
      setJobStatus(await api.getJobStatus());
    } catch {
      return;
    }
  }

  function newDoc(): void {
    api
      .postNewDoc()
      .then(() => {
        setProjectName("Untitled");
        return refetchTree();
      })
      .catch((err: unknown) =>
        pushToast("error", `New document failed: ${describe(err)}`),
      );
  }

  function openProject(): void {
    const input = el("input") as HTMLInputElement;
    input.type = "file";
    input.accept = ".ryp";
    input.style.display = "none";
    input.addEventListener("change", () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) return;
      api
        .openProject(file)
        .then(async (tree) => {
          setDocTree(tree);
          await refetchHistory();
          setProjectName(file.name);
          pushToast("success", `Opened ${file.name}`);
        })
        .catch((err: unknown) =>
          pushToast("error", `Opening project failed: ${describe(err)}`),
        );
    });
    document.body.appendChild(input);
    input.click();
  }

  function saveProject(): void {
    api
      .saveProject()
      .then((blob) => {
        downloadBlob(blob, "project.ryp");
        pushToast("success", "Project saved");
      })
      .catch((err: unknown) =>
        pushToast("error", `Saving project failed: ${describe(err)}`),
      );
  }

  function exportGcode(): void {
    api
      .exportGcode()
      .then((gcode) => {
        downloadBlob(
          new Blob([gcode], { type: "text/plain" }),
          "gcode.g",
        );
        pushToast("success", "G-code exported");
      })
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.status === 404) {
          pushToast("warning", "No G-code generated yet");
        } else {
          pushToast("error", `G-code export failed: ${describe(err)}`);
        }
      });
  }

  function undo(): void {
    docActions.undo();
  }

  function redo(): void {
    docActions.redo();
  }

  let rollTreeTimer: ReturnType<typeof setTimeout> | null = null;

  /** Batches tree refetches while the history timeline rolls. */
  function scheduleRollTreeRefetch(): void {
    if (rollTreeTimer) clearTimeout(rollTreeTimer);
    rollTreeTimer = setTimeout(() => {
      rollTreeTimer = null;
      void refetchTree();
    }, ROLL_TREE_REFETCH_MS);
  }

  function cancelRollTreeRefetch(): void {
    if (rollTreeTimer === null) return;
    clearTimeout(rollTreeTimer);
    rollTreeTimer = null;
  }

  async function runAutoLayout(anchor: HTMLElement): Promise<void> {
    let layouts: LayoutOption[];
    try {
      layouts = await api.getLayouts();
    } catch (err) {
      pushToast("error", `Loading layouts failed: ${describe(err)}`);
      return;
    }
    const decision = decideAutoLayout(layouts);
    if (decision.kind === "none") {
      pushToast("warning", "No layout strategies available");
      return;
    }
    const target =
      decision.kind === "run"
        ? decision.layout
        : await pickLayout(anchor, decision.layouts);
    if (!target) return;
    try {
      await api.runLayout(target.id);
      await refetchTree();
      pushToast("success", `${target.label} applied`);
    } catch (err) {
      pushToast("error", `Auto layout failed: ${describe(err)}`);
    }
  }

  /** Active layer uid from the local tree cache (first layer fallback). */
  function activeLayerUid(): string | null {
    const { layers } = docStore.getState();
    return layers.find((layer) => layer.is_active)?.uid ?? layers[0]?.uid ?? null;
  }

  function addStepToActiveLayer(): void {
    const uid = activeLayerUid();
    if (!uid) {
      pushToast("warning", "No layer to add a step to");
      return;
    }
    const layer = docStore
      .getState()
      .layers.find((candidate) => candidate.uid === uid);
    void openStepPicker(
      { uid, name: layer?.name ?? "Layer" },
      () => refetchTree(),
    );
  }

  const APP_SETTINGS_SECTIONS: SettingsSection[] = [
    {
      label: "General",
      fields: [
        {
          key: "theme",
          label: "Theme",
          kind: "choice",
          options: [
            { value: "system", label: "System" },
            { value: "light", label: "Light" },
            { value: "dark", label: "Dark" },
          ],
        },
        { key: "language", label: "Language", kind: "choice", options: [
          { value: "", label: "System default" },
          { value: "en", label: "English" },
          { value: "de", label: "Deutsch" },
          { value: "es", label: "Español" },
          { value: "fr", label: "Français" },
          { value: "hi", label: "हिन्दी" },
          { value: "pt", label: "Português" },
          { value: "uk", label: "Українська" },
          { value: "zh_CN", label: "中文" },
        ] },
        {
          key: "check_for_app_updates",
          label: "Check for updates",
          kind: "switch",
        },
      ],
      note:
        "Theme and language apply after reloading the page.",
    },
    {
      label: "Units",
      fields: [
        {
          key: "unit_preferences.length",
          label: "Length",
          kind: "choice",
          options: [
            { value: "mm", label: "mm" },
            { value: "cm", label: "cm" },
            { value: "m", label: "m" },
            { value: "in", label: "in" },
            { value: "ft", label: "ft" },
          ],
        },
        {
          key: "unit_preferences.speed",
          label: "Speed",
          kind: "choice",
          options: [
            { value: "mm/min", label: "mm/min" },
            { value: "mm/s", label: "mm/s" },
            { value: "in/min", label: "in/min" },
            { value: "in/s", label: "in/s" },
          ],
        },
        {
          key: "unit_preferences.acceleration",
          label: "Acceleration",
          kind: "choice",
          options: [
            { value: "mm/s²", label: "mm/s²" },
            { value: "cm/s²", label: "cm/s²" },
            { value: "m/s²", label: "m/s²" },
            { value: "in/s²", label: "in/s²" },
            { value: "ft/s²", label: "ft/s²" },
          ],
        },
      ],
    },
    {
      label: "Import",
      fields: [
        { key: "import_dpi", label: "Import DPI", kind: "number" },
        {
          key: "watch_source_files",
          label: "Watch source files",
          kind: "switch",
        },
      ],
    },
    {
      label: "Advanced",
      fields: [
        {
          key: "ops_color_mode",
          label: "Ops color source",
          kind: "choice",
          options: [
            { value: "laser", label: "Laser power" },
            { value: "layer", label: "Layer color" },
          ],
        },
        {
          key: "cache_budget_bytes",
          label: "Cache budget (bytes)",
          kind: "number",
        },
      ],
    },
  ];

  function runAppSettings(): void {
    void openSettingsDialog({
      title: "App settings",
      sections: APP_SETTINGS_SECTIONS,
      load: () =>
        api.getSettings() as unknown as Promise<Record<string, unknown>>,
      save: (changed) => {
        const mapped = { ...changed };
        if (mapped.language === "") mapped.language = null;
        return api.putSettings(
          mapped as Parameters<typeof api.putSettings>[0],
        ) as unknown as Promise<Record<string, unknown>>;
      },
      savedMessage: "Settings saved",
    });
  }

  function runMachineSettings(): void {
    const machineId = machineStore.getState().activeMachineId;
    if (!machineId) {
      pushToast("warning", "No active machine");
      return;
    }
    void openMachineSettingsDialog({
      machineId,
      onSaved: () => {
        void refetchViewport();
        void refetchMachine();
      },
    });
  }

  const docActions = new DocActions({
    api,
    refreshDoc: () => refetchTree(),
    onError: (message) => pushToast("error", message),
  });

  /** Machine commands shared by the ribbon's Machine tab. */
  const machineActions = {
    async homeAll(): Promise<void> {
      try {
        await api.homeMachine();
        await refetchMachine();
      } catch (err) {
        pushToast("error", `Homing failed: ${describe(err)}`);
      }
    },
    async setOriginHere(): Promise<void> {
      try {
        await api.setWorkOriginHere();
        await refetchMachine();
        await refetchViewport();
      } catch (err) {
        pushToast("error", `Set work origin failed: ${describe(err)}`);
      }
    },
    async goToOrigin(): Promise<void> {
      const origin =
        machineStore.getState().viewport?.workarea_origin_panel;
      if (!origin) return;
      try {
        await api.moveTo(origin.x, origin.y);
        await refetchMachine();
      } catch (err) {
        pushToast("error", `Move failed: ${describe(err)}`);
      }
    },
  };

  const shell = buildShell(root, {
    newDoc,
    openProject,
    saveProject,
    exportGcode,
    undo,
    redo,
    deleteSelected: () => docActions.deleteSelected(),
    duplicateSelected: () => docActions.duplicateSelected(),
    flip: (axis) => docActions.flipSelected(axis),
    align: (mode) => docActions.alignSelected(mode),
    autoLayout: (anchor) => void runAutoLayout(anchor),
    addStep: () => addStepToActiveLayer(),
    machineSettings: () => void runMachineSettings(),
    importFile: () => openImportDialog({ refreshDoc: () => refetchTree() }),
    setTool: () => undefined,
    viewPreset: (preset) => scene?.viewPreset(preset),
    frameSelection: () => {
      api
        .frame()
        .then(() => pushToast("info", "Framing the workpiece on the machine"))
        .catch((err: unknown) =>
          pushToast("error", `Frame failed: ${describe(err)}`),
        );
    },
    zoom: (deltaY) => zoomCanvas(deltaY),
    toggleToolpaths: () =>
      setShowToolpaths(!uiStore.getState().showToolpaths),
    startJob: () => {
      api
        .startJob()
        .catch((err: unknown) =>
          pushToast("error", `Start failed: ${describe(err)}`),
        );
    },
    openSettings: () => void runAppSettings(),
    homeAll: () => void machineActions.homeAll(),
    setOriginHere: () => void machineActions.setOriginHere(),
    goToOrigin: () => void machineActions.goToOrigin(),
    refreshDoc: () => refetchTree(),
    about: () => {
      const info = uiStore.getState().systemInfo;
      pushToast(
        "info",
        info
          ? `Rayforge ${info.version} · API ${info.api_version} · Python ${info.python_version}`
          : "Rayforge",
      );
    },
  });

  try {
    scene = new SceneManager(shell.viewportHost, api, {
      onClick: (hitUid, ev) => {
        if (hitUid) selectUids([hitUid], ev.shiftKey);
        else clearSelection();
      },
      onDragSelect: (uid) => selectUids([uid]),
      onPreview: (uid, transform) => {
        if (!uid || !transform) {
          clearTransformPreview();
          return;
        }
        setTransformPreview({
          uid,
          pos: transform.pos_mm,
          size: transform.size_mm,
          angleDeg: transform.angle_deg,
        });
      },
      onCommit: (items: CommitItem[]) => {
        commitTransforms(items);
      },
      getTransform: (uid) => {
        const found = findWorkpiece(docStore.getState().layers, uid);
        return found ? xformOf(found.workpiece) : null;
      },
    });
    shell.viewportHost.classList.add("has-canvas");
  } catch (err) {
    pushToast("error", `WebGL unavailable: ${describe(err)}`);
  }

  shell.viewportHost.appendChild(
    buildHistoryTimeline({
      api,
      onRollStart: () => undefined,
      onRollEnd: () => {
        cancelRollTreeRefetch();
        void refetchTree();
      },
      onError: (message) => pushToast("error", message),
    }).root,
  );

  function applyVisibilityFlags(): void {
    const state = uiStore.getState();
    scene?.setTravelVisible(state.showTravelMoves);
    scene?.setImageVisible(state.showWorkpieceImages);
    scene?.setNogoZonesVisible(state.showNogoZones);
    scene?.setStockVisible(state.showStock);
  }
  applyVisibilityFlags();
  uiStore.subscribe(applyVisibilityFlags);

  function commitTransforms(items: CommitItem[]): void {
    Promise.all(
      items.map((item) => api.putItemTransform(item.uid, item.transform)),
    )
      .then(() => refetchTree())
      .catch((err: unknown) => {
        scene?.discardPreviews();
        pushToast("error", `Transform failed: ${describe(err)}`);
      });
  }

  let lastLayers: unknown = null;
  docStore.subscribe((state) => {
    scene?.setSelection(state.selection);
    if (state.layers !== lastLayers) {
      lastLayers = state.layers;
      void scene?.syncTree({
        rev: state.rev,
        layers: state.layers,
        assets_count: state.assetsCount,
        stock_items: state.stocks,
      });
    }
  });

  /** Playhead position for a track distance, in panel mm (or null). */
  function trackPointPanel(
    track: Track,
    positionMm: number,
  ): ReturnType<typeof worldToPanel> | null {
    const matrix =
      machineStore.getState().viewport?.world_to_panel ?? null;
    const sample = sampleAt(track, positionMm);
    if (!sample || !matrix) return null;
    return worldToPanel(matrix, { x: sample.x, y: sample.y });
  }

  function syncOpsScene(): void {
    const ops = opsStore.getState();
    const job = jobStore.getState();
    const machine = machineStore.getState();

    if (machine.viewport) scene?.setViewport(machine.viewport);
    scene?.setBeam(
      machine.machineState?.beam_position_panel ??
        machine.viewport?.beam_position_panel ??
        null,
    );

    scene?.setOpsPreview(
      ops.available
        ? { generationId: ops.generationId ?? "", segments: ops.segments }
        : null,
    );
    scene?.setOpsVisible(uiStore.getState().showToolpaths && ops.available);

    const track = ops.track;
    let playhead: ReturnType<typeof worldToPanel> | null = null;
    if (track) {
      if (job.active) {
        playhead = trackPointPanel(track, job.progressFraction * track.totalDistance);
      } else if (ops.playing || ops.scrubbing) {
        playhead = trackPointPanel(track, ops.positionMm);
        scene?.setBeam(null);
      }
    }
    scene?.setPlayhead(playhead);
  }

  machineStore.subscribe(syncOpsScene);
  jobStore.subscribe(syncOpsScene);
  opsStore.subscribe(syncOpsScene);
  uiStore.subscribe((state) => {
    scene?.setOpsVisible(state.showToolpaths && opsStore.getState().available);
  });

  uiStore.subscribe((state) => {
    const pill = shell.topbar.connectionPill;
    const text = shell.topbar.connectionText;
    pill.classList.toggle("connected", state.wsStatus === "open");
    pill.classList.toggle("disconnected", state.wsStatus !== "open");
    const mock = state.mockMode ? " (mock)" : "";
    text.textContent =
      state.wsStatus === "open" ? `Live${mock}` : `Offline${mock}`;
  });
  shell.topbar.connectionPill.addEventListener("click", () => {
    pushToast("info", `Event stream: ${uiStore.getState().wsStatus}`);
  });

  const onKeyEvent = (ev: KeyboardEvent) => {
    docActions.handleKeydown(ev);
    if (ev.defaultPrevented) return;
    if (!(ev.ctrlKey || ev.metaKey)) return;
    const key = ev.key.toLowerCase();
    if (key === "n") {
      ev.preventDefault();
      newDoc();
    } else if (key === "s") {
      ev.preventDefault();
      saveProject();
    } else if (key === "o") {
      ev.preventDefault();
      openProject();
    } else if (key === "i") {
      ev.preventDefault();
      openImportDialog({ refreshDoc: () => refetchTree() });
    }
  };
  window.addEventListener("keydown", onKeyEvent);

  function handleEvent(ev: Parameters<typeof dispatchWsEvent>[0]): void {
    dispatchWsEvent(ev, {
      onDoc: (docEv) => {
        const plan = planDocEvent(docEv, docStore.getState().rev);
        if (docStore.getState().rolling && plan.refetchTree) {
          scheduleRollTreeRefetch();
        } else if (plan.refetchTree) {
          void refetchTree();
        } else {
          void refetchHistory();
        }
      },
      onViewport: (vp) => setViewport(normalizeViewport(vp)),
      onMachine: (st) => setMachineState(normalizeMachineState(st)),
      onJob: (job) => {
        const status = normalizeJobStatus(job);
        const wasActive = jobStore.getState().active;
        setJobStatus(status);
        if (shouldRefetchOps(wasActive, status.active)) void refreshOps();
      },
      onNotify: (level, message) => {
        const toastLevel =
          level === "error" || level === "warning" || level === "success"
            ? level
            : "info";
        pushToast(toastLevel, message);
      },
    });
  }

  function openEvents(): void {
    if (uiStore.getState().mockMode) {
      connection = startMockEvents(handleEvent);
      setWsStatus("open");
      return;
    }
    connection = connectWs(
      api.getWsUrl(),
      { onEvent: handleEvent, onStatus: setWsStatus },
      () => {
        void refetchViewport();
        void refetchTree();
        void refetchMachine();
        void refetchJob();
        void refreshOps();
      },
    );
  }

  async function boot(): Promise<void> {
    try {
      setSystemInfo(await api.getSystemInfo());
    } catch (err) {
      pushToast("warning", `Backend not reachable: ${describe(err)}`);
    }
    await refetchViewport();
    await refetchTree();
    try {
      setMachines(await api.getMachines());
    } catch {
      pushToast("warning", "Machine list unavailable");
    }
    await refetchMachine();
    await refetchJob();
    await refreshOps();
    openEvents();
  }

  const stopPlaybackTicker = startPlaybackTicker();
  void boot();

  return {
    destroy() {
      stopPlaybackTicker();
      cancelRollTreeRefetch();
      connection?.close();
      window.removeEventListener("keydown", onKeyEvent);
      scene?.dispose();
      root.classList.remove("app");
      root.replaceChildren();
    },
  };
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
