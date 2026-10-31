import { docStore } from "../stores/docStore";
import { jobStore } from "../stores/jobStore";
import { machineStore } from "../stores/machineStore";
import {
  setShowNogoZones,
  setShowStock,
  setShowToolpaths,
  setShowTravelMoves,
  setShowWorkpieceImages,
  uiStore,
} from "../stores/uiStore";
import { deriveJobCommands } from "./control/jobState";
import { clearElement, el } from "./ui/dom";
import { icon, type IconName } from "./ui/icons";

export interface RibbonActions {
  setTool: (tool: string) => void;
  newDoc: () => void;
  openProject: () => void;
  saveProject: () => void;
  importFile: () => void;
  exportGcode: () => void;
  duplicateSelected: () => void;
  deleteSelected: () => void;
  flip: (axis: "horizontal" | "vertical") => void;
  align: (
    mode:
      | "left"
      | "right"
      | "top"
      | "bottom"
      | "center_horizontal"
      | "center_vertical",
  ) => void;
  autoLayout: (anchor: HTMLElement) => void;
  addStep: () => void;
  machineSettings: () => void;
  frameSelection: () => void;
  viewPreset: (preset: "fit" | "top" | "front" | "iso") => void;
  zoom: (deltaY: number) => void;
  startJob: () => void;
  homeAll: () => void;
  setOriginHere: () => void;
  goToOrigin: () => void;
  about: () => void;
}

export interface AppToolbarRefs {
  root: HTMLElement;
}

interface RibbonItem {
  iconName: IconName;
  label: string;
  title: string;
  active?: boolean;
  disabled?: boolean;
  onClick?: (target: HTMLElement) => void;
}

interface RibbonGroup {
  label: string;
  items: RibbonItem[];
}

interface RibbonTab {
  id: string;
  label: string;
  groups: () => RibbonGroup[];
}

export function buildAppToolbar(actions: RibbonActions): AppToolbarRefs {
  const root = el("div", "floating-toolbar ribbon");
  let activeTool = "select";
  let activeTab: string = "model";

  function item(def: RibbonItem): HTMLElement {
    const btn = el("button", "ribbon-btn");
    btn.type = "button";
    btn.title = def.title;
    btn.appendChild(icon(def.iconName, 18));
    const label = el("span", "ribbon-btn-label");
    label.textContent = def.label;
    btn.appendChild(label);
    if (def.active) btn.classList.add("active");
    if (def.disabled) btn.disabled = true;
    if (def.onClick) {
      btn.addEventListener("click", () =>
        def.onClick?.(btn as HTMLElement),
      );
    }
    return btn;
  }

  function buildTabBar(tabs: RibbonTab[]): HTMLElement {
    const bar = el("div", "ribbon-tabs");
    for (const tab of tabs) {
      const btn = el("button", "ribbon-tab");
      btn.type = "button";
      btn.textContent = tab.label;
      if (tab.id === activeTab) btn.classList.add("active");
      btn.addEventListener("click", () => {
        activeTab = tab.id;
        rebuild();
      });
      bar.appendChild(btn);
    }
    return bar;
  }

  function buildBody(tab: RibbonTab): HTMLElement {
    const body = el("div", "ribbon-body");
    tab.groups().forEach((group, index) => {
      if (index > 0) body.appendChild(el("div", "ribbon-sep"));
      const groupEl = el("div", "ribbon-group");
      const items = el("div", "ribbon-items");
      for (const def of group.items) items.appendChild(item(def));
      const label = el("div", "ribbon-group-label");
      label.textContent = group.label;
      groupEl.append(items, label);
      body.appendChild(groupEl);
    });
    return body;
  }

  function rebuild(): void {
    const hasSelection = docStore.getState().selection.length > 0;
    const job = jobStore.getState();
    const { machineState, machines } = machineStore.getState();
    const hasContent = docStore
      .getState()
      .layers.some((layer) => layer.workpieces.length > 0);
    const jobCommands = deriveJobCommands({
      active: job.active,
      phase: job.phase,
      connected: machineState?.connected ?? false,
      hasContent,
    });
    const machineReady = machines.length > 0;
    const showToolpaths = uiStore.getState().showToolpaths;
    const showTravelMoves = uiStore.getState().showTravelMoves;
    const showWorkpieceImages = uiStore.getState().showWorkpieceImages;
    const showNogoZones = uiStore.getState().showNogoZones;
    const showStock = uiStore.getState().showStock;

    const tabs: RibbonTab[] = [
      {
        id: "model",
        label: "Model",
        groups: () => [
          {
            label: "Tools",
            items: (
              [
                ["select", "pointer", "Select"],
                ["move", "move", "Move"],
                ["rotate", "rotate", "Rotate"],
                ["scale", "scale", "Scale"],
              ] as const
            ).map(([id, iconName, label]) => ({
              iconName,
              label,
              title: label,
              active: activeTool === id,
              onClick: () => {
                activeTool = id;
                rebuild();
                actions.setTool(id);
              },
            })),
          },
          {
            label: "Project",
            items: [
              {
                iconName: "filePlus",
                title: "New project (Ctrl+N)",
                label: "New",
                onClick: actions.newDoc,
              },
              {
                iconName: "folder",
                title: "Open project (Ctrl+O)",
                label: "Open",
                onClick: actions.openProject,
              },
              {
                iconName: "save",
                title: "Save project (Ctrl+S)",
                label: "Save",
                onClick: actions.saveProject,
              },
              {
                iconName: "upload",
                title: "Import file (Ctrl+I)",
                label: "Import",
                onClick: actions.importFile,
              },
              {
                iconName: "save",
                title: "Export G-code",
                label: "Export",
                onClick: actions.exportGcode,
              },
            ],
          },
          {
            label: "Modify",
            items: [
              {
                iconName: "copy",
                title: "Duplicate selection (Ctrl+D)",
                label: "Duplicate",
                disabled: !hasSelection,
                onClick: actions.duplicateSelected,
              },
              {
                iconName: "trash",
                title: "Delete selection (Del)",
                label: "Delete",
                disabled: !hasSelection,
                onClick: actions.deleteSelected,
              },
              {
                iconName: "flipH",
                title: "Flip horizontally",
                label: "Flip H",
                disabled: !hasSelection,
                onClick: () => actions.flip("horizontal"),
              },
              {
                iconName: "flipV",
                title: "Flip vertically",
                label: "Flip V",
                disabled: !hasSelection,
                onClick: () => actions.flip("vertical"),
              },
            ],
          },
          {
            label: "Arrange",
            items: (
              [
                ["left", "alignLeft", "Left"],
                ["center_horizontal", "alignCenterH", "Center H"],
                ["right", "alignRight", "Right"],
                ["top", "alignTop", "Top"],
                ["center_vertical", "alignCenterV", "Center V"],
                ["bottom", "alignBottom", "Bottom"],
              ] as const
            ).map(([mode, iconName, label]) => ({
              iconName,
              label,
              title: `Align ${label}`,
              disabled: !hasSelection,
              onClick: () => actions.align(mode),
            })),
          },
          {
            label: "Layout",
            items: [
              {
                iconName: "grid",
                title: "Arrange all workpieces automatically",
                label: "Auto layout",
                disabled: !hasContent,
                onClick: (target) => actions.autoLayout(target),
              },
            ],
          },
        ],
      },
      {
        id: "machine",
        label: "Machine",
        groups: () => [
          {
            label: "Setup",
            items: [
              {
                iconName: "sliders",
                title: "Edit machine geometry and motion settings",
                label: "Machine settings",
                disabled: !machineReady,
                onClick: () => actions.machineSettings(),
              },
            ],
          },
          {
            label: "Positioning",
            items: [
              {
                iconName: "home",
                title: "Home all axes",
                label: "Home all",
                disabled: !machineReady,
                onClick: actions.homeAll,
              },
              {
                iconName: "crosshair",
                title: "Set work origin at the current position",
                label: "Set origin",
                disabled: !machineReady,
                onClick: actions.setOriginHere,
              },
              {
                iconName: "crosshairSmall",
                title: "Move to the work origin",
                label: "Go to origin",
                disabled: !machineReady,
                onClick: actions.goToOrigin,
              },
            ],
          },
        ],
      },
      {
        id: "job",
        label: "Job",
        groups: () => [
          {
            label: "Job",
            items: [
              {
                iconName: "zap",
                title: "Add a step to the active layer",
                label: "Add step",
                disabled: !hasContent,
                onClick: () => actions.addStep(),
              },
              {
                iconName: "play",
                title: "Start job",
                label: "Start",
                disabled: !jobCommands.startEnabled,
                onClick: actions.startJob,
              },
              {
                iconName: "frame",
                title: "Frame workarea (Home)",
                label: "Frame",
                onClick: actions.frameSelection,
              },
            ],
          },
          {
            label: "Output",
            items: [
              {
                iconName: "save",
                title: "Export G-code",
                label: "Export",
                onClick: actions.exportGcode,
              },
            ],
          },
        ],
      },
      {
        id: "view",
        label: "View",
        groups: () => [
          {
            label: "Views",
            items: [
              {
                iconName: "viewTop",
                title: "Top view (1)",
                label: "Top",
                onClick: () => actions.viewPreset("top"),
              },
              {
                iconName: "viewFront",
                title: "Front view (2)",
                label: "Front",
                onClick: () => actions.viewPreset("front"),
              },
              {
                iconName: "viewIso",
                title: "Isometric view (3)",
                label: "Iso",
                onClick: () => actions.viewPreset("iso"),
              },
              {
                iconName: "fit",
                title: "Zoom to fit (Home)",
                label: "Fit",
                onClick: () => actions.viewPreset("fit"),
              },
            ],
          },
          {
            label: "Show",
            items: [
              {
                iconName: "eye",
                title: "Toggle toolpaths",
                label: "Toolpaths",
                active: showToolpaths,
                onClick: () => setShowToolpaths(!showToolpaths),
              },
              {
                iconName: "eye",
                title: "Toggle travel moves",
                label: "Travels",
                active: showTravelMoves,
                onClick: () => setShowTravelMoves(!showTravelMoves),
              },
              {
                iconName: "eye",
                title: "Toggle workpiece images",
                label: "Images",
                active: showWorkpieceImages,
                onClick: () =>
                  setShowWorkpieceImages(!showWorkpieceImages),
              },
              {
                iconName: "eye",
                title: "Toggle no-go zones",
                label: "No-go zones",
                active: showNogoZones,
                onClick: () => setShowNogoZones(!showNogoZones),
              },
              {
                iconName: "eye",
                title: "Toggle stock material shapes",
                label: "Stock",
                active: showStock,
                onClick: () => setShowStock(!showStock),
              },
            ],
          },
          {
            label: "Zoom",
            items: [
              {
                iconName: "zoomOut",
                title: "Zoom out",
                label: "Out",
                onClick: () => actions.zoom(120),
              },
              {
                iconName: "zoomIn",
                title: "Zoom in",
                label: "In",
                onClick: () => actions.zoom(-120),
              },
            ],
          },
        ],
      },
      {
        id: "help",
        label: "Help",
        groups: () => [
          {
            label: "Help",
            items: [
              {
                iconName: "info",
                title: "About Rayforge",
                label: "About",
                onClick: actions.about,
              },
            ],
          },
        ],
      },
    ];

    const active = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];
    activeTab = active.id;
    clearElement(root);
    root.append(buildTabBar(tabs), buildBody(active));
  }

  rebuild();
  uiStore.subscribe(rebuild);
  docStore.subscribe(rebuild);
  machineStore.subscribe(rebuild);
  jobStore.subscribe(rebuild);
  return { root };
}
