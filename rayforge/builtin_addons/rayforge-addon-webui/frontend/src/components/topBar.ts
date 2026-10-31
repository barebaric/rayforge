import { docStore } from "../stores/docStore";
import {
  setShowNogoZones,
  setShowTravelMoves,
  setShowWorkpieceImages,
  uiStore,
} from "../stores/uiStore";
import { buildCommandPalette, type CommandEntry } from "./commandPalette";
import { button } from "./ui/Button";
import { el } from "./ui/dom";
import { icon } from "./ui/icons";

export interface TopBarActions {
  undo: () => void;
  redo: () => void;
  newDoc: () => void;
  openProject: () => void;
  saveProject: () => void;
  exportGcode: () => void;
  importFile: () => void;
  duplicateSelected: () => void;
  deleteSelected: () => void;
  viewPreset: (preset: "fit" | "top" | "front" | "iso") => void;
  frameSelection: () => void;
  toggleToolpaths: () => void;
  startJob: () => void;
  openSettings: () => void;
  about: () => void;
}

export interface TopBarRefs {
  root: HTMLElement;
  connectionPill: HTMLElement;
  connectionText: HTMLElement;
  projectTab: HTMLElement;
  commandInput: HTMLInputElement;
}

/**
 * Minimal top bar in the Solidcraft style: brand and connection pill
 * on the left, centered project tab, undo/redo and the command
 * palette on the right. Dropdown menus are gone; commands live in
 * the floating toolbar and this palette.
 */
export function buildTopBar(actions: TopBarActions): TopBarRefs {
  const root = el("div", "topbar");

  const brand = el("div", "topbar-brand");
  brand.append(icon("box", 16));
  const brandName = el("span");
  brandName.textContent = "Rayforge";
  brand.appendChild(brandName);

  const connectionPill = el("div", "conn-pill disconnected");
  const dot = el("span", "conn-dot");
  const connectionText = el("span", "conn-text");
  connectionText.textContent = "Disconnected";
  connectionPill.append(dot, connectionText);

  const left = el("div", "topbar-left");
  left.append(brand, connectionPill);
  root.appendChild(left);

  const projectTab = el("div", "project-tab");
  const projectName = el("span", "project-name");
  projectName.textContent = uiStore.getState().projectName;
  projectTab.appendChild(projectName);

  uiStore.subscribe((state) => {
    projectName.textContent = state.projectName;
  });
  docStore.subscribe((state) => {
    projectTab.title = state.dirty ? "Unsaved changes" : uiStore.getState().projectName;
    projectTab.classList.toggle("dirty", state.dirty);
  });

  const undoBtn = button({
    iconName: "undo",
    variant: "icon",
    title: "Undo (Ctrl+Z)",
    onClick: actions.undo,
  });
  const redoBtn = button({
    iconName: "redo",
    variant: "icon",
    title: "Redo (Ctrl+Y)",
    onClick: actions.redo,
  });
  const syncEditButtons = (): void => {
    const { canUndo, canRedo } = docStore.getState();
    undoBtn.disabled = !canUndo;
    redoBtn.disabled = !canRedo;
  };
  syncEditButtons();
  docStore.subscribe(syncEditButtons);

  const commands: CommandEntry[] = [
    { label: "New project", accelerator: "Ctrl+N", iconName: "filePlus", run: actions.newDoc },
    { label: "Open project", accelerator: "Ctrl+O", iconName: "folder", run: actions.openProject },
    { label: "Save project", accelerator: "Ctrl+S", iconName: "save", run: actions.saveProject },
    { label: "Export G-code", iconName: "save", run: actions.exportGcode },
    { label: "Import file", accelerator: "Ctrl+I", iconName: "upload", run: actions.importFile },
    { label: "Undo", accelerator: "Ctrl+Z", iconName: "undo", run: actions.undo },
    { label: "Redo", accelerator: "Ctrl+Y", iconName: "redo", run: actions.redo },
    { label: "Duplicate selection", accelerator: "Ctrl+D", iconName: "copy", run: actions.duplicateSelected },
    { label: "Delete selection", accelerator: "Del", iconName: "trash", run: actions.deleteSelected },
    { label: "Frame workarea", accelerator: "Home", iconName: "frame", run: actions.frameSelection },
    { label: "Fit view", iconName: "fit", run: () => actions.viewPreset("fit") },
    { label: "Top view", accelerator: "1", iconName: "viewTop", run: () => actions.viewPreset("top") },
    { label: "Front view", accelerator: "2", iconName: "viewFront", run: () => actions.viewPreset("front") },
    { label: "Isometric view", accelerator: "3", iconName: "viewIso", run: () => actions.viewPreset("iso") },
    { label: "Toggle toolpaths", iconName: "eye", run: actions.toggleToolpaths },
    {
      label: "Toggle travel moves",
      iconName: "eye",
      run: () => setShowTravelMoves(!uiStore.getState().showTravelMoves),
    },
    {
      label: "Toggle workpiece images",
      iconName: "eye",
      run: () =>
        setShowWorkpieceImages(!uiStore.getState().showWorkpieceImages),
    },
    {
      label: "Toggle no-go zones",
      iconName: "eye",
      run: () => setShowNogoZones(!uiStore.getState().showNogoZones),
    },
    { label: "Start job", iconName: "play", run: actions.startJob },
    { label: "About Rayforge", iconName: "info", run: actions.about },
  ];
  const palette = buildCommandPalette(commands);

  const settingsBtn = button({
    iconName: "sliders",
    variant: "icon",
    title: "App settings",
    onClick: actions.openSettings,
  });
  const actionsHost = el("div", "topbar-actions");
  actionsHost.append(undoBtn, redoBtn, settingsBtn, palette.root);

  root.append(projectTab, actionsHost);

  return {
    root,
    connectionPill,
    connectionText,
    projectTab,
    commandInput: palette.input,
  };
}
