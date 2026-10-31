import { createStore } from "zustand/vanilla";
import type { SystemInfo } from "../api/types";
import { isMockMode } from "../env";

export type ToastLevel = "info" | "success" | "warning" | "error";
export type LeftTab = "document" | "machine";

export interface Toast {
  id: number;
  level: ToastLevel;
  message: string;
}

interface UiState {
  activeLeftTab: LeftTab;
  toasts: Toast[];
  systemInfo: SystemInfo | null;
  wsStatus: "connecting" | "open" | "closed";
  mockMode: boolean;
  projectName: string;
  showToolpaths: boolean;
  showTravelMoves: boolean;
  showWorkpieceImages: boolean;
  showNogoZones: boolean;
  showStock: boolean;
}

const VIEW_SETTINGS_KEY = "rayforge.webui.view";

function readViewSetting(name: string, fallback: boolean): boolean {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const raw = localStorage.getItem(VIEW_SETTINGS_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return typeof parsed[name] === "boolean" ? (parsed[name] as boolean) : fallback;
  } catch {
    return fallback;
  }
}

function writeViewSetting(name: string, value: boolean): void {
  if (typeof localStorage === "undefined") return;
  try {
    const raw = localStorage.getItem(VIEW_SETTINGS_KEY);
    const parsed = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    localStorage.setItem(
      VIEW_SETTINGS_KEY,
      JSON.stringify({ ...parsed, [name]: value }),
    );
  } catch {
    return;
  }
}

export const uiStore = createStore<UiState>()(() => ({
  activeLeftTab: "document",
  toasts: [],
  systemInfo: null,
  wsStatus: "closed",
  mockMode: isMockMode(),
  projectName: "Untitled",
  showToolpaths: true,
  showTravelMoves: readViewSetting("showTravelMoves", true),
  showWorkpieceImages: readViewSetting("showWorkpieceImages", true),
  showNogoZones: readViewSetting("showNogoZones", true),
  showStock: readViewSetting("showStock", true),
}));

let toastSeq = 0;

export function pushToast(level: ToastLevel, message: string): number {
  const id = ++toastSeq;
  uiStore.setState((state) => ({
    toasts: [...state.toasts, { id, level, message }].slice(-5),
  }));
  return id;
}

export function dismissToast(id: number): void {
  uiStore.setState((state) => ({
    toasts: state.toasts.filter((toast) => toast.id !== id),
  }));
}

export function setLeftTab(tab: LeftTab): void {
  uiStore.setState({ activeLeftTab: tab });
}

export function setSystemInfo(info: SystemInfo): void {
  uiStore.setState({ systemInfo: info });
}

export function setWsStatus(status: UiState["wsStatus"]): void {
  uiStore.setState({ wsStatus: status });
}

/** Sets the display name shown in the top-bar project tab. */
export function setProjectName(name: string): void {
  uiStore.setState({ projectName: name });
}

/** Toggles toolpath preview visibility in the 3D scene. */
export function setShowToolpaths(show: boolean): void {
  uiStore.setState({ showToolpaths: show });
}

/** Toggles dashed travel-move lines inside the toolpath preview. */
export function setShowTravelMoves(show: boolean): void {
  writeViewSetting("showTravelMoves", show);
  uiStore.setState({ showTravelMoves: show });
}

/** Toggles workpiece raster textures (outlines stay visible). */
export function setShowWorkpieceImages(show: boolean): void {
  writeViewSetting("showWorkpieceImages", show);
  uiStore.setState({ showWorkpieceImages: show });
}

/** Toggles machine no-go zone overlays on the bed. */
export function setShowNogoZones(show: boolean): void {
  writeViewSetting("showNogoZones", show);
  uiStore.setState({ showNogoZones: show });
}

/** Toggles stock material shapes in the 3D scene. */
export function setShowStock(show: boolean): void {
  writeViewSetting("showStock", show);
  uiStore.setState({ showStock: show });
}
